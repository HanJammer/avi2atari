#!/usr/bin/env python3

# avi2atari.py - Atari 8-bit AVF Video Converter
# Copyright (C) 2026 HanJammer & Lumen
#
# A modern, all-in-one converter for AVF/BIN format on Atari 8-bit computers.
# Supports both PAL (50Hz) and NTSC (60Hz) output standards.
#
# Key Features:
# - Single file processing
# - Batch processing (directory support)
# - URL support (YouTube and direct video links) via yt-dlp
# - URL List processing via text file (--urllist filename.txt)
# - Downloads are saved to 'downloads/' folder and preserved
# - EBU R128 Audio Loudness Normalization (prevents silence or clipping on 8-bit DACs)
# - Error diffusion dithering (Floyd-Steinberg derivative)
# - Strict file structure integrity checks (prevents sync drift)
# - Auto-generation of silence for video-only inputs
# - Built-in test signal generator
#
# Original Concept: phaeron encvideo/encaudio/mux C++ sources)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import subprocess
import argparse
import glob
import time
import shutil
import numpy as np
import math
try:
    from numba import jit
except ImportError:
    # Numba is an accelerator, not a requirement: fall back to plain Python
    # (an order of magnitude slower, but correct) so mismatched numpy/numba
    # installs never block a conversion outright.
    def jit(*_args, **_kwargs):
        def _deco(fn):
            return fn
        return _deco
    print("WARNING: numba unavailable - encoding runs unaccelerated (slow).")

__version__ = "0.7"

import avf_format as avf

# Try importing yt_dlp for URL support
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# --- AVF FORMAT CONFIGURATION ---
# All container constants now live in avf_format (the shared PC-suite lib);
# these are aliases so the converter body reads the same as before.
HEIGHT = avf.HEIGHT
WIDTH = avf.WIDTH
FRAME_SIZE_BYTES = avf.FRAME_SIZE_APAC   # 17 sectors; byte-aligned or it drifts
C80_FRAME_SIZE = avf.FRAME_SIZE_C80      # 7680 video + 512 linear audio
PROFILE_APAC = avf.PROFILE_APAC
PROFILE_C80M3 = avf.PROFILE_C80M3
PROFILE_C80M4 = avf.PROFILE_C80M4
C80_HALFBLOCK = avf.C80_HALFBLOCK
SECONDS_PER_FRAME_PAL = avf.SECONDS_PER_FRAME_PAL
SECONDS_PER_FRAME_NTSC = avf.SECONDS_PER_FRAME_NTSC


# --- AVF2 METADATA HEADER --- (built by the shared avf_format lib)
def build_avf2_header(is_pal, frame_count, poster_data, meta, profile=PROFILE_APAC):
    # Thin wrapper over avf_format; the converter stamps its own version.
    m = dict(meta or {})
    m["converter"] = f"avi2atari {__version__}"
    return avf.build_avf2_header(is_pal, frame_count, poster_data, m, profile)


def probe_duration_seconds(input_file):
    # Best-effort media duration via ffprobe (ships with FFmpeg). None on failure.
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', input_file],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None

# --- PHASE TABLES (Precomputed) ---
# Atari generates colors via phase shift (Hue) and amplitude (Saturation).
# These tables map the digital color space to Atari's analog I/Q signal.
ITAB_PAL = np.zeros(16, dtype=np.float32)
QTAB_PAL = np.zeros(16, dtype=np.float32)
ITAB_NTSC = np.zeros(16, dtype=np.float32)
QTAB_NTSC = np.zeros(16, dtype=np.float32)

def init_tables():
    # Generates lookup tables for color conversion (Phase Shift).
    # PAL: 14 color cycles
    for i in range(15):
        ITAB_PAL[i+1] = 40.0 * math.cos(3.1415926535 * (float(i) + 0.5) / 7.0)
        QTAB_PAL[i+1] = 40.0 * math.sin(3.1415926535 * (float(i) + 0.5) / 7.0)
    # NTSC: 15 color cycles
    for i in range(15):
        ITAB_NTSC[i+1] = 40.0 * math.cos(3.1415926535 * float(i) / 7.5)
        QTAB_NTSC[i+1] = 40.0 * math.sin(3.1415926535 * float(i) / 7.5)

init_tables()

# --- AUDIO KERNEL (Numba JIT) ---
@jit(nopython=True, fastmath=True)
def encode_audio_chunk(audio_data):
    # Converts raw audio (8-bit Unsigned) to Atari POKEY format (4-bit PWM-ish).
    # Mapping: PC 0..255 -> Atari 0..100.
    # Uses simple linear scaling consistent with the original C++ encoder.
    out = np.zeros(len(audio_data), dtype=np.uint8)
    for i in range(len(audio_data)):
        val = int(audio_data[i])
        # Scaling: input 0-255 -> output 0-100
        out_val = (val * 100) // 255
        out[i] = out_val
    return out

# --- C80M4 VIDEO KERNEL (Numba JIT) ---
@jit(nopython=True, fastmath=True)
def encode_c80m4_frame(rgb):
    # rgb: float32 (48, 80, 3), 0..255. Serpentine Floyd-Steinberg dither
    # down to RGB444, then packed into the 24-row cell layout.
    q = np.zeros((48, 80, 3), dtype=np.uint8)
    work = rgb.copy()
    for y in range(48):
        if y % 2 == 0:
            xs, xe, xd = 0, 80, 1
        else:
            xs, xe, xd = 79, -1, -1
        for x in range(xs, xe, xd):
            for c in range(3):
                v = work[y, x, c]
                qq = int(round(v / 17.0))
                if qq < 0:
                    qq = 0
                if qq > 15:
                    qq = 15
                q[y, x, c] = qq
                e = v - qq * 17.0
                if 0 <= x + xd < 80:
                    work[y, x + xd, c] += e * 0.4375        # 7/16
                if y + 1 < 48:
                    work[y + 1, x, c] += e * 0.3125         # 5/16
                    if 0 <= x - xd < 80:
                        work[y + 1, x - xd, c] += e * 0.1875  # 3/16
                    if 0 <= x + xd < 80:
                        work[y + 1, x + xd, c] += e * 0.0625  # 1/16
    out = np.zeros(7680, dtype=np.uint8)
    for r in range(24):
        base = r * 320
        top = 2 * r          # top pixel -> BG (high nibble)
        bot = 2 * r + 1      # bottom pixel -> FG (low nibble)
        for x in range(80):
            out[base + x] = C80_HALFBLOCK
            out[base + 80 + x] = (q[top, x, 0] << 4) | q[bot, x, 0]
            out[base + 160 + x] = (q[top, x, 1] << 4) | q[bot, x, 1]
            out[base + 240 + x] = (q[top, x, 2] << 4) | q[bot, x, 2]
    return out.reshape((192, 40))

# --- VIDEO KERNEL (Numba JIT) ---
@jit(nopython=True, fastmath=True)
def encode_video_frame(y_plane, u_plane, v_plane, errfi, errfq, errfy, is_pal):
    # Main video encoding engine.
    # Performs:
    # 1. YUV -> Atari I/Q conversion (Phase/Amplitude)
    # 2. GTIA palette matching (16 hues / 16 luma levels)
    # 3. Dithering (Error diffusion)
    # 4. Bit packing (2 pixels per byte)
    output = np.zeros((HEIGHT, 40), dtype=np.uint8)

    # Local error buffers for dithering (reset per frame/line logic)
    perr = np.zeros(82, dtype=np.int32)
    nerr = np.zeros(82, dtype=np.int32)
    pcerr = np.zeros(82, dtype=np.float32)
    ncerr = np.zeros(82, dtype=np.float32)
    pserr = np.zeros(82, dtype=np.float32)
    nserr = np.zeros(82, dtype=np.float32)

    ITAB = ITAB_PAL if is_pal else ITAB_NTSC
    QTAB = QTAB_PAL if is_pal else QTAB_NTSC

    for y in range(HEIGHT):
        # Atari in AVF mode interleaves Chroma (Color) and Luma (Brightness) lines.
        # PAL: Even=Chroma, Odd=Luma
        # NTSC: Odd=Chroma, Even=Luma (due to different signal structure)
        is_chroma = (y % 2 == 0) if is_pal else (y % 2 != 0)

        if is_chroma:
            # --- CHROMA PROCESSING (COLOR) ---
            ncerr[:] = 0
            nserr[:] = 0
            # Serpentine scanning (left-to-right, then right-to-left) reduces dither artifacts
            direction = -1 if (y & 2) else 1
            start_x = 0 if direction == 1 else 79
            chroma_y = y // 2
            accum = 0

            # IMPORTANT - two distinct index spaces are in play here:
            #   i = scan index  (0..79, always advances in scan order)
            #   x = screen column (i counted from the left or the right)
            # The error buffers are indexed by SCAN index, never by column. That is
            # what makes the mirrored write pattern below correct: because the next
            # line is scanned in the opposite direction, "the pixel below" is at
            # scan index 79-i there, so 80-i is where its weight-5 share must land.
            # Indexing these buffers by column instead silently diffuses every
            # line's error onto the horizontally mirrored part of the next line.
            for i in range(80):
                x = start_x + (i * direction)
                u_val = int(u_plane[chroma_y, x]) - 128
                v_val = int(v_plane[chroma_y, x]) - 128
                fu = float(u_val)
                fv = float(v_val)

                # YUV to I/Q conversion matrix
                if is_pal:
                    red = 1.596 * fv
                    grn = -0.391 * fu - 0.813 * fv
                    blu = 2.017 * fu
                else:
                    red = 1.396 * fv
                    grn = -0.342 * fu - 0.711 * fv
                    blu = 1.765 * fu

                fi = (0.595*red - 0.274*grn - 0.321*blu) + pcerr[i + 1] / 25.0 + errfi[y, x]
                fq = (0.211*red - 0.522*grn + 0.311*blu) + pserr[i + 1] / 25.0 + errfq[y, x]

                satsq = fi*fi + fq*fq

                # Phase calculation (Hue)
                if is_pal:
                    rawhue = math.atan2(fq, fi) * (7.0 / 3.1415926535) - 0.5
                else:
                    rawhue = math.atan2(fq, fi) * (7.5 / 3.1415926535) + 2.0

                ihue = int(math.floor(rawhue + 0.5))
                cycle = 14 if is_pal else 15
                offset = 14000 if is_pal else 15000
                color = ((ihue + offset) % cycle) + 1
                out_color = 0

                # Saturation threshold (40.0^2 = 1600). Below this, color is rendered as gray.
                # This prevents excessive color noise in dark areas.
                if satsq > 1600.0:
                    out_color += color
                    sc = 40.0 / math.sqrt(satsq)
                    fi *= sc
                    fq *= sc

                # Quantization error calculation
                ierror = fi - ITAB[out_color]
                qerror = fq - QTAB[out_color]

                # Error propagation to next pixels and lines (Floyd-Steinberg variant).
                # errf* is the TEMPORAL term (9/25 of the error carried to the same
                # pixel in the next frame), so it is keyed by column - a given line
                # always scans the same way, so column and scan index stay in step
                # from frame to frame either way.
                errfi[y, x] = ierror * 0.36
                errfq[y, x] = qerror * 0.36

                pcerr[i + 2] += ierror * 7.0
                ncerr[81 - i] += ierror * 3.0
                ncerr[80 - i] += ierror * 5.0
                ncerr[79 - i] += ierror

                pserr[i + 2] += qerror * 7.0
                nserr[81 - i] += qerror * 3.0
                nserr[80 - i] += qerror * 5.0
                nserr[79 - i] += qerror

                # Packing 4-bit nibbles into a byte
                if direction < 0:
                    accum = ((accum >> 4) + (out_color << 4)) & 0xFF
                else:
                    accum = ((accum << 4) + out_color) & 0xFF

                if (i % 2) == 1:
                    output[y, x // 2] = accum

            pcerr[:] = ncerr
            pserr[:] = nserr

        else:
            # --- LUMA PROCESSING (BRIGHTNESS) ---
            nerr[:] = 0
            direction = -1 if (y & 2) else 1
            start_x = 0 if direction == 1 else 79
            accum = 0

            # Luma/chroma registration. Output line pair number n is carried by the
            # chroma line at 2n and the luma line at 2n+1 (PAL) or by the luma line
            # at 2n and the chroma line at 2n+1 (NTSC). The 4:2:0 chroma plane row n
            # covers source rows 2n and 2n+1, so the luma line must average that
            # SAME row pair or the two planes drift apart by one source line.
            # PAL luma rows are odd, so they have to look one row back.
            base_y = (y - 1) if is_pal else y
            if base_y < 0:
                base_y = 0
            if base_y > HEIGHT - 2:
                base_y = HEIGHT - 2

            # See the note in the chroma branch: error buffers use the SCAN index i,
            # pixel and output access use the screen column x.
            for i in range(80):
                x = start_x + (i * direction)
                base_x = x * 2

                # Luma sampling (2x2 averaging for stability)
                p0 = int(y_plane[base_y, base_x])
                p1 = int(y_plane[base_y, base_x + 1])
                p2 = int(y_plane[base_y + 1, base_x])
                p3 = int(y_plane[base_y + 1, base_x + 1])

                # Scale YUV range (16-235) to full range (0-255)
                val_a = (((p0 + p1 + p2 + p3 - 64) * 255 + 438) // 876)

                diff = (perr[i + 1] + errfy[y, x]) / 25.0
                val_a += int(diff)

                if val_a < 0: val_a = 0
                if val_a > 255: val_a = 255

                # Quantize to 16 brightness levels (4-bit)
                val_b = int((val_a + 8) // 17)
                if val_b < 0: val_b = 0
                if val_b > 15: val_b = 15

                val_c = val_b * 17
                err = val_a - val_c

                errfy[y, x] = float(err * 9)
                perr[i + 2] += int(err * 7)
                nerr[81 - i] += int(err * 3)
                nerr[80 - i] += int(err * 5)
                nerr[79 - i] += int(err)

                if direction < 0:
                    accum = ((accum >> 4) + (val_b << 4)) & 0xFF
                else:
                    accum = ((accum << 4) + val_b) & 0xFF

                if (i % 2) == 1:
                    output[y, x // 2] = accum

            temp = perr
            perr = nerr
            nerr = temp

    return output

# --- MUXING (STREAM INTERLEAVING) ---
def mux_frame(fout, vbuf, abuf, is_pal):
    # Writes one video and audio frame in the interleaved AVF format.
    # The format is rigid - 8704 bytes per frame.
    # Video is split into blocks of 3 lines, interleaved with audio data.
    v_flat = vbuf.flatten()

    # 1. VIDEO PART (Video Blocks)
    # Video: 192 lines / 3 = 64 blocks.
    # Each video block is padded with zeroes.
    for y in range(0, 192, 3):
        fout.write(b'\x00')
        fout.write(v_flat[y*40 : (y+1)*40])
        fout.write(b'\x00\x00\x00')
        fout.write(b'\x00')
        fout.write(v_flat[(y+1)*40 : (y+2)*40])
        fout.write(b'\x00\x00\x00')
        fout.write(v_flat[(y+2)*40 : (y+3)*40])

    # 2. AUDIO PART (Audio Blocks at the end of the frame)
    # Audio data is "smeared" in a specific pattern so the CPU can read it in time.
    off1 = 120 if is_pal else 70
    off2 = 52

    # Audio Loop 1 (32 audio lines)
    for y in range(32):
        fout.write(bytes([abuf[y]]))
        fout.write(bytes([abuf[y + 0*32 + off1]]))
        fout.write(bytes([abuf[y + 1*32 + off1]]))
        fout.write(bytes([abuf[y + 2*32 + off1]]))
        fout.write(bytes([abuf[y + 3*32 + off1]]))
        fout.write(bytes([abuf[y + 4*32 + off1]]))
        fout.write(bytes([abuf[y + 5*32 + off1]]))
        fout.write(bytes([abuf[y + 0*32 + off2]]))
        fout.write(bytes([abuf[y + 1*32 + off2]]))
        fout.write(b'\x00')

    # Audio Loop 2 (19 audio lines)
    # NOTE: The padding difference between PAL and NTSC is critical for sync!
    for y in range(19):
        fout.write(bytes([abuf[y+32]]))
        if is_pal:
            # PAL: 1 byte data + 1 byte data + 8 zeroes = 10 bytes
            fout.write(bytes([abuf[y + 2*32 + off2]]))
            fout.write(b'\x00' * 8)
        else:
            # NTSC: 1 byte data + 9 zeroes = 10 bytes
            fout.write(b'\x00' * 9)

    # End (Padding to fill the sector)
    fout.write(bytes([abuf[51]]))
    fout.write(b'\x00')

# --- DOWNLOADER UTILS ---
def download_media(url):
    # Downloads video from URL using yt-dlp.
    # Saves to 'downloads' directory.
    # Returns the path to the downloaded file and the video title.
    if not YT_DLP_AVAILABLE:
        print("ERROR: yt-dlp is not installed. Please install it with: pip install yt-dlp")
        sys.exit(1)

    print(f"--- Downloading from URL: {url} ---")

    # Ensure downloads directory exists
    download_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(download_dir, exist_ok=True)

    # Prefer a single muxed MP4 and force an mp4 container. Modern YouTube
    # serves DASH (separate video+audio) whose merged file often does NOT
    # match prepare_filename()'s guessed extension - the classic "downloaded
    # but no file" trap. merge_output_format pins it, and we also fall back
    # to the newest file actually written to the download dir.
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'restrictfilenames': True,   # ASCII-only filenames
    }

    before = set(os.listdir(download_dir))
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'video')
            filename = ydl.prepare_filename(info)
            if not os.path.exists(filename):
                # merge/postprocess changed the extension: take the newest
                # file that appeared in the download dir during this call.
                fresh = [os.path.join(download_dir, f)
                         for f in os.listdir(download_dir) if f not in before]
                cand = fresh or [os.path.join(download_dir, f)
                                 for f in os.listdir(download_dir)]
                if not cand:
                    print("Error downloading URL: no file was written")
                    return None, None
                filename = max(cand, key=os.path.getmtime)
            sanitized_title = "".join(
                c for c in title
                if c.isalpha() or c.isdigit() or c in ' .-_').strip()
            return filename, sanitized_title
        except Exception as e:
            print(f"Error downloading URL: {e}")
            return None, None

# --- MAIN PROCESSING LOGIC ---
def process_single_file(input_file, system, output_file, config):
    print(f"\n>>> Converting: {input_file} -> {output_file}")

    is_pal = (system == "PAL")
    fps = "49.86" if is_pal else "59.92"
    audio_rate = "15557" if is_pal else "15700" # Sample rates consistent with VirtualDub
    audio_chunk_size = 312 if is_pal else 262

    # Using temporary files instead of pipes for stability.
    pid = os.getpid()
    temp_vid = f"temp_v_{pid}.raw"
    temp_aud = f"temp_a_{pid}.raw"

    # --- 1. FFMPEG FILTERS ---
    # Force the output colour matrix to BT.601 ("smpte170m"). The downstream
    # YUV->I/Q conversion in encode_video_frame is hard-coded to BT.601
    # coefficients; modern HD content is usually tagged BT.709 and would yield
    # subtly off greens/skin tones if fed through unconverted. Letting libswscale
    # do the matrix conversion during the scale stage costs nothing extra.
    c80 = config['profile'] == PROFILE_C80M4
    if c80:
        # C80M4: 80x48 true-colour cells; the FPGA renders RGB directly,
        # so no BT.601 trip - plain RGB from the scaler.
        if config['aspect'] == 'fit':
            scale_filter = ("scale=80:48:force_original_aspect_ratio=decrease:flags=lanczos,"
                            "pad=80:48:(ow-iw)/2:(oh-ih)/2:black")
        else:
            scale_filter = "scale=80:48:flags=lanczos"
        vf_chain = (f"{scale_filter},"
                    f"eq=contrast={config['contrast']},"
                    f"hue=s={config['saturation']},"
                    f"fps={fps},format=rgb24")
        print("    Profile: AVF-C80 M4 (GTIA2RGB COL80, 80x48 RGB444)")
    common_scale_opts = "flags=lanczos:in_color_matrix=auto:out_color_matrix=bt601"
    if config['aspect'] == 'fit':
        # Preserve source aspect ratio, pad with black to 160x192 (letter/pillarbox).
        # Note: assumes square pixels for the source; ignores Atari's non-square
        # pixel aspect on a CRT (a known compromise - acceptable on modern displays).
        scale_filter = (
            f"scale=160:192:force_original_aspect_ratio=decrease:{common_scale_opts},"
            f"pad=160:192:(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        # Stretch (legacy behaviour): fill the frame regardless of source AR.
        scale_filter = f"scale=160:192:{common_scale_opts}"

    # Saturation is applied via the `hue` filter rather than `eq=saturation=`,
    # because `eq` saturates very weakly above 1.0 (clamp/curve in libavfilter)
    # while `hue=s=` scales chroma linearly - so values like 1.5 actually do
    # something visible. Contrast still goes through `eq` (no equivalent in `hue`).
    # Both filters are no-ops at 1.0, so default behaviour is preserved.
    if not c80:
        vf_chain = (
            f"{scale_filter},"
            f"eq=contrast={config['contrast']},"
            f"hue=s={config['saturation']},"
            f"fps={fps},format=yuv420p"
        )

    # Audio Filters: Loudnorm (EBU R128) or Manual Gain
    if config['loudnorm']:
        # EBU R128 normalization
        af_chain = "loudnorm=I=-16:TP=-1.5:LRA=11"
        print(f"    Audio: Loudness Normalization (I=-16 LUFS)")
    else:
        # Manual volume gain
        af_chain = f"volume={config['volume']}dB"
        print(f"    Audio: Manual Gain {config['volume']}dB")

    # --- 2. RAW EXTRACTION (FFmpeg) ---
    try:
        # Dump Video
        print(f"    [1/3] Extracting Video...")
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', input_file,
            '-vf', vf_chain,
            '-f', 'rawvideo', temp_vid
        ], check=True)

        # Dump Audio (Forced PCM Unsigned 8-bit)
        # Added TRY/EXCEPT to handle video-only files
        print(f"    [2/3] Extracting Audio...")
        use_dummy_audio = False
        try:
            subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', input_file,
                '-vn',
                '-af', af_chain,
                '-ar', audio_rate,
                '-ac', '1',
                '-c:a', 'pcm_u8',
                '-f', 'u8',
                temp_aud
            ], check=True)
        except subprocess.CalledProcessError:
            print("    WARNING: No audio stream detected (or extraction failed).")
            print("             Generating digital silence for compatibility.")
            use_dummy_audio = True

    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg extraction: {e}")
        return

    # --- 3. ENCODING & MUXING ---
    print(f"    [3/3] Encoding & Muxing...")

    fout = open(output_file, "wb")

    # The 8KB header is enabled by default. It is required for correct synchronization
    # by the "movplay" player, which always skips the first 16 sectors (regardless of media).
    if not config['no_header']:
        fout.write(b'\x00' * (16 * 512))

    # Dithering buffers
    errfi = np.zeros((HEIGHT, 80), dtype=np.float32)
    errfq = np.zeros((HEIGHT, 80), dtype=np.float32)
    errfy = np.zeros((HEIGHT, 80), dtype=np.float32)

    f_vid = open(temp_vid, "rb")

    # Open audio file only if it exists, otherwise we'll generate silence in loop
    f_aud = None
    if not use_dummy_audio:
        f_aud = open(temp_aud, "rb")

    frame_count = 0
    start_time = time.time()

    # AVF2: pick the poster frame index up front. Explicit --poster wins;
    # otherwise 10% of the source duration (ffprobe), otherwise 10 s.
    poster_idx = 0
    poster_data = None
    avf2_meta = None
    if config['avf2']:
        if config['poster'] is not None:
            poster_time = config['poster']
        else:
            duration = probe_duration_seconds(input_file)
            poster_time = duration * 0.10 if duration else 10.0
        poster_idx = max(0, int(round(poster_time * float(fps))))
        # Per-file metadata; untitled files fall back to the source basename
        # so batch conversions still produce self-describing AVF2 files.
        avf2_meta = {
            'title': config['title'] or os.path.splitext(os.path.basename(input_file))[0],
            'author': config['author'],
            'comment': config['comment'],
        }

    try:
        while True:
            if c80:
                rgb_data = f_vid.read(80 * 48 * 3)
                if len(rgb_data) != 11520: break
            else:
                y_data = f_vid.read(160 * 192)
                if len(y_data) != 30720: break

                u_data = f_vid.read(80 * 96)
                v_data = f_vid.read(80 * 96)

            # Read audio chunk or generate silence
            if not use_dummy_audio:
                a_data = f_aud.read(audio_chunk_size)
            else:
                a_data = b'' # Will be padded below

            if len(a_data) < audio_chunk_size:
                # Padding with silence (128 = silence in Unsigned 8-bit)
                # This handles both end-of-file and dummy-audio scenarios
                a_data += b'\x80' * (audio_chunk_size - len(a_data))

            a_chunk = np.frombuffer(a_data, dtype=np.uint8)

            if c80:
                rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape((48, 80, 3)).astype(np.float32)
                encoded_vid = encode_c80m4_frame(rgb)
            else:
                y_plane = np.frombuffer(y_data, dtype=np.uint8).reshape((192, 160))
                u_plane = np.frombuffer(u_data, dtype=np.uint8).reshape((96, 80))
                v_plane = np.frombuffer(v_data, dtype=np.uint8).reshape((96, 80))
                encoded_vid = encode_video_frame(y_plane, u_plane, v_plane, errfi, errfq, errfy, is_pal)
            encoded_aud = encode_audio_chunk(a_chunk)

            # AVF2 poster: keep overwriting until the chosen frame is reached,
            # so a clip shorter than the poster time still gets its last frame.
            if config['avf2'] and frame_count <= poster_idx:
                poster_data = encoded_vid.flatten().tobytes()

            # Both profiles share the exact APAC container (8704-byte
            # frames, same block/pad + audio mux). C80 differs ONLY in the
            # video CONTENT (COL80 M4 cells vs APAC luma/chroma) - so the
            # proven movplay streaming kernel and audio path are reused as
            # is, and the player just enables Mode 4 for a C80 profile.
            mux_frame(fout, encoded_vid, encoded_aud, is_pal)

            # --- INTEGRITY CHECK ---
            # Verifies if the file size matches the expected byte count exactly.
            # If not, aborts immediately to prevent generating a corrupted file.
            # Uses an explicit raise rather than assert so that running with
            # python -O does not silently disable the safety net.
            current_pos = fout.tell()
            header_offset = (16 * 512) if not config['no_header'] else 0
            expected_pos = header_offset + (frame_count + 1) * FRAME_SIZE_BYTES

            if current_pos != expected_pos:
                raise RuntimeError(
                    f"CRITICAL: byte size mismatch at frame {frame_count} "
                    f"(got {current_pos}, expected {expected_pos}). "
                    f"AVF stream is misaligned and would desync on hardware."
                )

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_count / elapsed
                print(f"    Encoding... Frame {frame_count} ({fps_proc:.1f} fps)", end='\r', flush=True)

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        f_vid.close()
        if f_aud: f_aud.close()

        # AVF2: patch the metadata header into the (so far zeroed) first
        # 16 sectors, now that frame_count and the poster are known.
        if config['avf2'] and frame_count > 0:
            header = build_avf2_header(is_pal, frame_count, poster_data, avf2_meta,
                                       config['profile'])
            fout.seek(0)
            fout.write(header)
            print(f"    AVF2 header: title='{avf2_meta['title']}', "
                  f"poster frame {min(poster_idx, frame_count - 1)}, "
                  f"{frame_count} frames")

        fout.close()

        # Clean up temporary files
        if os.path.exists(temp_vid): os.remove(temp_vid)
        if os.path.exists(temp_aud): os.remove(temp_aud)

        # NOTE: We do NOT remove the input file (even if downloaded), as per new requirements.

    print(f"\n    Done. Saved to {output_file}")


def generate_test_file(filename):
    # Generates a synthetic video file (SMPTE bars + 440Hz sine wave).
    print(f"--- Generating Test Signal: {filename} ---")
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'testsrc=duration=10:size=160x192:rate=50',
        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=10',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-ar', '44100', '-ac', '1',
        filename
    ]
    subprocess.run(cmd, check=True)
    print("    Test file created successfully.")

def main():
    avf.print_banner("Converter", __version__)
    parser = argparse.ArgumentParser(description="Atari 8-bit AVF Converter")

    # Input Options
    parser.add_argument("input", nargs='?', help="Input file, directory, or URL")
    parser.add_argument("--test-gen", action="store_true", help="Generate and convert a test signal")
    parser.add_argument("--urllist", action="store_true", help="Treat input argument as a text file containing URLs")

    # System Options
    parser.add_argument("--system", choices=['PAL', 'NTSC', 'BOTH'], default='BOTH', help="Target TV system")
    parser.add_argument("--out", help="Output base name; '-PAL.avf'/'-NTSC.avf' is appended "
                                      "(ignored in batch/list mode)")
    parser.add_argument("--version", action="version", version=f"avi2atari {__version__}")
    parser.add_argument("--no-header", action="store_true", help="Disable 8KB header (not recommended!)")
    parser.add_argument("--profile", choices=['apac', 'c80m4'], default='apac',
                        help="Video profile: 'apac' (classic, every Atari) or 'c80m4' "
                             "(AVF-C80: GTIA2RGB COL80 Mode 4, 80x48 true-colour RGB444; "
                             "implies --avf2). c80m4 needs a GTIA2RGB and the AV2PLAY "
                             "player (https://github.com/HanJammer/av2play) for true "
                             "colour; it also plays in legacy AVF players, but the "
                             "colours will be wrong.")

    # AVF2 metadata header (see docs/AVF2_HEADER.md). Fully backward
    # compatible: existing players skip the header region unconditionally.
    parser.add_argument("--avf2", action="store_true",
                        help="Write an AVF2 metadata header (title/author/poster) into the "
                             "8KB header region. Backward compatible with all players.")
    parser.add_argument("--title", default=None, help="AVF2: video title (ASCII, max 64 chars)")
    parser.add_argument("--author", default=None, help="AVF2: author of the conversion (ASCII, max 64 chars)")
    parser.add_argument("--comment", default=None, help="AVF2: free-text comment (ASCII, max 128 chars)")
    parser.add_argument("--poster", type=float, default=None, metavar="SECONDS",
                        help="AVF2: source timestamp of the poster frame "
                             "(default: 10%% of duration, or 10s if unknown)")

    # Image Options
    parser.add_argument("--saturation", type=float, default=1.0, help="Color saturation boost (default: 1.0)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast adjustment (default: 1.0)")
    parser.add_argument("--aspect", choices=['stretch', 'fit'], default='stretch',
                        help="How to fit source into 160x192: 'stretch' (legacy default, "
                             "fills the frame) or 'fit' (preserves source aspect ratio with "
                             "black bars). Use 'fit' for 16:9 content to avoid squashing.")

    # Audio Options
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--volume", type=float, default=12.0, help="Manual volume gain in dB (default: 12.0)")
    audio_group.add_argument("--loudnorm", action="store_true", help="Use EBU R128 loudness normalization (Recommended)")

    args = parser.parse_args()

    # If no arguments provided (and not test-gen), print help and exit
    if not args.input and not args.test_gen:
        parser.print_help()
        sys.exit(1)

    # Every mode shells out to FFmpeg, so fail here with a useful message rather
    # than a bare FileNotFoundError traceback from the first subprocess call.
    if shutil.which('ffmpeg') is None:
        print("ERROR: 'ffmpeg' was not found on your PATH.")
        print("       Install it and make sure the 'bin' folder is on PATH - see the README.")
        sys.exit(1)

    if args.avf2 and args.no_header:
        print("ERROR: --avf2 requires the 8KB header; it cannot be combined with --no-header.")
        sys.exit(1)

    if args.profile == 'c80m4':
        if args.no_header:
            print("ERROR: --profile c80m4 requires the AVF2 header (players must be able "
                  "to identify the profile); it cannot be combined with --no-header.")
            sys.exit(1)
        if not args.avf2:
            print("NOTE: --profile c80m4 implies --avf2 (the profile id lives in the header).")
            args.avf2 = True

    # Config dictionary
    config = {
        'no_header': args.no_header,
        'saturation': args.saturation,
        'contrast': args.contrast,
        'aspect': args.aspect,
        'volume': args.volume,
        'loudnorm': args.loudnorm,
        'avf2': args.avf2,
        'profile': PROFILE_C80M4 if args.profile == 'c80m4' else PROFILE_APAC,
        'title': args.title,
        'author': args.author,
        'comment': args.comment,
        'poster': args.poster
    }

    # MODE 1: Test Generator
    if args.test_gen:
        test_file = "test_tone.mp4"
        generate_test_file(test_file)
        if args.system in ['BOTH', 'PAL']:
            process_single_file(test_file, 'PAL', "test_tone-PAL.avf", config)
        if args.system in ['BOTH', 'NTSC']:
            process_single_file(test_file, 'NTSC', "test_tone-NTSC.avf", config)
        return

    # Determine input type
    input_arg = args.input

    # MODE 2: URL List File
    if args.urllist:
        if not os.path.exists(input_arg):
            print(f"ERROR: URL list file '{input_arg}' not found.")
            return

        print(f"--- Processing URL List: {input_arg} ---")
        with open(input_arg, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split('\t')
            url = parts[0]
            custom_name = parts[1] if len(parts) > 1 else None

            dl_path, title = download_media(url)
            if dl_path:
                base_name = custom_name if custom_name else title
                if args.system in ['BOTH', 'PAL']:
                    process_single_file(dl_path, 'PAL', f"{base_name}-PAL.avf", config)
                if args.system in ['BOTH', 'NTSC']:
                    process_single_file(dl_path, 'NTSC', f"{base_name}-NTSC.avf", config)
        return

    # MODE 3: Direct URL (YouTube or File)
    if input_arg.startswith("http://") or input_arg.startswith("https://"):
        dl_path, title = download_media(input_arg)
        if dl_path:
            base_name = args.out if args.out else title
            if args.system in ['BOTH', 'PAL']:
                process_single_file(dl_path, 'PAL', f"{base_name}-PAL.avf", config)
            if args.system in ['BOTH', 'NTSC']:
                process_single_file(dl_path, 'NTSC', f"{base_name}-NTSC.avf", config)
        return

    # MODE 4: Directory Batch
    if os.path.isdir(input_arg):
        # Expanded extension list for batch processing
        extensions = ['*.mp4', '*.mkv', '*.avi', '*.flv', '*.webm', '*.wmv', '*.mov', '*.mpg', '*.mpeg', '*.m4v']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_arg, ext)))

        print(f"--- Batch Mode: Found {len(files)} video files in '{input_arg}' ---")

        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            if args.system in ['BOTH', 'PAL']:
                process_single_file(f, 'PAL', os.path.join(input_arg, f"{base}-PAL.avf"), config)
            if args.system in ['BOTH', 'NTSC']:
                process_single_file(f, 'NTSC', os.path.join(input_arg, f"{base}-NTSC.avf"), config)
        return

    # MODE 5: Single Local File
    if os.path.exists(input_arg):
        base = args.out if args.out else os.path.splitext(os.path.basename(input_arg))[0]
        if args.system in ['BOTH', 'PAL']:
            process_single_file(input_arg, 'PAL', f"{base}-PAL.avf", config)
        if args.system in ['BOTH', 'NTSC']:
            process_single_file(input_arg, 'NTSC', f"{base}-NTSC.avf", config)
    else:
        print(f"ERROR: Input '{input_arg}' not found.")

if __name__ == "__main__":
    main()
