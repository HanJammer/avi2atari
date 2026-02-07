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
from numba import jit

# Try importing yt_dlp for URL support
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# --- AVF FORMAT CONFIGURATION ---
# Target resolution used by the player (scaled via Display List interrupts)
HEIGHT = 192
WIDTH = 160

# Frame size in bytes on disk.
# The player reads data in blocks of 17 sectors (17 * 512 = 8704 bytes).
# Every byte must be perfectly aligned, otherwise the video will drift or turn into noise.
FRAME_SIZE_BYTES = 8704

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

                fi = (0.595*red - 0.274*grn - 0.321*blu) + pcerr[x + 1] / 25.0 + errfi[y, x]
                fq = (0.211*red - 0.522*grn + 0.311*blu) + pserr[x + 1] / 25.0 + errfq[y, x]

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

                # Error propagation to next pixels and lines (Floyd-Steinberg variant)
                errfi[y, x] = ierror * 0.36
                errfq[y, x] = qerror * 0.36

                pcerr[x + 2] += ierror * 7.0
                ncerr[81 - x] += ierror * 3.0
                ncerr[80 - x] += ierror * 5.0
                ncerr[79 - x] += ierror

                pserr[x + 2] += qerror * 7.0
                nserr[81 - x] += qerror * 3.0
                nserr[80 - x] += qerror * 5.0
                nserr[79 - x] += qerror

                # Packing 4-bit nibbles into a byte
                if direction < 0:
                    accum = (accum >> 4) + (out_color << 4)
                else:
                    accum = (accum << 4) + out_color

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
            
            for i in range(80):
                x = start_x + (i * direction)
                base_x = x * 2
                
                # Luma sampling (2x2 averaging for stability)
                p0 = int(y_plane[y, base_x])
                p1 = int(y_plane[y, base_x + 1])
                y_next = y + 1 if y + 1 < HEIGHT else y
                p2 = int(y_plane[y_next, base_x])
                p3 = int(y_plane[y_next, base_x + 1])
                
                # Scale YUV range (16-235) to full range (0-255)
                val_a = (((p0 + p1 + p2 + p3 - 64) * 255 + 438) // 876)
                
                diff = (perr[x + 1] + errfy[y, x]) / 25.0
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
                perr[x + 2] += int(err * 7)
                nerr[81 - x] += int(err * 3)
                nerr[80 - x] += int(err * 5)
                nerr[79 - x] += int(err)
                
                if direction < 0:
                    accum = (accum >> 4) + (val_b << 4)
                else:
                    accum = (accum << 4) + val_b
                
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
    
    # Configure yt-dlp to download to 'downloads' folder with clean filename
    # %(title)s.%(ext)s - uses video title
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'restrictfilenames': True, # ASCII only filenames
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            title = info.get('title', 'video')
            # Sanitize title for output usage
            sanitized_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in ' .-_']).strip()
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
    vf_chain = f"scale=160:192:flags=lanczos,eq=saturation={config['saturation']}:contrast={config['contrast']},fps={fps},format=yuv420p"
    
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
    
    try:
        while True:
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

            y_plane = np.frombuffer(y_data, dtype=np.uint8).reshape((192, 160))
            u_plane = np.frombuffer(u_data, dtype=np.uint8).reshape((96, 80))
            v_plane = np.frombuffer(v_data, dtype=np.uint8).reshape((96, 80))
            a_chunk = np.frombuffer(a_data, dtype=np.uint8)

            encoded_vid = encode_video_frame(y_plane, u_plane, v_plane, errfi, errfq, errfy, is_pal)
            encoded_aud = encode_audio_chunk(a_chunk)

            mux_frame(fout, encoded_vid, encoded_aud, is_pal)

            # --- INTEGRITY CHECK ---
            # Verifies if the file size matches the expected byte count exactly.
            # If not, aborts immediately to prevent generating a corrupted file.            
            current_pos = fout.tell()
            header_offset = (16 * 512) if not config['no_header'] else 0
            expected_pos = header_offset + (frame_count + 1) * FRAME_SIZE_BYTES
            
            assert current_pos == expected_pos, f"CRITICAL ERROR at Frame {frame_count}. Byte size mismatch!"

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_count / elapsed
                print(f"    Encoding... Frame {frame_count} ({fps_proc:.1f} fps)", end='\r')

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        f_vid.close()
        if f_aud: f_aud.close()
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
    parser = argparse.ArgumentParser(description="Atari 8-bit AVF Converter")
    
    # Input Options
    parser.add_argument("input", nargs='?', help="Input file, directory, or URL")
    parser.add_argument("--test-gen", action="store_true", help="Generate and convert a test signal")
    parser.add_argument("--urllist", action="store_true", help="Treat input argument as a text file containing URLs")

    # System Options
    parser.add_argument("--system", choices=['PAL', 'NTSC', 'BOTH'], default='BOTH', help="Target TV system")
    parser.add_argument("--out", help="Output filename (ignored in batch/list mode)")
    parser.add_argument("--no-header", action="store_true", help="Disable 8KB header (not recommended!)")

    # Image Options
    parser.add_argument("--saturation", type=float, default=1.0, help="Color saturation boost (default: 1.0)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast adjustment (default: 1.0)")

    # Audio Options
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--volume", type=float, default=12.0, help="Manual volume gain in dB (default: 12.0)")
    audio_group.add_argument("--loudnorm", action="store_true", help="Use EBU R128 loudness normalization (Recommended)")

    args = parser.parse_args()

    # If no arguments provided (and not test-gen), print help and exit
    if not args.input and not args.test_gen:
        parser.print_help()
        sys.exit(1)

    # Config dictionary
    config = {
        'no_header': args.no_header,
        'saturation': args.saturation,
        'contrast': args.contrast,
        'volume': args.volume,
        'loudnorm': args.loudnorm
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
