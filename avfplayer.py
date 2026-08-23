#!/usr/bin/env python3
#
# avfplayer.py - AVF Video Player
# Copyright (C) 2026 HanJammer & Lumen
#
# A simple Python AVF video file (Atari Video Format) player for modern computers.
# Supports both PAL (50Hz) and NTSC (60Hz) input standards.
# 
# Key Features:
# - Hardware-adaptive audio sync
# - GTIA Palette emulation (YIQ/YUV colorspace)
# - Real-time Phase/Saturation adjustment
# - CRT Scanline emulation
# - Horizontal Blending (Blur)
# - Looped playback
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
import argparse
import numpy as np
import math

# pygame is imported lazily in main() (only playback needs it), so header
# inspection works on machines without pygame installed.
pygame = None

__version__ = "0.5"

import avf_format as avf

# --- CONSTANTS (shared PC-suite lib) ---
WIDTH, HEIGHT = avf.WIDTH, avf.HEIGHT
FRAME_SIZE_BYTES = avf.FRAME_SIZE_APAC
HEADER_SIZE = avf.HEADER_SIZE


def parse_avf2(path):
    """AVF2 metadata + poster via the shared lib. Returns (meta, poster);
    (None, None) for Legacy/v1, header-less, or bad-checksum files (the
    player then treats it as plain Legacy playback)."""
    size = os.path.getsize(path)
    if size % FRAME_SIZE_BYTES == 0 and size % avf.FRAME_SIZE_C80 != 0:
        return None, None  # header-less APAC file
    with open(path, "rb") as f:
        buf = f.read(HEADER_SIZE)
    meta, problems = avf.parse_avf2_header(buf)
    if meta is None:
        return None, None
    if not meta.get("checksum_ok", True):
        print("[!] AVF2 magic found but checksum is invalid - ignoring metadata.")
        return None, None
    return meta, avf.poster_from_header(buf, meta)

class AVFPlayer:
    def __init__(self, filename, system='PAL', scale=3, debug=False,
                 meta=None, poster=None, show_poster=False):
        # 1. AUDIO INITIALIZATION
        # We request 44.1kHz Stereo, but we must accept what the OS gives us.
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=4096)
        pygame.init()
        
        # Check actual hardware parameters (critical for sync on modern PCs)
        self.mix_freq, self.mix_bits, self.mix_chans = pygame.mixer.get_init()
        print(f"[*] Audio Hardware: {self.mix_freq}Hz, {self.mix_bits}bit, Ch: {self.mix_chans}")
        
        self.system = system.upper()
        self.is_pal = (self.system == 'PAL')
        self.fps = 49.86 if self.is_pal else 59.92
        self.debug_mode = debug
        self.filename = filename
        self.looping = False

        # --- AVF2 METADATA (optional) ---
        self.meta = meta
        self.poster = poster
        self.show_poster = show_poster
        self.profile = meta["profile"] if meta else 0
        if meta:
            print(f"[*] AVF2: '{meta['title']}' by {meta['author'] or '?'} "
                  f"({meta['system']}, {meta['duration_ms']/1000.0:.1f}s, "
                  f"{meta['profile_name']}, {meta['converter']})")
        
        # --- CRT EFFECTS ---
        self.show_scanlines = True
        self.enable_blending = True
        
        # --- DISPLAY ---
        self.scale_x = scale * 2 # Atari pixels are wide
        self.scale_y = scale
        self.window_w = WIDTH * self.scale_x
        self.window_h = HEIGHT * self.scale_y
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        if meta and meta.get("title"):
            caption = meta["title"] + (f" - {meta['author']}" if meta.get("author") else "")
        else:
            caption = os.path.basename(filename)
        pygame.display.set_caption(f"Python AVF Player | {caption}")
        
        # --- COLORS (GTIA) ---
        # The palette is the exact inverse of the avi2atari/phaeron encoder's
        # colour model (see _generate_gtia_palette), so the neutral defaults
        # below reproduce the source colours. Both stay live-tunable.
        self.phase_shift = 0.0      # Hue rotation in radians (0 = encoder-exact)
        self.saturation = 1.0       # Chroma gain (1.0 = Atari's amplitude of 40/255)
        self.palette = self._generate_gtia_palette()
        
        # --- DATA PROCESSING ---
        print(f"[*] Processing data...")
        self.video_frames, self.final_sound_obj, self.viz_array = self._load_process_full()
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 10, bold=True)
        
        # Scanline Mask (Pre-calculated for performance)
        # Resolution: 192 lines. We darken every second line.
        self.scanline_mask = np.ones((192, 1, 1), dtype=np.float32)
        self.scanline_mask[1::2, :, :] = 0.6 # Darken factor (0.0=Black, 1.0=Transparent)

    def _generate_gtia_palette(self):
        # Generates the 256-color palette (16 Hue * 16 Luma) as the EXACT
        # inverse of the encoder's colour model (avi2atari.py / phaeron's
        # encvideo50n/60n C++ sources).
        #
        # The encoder works in YIQ, not YUV: it computes
        #   fi = 0.595 R' - 0.274 G' - 0.321 B'   (the I axis)
        #   fq = 0.211 R' - 0.522 G' + 0.311 B'   (the Q axis)
        # and quantises every saturated pixel to a hue index whose I/Q
        # coordinates are:
        #   PAL : 14 hues, ITAB[c] = 40*cos(pi*(c-0.5)/7),  c = 1..14
        #   NTSC: 15 hues, ITAB[c] = 40*cos(pi*(c-1)/7.5),  c = 1..15
        # with chroma amplitude fixed at exactly 40 (out of 255) and
        # luma = nibble * 17. Decoding is therefore: rebuild I/Q from the
        # hue index, then apply the standard YIQ->RGB matrix.
        #
        # NTSC quirk: the encoder's hue *selection* (atan2*(7.5/pi) + 2.0)
        # emits indices shifted +2 steps relative to its own ITAB/QTAB
        # tables, so we subtract 2 steps (-48 degrees) when decoding.
        # Confirmed numerically: a round-trip grid search over the RGB cube
        # puts the error optimum at exactly -2 hue steps.
        #
        # phase_shift rotates the whole colour wheel (radians, 0 = neutral),
        # saturation scales chroma amplitude (1.0 = the Atari's 40/255;
        # ~2.0 approximates the punchier look of the original source).
        palette = np.zeros((256, 3), dtype=np.uint8)
        amp = 40.0 * self.saturation

        for chroma in range(16):
            if chroma == 0:
                # Grayscale row - no chroma
                i_val = 0.0
                q_val = 0.0
            else:
                if self.is_pal:
                    angle = math.pi * (chroma - 0.5) / 7.0
                else:
                    angle = math.pi * (chroma - 1 - 2) / 7.5
                angle += self.phase_shift
                i_val = amp * math.cos(angle)
                q_val = amp * math.sin(angle)

            for luma in range(16):
                y = luma * 17.0

                # YIQ -> RGB (standard matrix)
                r = y + 0.9563 * i_val + 0.6210 * q_val
                g = y - 0.2721 * i_val - 0.6474 * q_val
                b = y - 1.1070 * i_val + 1.7046 * q_val

                # Index: HighNibble=Chroma, LowNibble=Luma
                idx = (chroma << 4) | luma
                palette[idx] = [
                    max(0, min(255, int(round(r)))),
                    max(0, min(255, int(round(g)))),
                    max(0, min(255, int(round(b))))
                ]

        return palette

    def _decode_frame_gtia(self, luma_block, chroma_block):
        # Combines Luma and Chroma nibbles into palette indices.
        indices = (chroma_block.astype(np.int32) << 4) | luma_block.astype(np.int32)
        return self.palette[indices]

    def _load_process_full(self):
        # 1. READ FILE. Both profiles use the classic 8704-byte APAC mux
        # (C80 differs only in video content, decoded by _blit_frame_c80).
        frame_size = avf.frame_size(self.profile)
        size = os.path.getsize(self.filename)
        with open(self.filename, "rb") as f:
            if size % frame_size != 0: f.seek(HEADER_SIZE)
            raw = f.read()

        num_frames = len(raw) // frame_size
        video_frames = []
        audio_chunks = []

        off1, off2 = (120, 52) if self.is_pal else (70, 52)
        audio_len = 312 if self.is_pal else 262

        # 2. DEMUX LOOP
        for i in range(num_frames):
            base = i * frame_size
            chunk = raw[base : base+frame_size]

            # --- Video extraction ---
            v = np.zeros((HEIGHT, 40), dtype=np.uint8)
            for b in range(64):
                bb = b*128; y=b*3
                v[y] = np.frombuffer(chunk[bb+1:bb+41], dtype=np.uint8)
                v[y+1] = np.frombuffer(chunk[bb+45:bb+85], dtype=np.uint8)
                v[y+2] = np.frombuffer(chunk[bb+88:bb+128], dtype=np.uint8)
            video_frames.append(v)
            
            # --- Audio extraction ---
            ptr = 8192; ac = np.full(512, 50, dtype=np.uint8)
            for y in range(32):
                if ptr+9>=len(chunk): break
                ac[y]=chunk[ptr]; ac[y+off1]=chunk[ptr+1]
                ac[y+32+off1]=chunk[ptr+2]; ac[y+64+off1]=chunk[ptr+3]
                ac[y+96+off1]=chunk[ptr+4]; ac[y+128+off1]=chunk[ptr+5]
                ac[y+160+off1]=chunk[ptr+6]; ac[y+off2]=chunk[ptr+7]
                ac[y+32+off2]=chunk[ptr+8]; ptr+=10
            for y in range(19):
                if ptr>=len(chunk): break
                ac[y+32]=chunk[ptr]; ptr+=1
                if self.is_pal:
                    if ptr<len(chunk): ac[y+64+off2]=chunk[ptr]; ptr+=1
                    ptr+=8
                else: ptr+=9
            if ptr<len(chunk): ac[51]=chunk[ptr]
            audio_chunks.append(ac[:audio_len])

        # 3. AUDIO SYNC & RESAMPLING
        raw_audio = np.concatenate(audio_chunks).astype(np.float32) if audio_chunks else np.zeros(1000)
        # Center to 0 (Atari uses 0-100, silence ~50)
        raw_audio = np.clip(raw_audio, 0, 100) - 50.0
        
        # Calculate precise duration
        vid_dur = num_frames / self.fps
        tgt_samples = int(vid_dur * self.mix_freq)
        
        # Time Stretch Interpolation
        resampled = np.interp(
            np.linspace(0, 1, tgt_samples),
            np.linspace(0, 1, len(raw_audio)),
            raw_audio
        )
        # Scaling & Clipping to int16
        resampled = np.clip(resampled * 500.0, -32000, 32000).astype(np.int16)
        
        # Hardware Channel Mapping (Mono -> N-Channels)
        if self.mix_chans > 1:
            final = np.tile(resampled.reshape(-1, 1), (1, self.mix_chans))
        else:
            final = resampled
            
        final = np.ascontiguousarray(final)
        snd = pygame.sndarray.make_sound(final)
        
        # Visualization array (take only first channel)
        viz = final[:, 0] if self.mix_chans > 1 else final
        
        return video_frames, snd, viz

    def _blit_frame_c80(self, vf):
        # AVF-C80 M4 profile: 24 cell rows x [chars|R|G|B] planes of 40+40
        # bytes each; every RGB byte packs BG in the high nibble (top pixel)
        # and FG in the low nibble (bottom pixel) -> an 80x48 RGB444 image.
        a = vf.reshape(24, 8, 40)
        Rb = a[:, 2:4, :].reshape(24, 80).astype(np.int32)
        Gb = a[:, 4:6, :].reshape(24, 80).astype(np.int32)
        Bb = a[:, 6:8, :].reshape(24, 80).astype(np.int32)
        rgb = np.zeros((48, 80, 3), dtype=np.float32)
        rgb[0::2, :, 0] = (Rb >> 4) * 17;  rgb[1::2, :, 0] = (Rb & 15) * 17
        rgb[0::2, :, 1] = (Gb >> 4) * 17;  rgb[1::2, :, 1] = (Gb & 15) * 17
        rgb[0::2, :, 2] = (Bb >> 4) * 17;  rgb[1::2, :, 2] = (Bb & 15) * 17
        rgb_192 = np.repeat(np.repeat(rgb, 4, axis=0), 2, axis=1)
        if self.show_scanlines:
            rgb_192 = (rgb_192 * self.scanline_mask).astype(np.uint8)
        else:
            rgb_192 = rgb_192.astype(np.uint8)
        surf = pygame.surfarray.make_surface(rgb_192.swapaxes(0, 1))
        self.screen.blit(pygame.transform.scale(surf, (self.window_w, self.window_h)), (0, 0))

    def _blit_frame(self, vf):
        # Full render pipeline for one 192x40 frame (video or AVF2 poster):
        # line split, nibble unpack, palette decode, upscale, blend, scanlines.
        if self.profile == 2:
            return self._blit_frame_c80(vf)
        chroma_line = (vf[0::2] if self.is_pal else vf[1::2])
        luma_line   = (vf[1::2] if self.is_pal else vf[0::2])

        h_proc = min(len(chroma_line), len(luma_line))

        # Unpack nibbles (96 lines, 80 bytes -> 160 pixels)
        c_unp = np.stack([(chroma_line[:h_proc]>>4)&0xF, chroma_line[:h_proc]&0xF], axis=-1).reshape(h_proc, 80)
        l_unp = np.stack([(luma_line[:h_proc]>>4)&0xF, luma_line[:h_proc]&0xF], axis=-1).reshape(h_proc, 80)

        # 1. Base RGB Decoding (96 x 80)
        rgb_base = self._decode_frame_gtia(l_unp, c_unp).astype(np.float32)

        # Scale horizontally x2 (96 x 160)
        rgb_wide = np.repeat(rgb_base, 2, axis=1)

        # 2. Horizontal Blending (Blur)
        if self.enable_blending:
            blended = np.zeros_like(rgb_wide)
            blended[:, 1:] = (rgb_wide[:, 1:] + rgb_wide[:, :-1]) * 0.5
            blended[:, 0] = rgb_wide[:, 0]
            rgb_wide = blended

        # Scale vertically x2 (192 x 160)
        rgb_192 = np.repeat(rgb_wide, 2, axis=0)

        # 3. Scanlines (Vertical lines)
        if self.show_scanlines:
            rgb_192 = (rgb_192 * self.scanline_mask).astype(np.uint8)
        else:
            rgb_192 = rgb_192.astype(np.uint8)

        # Blit to screen
        surf = pygame.surfarray.make_surface(rgb_192.swapaxes(0, 1))
        self.screen.blit(pygame.transform.scale(surf, (self.window_w, self.window_h)), (0,0))

    def _show_poster(self):
        # Static AVF2 poster view; any key starts playback, ESC quits.
        # S/B and the phase/saturation keys work here too, so the poster
        # doubles as a colour-tuning screen.
        vf = np.frombuffer(self.poster, dtype=np.uint8).reshape(HEIGHT, 40)
        while True:
            self._blit_frame(vf)
            lbl = self.font.render(
                "AVF2 POSTER - any key: play | ESC: quit | S/B/[/] tune", True, (255,255,0))
            self.screen.blit(lbl, (10, 10))
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit(0)
                    mods = pygame.key.get_mods()
                    if e.key == pygame.K_s: self.show_scanlines = not self.show_scanlines
                    elif e.key == pygame.K_b: self.enable_blending = not self.enable_blending
                    elif e.key == pygame.K_RIGHTBRACKET and (mods & pygame.KMOD_SHIFT):
                        self.saturation = min(2.0, self.saturation + 0.05); self.palette = self._generate_gtia_palette()
                    elif e.key == pygame.K_LEFTBRACKET and (mods & pygame.KMOD_SHIFT):
                        self.saturation = max(0.0, self.saturation - 0.05); self.palette = self._generate_gtia_palette()
                    elif e.key == pygame.K_RIGHTBRACKET:
                        self.phase_shift += 0.05; self.palette = self._generate_gtia_palette()
                    elif e.key == pygame.K_LEFTBRACKET:
                        self.phase_shift -= 0.05; self.palette = self._generate_gtia_palette()
                    else:
                        return
            self.clock.tick(30)

    def run(self):
        if self.show_poster and self.poster is not None:
            self._show_poster()
        print("[*] PLAYER STARTED. Controls:")
        print("    [ S ]          Scanlines (Toggle)")
        print("    [ B ]          Blending (Toggle)")
        print("    [ [ ] / [ ] ]  Phase +/- 0.05")
        print("    [ Shift+[] ]   Saturation +/- 0.05")
        print("    [ L ]          Loop")
        print("    [ D ]          Oscilloscope")
        
        running = True
        while running:
            self.final_sound_obj.play()
            start_ticks = pygame.time.get_ticks()
            paused = False; pause_start = 0
            
            loop_active = True
            while loop_active and running:
                if not paused:
                    ticks = pygame.time.get_ticks() - start_ticks
                    f_idx = int((ticks/1000.0) * self.fps)
                
                # Handle End of File
                if f_idx >= len(self.video_frames):
                    self.final_sound_obj.stop()
                    if self.looping: 
                        loop_active = False # Triggers restart
                        continue 
                    else: 
                        running = False 
                        break

                # Input Handling
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: running=False; loop_active=False
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_ESCAPE: running=False; loop_active=False
                        if e.key == pygame.K_l: self.looping = not self.looping
                        if e.key == pygame.K_d: self.debug_mode = not self.debug_mode
                        if e.key == pygame.K_s: self.show_scanlines = not self.show_scanlines
                        if e.key == pygame.K_b: self.enable_blending = not self.enable_blending
                        if e.key == pygame.K_SPACE:
                            if paused: pygame.mixer.unpause(); start_ticks += (pygame.time.get_ticks()-pause_start); paused=False
                            else: pygame.mixer.pause(); pause_start = pygame.time.get_ticks(); paused=True
                        
                        # Live Tuning
                        mods = pygame.key.get_mods()
                        regen = False
                        if mods & pygame.KMOD_SHIFT:
                            if e.key == pygame.K_RIGHTBRACKET: self.saturation = min(2.0, self.saturation + 0.05); regen=True
                            if e.key == pygame.K_LEFTBRACKET: self.saturation = max(0.0, self.saturation - 0.05); regen=True
                        else:
                            if e.key == pygame.K_LEFTBRACKET: self.phase_shift -= 0.05; regen=True
                            if e.key == pygame.K_RIGHTBRACKET: self.phase_shift += 0.05; regen=True
                        
                        if regen: self.palette = self._generate_gtia_palette()

                if paused: self.clock.tick(10); continue

                # --- RENDER PIPELINE ---
                self._blit_frame(self.video_frames[f_idx])

                # GUI Overlays
                if self.debug_mode: self._draw_oscilloscope(ticks)
                else: self._draw_progressbar(f_idx)
                
                status = f"S:{'ON' if self.show_scanlines else 'OFF'} | B:{'ON' if self.enable_blending else 'OFF'}"
                lbl = self.font.render(f"{status} | Ph: {self.phase_shift:.2f} | Sat: {self.saturation:.2f}", True, (255,255,0))
                self.screen.blit(lbl, (10, 10))

                pygame.display.flip()
                self.clock.tick(self.fps * 1.5)
        pygame.quit()

    def _draw_oscilloscope(self, ms):
        idx = int((ms/1000.0)*self.mix_freq); w=1000
        if idx < len(self.viz_array):
            c = self.viz_array[idx:idx+w]
            s=pygame.Surface((self.window_w, 100)); s.set_alpha(150); s.fill((0,0,0))
            pygame.draw.line(s,(50,50,50),(0,50),(self.window_w,50))
            pts=[]
            if len(c)>0:
                step=max(1, len(c)/self.window_w)
                for i in range(0,len(c),int(step)):
                    pts.append(((i/len(c))*self.window_w, 50-(c[i]/32000*50)))
                if len(pts)>1: pygame.draw.lines(s,(0,255,0),False,pts,2)
            self.screen.blit(s,(0,self.window_h-100))

    def _draw_progressbar(self, idx):
        y=self.window_h-10; w=self.window_w
        pygame.draw.rect(self.screen,(50,50,50),(0,y,w,10))
        pygame.draw.rect(self.screen,(0,100,255),(0,y, (idx/len(self.video_frames))*w, 10))
        if self.looping: t=self.font.render("LOOP",1,(0,255,0)); self.screen.blit(t,(w-50,10))

def main():
    avf.print_banner("Player", __version__)
    parser = argparse.ArgumentParser(description="AVF Video Player")
    parser.add_argument("file", help="Input AVF file")
    parser.add_argument("system", nargs="?", default=None,
                        help="TV System (PAL/NTSC). Default: from the AVF2 header "
                             "if present, otherwise PAL")
    parser.add_argument("--scale", type=int, default=3, help="Window scale factor (default 3)")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlay")
    parser.add_argument("--info", action="store_true",
                        help="Print AVF2 metadata (if any) and exit without playing")
    parser.add_argument("--poster", action="store_true",
                        help="Show the AVF2 poster frame before playback (any key starts)")
    parser.add_argument("--version", action="version", version=f"avfplayer {__version__}")

    args = parser.parse_args()
    if not os.path.exists(args.file):
        print("Error: File not found.")
        return 2

    meta, poster = parse_avf2(args.file)

    if args.info:
        if meta is None:
            print(f"{args.file}: no AVF2 metadata (plain AVF v1 or header-less file)")
        else:
            print(f"{args.file}: AVF2 v{meta['version']}")
            print(f"  System:    {meta['system']}")
            print(f"  Profile:   {meta['profile_name']}")
            print(f"  Frames:    {meta['frame_count']}  ({meta['duration_ms']/1000.0:.1f}s)")
            print(f"  Date:      {meta['date']}")
            print(f"  Title:     {meta['title']}")
            print(f"  Author:    {meta['author']}")
            print(f"  Converter: {meta['converter']}")
            print(f"  Comment:   {meta['comment']}")
            print(f"  Poster:    {'yes' if poster is not None else 'no'}")
        return 0

    global pygame
    import pygame
    import pygame.sndarray

    system = args.system or (meta["system"] if meta else "PAL")
    if meta and args.system and args.system.upper() != meta["system"]:
        print(f"[!] Requested {args.system.upper()} but the AVF2 header says "
              f"{meta['system']} - using {args.system.upper()} as requested.")

    AVFPlayer(args.file, system, args.scale, args.debug,
              meta=meta, poster=poster, show_poster=args.poster).run()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
