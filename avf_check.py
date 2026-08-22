#!/usr/bin/env python3

# avf_check.py - Structural validator for AVF files produced by avi2atari.py
# Copyright (C) 2026 HanJammer & Lumen
#
# Verifies that the binary layout of an AVF file matches what movplay/AVFplay
# expects on Atari 8-bit. This catches structural drift WITHOUT needing to
# boot an emulator. Use it before/after encoder changes to confirm that no
# byte-layout regression has been introduced.
#
# Checks:
#   - Optional 8 KB (16 sectors) header is present and zeroed
#   - Total payload length is an exact multiple of 8704 (frame size)
#   - In every frame, the fixed padding bytes inside video blocks are zero
#   - Audio block tail padding is zero
#
# Usage:
#   python avf_check.py file.avf [--no-header] [--system PAL|NTSC]
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
import argparse

import avf_format as avf

__version__ = "0.6"

FRAME_SIZE = avf.FRAME_SIZE_APAC
HEADER_SIZE = avf.HEADER_SIZE


parse_avf2_header = avf.parse_avf2_header


def check_header(buf):
    if len(buf) < HEADER_SIZE:
        return [f"file shorter than expected header ({len(buf)} < {HEADER_SIZE})"]
    nz = sum(1 for b in buf[:HEADER_SIZE] if b != 0)
    if nz:
        return [f"header is not all zeroes ({nz} non-zero bytes in first {HEADER_SIZE})"]
    return []


def check_video_block_padding(frame, frame_idx):
    # Each frame begins with 64 video sub-blocks of 128 bytes (64 * 128 = 8192).
    # Layout per block (see mux_frame in avi2atari.py):
    #   [0]      : 0x00
    #   [1..40]  : line N
    #   [41..43] : 0x00 0x00 0x00
    #   [44]     : 0x00
    #   [45..84] : line N+1
    #   [85..87] : 0x00 0x00 0x00
    #   [88..127]: line N+2
    errors = []
    for b in range(64):
        base = b * 128
        zeros_at = [0, 41, 42, 43, 44, 85, 86, 87]
        for off in zeros_at:
            if frame[base + off] != 0:
                errors.append(
                    f"frame {frame_idx}, video block {b}, offset {off}: "
                    f"expected 0x00, got 0x{frame[base + off]:02x}"
                )
                if len(errors) > 5:
                    errors.append("  (more video padding errors suppressed)")
                    return errors
    return errors


def check_audio_block_padding(frame, frame_idx, is_pal):
    # Audio area starts at offset 8192 inside the frame and is 512 bytes long.
    # Layout: 32 lines of 10 bytes (loop 1) + 19 lines of 10 bytes (loop 2)
    #         + 2 trailing bytes = 320 + 190 + 2 = 512.
    audio_base = 8192
    errors = []

    # Loop 1: each 10-byte line ends with a single zero byte at offset 9.
    for y in range(32):
        if frame[audio_base + y * 10 + 9] != 0:
            errors.append(
                f"frame {frame_idx}, audio loop1 line {y}, "
                f"offset {audio_base + y * 10 + 9}: expected 0x00"
            )
            break

    # Loop 2: 19 lines of 10 bytes starting at audio_base + 320.
    # PAL: bytes [2..9] are zero (8 zeroes). NTSC: bytes [1..9] are zero (9 zeroes).
    loop2_base = audio_base + 320
    pad_start = 2 if is_pal else 1
    for y in range(19):
        line_off = loop2_base + y * 10
        for k in range(pad_start, 10):
            if frame[line_off + k] != 0:
                errors.append(
                    f"frame {frame_idx}, audio loop2 line {y}, "
                    f"offset {line_off + k}: expected 0x00 (system={'PAL' if is_pal else 'NTSC'})"
                )
                break
        if errors and "loop2" in errors[-1]:
            break

    # Trailing 2 bytes: a data byte (any value) followed by 0x00.
    if frame[audio_base + 511] != 0:
        errors.append(
            f"frame {frame_idx}, audio trailing byte at offset {audio_base + 511}: expected 0x00"
        )

    return errors


def validate(path, has_header, system):
    is_pal = (system == "PAL")
    size = os.path.getsize(path)

    with open(path, "rb") as f:
        data = f.read()

    problems = []

    meta = None
    if has_header:
        meta, hdr_problems = parse_avf2_header(data)
        if meta is None:
            # Plain AVF v1: header must be all zeroes.
            problems += check_header(data)
        else:
            problems += hdr_problems
        payload = data[HEADER_SIZE:]
    else:
        payload = data

    # C80 profiles use 8192-byte frames (7680 raw video + 512 linear
    # audio, no pads); APAC keeps the classic 8704-byte mux.
    c80 = bool(meta) and avf.is_c80(meta["profile"])
    frame_size = 8192 if c80 else FRAME_SIZE

    if len(payload) % frame_size != 0:
        problems.append(
            f"payload length {len(payload)} is not a multiple of frame size {frame_size} "
            f"(remainder {len(payload) % frame_size})"
        )

    n_frames = len(payload) // frame_size

    # AVF2 cross-checks: the header must agree with reality.
    if meta is not None:
        if meta["frame_count"] != n_frames:
            problems.append(
                f"AVF2 frame_count is {meta['frame_count']} but payload holds {n_frames} frames"
            )
        if meta["system"] != system:
            problems.append(
                f"AVF2 header says {meta['system']} but validation was requested for {system}"
            )
    sample_count = 0
    sample_step = max(1, n_frames // 20)  # sample up to ~20 frames across the file

    for idx in range(0, n_frames, sample_step):
        frame = payload[idx * frame_size:(idx + 1) * frame_size]
        if c80:
            continue        # C80 frames carry no pad structure to verify
        problems += check_video_block_padding(frame, idx)
        problems += check_audio_block_padding(frame, idx, is_pal)
        sample_count += 1
        if len(problems) > 30:
            problems.append("  (further frame errors suppressed)")
            break

    print(f"File: {path}")
    print(f"  Size:   {size} bytes")
    if has_header:
        header_kind = "AVF2 metadata (8192 B)" if meta is not None else "AVF v1, zeroed (8192 B)"
    else:
        header_kind = "none"
    print(f"  Header: {header_kind}")
    print(f"  Frames: {n_frames}  (sampled {sample_count})")
    print(f"  System: {system}")
    if meta is not None:
        secs = meta["duration_ms"] / 1000.0
        print(f"  AVF2:   v{meta['version']}  {meta['system']}  "
              f"{meta['frame_count']} frames  {secs:.1f}s  "
              f"poster={'yes' if meta['poster'] else 'no'}  date={meta['date']}")
        print(f"    Profile:   {meta['profile_name']}"
              + ("  (poster in profile encoding)" if meta['poster_c80'] else ""))
        print(f"    Title:     {meta['title']}")
        print(f"    Author:    {meta['author']}")
        print(f"    Converter: {meta['converter']}")
        if meta["comment"]:
            print(f"    Comment:   {meta['comment']}")

    if problems:
        print(f"  RESULT: FAIL ({len(problems)} issue(s))")
        for p in problems:
            print(f"    - {p}")
        return 1
    else:
        print(f"  RESULT: OK")
        return 0


def main():
    avf.print_banner("Validator", __version__)
    ap = argparse.ArgumentParser(description="AVF structural validator")
    ap.add_argument("file", help="AVF file to validate")
    ap.add_argument("--no-header", action="store_true",
                    help="File was produced with --no-header (skip header check)")
    ap.add_argument("--system", choices=["PAL", "NTSC"], default="PAL",
                    help="Target system (affects loop2 padding pattern). Default: PAL")
    ap.add_argument("--version", action="version", version=f"avf_check {__version__}")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}")
        sys.exit(2)

    sys.exit(validate(args.file, has_header=not args.no_header, system=args.system))


if __name__ == "__main__":
    main()
