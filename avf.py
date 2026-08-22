#!/usr/bin/env python3
"""avf - umbrella CLI for the avi2atari PC suite.

    avf convert <video> [opts]   convert video to AVF/AVF2 (avi2atari)
    avf check   <file.avf>       validate the binary layout (avf_check)
    avf play    <file.avf>       play in a window (avfplayer; needs pygame)
    avf meta    <file.avf>       show AVF2 metadata (read-only)

Each subcommand forwards its own --help. The suite shares one format
library (avf_format), so the converter, checker and player can never
drift on the container layout again.
"""
import sys

USAGE = __doc__


def meta_cmd(argv):
    import argparse
    import os
    import avf_format as avf
    ap = argparse.ArgumentParser(prog="avf meta",
                                 description="Show AVF2 metadata (read-only)")
    ap.add_argument("file")
    a = ap.parse_args(argv)
    if not os.path.exists(a.file):
        print("Error: file not found."); return 2
    with open(a.file, "rb") as f:
        buf = f.read(avf.HEADER_SIZE)
    meta, problems = avf.parse_avf2_header(buf)
    if meta is None:
        print(f"{a.file}: no AVF2 metadata (Legacy/v1 or header-less file)")
        return 0
    secs = meta["duration_ms"] / 1000.0
    print(f"{a.file}: AVF2 v{meta['version']}")
    print(f"  System:    {meta['system']}")
    print(f"  Profile:   {meta['profile_name']}")
    print(f"  Frames:    {meta['frame_count']}  ({secs:.1f}s)")
    print(f"  Date:      {meta['date']}")
    print(f"  Title:     {meta['title']}")
    print(f"  Author:    {meta['author']}")
    print(f"  Converter: {meta['converter']}")
    print(f"  Comment:   {meta['comment']}")
    print(f"  Poster:    {'yes' if meta['poster'] else 'no'}"
          + ("  (profile-encoded)" if meta['poster_c80'] else ""))
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd = sys.argv[1]
    argv = sys.argv[2:]
    # each tool parses sys.argv[1:]; present a clean prog name to argparse
    sys.argv = [f"avf {cmd}"] + argv
    if cmd == "convert":
        import avi2atari
        return avi2atari.main() or 0
    if cmd == "check":
        import avf_check
        return avf_check.main() or 0
    if cmd == "play":
        import avfplayer
        return avfplayer.main() or 0
    if cmd == "meta":
        return meta_cmd(argv)
    print(f"unknown command: {cmd}\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main() or 0)
