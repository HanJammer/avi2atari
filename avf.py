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
    avf.print_banner("Metadata", avf.__version__)
    ap = argparse.ArgumentParser(
        prog="avf meta",
        description="Show or edit AVF2 metadata; upgrade Legacy (v1) to AVF2.")
    ap.add_argument("file")
    ap.add_argument("--set-title", metavar="T", help="set AVF2 title")
    ap.add_argument("--set-author", metavar="A", help="set AVF2 author")
    ap.add_argument("--set-comment", metavar="C", help="set AVF2 comment")
    ap.add_argument("--to-avf2", action="store_true",
                    help="upgrade a Legacy (v1) file to AVF2")
    ap.add_argument("--system", choices=["PAL", "NTSC"],
                    help="TV system for --to-avf2 (REQUIRED; v1 has no marker)")
    ap.add_argument("--poster-frame", type=int, metavar="N",
                    help="--to-avf2: use frame N's picture as the poster")
    ap.add_argument("--out", metavar="FILE",
                    help="write to FILE instead of editing in place / the "
                         "default <base>-avf2.avf")
    a = ap.parse_args(argv)
    if not os.path.exists(a.file):
        print("Error: file not found."); return 2

    updates = {}
    if a.set_title is not None:
        updates["title"] = a.set_title
    if a.set_author is not None:
        updates["author"] = a.set_author
    if a.set_comment is not None:
        updates["comment"] = a.set_comment

    info = avf.classify(a.file)

    # ---- upgrade Legacy -> AVF2 -------------------------------------------
    if a.to_avf2:
        if info["kind"] == "avf2":
            print("Already AVF2 - use --set-title/--set-author/--set-comment.")
            return 2
        if info["kind"] == "unknown":
            print("Unrecognised AVF layout - not a plain Legacy (v1) stream?")
            return 2
        if not a.system:
            print("ERROR: --to-avf2 needs --system PAL|NTSC "
                  "(a Legacy stream carries no TV-system marker).")
            return 2
        data = avf.legacy_to_avf2_bytes(a.file, a.system == "PAL", updates,
                                        a.poster_frame)
        out = a.out or (os.path.splitext(a.file)[0] + "-avf2.avf")
        with open(out, "wb") as f:
            f.write(data)
        n = (len(data) - avf.HEADER_SIZE) // avf.FRAME_SIZE_APAC
        print(f"Upgraded Legacy -> AVF2: {out}")
        print(f"  {a.system}, {n} frames"
              + (f", poster from frame {a.poster_frame}" if a.poster_frame
                 is not None else ", no poster"))
        return 0

    # ---- edit existing AVF2 fields ----------------------------------------
    if updates:
        if info["kind"] != "avf2":
            print("Not an AVF2 file - upgrade it first with "
                  "--to-avf2 --system PAL|NTSC.")
            return 2
        new_header = avf.set_metadata_bytes(info["header_bytes"], updates)
        if a.out:
            with open(a.file, "rb") as f:
                f.seek(avf.HEADER_SIZE)
                payload = f.read()
            with open(a.out, "wb") as f:
                f.write(new_header + payload)
            print(f"Wrote {a.out} with updated metadata.")
        else:
            with open(a.file, "r+b") as f:      # in place: only the 8 KB header
                f.seek(0)
                f.write(new_header)
            print(f"Updated metadata in place: {a.file}")
        for k, v in updates.items():
            print(f"  {k}: {v!r}")
        return 0

    # ---- read-only display ------------------------------------------------
    meta = info["meta"]
    if meta is None:
        kind = {"legacy_headerless": "Legacy (v1), header-less",
                "legacy_zero": "Legacy (v1), zeroed 8 KB header",
                "unknown": "unrecognised"}.get(info["kind"], info["kind"])
        print(f"{a.file}: {kind} - {info['n_frames']} frames, no metadata")
        print("  Upgrade with: avf meta <file> --to-avf2 --system PAL|NTSC")
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
    if info["problems"]:
        print("  PROBLEMS:")
        for p in info["problems"]:
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
