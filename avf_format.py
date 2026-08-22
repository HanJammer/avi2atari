#!/usr/bin/env python3
"""avf_format - shared AVF / AVF2 container primitives for the PC suite.

One home for the format so the converter, checker, player and web app can
never drift apart again (the profile byte once lived in three near-copies
of parse_avf2). Pure standard library - numpy stays optional and is only
needed by the frame decoders in the tools, not here.

Layout reference: avi2atari docs/AVF_FORMAT.md and docs/AVF2_HEADER.md.
"""
import os
import time

__version__ = "1.1"

# --- geometry ---------------------------------------------------------------
WIDTH, HEIGHT = 160, 192          # APAC luma grid
HEADER_SIZE = 16 * 512            # 8192: the region every player skips
VIDEO_BYTES = 7680                # picture payload per frame (both profiles)
FRAME_SIZE_APAC = 8704            # APAC: video + padded/smeared audio mux
FRAME_SIZE_C80 = 8192             # C80: 7680 raw video + 512 linear audio

# --- video profiles (AVF2 header byte 7) ------------------------------------
PROFILE_APAC = 0
PROFILE_C80M3 = 1                 # reserved, not implemented
PROFILE_C80M4 = 2
PROFILE_NAMES = {0: "APAC", 1: "C80M3", 2: "C80M4"}
C80_HALFBLOCK = 0x01             # C80 cell glyph: rows 0-3 BG, rows 4-7 FG

# --- TV systems (AVF2 header byte 5) ----------------------------------------
SYSTEM_NAMES = {0: "PAL", 1: "NTSC"}
# Exact frame periods (POKEY line rate * lines / colour clock).
SECONDS_PER_FRAME_PAL = 114.0 * 312.0 / 1773447.5
SECONDS_PER_FRAME_NTSC = 114.0 * 262.0 / 1789772.5
FPS_PAL = 1.0 / SECONDS_PER_FRAME_PAL      # ~49.86
FPS_NTSC = 1.0 / SECONDS_PER_FRAME_NTSC    # ~59.92

# --- APAC audio mux parameters (used by the player/checker demux) -----------
AUDIO_CHUNK_PAL = 312
AUDIO_CHUNK_NTSC = 262
AUDIO_OFF1_PAL = 120
AUDIO_OFF1_NTSC = 70
AUDIO_OFF2 = 52


def frame_size(profile):
    """Bytes per muxed frame for a video profile."""
    return FRAME_SIZE_C80 if profile in (PROFILE_C80M3, PROFILE_C80M4) \
        else FRAME_SIZE_APAC


def is_c80(profile):
    return profile in (PROFILE_C80M3, PROFILE_C80M4)


def seconds_per_frame(is_pal):
    return SECONDS_PER_FRAME_PAL if is_pal else SECONDS_PER_FRAME_NTSC


def checksum(sector0):
    """AVF2 sector-0 checksum: 16-bit sum of bytes 0..509."""
    return sum(sector0[0:510]) & 0xFFFF


def _avf2_text(value, length):
    """Fixed-width NUL-padded ASCII field; non-ASCII dropped."""
    raw = (value or "").encode("ascii", "ignore")[:length]
    return raw + b"\x00" * (length - len(raw))


def build_avf2_header(is_pal, frame_count, poster_data, meta,
                      profile=PROFILE_APAC):
    """Build the 8192-byte AVF2 header (sector 0 + 7680-byte poster).

    meta: dict with optional title/author/comment. poster_data: 7680 bytes
    or None. profile: PROFILE_* (poster is flagged as profile-encoded when
    non-APAC).
    """
    spf = seconds_per_frame(is_pal)
    duration_ms = int(round(frame_count * spf * 1000.0))
    meta = meta or {}

    s0 = bytearray(512)
    s0[0:4] = b"AVF2"
    s0[4] = 1                                       # header_version
    s0[5] = 0 if is_pal else 1                      # system
    flags = 1 if poster_data else 0                 # bit0 = poster present
    if poster_data and profile != PROFILE_APAC:
        flags |= 2                                  # bit1 = poster in-profile
    s0[6] = flags
    s0[7] = profile                                 # video profile
    s0[8:12] = int(frame_count).to_bytes(4, "little")
    s0[12:16] = int(duration_ms).to_bytes(4, "little")
    s0[16:24] = time.strftime("%Y%m%d").encode("ascii")
    s0[64:128] = _avf2_text(meta.get("title"), 64)
    s0[128:192] = _avf2_text(meta.get("author"), 64)
    s0[192:224] = _avf2_text(meta.get("converter")
                             or f"avi2atari {meta.get('_conv_ver', '')}".strip(),
                             32)
    s0[224:352] = _avf2_text(meta.get("comment"), 128)
    s0[510:512] = checksum(s0).to_bytes(2, "little")

    poster = poster_data if poster_data else b"\x00" * VIDEO_BYTES
    if len(poster) != VIDEO_BYTES:
        raise ValueError(f"poster must be {VIDEO_BYTES} bytes, got {len(poster)}")
    return bytes(s0) + poster


def _text(s0, lo, hi):
    raw = bytes(s0[lo:hi]).rstrip(b"\x00")
    return "".join(chr(b) if 0x20 <= b <= 0x7E else "?" for b in raw)


def parse_avf2_header(buf):
    """Parse an AVF2 header. Returns (meta dict, problems list).

    (None, []) when the buffer is not AVF2 (a plain Legacy/v1 zero header
    is expected instead). meta carries everything a caller needs, including
    'profile'/'profile_name', 'poster', 'poster_c80', 'checksum_ok'.
    """
    if len(buf) < HEADER_SIZE or buf[0:4] != b"AVF2":
        return None, []

    problems = []
    s0 = buf[:512]

    stored = int.from_bytes(s0[510:512], "little")
    actual = checksum(s0)
    checksum_ok = (stored == actual)
    if not checksum_ok:
        problems.append(f"AVF2 checksum mismatch (stored 0x{stored:04x}, "
                        f"computed 0x{actual:04x})")

    version = s0[4]
    if version < 1:
        problems.append(f"AVF2 header_version is {version}, expected >= 1")
    if s0[5] not in (0, 1):
        problems.append(f"AVF2 system byte is {s0[5]}, expected 0/1")

    profile = s0[7]
    if profile not in PROFILE_NAMES:
        problems.append(f"AVF2 profile byte is {profile}, expected 0/1/2")
    if (s0[6] & 2) and profile == PROFILE_APAC:
        problems.append("AVF2 flags bit1 (profile-encoded poster) set on "
                        "an APAC file")

    meta = {
        "version": version,
        "system": SYSTEM_NAMES.get(s0[5], f"?{s0[5]}"),
        "is_pal": s0[5] == 0,
        "poster": bool(s0[6] & 1),
        "poster_c80": bool(s0[6] & 2),
        "profile": profile,
        "profile_name": PROFILE_NAMES.get(profile, f"?{profile}"),
        "frame_count": int.from_bytes(s0[8:12], "little"),
        "duration_ms": int.from_bytes(s0[12:16], "little"),
        "date": _text(s0, 16, 24),
        "title": _text(s0, 64, 128),
        "author": _text(s0, 128, 192),
        "converter": _text(s0, 192, 224),
        "comment": _text(s0, 224, 352),
        "checksum_ok": checksum_ok,
    }

    if not meta["poster"]:
        nz = sum(1 for b in buf[512:HEADER_SIZE] if b != 0)
        if nz:
            problems.append(f"AVF2 poster flag is 0 but poster area has "
                            f"{nz} non-zero bytes")

    return meta, problems


def poster_from_header(buf, meta):
    """The 7680-byte poster if the header flags one, else None."""
    if meta and meta.get("poster") and len(buf) >= HEADER_SIZE:
        return buf[512:HEADER_SIZE]
    return None


# ===========================================================================
# Metadata editing + Legacy(v1) -> AVF2 upgrade (used by `avf meta` and web)
# ===========================================================================

def demux_apac_video(frame):
    """Extract the 7680 raw video bytes from one 8704-byte APAC frame
    (inverse of the converter's 64 x 128-byte block mux)."""
    if len(frame) < FRAME_SIZE_APAC:
        raise ValueError("APAC frame must be 8704 bytes")
    out = bytearray()
    for b in range(64):
        blk = frame[b * 128:(b + 1) * 128]
        out += blk[1:41] + blk[45:85] + blk[88:128]   # line0, line1, line2
    return bytes(out)


def classify(path):
    """Inspect an AVF file. Returns a dict:
      kind: 'avf2' | 'legacy_zero' | 'legacy_headerless' | 'unknown'
      profile, meta (or None), header_bytes (8192 or None), payload_off,
      frame_size, n_frames, is_pal (best guess or None).
    Legacy = plain AVF v1 (no populated AVF2 header)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)

    meta, problems = parse_avf2_header(head)
    if meta is not None:
        prof = meta["profile"]
        fs = frame_size(prof)
        payload = size - HEADER_SIZE
        return {
            "kind": "avf2", "profile": prof, "meta": meta,
            "header_bytes": head, "payload_off": HEADER_SIZE,
            "frame_size": fs, "n_frames": payload // fs if fs else 0,
            "is_pal": meta["is_pal"], "problems": problems,
        }

    # No AVF2 header. Legacy is always APAC (C80 requires the header).
    fs = FRAME_SIZE_APAC
    if size % fs == 0:
        return {"kind": "legacy_headerless", "profile": PROFILE_APAC,
                "meta": None, "header_bytes": None, "payload_off": 0,
                "frame_size": fs, "n_frames": size // fs, "is_pal": None,
                "problems": []}
    if size >= HEADER_SIZE and (size - HEADER_SIZE) % fs == 0 \
            and not any(head):
        return {"kind": "legacy_zero", "profile": PROFILE_APAC,
                "meta": None, "header_bytes": head, "payload_off": HEADER_SIZE,
                "frame_size": fs, "n_frames": (size - HEADER_SIZE) // fs,
                "is_pal": None, "problems": []}
    return {"kind": "unknown", "profile": None, "meta": None,
            "header_bytes": None, "payload_off": 0, "frame_size": fs,
            "n_frames": 0, "is_pal": None, "problems": ["unrecognised layout"]}


def set_metadata_bytes(header8192, updates):
    """Return a new 8192-byte AVF2 header with title/author/comment updated
    and the checksum recomputed. Poster, profile, system, frames, duration,
    date and converter are preserved."""
    s0 = bytearray(header8192[:512])
    if s0[0:4] != b"AVF2":
        raise ValueError("not an AVF2 header")
    if "title" in updates:
        s0[64:128] = _avf2_text(updates["title"], 64)
    if "author" in updates:
        s0[128:192] = _avf2_text(updates["author"], 64)
    if "comment" in updates:
        s0[224:352] = _avf2_text(updates["comment"], 128)
    s0[510:512] = checksum(s0).to_bytes(2, "little")
    return bytes(s0) + bytes(header8192[512:HEADER_SIZE])


def legacy_to_avf2_bytes(path, is_pal, meta=None, poster_frame=None):
    """Read a Legacy (v1) APAC file and return the AVF2-upgraded bytes.

    is_pal is a REQUIRED manual choice - v1 streams carry no TV-system
    marker. poster_frame (int) extracts that frame's picture as the poster;
    None means no poster. Raises if the file is already AVF2 or unrecognised.
    """
    info = classify(path)
    if info["kind"] == "avf2":
        raise ValueError("file is already AVF2 (use metadata editing)")
    if info["kind"] == "unknown":
        raise ValueError("unrecognised AVF layout - not a plain v1 stream?")
    with open(path, "rb") as f:
        f.seek(info["payload_off"])
        payload = f.read()
    n_frames = len(payload) // FRAME_SIZE_APAC

    poster = None
    if poster_frame is not None and n_frames:
        idx = max(0, min(poster_frame, n_frames - 1))
        frame = payload[idx * FRAME_SIZE_APAC:(idx + 1) * FRAME_SIZE_APAC]
        poster = demux_apac_video(frame)

    header = build_avf2_header(is_pal, n_frames, poster, meta or {},
                              PROFILE_APAC)
    return header + payload


# ===========================================================================
# Shared CLI banner for the suite
# ===========================================================================
BANNER_WIDTH = 78


_banner_shown = False


def print_banner(tool, version):
    """Print the suite banner to stderr, once per process (keeps stdout
    clean for piping; the umbrella + a sub-tool never double it)."""
    global _banner_shown
    if _banner_shown:
        return
    _banner_shown = True
    import sys
    rule = "-" * BANNER_WIDTH
    line = (f"AVI2ATARI AVF Suite - {tool} {version} "
            f"- HanJammer / RustyBits 2026")
    print(f"{rule}\n{line}\n{rule}", file=sys.stderr)


# ===========================================================================
# Accurate RGB decoders (shared by the web preview; the exact inverse of the
# encoder's YIQ model - ported from avfplayer's _generate_gtia_palette).
# numpy is imported lazily so header-only tools stay dependency-light.
# ===========================================================================
def gtia_palette(is_pal, phase_shift=0.0, saturation=1.0):
    import math
    import numpy as np
    pal = np.zeros((256, 3), dtype=np.uint8)
    amp = 40.0 * saturation
    for chroma in range(16):
        if chroma == 0:
            iv = qv = 0.0
        else:
            ang = (math.pi * (chroma - 0.5) / 7.0) if is_pal \
                else (math.pi * (chroma - 1 - 2) / 7.5)
            ang += phase_shift
            iv = amp * math.cos(ang)
            qv = amp * math.sin(ang)
        for luma in range(16):
            y = luma * 17.0
            r = y + 0.9563 * iv + 0.6210 * qv
            g = y - 0.2721 * iv - 0.6474 * qv
            b = y - 1.1070 * iv + 1.7046 * qv
            pal[(chroma << 4) | luma] = [
                min(255, max(0, int(round(v)))) for v in (r, g, b)]
    return pal


def decode_apac_rgb(video_bytes, is_pal, phase_shift=0.0, saturation=1.0):
    """7680 APAC bytes -> (96, 80, 3) uint8, encoder-exact GTIA colours."""
    import numpy as np
    pal = gtia_palette(is_pal, phase_shift, saturation)
    vf = np.frombuffer(bytes(video_bytes[:VIDEO_BYTES]), np.uint8).reshape(192, 40)
    chroma = vf[0::2] if is_pal else vf[1::2]
    luma = vf[1::2] if is_pal else vf[0::2]
    h = min(len(chroma), len(luma))
    c = np.stack([(chroma[:h] >> 4) & 0xF, chroma[:h] & 0xF], -1).reshape(h, 80)
    l = np.stack([(luma[:h] >> 4) & 0xF, luma[:h] & 0xF], -1).reshape(h, 80)
    return pal[(c.astype(np.int32) << 4) | l.astype(np.int32)]


def decode_c80_rgb(video_bytes):
    """7680 C80-M4 bytes -> (48, 80, 3) uint8 true-colour RGB444."""
    import numpy as np
    a = np.frombuffer(bytes(video_bytes[:VIDEO_BYTES]), np.uint8).reshape(24, 8, 40)
    R = a[:, 2:4, :].reshape(24, 80).astype(np.int32)
    G = a[:, 4:6, :].reshape(24, 80).astype(np.int32)
    B = a[:, 6:8, :].reshape(24, 80).astype(np.int32)
    out = np.zeros((48, 80, 3), dtype=np.uint8)
    out[0::2, :, 0] = (R >> 4) * 17; out[1::2, :, 0] = (R & 15) * 17
    out[0::2, :, 1] = (G >> 4) * 17; out[1::2, :, 1] = (G & 15) * 17
    out[0::2, :, 2] = (B >> 4) * 17; out[1::2, :, 2] = (B & 15) * 17
    return out


def decode_frame_rgb(video_bytes, profile, is_pal):
    """Dispatch by profile -> (H, W, 3) uint8."""
    if is_c80(profile):
        return decode_c80_rgb(video_bytes)
    return decode_apac_rgb(video_bytes, is_pal)


def inject_poster(header8192, poster7680, profile=None):
    """Return a new 8 KB AVF2 header with sectors 1-15 replaced by
    poster7680 and the poster flags/checksum updated. profile defaults to
    the header's own byte 7 (sets flags bit1 for non-APAC)."""
    if len(poster7680) != VIDEO_BYTES:
        raise ValueError(f"poster must be {VIDEO_BYTES} bytes")
    s0 = bytearray(header8192[:512])
    if s0[0:4] != b"AVF2":
        raise ValueError("not an AVF2 header")
    prof = s0[7] if profile is None else profile
    flags = (s0[6] | 1)
    flags = (flags | 2) if prof != PROFILE_APAC else (flags & ~2)
    s0[6] = flags
    s0[510:512] = checksum(s0).to_bytes(2, "little")
    return bytes(s0) + bytes(poster7680)
