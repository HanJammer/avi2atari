#!/usr/bin/env python3
"""Regression tests for AVF2 v2 / C80_BTC (spec/AVF2_HEADER.md section 5).

Self-contained (no test framework): run with
    python tests/test_btc.py
Exit code 0 = all passed. Covers the checklist from the Codex
implementation review (docs in av2play/docs/CODEX_REVIEW_C80_BTC*.md).
"""
import os
import sys
import io
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import avf_format as avf

PASSED = 0


def ok(name):
    global PASSED
    PASSED += 1
    print(f"  ok  {name}")


def classic_poster_bytes():
    """A minimal valid classic-C80 poster (constant half-block chars)."""
    p = np.zeros((24, 8, 40), np.uint8)
    p[:, 0:2, :] = avf.C80_HALFBLOCK
    return p.tobytes()



def _state():
    """Fresh temporal-state arrays for encode_c80btc_frame."""
    return (np.zeros((24, 80), np.uint8),
            np.zeros((24, 80, 3), np.uint8),
            np.zeros((24, 80, 3), np.uint8))

# 1. valid header ----------------------------------------------------------
def test_header_valid():
    h = avf.build_avf2_header(True, 42, classic_poster_bytes(), {"title": "x"},
                              avf.PROFILE_C80M4, c80_btc=True)
    m, prob = avf.parse_avf2_header(h)
    assert m is not None and not prob, prob
    assert m["version"] == 2
    assert m["profile"] == avf.PROFILE_C80M4
    assert m["flags"] & avf.FLAG_C80_BTC
    assert m["c80_btc"] and m["checksum_ok"]
    assert m["profile_name"] == "C80M4+BTC"
    ok("header: version 2, profile 2, FLAG_C80_BTC, checksum")


# 2. illegal combinations --------------------------------------------------
def test_illegal_combos():
    try:
        avf.build_avf2_header(True, 1, None, {}, avf.PROFILE_APAC, c80_btc=True)
        raise AssertionError("builder accepted BTC on APAC")
    except ValueError:
        pass
    h = bytearray(avf.build_avf2_header(True, 1, classic_poster_bytes(), {},
                                        avf.PROFILE_C80M4, c80_btc=True))
    h[4] = 1                                    # forge version 1
    h[510:512] = avf.checksum(h[:512]).to_bytes(2, "little")
    _, prob = avf.parse_avf2_header(bytes(h))
    assert any("header_version >= 2" in p for p in prob), prob
    h[4] = 2
    h[7] = avf.PROFILE_APAC                     # forge profile 0
    h[510:512] = avf.checksum(h[:512]).to_bytes(2, "little")
    _, prob = avf.parse_avf2_header(bytes(h))
    assert any("expected 2 (C80M4)" in p for p in prob), prob
    ok("illegal: builder rejects BTC+APAC; parser flags v1+BTC and profile!=2")


# 3. known-mask decoder geometry ------------------------------------------
def test_mask_geometry():
    vid = np.zeros((24, 8, 40), np.uint8)
    # cell (0,0): halfblock $F0, BG=red F00, FG=blue 00F
    vid[0, 0, 0] = 0xF0
    vid[0, 2, 0] = 0xF0            # R: BG=F FG=0
    vid[0, 6, 0] = 0x0F            # B: BG=0 FG=F
    # cell (0,2): mask $01 (bit0 only), FG=red
    vid[0, 0, 2] = 0x01
    vid[0, 2, 2] = 0x0F            # R: BG=0 FG=F
    d = avf.decode_c80btc_rgb(vid.tobytes())
    assert d.shape == (96, 160, 3)
    assert tuple(d[0, 0]) == (255, 0, 0)      # mask row 0 -> BG red
    assert tuple(d[2, 0]) == (0, 0, 255)      # mask row 2 -> FG blue
    assert tuple(d[3, 1]) == (0, 0, 255)      # right column, row 3 -> FG
    assert tuple(d[0, 4]) == (255, 0, 0)      # bit0 = TOP-LEFT subpixel = FG
    assert tuple(d[0, 5]) == (0, 0, 0)        # bit1 clear -> top-right = BG
    assert tuple(d[1, 4]) == (0, 0, 0)        # mask row 1 = BG
    ok("mask geometry: bit0 top-left, bit1 top-right, BG=hi/FG=lo nibbles")


# 4. encoder -> decoder round-trip ----------------------------------------
def test_roundtrip():
    import avi2atari as a
    # source built exactly on the RGB444 grid, two colours per 2x4 block:
    # top mask row colour A, rest colour B -> representable losslessly
    rgb = np.zeros((96, 160, 3), np.float32)
    A = (17.0 * 12, 17.0 * 3, 17.0 * 7)
    B = (17.0 * 1, 17.0 * 14, 17.0 * 9)
    for k in range(3):
        rgb[:, :, k] = B[k]
        rgb[0::4, :, k] = A[k]                 # every block's top mask row
    leg_bg = np.zeros((24, 80, 3), np.uint8)
    leg_fg = np.zeros((24, 80, 3), np.uint8)
    enc = a.encode_c80btc_frame(rgb, leg_bg, leg_fg, *_state(), 0, 0.0)
    dec = avf.decode_c80btc_rgb(enc.tobytes()).astype(np.float32)
    assert np.array_equal(dec, rgb), "lossless-representable frame diverged"
    ok("round-trip: exactly-representable 160x96 frame is lossless")


# 5. BTC error never exceeds the halfblock baseline ------------------------
def test_beats_baseline():
    import avi2atari as a
    rng = np.random.default_rng(1234)
    rgb = (rng.random((96, 160, 3)) * 255).astype(np.float32)
    rgb48 = rgb.reshape(48, 2, 80, 2, 3).mean(axis=(1, 3))
    classic = a.encode_c80m4_frame(rgb48)
    ca = classic.reshape(24, 8, 40)
    leg = np.stack([ca[:, 2:4, :].reshape(24, 80),
                    ca[:, 4:6, :].reshape(24, 80),
                    ca[:, 6:8, :].reshape(24, 80)], -1)
    leg_bg = ((leg >> 4) & 0xF).astype(np.uint8)
    leg_fg = (leg & 0xF).astype(np.uint8)
    enc = a.encode_c80btc_frame(rgb, leg_bg, leg_fg, *_state(), 0, 0.0)
    btc_rec = avf.decode_c80btc_rgb(enc.tobytes()).astype(np.float64)
    base_rec = np.repeat(np.repeat(
        avf.decode_c80_rgb(classic.tobytes()).astype(np.float64), 2, 0), 2, 1)
    btc_sse = float(((btc_rec - rgb) ** 2).sum())
    base_sse = float(((base_rec - rgb) ** 2).sum())
    assert btc_sse <= base_sse, (btc_sse, base_sse)
    ok(f"quality: BTC SSE {btc_sse:.3e} <= baseline SSE {base_sse:.3e} "
       f"(ratio {btc_sse/base_sse:.3f})")


# 6. metadata tools preserve the BTC bit ----------------------------------
def test_tools_preserve_flag():
    h = avf.build_avf2_header(True, 7, classic_poster_bytes(), {"title": "a"},
                              avf.PROFILE_C80M4, c80_btc=True)
    h2 = avf.set_metadata_bytes(h, {"title": "b", "author": "c"})
    m2, prob2 = avf.parse_avf2_header(h2)
    assert m2["c80_btc"] and m2["checksum_ok"] and not prob2, prob2
    h3 = avf.inject_poster(h, classic_poster_bytes())
    m3, prob3 = avf.parse_avf2_header(h3)
    assert m3["c80_btc"] and m3["poster"] and m3["poster_c80"] and not prob3
    ok("tools: set_metadata_bytes + inject_poster keep BTC bit + checksum")


# 7. poster must be classic; poisoned poster is flagged --------------------
def test_poster_classic_enforced():
    good = avf.build_avf2_header(True, 7, classic_poster_bytes(), {},
                                 avf.PROFILE_C80M4, c80_btc=True)
    _, prob = avf.parse_avf2_header(good)
    assert not prob, prob
    poisoned = np.frombuffer(good[512:], np.uint8).reshape(24, 8, 40).copy()
    poisoned[:, 0:2, :] = 0x5A                  # masks in the poster
    bad = good[:512] + poisoned.tobytes()
    _, prob = avf.parse_avf2_header(bad)
    assert any("must be classic C80" in p for p in prob), prob
    ok("poster: classic passes; BTC-mask char plane is flagged")


# 8. preview routing: poster -> classic decode, frame -> BTC decode --------
def test_preview_routing():
    import avi2atari as a
    rgb = np.zeros((96, 160, 3), np.float32)
    leg = np.zeros((24, 80, 3), np.uint8)
    enc = a.encode_c80btc_frame(rgb, leg, leg, *_state(), 0, 0.0)
    abuf = np.zeros(320, np.uint8)
    buf = io.BytesIO()
    a.mux_frame(buf, enc, abuf, True)
    frame = buf.getvalue()
    assert len(frame) == avf.FRAME_SIZE_APAC
    header = avf.build_avf2_header(True, 1, classic_poster_bytes(), {},
                                   avf.PROFILE_C80M4, c80_btc=True)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.avf")
        with open(p, "wb") as f:
            f.write(header + frame)
        info = avf.classify(p)
        assert info["kind"] == "avf2" and info["meta"]["c80_btc"]
        data = header + frame
        # poster path: classic decode (48, 80, 3)
        pv = avf.poster_from_header(data[:avf.HEADER_SIZE], info["meta"])
        prgb = avf.decode_frame_rgb(pv, info["profile"], True, False)
        assert prgb.shape == (48, 80, 3)
        # stream path: BTC decode (96, 160, 3)
        vid = avf.demux_apac_video(frame)
        frgb = avf.decode_frame_rgb(vid, info["profile"], True,
                                    info["meta"]["c80_btc"])
        assert frgb.shape == (96, 160, 3)
    ok("routing: poster decodes classic 48x80, stream decodes BTC 96x160")


# 9. temporal hysteresis: fewer changes, bounded quality cost --------------
def test_temporal_hysteresis():
    import avi2atari as a
    rng = np.random.default_rng(7)
    base = (rng.random((96, 160, 3)) * 255).astype(np.float32)
    noise1 = rng.normal(0, 6, base.shape).astype(np.float32)
    noise2 = rng.normal(0, 6, base.shape).astype(np.float32)
    f1 = np.clip(base + noise1, 0, 255)
    f2 = np.clip(base + noise2, 0, 255)
    leg = np.zeros((24, 80, 3), np.uint8)
    K = 0.15

    # stateless: two independent frames
    e1 = a.encode_c80btc_frame(f1, leg, leg, *_state(), 0, 0.0)
    e2 = a.encode_c80btc_frame(f2, leg, leg, *_state(), 0, 0.0)
    stateless_changes = int((e1 != e2).sum())
    stateless_sse = float(((avf.decode_c80btc_rgb(e2.tobytes())
                            .astype(np.float64) - f2) ** 2).sum())

    # hysteresis: same two frames through carried state
    pm, pb, pf = _state()
    h1 = a.encode_c80btc_frame(f1, leg, leg, pm, pb, pf, 0, K)
    h2 = a.encode_c80btc_frame(f2, leg, leg, pm, pb, pf, 1, K)
    hyst_changes = int((h1 != h2).sum())
    hyst_sse = float(((avf.decode_c80btc_rgb(h2.tobytes())
                       .astype(np.float64) - f2) ** 2).sum())

    assert np.array_equal(h1, e1), "first frame must be stateless-identical"
    assert hyst_changes < stateless_changes, (hyst_changes, stateless_changes)
    assert hyst_sse <= stateless_sse * (1.0 + K) + 1e-6, (hyst_sse, stateless_sse)
    ok(f"temporal: changes {stateless_changes} -> {hyst_changes} "
       f"(-{100 - 100 * hyst_changes // max(1, stateless_changes)}%), "
       f"SSE ratio {hyst_sse / stateless_sse:.3f} <= 1+K")

if __name__ == "__main__":
    for t in (test_header_valid, test_illegal_combos, test_mask_geometry,
              test_roundtrip, test_beats_baseline, test_tools_preserve_flag,
              test_poster_classic_enforced, test_preview_routing,
              test_temporal_hysteresis):
        t()
    print(f"ALL {PASSED} TESTS PASSED")
