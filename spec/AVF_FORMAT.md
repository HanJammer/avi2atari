# AVF Container Format - Specification (v1)

This document describes the on-disk AVF ("Atari Video Format") stream format
played by `movplay` / `AVFPLAY` on Atari 8-bit computers and produced by
`avi2atari.py`. It covers both the **PAL (50 Hz)** and **NTSC (60 Hz)**
variants. Everything below was verified against three independent sources:
phaeron's original C++ tools (`encvideo50n.cpp`, `encvideo60n.cpp`,
`encaudio60.cpp`, `mux50n.cpp`, `mux60n.cpp`), the `movplay` 6502 player
source, and the byte-level behaviour of `avi2atari.py` / `avf_check.py`.

AVF v1 has **no magic number, no header fields and no metadata** - it is a
bare, rigidly aligned stream. (A backward-compatible metadata extension is
proposed in [AVF2_HEADER.md](AVF2_HEADER.md).)

---

## 1. Design context

AVF is not a "file format" in the usual sense - it is a **beam-synchronous
stream**. The player races the electron beam: video bytes are DMA-fetched by
ANTIC from a 4 KB ring buffer that the CPU refills from mass storage *while
the frame is being displayed*, and audio samples are consumed at a rate of
**exactly one sample per scan line**. There is no seeking, no resync marker
and no tolerance: if the stream drifts by a single byte, the rest of the file
plays as noise. This is why:

- every frame is exactly **8704 bytes = 17 disk sectors** (17 * 512),
- the file must be stored **contiguously** on the medium (fragmentation
  breaks streaming),
- all padding bytes have fixed, known values.

Storage bandwidth required:

| System | Frame rate | Data rate |
|--------|-----------:|----------:|
| PAL    | 49.86 fps  | ~434 KB/s |
| NTSC   | 59.92 fps  | ~521 KB/s |

The frame rates are the machines' true vertical frequencies: the Atari PAL
frame is 312 scan lines of 114 machine cycles at 1.7734 MHz (-> 49.86 Hz), the
NTSC frame is 262 lines at 1.7898 MHz (-> 59.92 Hz).

## 2. Global file layout

```
+--------------------------+
| Header: 8192 bytes       |  16 sectors, all 0x00 in AVF v1
+--------------------------+
| Frame 0: 8704 bytes      |  17 sectors
+--------------------------+
| Frame 1: 8704 bytes      |
+--------------------------+
| ...                      |
+--------------------------+
```

- **Header (8192 B).** `movplay` unconditionally skips the first 16 sectors
  of the stream before playback, so the file starts with an 8192-byte block
  that the player never interprets. In AVF v1 it is written as zeroes.
  `avi2atari.py --no-header` omits it (for loaders that do not skip it) -
  such files are not playable by standard `movplay`.
- **Frames.** The payload is an exact multiple of 8704 bytes. Total file size
  is `8192 + n * 8704`.

A useful detection property: `8192 mod 8704 != 0`, so `size mod 8704 == 0`
reliably distinguishes header-less files from headered ones.

## 3. Frame layout (8704 bytes)

Each frame consists of a **video area** (8192 B) followed by an **audio
area** (512 B):

```
offset 0    ... 8191 : video  - 64 blocks * 128 bytes
offset 8192 ... 8703 : audio  - 32*10 + 19*10 + 2 bytes
```

### 3.1 Video area - 64 blocks * 128 bytes

The visible picture is 192 scan lines of 40 bytes each (7680 B of pixel
data). Lines are grouped three to a block; each 128-byte block has this
fixed layout (offsets relative to block start, block *b* carries lines
*3b*, *3b+1*, *3b+2*, top to bottom):

| Offset  | Size | Content            |
|---------|-----:|--------------------|
| 0       | 1    | `0x00` pad         |
| 1-40    | 40   | line *3b*          |
| 41-43   | 3    | `0x00` pad         |
| 44      | 1    | `0x00` pad         |
| 45-84   | 40   | line *3b+1*        |
| 85-87   | 3    | `0x00` pad         |
| 88-127  | 40   | line *3b+2*        |

The pad bytes exist to keep the per-line stride compatible with the player's
display list fetch pattern; they must be zero (see `avf_check.py`).

### 3.2 Pixel encoding

**Display technique.** The player displays each line pair using two GTIA
modes, switched per scan line by writing `PRIOR` mid-frame (the display list
is built from ANTIC mode-2 instructions abused via VSCROL so that each
fetches one scan line):

- **luma lines** are shown in GTIA mode 9 (16 luminances, 80 pixels/line),
- **chroma lines** are shown in GTIA mode 11 (16 hues, 80 pixels/line).

On a composite/RF display (or with GTIA2RGB's BLENDING enabled) adjacent
chroma and luma lines visually fuse into full-colour pixels - the classic
APAC ("Any Point, Any Colour") technique. Effective resolution is therefore
**80 * 96 colour cells**, displayed as 192 interleaved scan lines.

**Line parity differs between systems:**

| System | Even lines (0,2,4...) | Odd lines (1,3,5...) |
|--------|---------------------|--------------------|
| PAL    | chroma              | luma               |
| NTSC   | luma                | chroma             |

**Nibble packing.** Each 40-byte line holds 80 four-bit pixels; the **high
nibble is the left pixel** of each pair.

**Luma values** (luma lines): 0-15, displayed luminance ~ `nibble * 17` on a
0-255 scale. The encoder converts limited-range BT.601 luma (16-235) to full
range before quantisation.

**Chroma values** (chroma lines): 0 = grey (no colour), 1-14 (PAL) or 1-15
(NTSC) = hue index. The encoder works in **YIQ** space and quantises every
sufficiently saturated pixel (chroma amplitude > 40/255) to the nearest hue
with amplitude normalised to exactly 40/255. The hue tables are:

```
PAL  (14 hues):  I(c) = 40*cos(pi(c-0.5)/7)    Q(c) = 40*sin(pi(c-0.5)/7)    c = 1...14
NTSC (15 hues):  I(c) = 40*cos(pi(c-1)/7.5)    Q(c) = 40*sin(pi(c-1)/7.5)    c = 1...15
```

Hue index selection:

```
PAL:   c = ((round(atan2(Q,I)*7/pi - 0.5) + 14000) mod 14) + 1
NTSC:  c = ((round(atan2(Q,I)*7.5/pi + 2.0) + 15000) mod 15) + 1
```

> **NTSC hue quirk.** In the original NTSC encoder the hue *selection*
> formula (`+2.0` term) is offset by **+2 hue steps (+48 degree)** relative to its
> own I/Q hue table. The emitted indices inherit this offset. A decoder that
> wants to reproduce the encoder's intended colours must subtract 2 hue steps
> when mapping NTSC indices back to angles (this is what `avfplayer` does).
> Whether the +2 offset matches real NTSC GTIA hardware phase has not been
> verified on hardware; PAL has no such inconsistency.

**Colour matrix.** All RGB<->YUV conversion on the encoder side uses BT.601
coefficients; sources tagged BT.709 are converted during scaling.

**Dithering** (encoder behaviour, not part of the format): serpentine
error-diffusion with spatial weights 7/25, 5/25, 3/25, 1/25 and a temporal
term of 9/25 carried to the same pixel of the *next frame*, applied
independently to luma and to the I/Q chroma error.

### 3.3 Audio area - 512 bytes

**Sample format.** Unsigned bytes in the range **0-100** (silence ~ 50),
played through POKEY as ~6.5-bit PWM (or through a Covox DAC in Covox builds
of the player). The PC-side conversion is linear: `out = in * 100 / 255`
from unsigned 8-bit input.

**Sample rate = horizontal line rate.** The player consumes exactly one
sample per scan line, so:

| System | Line rate (~ sample rate) | Samples per frame |
|--------|---------------------------|------------------:|
| PAL    | 15 557 Hz                 | 312               |
| NTSC   | 15 700 Hz                 | 262               |

**Interleave map.** The audio block is not sequential: samples are laid out
in the order the CPU needs them during the frame (it reads up to 10 bytes
per scan line in a fixed schedule; some are consumed immediately, some are
buffered for the next frame). Writing `s(k)` for the *k*-th sample of the
frame's audio chunk, the 512-byte area is, byte for byte:

**Loop 1 - 32 lines * 10 bytes** (offsets 8192 + 10*y, y = 0...31):

| Byte | PAL        | NTSC       |
|-----:|------------|------------|
| 0    | s(y)       | s(y)       |
| 1    | s(y+120)   | s(y+70)    |
| 2    | s(y+152)   | s(y+102)   |
| 3    | s(y+184)   | s(y+134)   |
| 4    | s(y+216)   | s(y+166)   |
| 5    | s(y+248)   | s(y+198)   |
| 6    | s(y+280)   | s(y+230)   |
| 7    | s(y+52)    | s(y+52)    |
| 8    | s(y+84)    | s(y+84)    |
| 9    | `0x00`     | `0x00`     |

**Loop 2 - 19 lines * 10 bytes** (offsets 8512 + 10*y, y = 0...18):

| Byte | PAL        | NTSC       |
|-----:|------------|------------|
| 0    | s(y+32)    | s(y+32)    |
| 1    | s(y+116)   | `0x00`     |
| 2-9  | `0x00` *8  | `0x00` *8 (bytes 1-9 are nine zeroes) |

**Tail - 2 bytes** (offsets 8702-8703): `s(51)`, `0x00`.

This covers all 312 (PAL) / 262 (NTSC) samples of the frame. Note that some
ranges are intentionally emitted **twice** (PAL: s(120)-s(134); NTSC:
s(70)-s(115)) - the read schedule needs them at two different points of the
frame and duplicating them is cheaper than buffering.

The PAL/NTSC difference in loop 2, byte 1 (sample vs zero) is one of the two
structural differences between the variants (the other being the loop-1
offsets), and is what `avf_check.py --system` verifies.

## 4. PAL vs NTSC summary

| Property                | PAL              | NTSC             |
|-------------------------|------------------|------------------|
| Frame rate              | 49.86 fps        | 59.92 fps        |
| Chroma lines            | even             | odd              |
| Hue count               | 14               | 15               |
| Hue table angle         | pi(c-0.5)/7      | pi(c-1)/7.5      |
| Hue selection offset    | -0.5             | +2.0 (see quirk) |
| YUV->RGB pre-matrix      | 1.596/-0.391/-0.813/2.017 | 1.396/-0.342/-0.711/1.765 |
| Audio samples per frame | 312              | 262              |
| Audio sample rate       | 15 557 Hz        | 15 700 Hz        |
| Loop-1 spread offset    | +120             | +70              |
| Loop-2 second sample    | s(y+116)         | none (zero)      |
| Frame size              | 8704 B           | 8704 B           |
| Data rate               | ~434 KB/s        | ~521 KB/s        |

The frame *structure* (8704 B, 64*128 video blocks, 32+19+2 audio lines) is
identical in both variants; a PAL file played as NTSC (or vice versa) stays
in sync structurally but shows wrong line parity, wrong hues and wrong pitch.

## 5. Validation

`avf_check.py` (in this repository) verifies the structural invariants of a
finished file without an emulator: header presence/zeroing, payload being an
exact multiple of 8704, video block pad bytes, and the system-specific audio
tail padding. It does not (cannot) validate pixel or sample *values*, which
have no redundancy.

## 6. References

- phaeron (Avery Lee) - `encvideo50n/60n.cpp`, `encaudio60.cpp`,
  `mux50n/60n.cpp` (original Windows tool chain, 2013)
- `movplay.s` / AVFPLAY - the 6502 player (Avery Lee; AVG CART port and
  sound fixes by Jakub Husak: <https://github.com/jhusak/avgcart_avfplay>)
- `avi2atari.py` - this repository's encoder
- `avfplayer` - PC-side reference player:
  <https://github.com/HanJammer/avfplayer>


## AVF-C80 profile (GTIA2RGB COL80 Mode 4)

A true-colour variant selected by AVF2 header byte 7 = 2. It uses the
**identical container** to APAC - same 8 KB header, same 8704-byte
frames, same block/pad video mux and the same audio interleave - so the
proven `movplay` streaming kernel and audio path are reused unchanged.
Only the **video content** differs:

- Each frame's 7680 video bytes are 24 cell rows of
  `[chars 80 | R 80 | G 80 | B 80]` (80 bytes per plane, laid out across
  the 8 scan lines of a COL80 character row). Every cell uses the
  half-block glyph (top half = BG nibble = top pixel, bottom half = FG
  nibble = bottom pixel), giving 80x48 true-colour RGB444.
- The GTIA2RGB, in COL80 Mode 4, forms the 80 cells from the fetched
  byte sequence and assigns planes by scan-line position; it only
  watches ANTIC's AN0..AN2, so a normal character-mode display list
  (as APAC streaming already uses) drives it.

Because the container is byte-for-byte an APAC file apart from the video
content and the profile byte, `avf_check` validates C80 with the same
structural checks, and any APAC-capable player streams the bytes fine -
it just needs COL80 Mode 4 enabled (and the 8-line DL offset the
GTIA2RGB adds) to render them as colour cells rather than luma/chroma.

APAC (profile 0) remains the default. A player that does not recognise
the profile byte must not enable Mode 4, or it would show the cell data
as APAC noise.
