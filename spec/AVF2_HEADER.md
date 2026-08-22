# AVF2 - Metadata Header Extension (Proposal, draft 1)

Status: **implemented in the PC tools** (avi2atari 0.4 `--avf2` writer,
avf_check 0.4 validator, avfplayer 0.3 reader) - layout below is now the
working spec, version-frozen as header_version 1. Not yet regression-tested
on real hardware with stock AVFPLAY (MVP step 5).

## 1. Motivation

AVF v1 carries zero metadata: not even a magic number, the target TV system,
or a title. Every consumer has to guess the system from the filename suffix
convention (`-PAL.avf` / `-NTSC.avf`) and has nothing to show in a file
browser. The goal is an ID3-like header carrying:

- format identification + target system (PAL/NTSC),
- video title,
- author (of the conversion),
- converter name and version,
- a thumbnail / poster image,

while remaining **100% playable by every existing player**.

## 2. The key observation

`movplay`/`AVFPLAY` **unconditionally skip the first 16 sectors (8192
bytes)** of the stream. AVF v1 fills that region with zeroes, but no player
ever looks at it. That gives us 8192 bytes of free, already-reserved space:

> **AVF2 = AVF v1 with the 8 KB header populated instead of zeroed.**
> The payload (frames from offset 8192 on) is byte-identical to v1.

Compatibility matrix:

| Consumer                        | AVF v1 (zero header) | AVF2 |
|---------------------------------|----------------------|------|
| `movplay` / `AVFPLAY` (Atari)   | plays                | plays (header skipped) |
| `avfplayer` (PC)                | plays                | plays (header skipped); can render metadata |
| `avf_check.py` (current)        | OK                   | FAIL (expects zero header) -> needs the v2 update below |
| future menu player              | shows filename only  | shows title/author/poster |

The only tool that must change semantics is our own validator, which today
treats "header is all zeroes" as an invariant. Under AVF2 the rule becomes
"header is all zeroes **or** begins with a valid AVF2 block".

## 3. Header layout

The 8192-byte header is divided into a **metadata sector** (sector 0) and a
**poster image** (sectors 1-15).

All integers are **little-endian** (natural for both the 6502 and x86).
All text fields are ASCII 0x20-0x7E, NUL-padded, not necessarily
NUL-terminated (a field may use its full length). For direct rendering on
the Atari, writers SHOULD restrict text to 0x20-0x7A.

### 3.1 Sector 0 - metadata (512 bytes)

| Offset | Size | Field            | Contents |
|-------:|-----:|------------------|----------|
| 0      | 4    | magic            | `"AVF2"` (41 56 46 32) |
| 4      | 1    | header_version   | 1 |
| 5      | 1    | system           | 0 = PAL, 1 = NTSC |
| 6      | 1    | flags            | bit 0: poster present; bit 1: poster encoded in the video profile (non-APAC); bits 2-7 reserved (0) |
| 7      | 1    | profile          | video profile: 0 = APAC, 1 = C80M3 (reserved), 2 = C80M4 (GTIA2RGB COL80 Mode 4). Legacy/v1 and pre-0.5 AVF2 files carry 0. |
| 8      | 4    | frame_count      | number of 8704-byte frames (u32) |
| 12     | 4    | duration_ms      | playback length in milliseconds (u32; redundant with frame_count, precomputed for 6502 convenience) |
| 16     | 8    | date             | conversion date, ASCII `YYYYMMDD` |
| 24     | 40   | reserved         | 0 |
| 64     | 64   | title            | video title |
| 128    | 64   | author           | author of the conversion |
| 192    | 32   | converter        | e.g. `"avi2atari 0.3"` |
| 224    | 128  | comment          | free text |
| 352    | 158  | reserved         | 0 |
| 510    | 2    | checksum         | u16: sum of bytes 0-509 modulo 65536 |

Rationale for the checksum choice: a 16-bit byte-sum is computable on a 6502
in a dozen lines and catches the realistic failure mode (a header written by
something that is not an AVF2 writer, or a torn/partial write). This is
integrity *identification*, not tamper protection.

Versioning rules:

- Readers MUST accept any `header_version >= 1` and ignore fields and flag
  bits they do not know; future versions may only assign meaning to bytes
  that are `reserved`/zero in version 1.
- Readers MUST treat a sector 0 that fails magic or checksum as "no
  metadata" (i.e. fall back to AVF v1 behaviour), not as a fatal error -
  the stream itself is still playable.

### 3.2 Sectors 1-15 - poster frame (7680 bytes)

7680 bytes is **exactly one full AVF video frame's pixel payload**:
192 lines * 40 bytes, stored top-to-bottom **without** the 128-byte-block
padding of the in-stream video area.

- Same nibble semantics, same chroma/luma line parity as the stream itself
  (per the `system` field).
- Recommended content: a representative frame of the video ("poster"),
  selected or supplied at conversion time.
- If flags bit 0 = 0, sectors 1-15 are all zero and consumers show a
  placeholder.

Why this encoding and not a smaller thumbnail or a PC-side format (PNG etc.):

1. **The Atari can render it with the player's own display code** - a menu
   player already contains an APAC renderer (that is literally what
   playback is), so "show the poster" is: stream 15 sectors into the
   framebuffer with un-padded lines. No decoder, no scaling.
2. **PC tools can render it with the player's own palette code**
   (`avfplayer` shares the decode path).
3. It is the *only* size that exactly fills the remaining 15 sectors -
   no layout decisions, no wasted space, full 160*192 quality.

A menu player that wants a smaller preview can decimate lines/columns while
copying (e.g. every other line pair -> 80*96 half-size preview).

## 4. Tooling impact

### `avi2atari.py` (writer)

- New options: `--title`, `--author`, `--comment`, `--poster
  <seconds|image>` (default: frame at 10% of duration), `--avf1` to force
  the legacy zero header.
- Default output becomes AVF2 **after** the format is validated on real
  hardware with all three players in circulation (movplay ATR, AVFPLAY
  AVG/SUB builds, Covox builds). Until then, AVF2 writing sits behind
  `--avf2`.
- Implementation notes: the muxer already writes the 8 KB header first; the
  only change is filling the buffer before writing. The poster frame is one
  extra `encode_video_frame()` call on the chosen source frame (dither
  buffers zeroed, temporal term discarded).

### `avf_check.py` (validator)

- Accept both header forms: all-zero (v1) or magic+checksum-valid AVF2
  block (and then cross-check `system` against `--system`, `frame_count`
  against actual payload size - two new *free* validations we cannot do
  today).
- Report metadata in the summary output.

### `avfplayer` (PC player)

- Parse the header if present; set the window title to `title - author`;
  `--info` flag to dump metadata; optionally show the poster before
  playback starts.
- Falls back cleanly on v1 files (current behaviour).

### Atari menu player

See [PLAYER_CONCEPT.md](PLAYER_CONCEPT.md) - the header is designed so that
the menu's right-hand panel is: sector 0 -> text fields, sectors 1-15 ->
poster render.

## 5. MVP and sequencing

1. **Freeze this layout** (after review).
2. `avf_check.py`: add AVF2 acceptance (small, testable, no risk to
   playback - do this first so the ecosystem can validate files).
3. `avi2atari.py`: `--avf2 --title/--author/--poster` writer.
4. `avfplayer`: metadata display + poster rendering (also serves as the
   reference reader implementation).
5. Hardware regression test: AVF2 file on 130XE + SUB Cart with the stock
   AVFPLAY - confirm the header is genuinely ignored.
6. Flip `avi2atari` default to AVF2.

## 6. Open questions

- **Character set for `title`.** ASCII-only is safe everywhere; Polish
  titles lose diacritics. Option: keep ASCII in the fixed fields, allow a
  UTF-8 copy of the title in a future v2 TLV area (bytes 352-509 are
  reserved for exactly this kind of growth).
- **Poster frame choice.** Fixed 10% point is a decent default; scene-aware
  selection (max luma variance) is cheap on the PC side and could be the
  default instead.
- **Should `duration_ms` count PAL frames at exactly 49.86?** Proposal:
  `duration_ms = round(frame_count * 1000 * 114 * 312 / 1773447.5)` for PAL
  and `* 114 * 262 / 1789772.5` for NTSC, i.e. true hardware timing.
- **`--no-header` files**: AVF2 is by definition impossible there; the
  writer should reject `--avf2 --no-header`.
