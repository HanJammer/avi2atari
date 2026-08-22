#!/usr/bin/env python3
"""AVI2ATARI AVF Suite - web backend (FastAPI).

Reuses the exact CLI converter (shelled out per job in an isolated tmpfs
dir) and the shared avf_format library for metadata/preview - no format
logic is duplicated here. RAM-only: scratch lives on tmpfs (/dev/shm in
the container); nothing is persisted, only request IP + filenames are
logged (see the disclaimer in the UI).
"""
import io
import os
import subprocess
import sys
import tempfile
import zipfile
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import (JSONResponse, StreamingResponse, FileResponse,
                               Response)
from fastapi.staticfiles import StaticFiles

# the suite lives one dir up; import the shared library from there
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import avf_format as avf   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")

# --- configuration (environment) -------------------------------------------
def _env_bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


YOUTUBE_ENABLED = _env_bool("YOUTUBE_ENABLED", False)
LOG_REQUESTS = _env_bool("LOG_REQUESTS", True)
SERVER_MODE = _env_bool("SERVER_MODE", False)      # show the "hosted" disclaimer
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "512"))
SCRATCH_DIR = os.environ.get("SCRATCH_DIR") or (
    "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("avf-suite")

app = FastAPI(title="AVI2ATARI AVF Suite")


def client_ip(request):
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "?")


def audit(request, action, name):
    if LOG_REQUESTS:
        log.info("%s | %s | %s", client_ip(request), action, name)


def job_dir():
    return tempfile.TemporaryDirectory(dir=SCRATCH_DIR, prefix="avfjob_")


# ---------------------------------------------------------------------------
# converter (shell out to the CLI, isolated per job)
# ---------------------------------------------------------------------------
def build_convert_argv(inp, out_base, opts):
    argv = [sys.executable, os.path.join(REPO, "avi2atari.py"), inp,
            "--system", opts["system"], "--out", out_base]
    if opts["profile"] == "c80m4":
        argv += ["--profile", "c80m4"]         # implies --avf2
    elif opts["avf2"]:
        argv += ["--avf2"]
    for k in ("title", "author", "comment"):
        if opts.get(k):
            argv += [f"--{k}", opts[k]]
    argv += ["--aspect", opts.get("aspect", "fit")]
    if opts.get("contrast"):
        argv += ["--contrast", str(opts["contrast"])]
    if opts.get("saturation"):
        argv += ["--saturation", str(opts["saturation"])]
    if opts.get("loudnorm"):
        argv += ["--loudnorm"]
    elif opts.get("volume") is not None:
        argv += ["--volume", str(opts["volume"])]
    if opts.get("poster") is not None:
        argv += ["--poster", str(opts["poster"])]
    return argv


def run_convert(input_bytes, filename, opts):
    """Convert in an isolated tmpfs job dir. Returns (payload_bytes,
    download_name, media_type). One .avf streams as-is; several (e.g.
    --system BOTH) are zipped."""
    with job_dir() as d:
        ext = os.path.splitext(filename)[1] or ".bin"
        inp = os.path.join(d, "input" + ext)
        with open(inp, "wb") as f:
            f.write(input_bytes)
        argv = build_convert_argv(inp, os.path.join(d, "out"), opts)
        r = subprocess.run(argv, cwd=d, capture_output=True, text=True,
                           timeout=int(os.environ.get("CONVERT_TIMEOUT", "1800")))
        avfs = sorted(x for x in os.listdir(d) if x.lower().endswith(".avf"))
        if not avfs:
            raise HTTPException(500, "conversion produced no output:\n"
                                + (r.stderr or r.stdout or "")[-2000:])
        base = os.path.splitext(filename)[0]
        if len(avfs) == 1:
            with open(os.path.join(d, avfs[0]), "rb") as f:
                return f.read(), base + "-" + avfs[0].split("-")[-1], \
                    "application/octet-stream"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for a in avfs:
                z.write(os.path.join(d, a), base + "-" + a.split("-")[-1])
        return buf.getvalue(), base + "-avf.zip", "application/zip"


# ---------------------------------------------------------------------------
# preview: decode a frame/poster to PNG (server-side, no pygame)
# ---------------------------------------------------------------------------
def _gtia_rgb(hue, lum):
    import colorsys
    v = lum / 15.0
    if hue == 0:
        c = int(v * 255)
        return (c, c, c)
    ang = ((hue - 1) * 24 + 30) / 360.0
    r, g, b = colorsys.hsv_to_rgb(ang % 1.0, 0.8, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def render_png(video_bytes, profile, is_pal):
    from PIL import Image
    if avf.is_c80(profile):
        img = Image.new("RGB", (80, 48))
        px = img.load()
        for r in range(24):
            base = r * 320
            for x in range(80):
                R, G, B = (video_bytes[base + 80 + x], video_bytes[base + 160 + x],
                           video_bytes[base + 240 + x])
                px[x, 2 * r] = ((R >> 4) * 17, (G >> 4) * 17, (B >> 4) * 17)
                px[x, 2 * r + 1] = ((R & 15) * 17, (G & 15) * 17, (B & 15) * 17)
        out = img.resize((640, 384), Image.NEAREST)
    else:
        rows = [video_bytes[i * 40:(i + 1) * 40] for i in range(192)]
        img = Image.new("RGB", (80, 96))
        px = img.load()
        for row in range(96):
            cr = rows[row * 2] if is_pal else rows[row * 2 + 1]
            lr = rows[row * 2 + 1] if is_pal else rows[row * 2]
            for bx in range(40):
                for n in range(2):
                    h = (cr[bx] >> (4 * (1 - n))) & 0xF
                    lm = (lr[bx] >> (4 * (1 - n))) & 0xF
                    px[bx * 2 + n, row] = _gtia_rgb(h, lm)
        out = img.resize((640, 384), Image.NEAREST)
    b = io.BytesIO()
    out.save(b, "PNG")
    return b.getvalue()


def frame_video(data, info, index):
    """Raw 7680 video bytes of frame `index` (or the poster if index<0)."""
    if index < 0 and info["meta"] and info["meta"].get("poster"):
        return avf.poster_from_header(data[:avf.HEADER_SIZE], info["meta"]), \
            info["profile"]
    fs = info["frame_size"]
    off = info["payload_off"] + max(0, index) * fs
    frame = data[off:off + fs]
    if len(frame) < fs:
        raise HTTPException(400, "frame index out of range")
    if avf.is_c80(info["profile"]):
        return frame[:avf.VIDEO_BYTES], info["profile"]
    return avf.demux_apac_video(frame), info["profile"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/config")
def api_config():
    return {
        "youtube_enabled": YOUTUBE_ENABLED,
        "server_mode": SERVER_MODE,
        "max_upload_mb": MAX_UPLOAD_MB,
        "suite_version": avf.__version__,
    }


async def _read_upload(f):
    data = await f.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB} MB")
    return data


@app.post("/api/convert")
async def api_convert(request: Request, file: UploadFile = File(...),
                      system: str = Form("PAL"), profile: str = Form("apac"),
                      avf2: bool = Form(True), title: str = Form(""),
                      author: str = Form(""), comment: str = Form(""),
                      aspect: str = Form("fit"), contrast: float = Form(1.0),
                      saturation: float = Form(1.0), loudnorm: bool = Form(True),
                      volume: float = Form(12.0), poster: str = Form("")):
    data = await _read_upload(file)
    audit(request, f"convert:{system}:{profile}", file.filename)
    opts = dict(system=system, profile=profile, avf2=avf2, title=title,
                author=author, comment=comment, aspect=aspect,
                contrast=contrast, saturation=saturation, loudnorm=loudnorm,
                volume=volume, poster=float(poster) if poster else None)
    payload, name, mtype = run_convert(data, file.filename, opts)
    return StreamingResponse(io.BytesIO(payload), media_type=mtype,
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


@app.post("/api/convert_url")
async def api_convert_url(request: Request, url: str = Form(...),
                          system: str = Form("PAL"), profile: str = Form("apac"),
                          avf2: bool = Form(True), title: str = Form(""),
                          author: str = Form(""), comment: str = Form("")):
    if not YOUTUBE_ENABLED:
        raise HTTPException(403, "URL downloading is disabled on this server")
    audit(request, f"convert_url:{system}:{profile}", url)
    with job_dir() as d:
        argv = [sys.executable, os.path.join(REPO, "avi2atari.py"), url,
                "--system", system, "--out", "out"]
        if profile == "c80m4":
            argv += ["--profile", "c80m4"]
        elif avf2:
            argv += ["--avf2"]
        for k, v in (("title", title), ("author", author), ("comment", comment)):
            if v:
                argv += [f"--{k}", v]
        r = subprocess.run(argv, cwd=d, capture_output=True, text=True,
                           timeout=int(os.environ.get("CONVERT_TIMEOUT", "3600")))
        avfs = sorted(x for x in os.listdir(d) if x.lower().endswith(".avf"))
        if not avfs:
            raise HTTPException(500, "conversion produced no output:\n"
                                + (r.stderr or r.stdout or "")[-2000:])
        if len(avfs) == 1:
            with open(os.path.join(d, avfs[0]), "rb") as f:
                payload = f.read()
            return StreamingResponse(io.BytesIO(payload),
                                     media_type="application/octet-stream",
                                     headers={"Content-Disposition":
                                              f'attachment; filename="{avfs[0]}"'})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for a in avfs:
                z.write(os.path.join(d, a), a)
        return StreamingResponse(io.BytesIO(buf.getvalue()),
                                 media_type="application/zip",
                                 headers={"Content-Disposition":
                                          'attachment; filename="avf.zip"'})


@app.post("/api/meta")
async def api_meta(request: Request, file: UploadFile = File(...)):
    data = await _read_upload(file)
    audit(request, "meta", file.filename)
    with job_dir() as d:
        p = os.path.join(d, "f.avf")
        with open(p, "wb") as f:
            f.write(data)
        info = avf.classify(p)
    meta = info["meta"]
    return JSONResponse({
        "kind": info["kind"], "profile": info["profile"],
        "profile_name": avf.PROFILE_NAMES.get(info["profile"], "?"),
        "n_frames": info["n_frames"], "is_pal": info["is_pal"],
        "meta": meta, "problems": info["problems"],
    })


@app.post("/api/meta_edit")
async def api_meta_edit(request: Request, file: UploadFile = File(...),
                        title: str = Form(None), author: str = Form(None),
                        comment: str = Form(None)):
    data = await _read_upload(file)
    audit(request, "meta_edit", file.filename)
    meta, problems = avf.parse_avf2_header(data[:avf.HEADER_SIZE])
    if meta is None:
        raise HTTPException(400, "not an AVF2 file - upgrade Legacy first")
    updates = {k: v for k, v in
               (("title", title), ("author", author), ("comment", comment))
               if v is not None}
    new_header = avf.set_metadata_bytes(data[:avf.HEADER_SIZE], updates)
    out = new_header + data[avf.HEADER_SIZE:]
    name = os.path.splitext(file.filename)[0] + "-edited.avf"
    return StreamingResponse(io.BytesIO(out), media_type="application/octet-stream",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


@app.post("/api/upgrade")
async def api_upgrade(request: Request, file: UploadFile = File(...),
                      system: str = Form("PAL"), title: str = Form(None),
                      author: str = Form(None), comment: str = Form(None),
                      poster_frame: str = Form("")):
    data = await _read_upload(file)
    audit(request, f"upgrade:{system}", file.filename)
    with job_dir() as d:
        p = os.path.join(d, "f.avf")
        with open(p, "wb") as f:
            f.write(data)
        updates = {k: v for k, v in
                   (("title", title), ("author", author), ("comment", comment))
                   if v is not None}
        try:
            out = avf.legacy_to_avf2_bytes(
                p, system == "PAL", updates,
                int(poster_frame) if poster_frame else None)
        except ValueError as e:
            raise HTTPException(400, str(e))
    name = os.path.splitext(file.filename)[0] + "-avf2.avf"
    return StreamingResponse(io.BytesIO(out), media_type="application/octet-stream",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


@app.post("/api/preview")
async def api_preview(request: Request, file: UploadFile = File(...),
                      frame: int = Form(-1)):
    data = await _read_upload(file)
    with job_dir() as d:
        p = os.path.join(d, "f.avf")
        with open(p, "wb") as f:
            f.write(data)
        info = avf.classify(p)
    if info["kind"] == "unknown":
        raise HTTPException(400, "unrecognised AVF file")
    video, profile = frame_video(data, info, frame)
    is_pal = info["is_pal"] if info["is_pal"] is not None else True
    return Response(render_png(video, profile, is_pal), media_type="image/png")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    avf.print_banner("Web", avf.__version__)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
