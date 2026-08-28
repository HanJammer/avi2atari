#!/usr/bin/env python3
"""AVI2ATARI AVF Suite - web backend (FastAPI).

Reuses the exact CLI converter (shelled out per job in an isolated tmpfs
dir) and the shared avf_format library for metadata/preview/decoding - no
format logic is duplicated here. RAM-only: scratch lives on tmpfs
(/dev/shm in the container); nothing is persisted, only request IP +
file names / URLs are logged (see the disclaimer in the UI).
"""
import io
import os
import re
import random
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
import logging
import collections

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import (JSONResponse, StreamingResponse, FileResponse,
                               Response)
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import avf_format as avf   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
CONVERTER = os.path.join(REPO, "avi2atari.py")


def _env_bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _converter_version():
    """avi2atari's __version__ without importing it (numpy/numba-free)."""
    try:
        with open(CONVERTER, encoding="utf-8") as f:
            m = re.search(r'__version__\s*=\s*"([^"]+)"', f.read())
        return m.group(1) if m else ""
    except OSError:
        return ""


CONVERTER_VERSION = _converter_version()

YOUTUBE_ENABLED = _env_bool("YOUTUBE_ENABLED", False)
LOG_REQUESTS = _env_bool("LOG_REQUESTS", True)
SERVER_MODE = _env_bool("SERVER_MODE", False)
LOCAL_PLAYBACK = _env_bool("LOCAL_PLAYBACK", False)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "512"))
CONVERT_TIMEOUT = int(os.environ.get("CONVERT_TIMEOUT", "1800"))
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


# ===========================================================================
# job model (async conversion with progress); everything lives in RAM/tmpfs
# ===========================================================================
JOBS = {}
JOBS_LOCK = threading.Lock()
_FRAME_RE = re.compile(r"Frame (\d+)")


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.state = "running"          # running | done | error
        self.percent = 0
        self.msg = "starting..."
        self.error = None
        self.payload = None
        self.name = None
        self.mtype = "application/octet-stream"
        self.proc = None
        self.cancelled = False
        self.tail = []
        self.ts = time.time()


def _new_job():
    j = Job()
    with JOBS_LOCK:
        for k in [k for k, v in JOBS.items() if time.time() - v.ts > 1800]:
            JOBS.pop(k, None)
        JOBS[j.id] = j
    return j


def _ffprobe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], capture_output=True, text=True,
            timeout=60).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _run_converter(argv, cwd, job, base_frac, span, total):
    # -u (unbuffered) + reading char-by-char so the converter's CR-terminated
    # progress line updates the bar live instead of arriving only at the end.
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=0)
    job.proc = p
    tail = collections.deque(maxlen=20)
    buf, CR, LF = "", chr(13), chr(10)
    while True:
        ch = p.stdout.read(1)
        if ch == "":
            break
        if ch == CR or ch == LF:
            if buf.strip():
                tail.append(buf.strip())
                m = _FRAME_RE.search(buf)
                if m and total:
                    frac = min(1.0, int(m.group(1)) / total)
                    job.percent = int((base_frac + frac * span) * 100)
                    job.msg = f"encoding frame {m.group(1)}/{total}"
                elif "Extracting" in buf or "Downloading" in buf:
                    job.msg = buf.strip()[:80]
            buf = ""
        else:
            buf += ch
    p.wait(timeout=CONVERT_TIMEOUT)
    job.proc = None
    job.tail = list(tail)
    return p.returncode


def _collect(d, base):
    avfs = sorted(x for x in os.listdir(d) if x.lower().endswith(".avf"))
    if not avfs:
        return None
    if len(avfs) == 1:
        with open(os.path.join(d, avfs[0]), "rb") as f:
            return f.read(), base + "-" + avfs[0].split("-")[-1], \
                "application/octet-stream"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for a in avfs:
            z.write(os.path.join(d, a), a)
    return buf.getvalue(), base + "-avf.zip", "application/zip"


def _resolve_poster(opts, dur):
    """--poster seconds for this file, from the poster mode."""
    mode = opts.get("poster_mode", "none")
    if mode == "frame":
        return opts.get("poster_secs", 0.0)
    if mode == "random" and dur > 0:
        return round(random.uniform(0, dur * 0.9), 2)
    return None


def _conv_args(inp, out_base, o, poster):
    argv = [sys.executable, "-u", CONVERTER, inp, "--system", o["system"],
            "--out", out_base, "--aspect", o.get("aspect", "fit")]
    if o["profile"] in ("c80m4", "c80btc"):
        argv += ["--profile", o["profile"]]
        if o["profile"] == "c80btc" and o.get("btc_stability") is not None:
            argv += ["--btc-stability", str(o["btc_stability"])]
        if o["profile"] == "c80btc" and o.get("btc_dither", "none") != "none":
            argv += ["--btc-dither", o["btc_dither"],
                     "--btc-dither-strength",
                     str(o.get("btc_dither_strength", 0.5))]
            if o.get("btc_dither_gate"):
                argv += ["--btc-dither-gate"]
    else:
        argv += ["--avf", "2" if o.get("avf2", True) else "1"]
    for k in ("title", "author", "comment"):
        if o.get(k):
            argv += [f"--{k}", o[k]]
    if o.get("contrast") is not None:
        argv += ["--contrast", str(o["contrast"])]
    if o.get("saturation") is not None:
        argv += ["--saturation", str(o["saturation"])]
    if o.get("loudnorm"):
        argv += ["--loudnorm"]
    elif o.get("volume") is not None:
        argv += ["--volume", str(o["volume"])]
    if poster is not None:
        argv += ["--poster", str(poster)]
    return argv


def _job_convert_files(job, files, opts):
    try:
        with tempfile.TemporaryDirectory(dir=SCRATCH_DIR, prefix="avfjob_") as d:
            n = len(files)
            outdir = os.path.join(d, "out")
            os.makedirs(outdir)
            for i, (fn, data) in enumerate(files):
                ext = os.path.splitext(fn)[1] or ".bin"
                inp = os.path.join(d, f"in{i}{ext}")
                with open(inp, "wb") as f:
                    f.write(data)
                dur = _ffprobe_duration(inp)
                fps = avf.FPS_PAL if opts["system"] != "NTSC" else avf.FPS_NTSC
                total = int(dur * fps) if dur else 0
                base = os.path.splitext(os.path.basename(fn))[0]
                job.msg = f"[{i+1}/{n}] {base}"
                poster = _resolve_poster(opts, dur)
                o2 = dict(opts)
                if not o2.get("title"):
                    o2["title"] = base          # default title = input file name
                rc = _run_converter(
                    _conv_args(inp, os.path.join(outdir, base), o2, poster),
                    d, job, i / n, 1.0 / n, total)
                if rc != 0:
                    raise RuntimeError(f"failed on {fn}: "
                                       + " | ".join(job.tail[-4:]))
            base = os.path.splitext(os.path.basename(files[0][0]))[0] \
                if n == 1 else "avf_batch"
            if opts.get("poster_image"):
                pimg, pname = opts["poster_image"]
                poster = encode_image_poster(pimg, pname, opts["system"],
                                             opts["profile"])
                _apply_image_poster(outdir, poster)
            res = _collect(outdir, base)
            if not res:
                raise RuntimeError("no .avf produced: "
                                   + " | ".join(job.tail[-6:]))
            job.payload, job.name, job.mtype = res
        job.percent = 100; job.state = "done"; job.msg = "done"
    except Exception as e:
        job.state = "error"; job.error = str(e); job.msg = "error"


def _job_convert_urls(job, urls, opts):
    try:
        with tempfile.TemporaryDirectory(dir=SCRATCH_DIR, prefix="avfjob_") as d:
            outdir = os.path.join(d, "out")
            os.makedirs(outdir)
            n = len(urls)
            for i, url in enumerate(urls):
                job.msg = f"[{i+1}/{n}] downloading..."
                argv = [sys.executable, "-u", CONVERTER, url, "--system",
                        opts["system"], "--out", os.path.join(outdir, f"url{i}")]
                if opts["profile"] in ("c80m4", "c80btc"):
                    argv += ["--profile", opts["profile"]]
                    if opts["profile"] == "c80btc" \
                            and opts.get("btc_stability") is not None:
                        argv += ["--btc-stability", str(opts["btc_stability"])]
                    if opts["profile"] == "c80btc" \
                            and opts.get("btc_dither", "none") != "none":
                        argv += ["--btc-dither", opts["btc_dither"],
                                 "--btc-dither-strength",
                                 str(opts.get("btc_dither_strength", 0.5))]
                        if opts.get("btc_dither_gate"):
                            argv += ["--btc-dither-gate"]
                else:
                    argv += ["--avf", "2" if opts.get("avf2", True) else "1"]
                for k in ("title", "author", "comment"):
                    if opts.get(k):
                        argv += [f"--{k}", opts[k]]
                rc = _run_converter(argv, d, job, i / n, 1.0 / n, 0)
                if rc != 0:
                    raise RuntimeError(f"failed on {url}: "
                                       + " | ".join(job.tail[-4:]))
            res = _collect(outdir, "avf_urls")
            if not res:
                raise RuntimeError("no .avf produced: "
                                   + " | ".join(job.tail[-6:]))
            job.payload, job.name, job.mtype = res
        job.percent = 100; job.state = "done"; job.msg = "done"
    except Exception as e:
        job.state = "error"; job.error = str(e); job.msg = "error"


def _job_testtone(job, opts):
    try:
        with tempfile.TemporaryDirectory(dir=SCRATCH_DIR, prefix="avfjob_") as d:
            argv = [sys.executable, "-u", CONVERTER, "--test-gen",
                    "--system", opts["system"]]
            if opts["profile"] in ("c80m4", "c80btc"):
                argv += ["--profile", opts["profile"]]
            else:
                argv += ["--avf", "2"]
            job.msg = "generating test signal..."
            rc = _run_converter(argv, d, job, 0.0, 1.0, 500)
            if rc != 0:
                raise RuntimeError("test-gen failed")
            res = _collect(d, "test_tone")
            if not res:
                raise RuntimeError("no .avf produced: "
                                   + " | ".join(job.tail[-6:]))
            job.payload, job.name, job.mtype = res
        job.percent = 100; job.state = "done"; job.msg = "done"
    except Exception as e:
        job.state = "error"; job.error = str(e); job.msg = "error"


def encode_image_poster(image_bytes, filename, system, profile):
    """Encode a still image into a 7680-byte poster of the given profile by
    running the real converter on a 1-frame clip and lifting its poster."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR, prefix="avfimg_") as d:
        ext = os.path.splitext(filename)[1] or ".png"
        img = os.path.join(d, "img" + ext)
        with open(img, "wb") as f:
            f.write(image_bytes)
        clip = os.path.join(d, "clip.mp4")
        fps = "25" if system != "NTSC" else "30"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
                        "-i", img, "-t", "0.2", "-r", fps, "-pix_fmt",
                        "yuv420p", clip], check=True, timeout=120)
        argv = [sys.executable, CONVERTER, clip, "--system", system,
                "--out", os.path.join(d, "p"), "--poster", "0"]
        # the poster is ALWAYS classic C80 for C80 files (spec section 5),
        # so the still is encoded with plain c80m4 even for c80btc streams
        argv += ["--profile", "c80m4"] if profile in ("c80m4", "c80btc") else ["--avf2"]
        subprocess.run(argv, cwd=d, capture_output=True, text=True,
                       timeout=CONVERT_TIMEOUT)
        avfs = [x for x in os.listdir(d) if x.lower().endswith(".avf")]
        if not avfs:
            raise RuntimeError("poster encode produced no AVF")
        with open(os.path.join(d, avfs[0]), "rb") as f:
            head = f.read(avf.HEADER_SIZE)
        meta, _ = avf.parse_avf2_header(head)
        poster = avf.poster_from_header(head, meta)
        if not poster:
            raise RuntimeError("poster encode had no poster")
        return poster


def _apply_image_poster(outdir, poster):
    for a in os.listdir(outdir):
        if not a.lower().endswith(".avf"):
            continue
        p = os.path.join(outdir, a)
        with open(p, "rb") as f:
            data = f.read()
        head = avf.inject_poster(data[:avf.HEADER_SIZE], poster)
        with open(p, "wb") as f:
            f.write(head + data[avf.HEADER_SIZE:])


def _spawn(fn, *a):
    j = _new_job()
    threading.Thread(target=fn, args=(j,) + a, daemon=True).start()
    return j.id


# ===========================================================================
# preview / decode
# ===========================================================================
def render_png(video_bytes, profile, is_pal, c80_btc=False):
    from PIL import Image
    rgb = avf.decode_frame_rgb(video_bytes, profile, is_pal, c80_btc)
    img = Image.fromarray(rgb, "RGB").resize((640, 384), Image.NEAREST)
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()


def frame_video(data, info, index):
    meta = info["meta"] or {}
    if index < 0 and meta.get("poster"):
        # posters are ALWAYS classic-encoded, even in C80_BTC files (spec 5)
        return avf.poster_from_header(data[:avf.HEADER_SIZE], info["meta"]), \
            info["profile"], False
    fs = info["frame_size"]
    off = info["payload_off"] + max(0, index) * fs
    frame = data[off:off + fs]
    if len(frame) < fs:
        raise HTTPException(400, "frame index out of range")
    # All profiles use the APAC container; unwrap the block/pad mux to the
    # 7680 video bytes (the render dispatches by profile + BTC flag).
    return avf.demux_apac_video(frame), info["profile"], \
        bool(meta.get("c80_btc"))


# ===========================================================================
# API
# ===========================================================================
@app.get("/api/config")
def api_config():
    return {"youtube_enabled": YOUTUBE_ENABLED, "server_mode": SERVER_MODE,
            "local_playback": LOCAL_PLAYBACK, "max_upload_mb": MAX_UPLOAD_MB,
            "suite_version": avf.__version__,
            "converter_version": CONVERTER_VERSION}


@app.post("/api/convert")
async def api_convert(request: Request, files: list[UploadFile] = File(...),
                      system: str = Form("PAL"), profile: str = Form("apac"),
                      avf2: bool = Form(True), title: str = Form(""),
                      author: str = Form(""), comment: str = Form(""),
                      aspect: str = Form("fit"), contrast: float = Form(1.0),
                      saturation: float = Form(1.0), loudnorm: bool = Form(True),
                      volume: float = Form(12.0), poster_mode: str = Form("none"),
                      poster_frame: str = Form(""),
                      btc_stability: float = Form(None),
                      btc_dither: str = Form("none"),
                      btc_dither_strength: float = Form(0.5),
                      btc_dither_gate: bool = Form(False),
                      poster_image: UploadFile = File(None)):
    payloads, total = [], 0
    for f in files:
        data = await f.read()
        total += len(data)
        if total > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"uploads exceed {MAX_UPLOAD_MB} MB")
        payloads.append((f.filename, data))
    audit(request, f"convert:{system}:{profile}:{len(payloads)}f",
          ",".join(n for n, _ in payloads))
    opts = dict(system=system, profile=profile, avf2=avf2, title=title,
                btc_stability=btc_stability, btc_dither=btc_dither,
                btc_dither_strength=btc_dither_strength,
                btc_dither_gate=btc_dither_gate,
                author=author, comment=comment, aspect=aspect,
                contrast=contrast, saturation=saturation, loudnorm=loudnorm,
                volume=volume, poster_mode=poster_mode)
    if poster_mode == "frame" and poster_frame:
        fps = avf.FPS_PAL if system != "NTSC" else avf.FPS_NTSC
        opts["poster_secs"] = float(poster_frame) / fps
    if poster_mode == "image" and poster_image is not None:
        opts["poster_mode"] = "none"
        opts["poster_image"] = (await poster_image.read(), poster_image.filename)
    return {"job": _spawn(_job_convert_files, payloads, opts)}


@app.post("/api/convert_url")
async def api_convert_url(request: Request, urls: str = Form(...),
                          system: str = Form("PAL"), profile: str = Form("apac"),
                          avf2: bool = Form(True), title: str = Form(""),
                          author: str = Form(""), comment: str = Form(""),
                          btc_stability: float = Form(None),
                          btc_dither: str = Form("none"),
                          btc_dither_strength: float = Form(0.5),
                          btc_dither_gate: bool = Form(False)):
    if not YOUTUBE_ENABLED:
        raise HTTPException(403, "URL downloading is disabled on this server")
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    if not url_list:
        raise HTTPException(400, "no URLs given")
    audit(request, f"convert_url:{system}:{len(url_list)}", " ".join(url_list))
    opts = dict(system=system, profile=profile, avf2=avf2, title=title,
                btc_stability=btc_stability, btc_dither=btc_dither,
                btc_dither_strength=btc_dither_strength,
                btc_dither_gate=btc_dither_gate,
                author=author, comment=comment)
    return {"job": _spawn(_job_convert_urls, url_list, opts)}


@app.post("/api/testtone")
async def api_testtone(request: Request, system: str = Form("PAL"),
                       profile: str = Form("apac")):
    audit(request, f"testtone:{system}:{profile}", "-")
    return {"job": _spawn(_job_testtone, dict(system=system, profile=profile))}


@app.get("/api/progress/{job_id}")
def api_progress(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return {"state": j.state, "percent": j.percent, "msg": j.msg,
            "error": j.error, "name": j.name}


@app.post("/api/cancel/{job_id}")
def api_cancel(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    j.cancelled = True
    if j.proc:
        try:
            j.proc.terminate()
        except Exception:
            pass
    j.state = "error"; j.error = "cancelled"; j.msg = "cancelled"
    return {"ok": True}


@app.get("/api/result/{job_id}")
def api_result(job_id: str):
    j = JOBS.get(job_id)
    if not j or j.state != "done" or j.payload is None:
        raise HTTPException(404, "result not ready")
    payload, name, mtype = j.payload, j.name, j.mtype
    JOBS.pop(job_id, None)
    return StreamingResponse(io.BytesIO(payload), media_type=mtype,
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


@app.post("/api/play")
async def api_play(request: Request, file: UploadFile = File(...),
                   system: str = Form("")):
    if not LOCAL_PLAYBACK:
        raise HTTPException(403, "server-side playback is disabled")
    data = await file.read()
    d = tempfile.mkdtemp(dir=SCRATCH_DIR, prefix="avfplay_")
    p = os.path.join(d, "play.avf")
    with open(p, "wb") as f:
        f.write(data)
    argv = [sys.executable, os.path.join(REPO, "avfplayer.py"), p]
    if system:
        argv.append(system)
    subprocess.Popen(argv)
    audit(request, "play", file.filename)
    return {"ok": True}


@app.post("/api/meta")
async def api_meta(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    audit(request, "meta", file.filename)
    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR) as d:
        p = os.path.join(d, "f.avf")
        with open(p, "wb") as f:
            f.write(data)
        info = avf.classify(p)
    m = info["meta"] or {}
    return JSONResponse({
        "kind": info["kind"], "profile": info["profile"],
        "profile_name": m.get("profile_name")
        or avf.PROFILE_NAMES.get(info["profile"], "?"),
        "n_frames": info["n_frames"], "is_pal": info["is_pal"],
        "meta": info["meta"], "problems": info["problems"]})


@app.post("/api/meta_edit")
async def api_meta_edit(request: Request, file: UploadFile = File(...),
                        title: str = Form(None), author: str = Form(None),
                        comment: str = Form(None)):
    data = await file.read()
    audit(request, "meta_edit", file.filename)
    meta, _ = avf.parse_avf2_header(data[:avf.HEADER_SIZE])
    if meta is None:
        raise HTTPException(400, "not an AVF2 file - upgrade Legacy first")
    updates = {k: v for k, v in (("title", title), ("author", author),
                                 ("comment", comment)) if v is not None}
    out = avf.set_metadata_bytes(data[:avf.HEADER_SIZE], updates) \
        + data[avf.HEADER_SIZE:]
    name = os.path.splitext(file.filename)[0] + "-edited.avf"
    return StreamingResponse(io.BytesIO(out), media_type="application/octet-stream",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


@app.post("/api/upgrade")
async def api_upgrade(request: Request, file: UploadFile = File(...),
                      system: str = Form("PAL"), title: str = Form(None),
                      author: str = Form(None), comment: str = Form(None),
                      poster_frame: str = Form(""),
                      poster_image: UploadFile = File(None)):
    data = await file.read()
    audit(request, f"upgrade:{system}", file.filename)
    pimg = await poster_image.read() if poster_image is not None else None
    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR) as d:
        p = os.path.join(d, "f.avf")
        with open(p, "wb") as f:
            f.write(data)
        upd = {k: v for k, v in (("title", title), ("author", author),
                                 ("comment", comment)) if v is not None}
        try:
            out = avf.legacy_to_avf2_bytes(
                p, system == "PAL", upd,
                None if pimg else (int(poster_frame) if poster_frame else None))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if pimg:
            poster = encode_image_poster(pimg, poster_image.filename,
                                         system, "apac")
            out = avf.inject_poster(out[:avf.HEADER_SIZE], poster)                 + out[avf.HEADER_SIZE:]
    name = os.path.splitext(file.filename)[0] + "-avf2.avf"
    return StreamingResponse(io.BytesIO(out), media_type="application/octet-stream",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{name}"'})


@app.post("/api/preview")
async def api_preview(request: Request, file: UploadFile = File(...),
                      frame: int = Form(-1)):
    data = await file.read()
    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR) as d:
        p = os.path.join(d, "f.avf")
        with open(p, "wb") as f:
            f.write(data)
        info = avf.classify(p)
    if info["kind"] == "unknown":
        raise HTTPException(400, "unrecognised AVF file")
    video, profile, c80_btc = frame_video(data, info, frame)
    is_pal = info["is_pal"] if info["is_pal"] is not None else True
    return Response(render_png(video, profile, is_pal, c80_btc),
                    media_type="image/png")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    global YOUTUBE_ENABLED, LOCAL_PLAYBACK
    if os.environ.get("YOUTUBE_ENABLED") is None:
        YOUTUBE_ENABLED = True
    if os.environ.get("LOCAL_PLAYBACK") is None:
        LOCAL_PLAYBACK = True
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    avf.print_banner("Web", avf.__version__)
    print(f"AVF Suite web UI on http://{host}:{port}  "
          f"(YouTube: {'on' if YOUTUBE_ENABLED else 'off'}, "
          f"local playback: {'on' if LOCAL_PLAYBACK else 'off'})")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
