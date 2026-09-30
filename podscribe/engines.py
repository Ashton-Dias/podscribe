"""Transcription engines. MLX (Apple GPU) on Apple Silicon, faster-whisper (CUDA / CPU int8) elsewhere.
Both return (segments, duration) with segments = [{"start", "end", "text"}]."""
import contextlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from . import PodscribeError

MODELS = ("small", "medium", "large-v3", "large-v3-turbo")
DEFAULT_MODEL = "large-v3-turbo"
ENGINES = ("mlx", "faster-whisper")
MLX_REPOS = {
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
MLX_SIZE_GB = {"small": 0.5, "medium": 1.5, "large-v3": 3.1, "large-v3-turbo": 1.6}

# chunking + quiet-passage protection for the MLX engine (it has no built-in VAD)
CHUNK_SECONDS = 240   # target chunk length; cuts snap to the nearest silence
CUT_WINDOW = 45       # how far from the target a silence may be to use it as the cut
PAD = 2.0             # extra audio on each side of a chunk so words at the cut aren't clipped
SILENCE_DB = -35
SILENCE_MIN = 0.6     # seconds
SILENT_SEGMENT_MIN = 1.0  # only silences at least this long are used to reject segments
NO_SPEECH_MAX = 0.6
MIN_SPEECH_S = 1.5    # a segment with at least this much VAD speech is kept (and trimmed to it)


# ---------------------------------------------------------------- selection
def select_engine() -> str:
    forced = os.environ.get("PODSCRIBE_ENGINE", "").strip().lower()
    if forced:
        if forced not in ENGINES:
            raise PodscribeError(f"PODSCRIBE_ENGINE must be one of: {', '.join(ENGINES)} (got '{forced}').")
        return forced
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            print("mlx-whisper is not installed; using faster-whisper on CPU (pip install mlx-whisper for the Apple GPU).")
    return "faster-whisper"


def run(engine: str, audio: Path, model: str, progress, log=print) -> tuple[list[dict], float]:
    if model not in MODELS:
        raise PodscribeError(f"Unknown model '{model}'. Choose one of: {', '.join(MODELS)}.")
    if engine == "mlx":
        return _mlx(audio, model, progress, log)
    return _faster_whisper(audio, model, progress, log)


# ---------------------------------------------------------------- faster-whisper
def _fw_model_name(model: str, log) -> str:
    from faster_whisper.utils import _MODELS
    if model in _MODELS:
        return model
    log(f"This faster-whisper has no '{model}'; using large-v3 instead.")
    return "large-v3"


def _faster_whisper(audio, model, progress, log):
    from faster_whisper import WhisperModel
    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False
    device, compute = ("cuda", "float16") if cuda else ("cpu", "int8")
    progress(None, None)  # model loading: no measurable progress yet
    log(f"Loading Whisper '{model}' with faster-whisper on {device} ({compute})...")
    try:
        wm = WhisperModel(_fw_model_name(model, log), device=device, compute_type=compute)
        seg_iter, info = wm.transcribe(str(audio), vad_filter=True)
        segments = []
        for seg in seg_iter:
            segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
            progress(seg.end, info.duration)
    except PodscribeError:
        raise
    except Exception as e:
        raise PodscribeError(f"Transcription failed: {e}")
    progress(info.duration, info.duration)
    return segments, info.duration


# ---------------------------------------------------------------- ffmpeg helpers
def _need_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise PodscribeError("ffmpeg not found. Install it with: brew install ffmpeg")


def audio_duration(path: Path) -> float:
    _need_ffmpeg()
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        raise PodscribeError(f"Could not read audio file {path.name}: {p.stderr.strip()[:200] or 'unsupported format'}")


def parse_silences(stderr: str, duration: float) -> list[tuple[float, float]]:
    silences, start = [], None
    for line in stderr.splitlines():
        if m := re.search(r"silence_start: (-?[\d.]+)", line):
            start = max(0.0, float(m.group(1)))
        elif (m := re.search(r"silence_end: ([\d.]+)", line)) and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    if start is not None:
        silences.append((start, duration))
    return silences


def detect_silences(path: Path, duration: float) -> list[tuple[float, float]]:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                        f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}", "-f", "null", "-"],
                       capture_output=True, text=True)
    return parse_silences(p.stderr, duration)


def extract(path: Path, start: float, end: float, out: Path) -> None:
    p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                        "-i", str(path), "-ac", "1", "-ar", "16000", str(out)], capture_output=True, text=True)
    if p.returncode != 0:
        raise PodscribeError(f"ffmpeg could not decode the audio: {p.stderr.strip()[:200]}")


# ---------------------------------------------------------------- chunking (pure, unit-tested)
def plan_cuts(duration: float, silences, target: float = CHUNK_SECONDS, window: float = CUT_WINDOW) -> list[float]:
    """Cut points [0, c1, ..., duration]. Each cut snaps to the middle of the silence nearest the target."""
    cuts, pos = [0.0], 0.0
    while duration - pos > target * 1.25:
        goal = pos + target
        mids = [(s + e) / 2 for s, e in silences if goal - window <= (s + e) / 2 <= goal + window]
        cut = min(mids, key=lambda c: abs(c - goal)) if mids else goal
        cuts.append(cut)
        pos = cut
    cuts.append(duration)
    return cuts


def chunk_bounds(cuts: list[float], silences, duration: float) -> list[tuple[float, float, float, float]]:
    """(own_lo, own_hi, audio_start, audio_end) per chunk. A cut inside a silence needs no padding (no word can be
    clipped, and no audio is transcribed twice); a hard cut with no silence nearby gets PAD seconds of context."""
    def in_silence(c):
        return any(a <= c <= b for a, b in silences)
    out = []
    for lo, hi in zip(cuts, cuts[1:]):
        pad_lo = 0.0 if lo == 0 or in_silence(lo) else PAD
        pad_hi = 0.0 if hi >= duration or in_silence(hi) else PAD
        out.append((lo, hi, max(0.0, lo - pad_lo), min(duration, hi + pad_hi)))
    return out


def shift_and_keep(segs: list[dict], offset: float, lo: float, hi: float, last: bool) -> list[dict]:
    """Shift chunk-relative segments to absolute time and keep those whose midpoint lies in [lo, hi)
    (the padded overlap belongs to whichever chunk owns the segment's midpoint, so nothing is duplicated)."""
    out = []
    for s in segs:
        a, b = s["start"] + offset, s["end"] + offset
        mid = (a + b) / 2
        if lo <= mid < hi or (last and mid >= lo):
            out.append({**s, "start": round(a, 2), "end": round(b, 2)})
    return out


def coverage(a: float, b: float, silences, min_len: float = 0.0) -> float:
    """Fraction of [a, b] that lies inside silences of at least min_len seconds."""
    if b <= a:
        return 0.0
    covered = sum(max(0.0, min(b, e) - max(a, s)) for s, e in silences if e - s >= min_len)
    return covered / (b - a)


def merge_silences(silences, gap: float = 0.5) -> list[tuple[float, float]]:
    """Join silences separated by less than `gap` seconds (near-threshold noise makes ffmpeg report them in pieces)."""
    out = []
    for a, b in sorted(silences):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def detect_speech(path: Path) -> list[tuple[float, float]]:
    """Speech regions (seconds) from the Silero VAD bundled with faster-whisper: the same filter
    faster-whisper applies with vad_filter=True. Catches loud music/noise that silence detection can't."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    audio = decode_audio(str(path), sampling_rate=16000)
    return [(t["start"] / 16000, t["end"] / 16000) for t in get_speech_timestamps(audio, VadOptions())]


def _speech_overlap(a: float, b: float, speech) -> list[tuple[float, float]]:
    return [(max(a, s), min(b, e)) for s, e in speech if min(b, e) > max(a, s)]


def clean_segments(segs: list[dict], silences, speech=None) -> list[dict]:
    """Drop Whisper hallucinations. With `speech` (VAD regions): drop segments that are mostly outside speech and
    trim the rest to the speech they overlap. Always: drop text inside long silences, high no-speech + low
    confidence, and 3rd+ identical repeats; trim a start that sits inside a long silence."""
    silences = merge_silences(silences)
    out, last, run = [], None, 0
    for s in segs:
        text = s["text"].strip()
        if not text:
            continue
        if speech is not None:
            ov = _speech_overlap(s["start"], s["end"], speech)
            spoken = sum(e - b for b, e in ov)
            # Whisper can stretch a real segment over noise before it, so keep anything with >= 1.5 s of speech
            if not ov or (spoken < MIN_SPEECH_S and spoken / max(s["end"] - s["start"], 1e-6) < 0.3):
                continue
            s = {**s, "start": round(ov[0][0], 2), "end": round(ov[-1][1], 2)}
        for a, b in silences:
            if b - a >= SILENT_SEGMENT_MIN and a < s["start"] < b < s["end"]:
                s = {**s, "start": round(b, 2)}
        if s.get("no_speech_prob", 0) > NO_SPEECH_MAX and s.get("avg_logprob", 0) < -1.0:
            continue
        if coverage(s["start"], s["end"], silences, SILENT_SEGMENT_MIN) >= 0.8:
            continue
        key = re.sub(r"\W+", " ", text.lower()).strip()
        run = run + 1 if key == last else 1
        last = key
        if run >= 3:
            continue
        out.append({"start": s["start"], "end": s["end"], "text": text})
    return out


# ---------------------------------------------------------------- MLX
def _hf_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return any(isinstance(try_to_load_from_cache(repo, f), str) for f in ("weights.safetensors", "weights.npz"))
    except Exception:
        return False


def _ensure_model(model: str, log) -> str:
    repo = MLX_REPOS[model]
    if not _hf_cached(repo):
        log(f"Downloading model (one time, ~{MLX_SIZE_GB[model]:g} GB)...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo)
        except Exception as e:
            raise PodscribeError(f"Could not download the '{model}' model from Hugging Face ({e}). Check your connection and retry.")
    return repo


def _total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def _free_mlx():
    try:
        import mlx.core as mx
        import mlx_whisper.transcribe as t
        t.ModelHolder.model = None
        t.ModelHolder.model_path = None
        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
    except Exception:
        pass


_HOOK_LOCK = threading.Lock()


def _mlx_tqdm_holder():
    """The `tqdm` module object inside mlx_whisper's transcribe module, or None if this mlx-whisper has none.
    Note: `import mlx_whisper.transcribe` yields the *function* of that name, so go through sys.modules."""
    holder = getattr(sys.modules.get("mlx_whisper.transcribe"), "tqdm", None)
    return holder if holder is not None and hasattr(holder, "tqdm") else None


@contextlib.contextmanager
def _frame_progress(on_frames):
    """mlx_whisper reports progress through an internal tqdm bar (one update per 30 s window of audio) that is only
    enabled with verbose=False. Swap in a silent subclass that calls on_frames(done, total) and restore it afterwards.
    Yields True if the hook is installed, False if this mlx-whisper has no such bar (progress then stays per-chunk)."""
    holder = _mlx_tqdm_holder()
    if holder is None:
        yield False
        return
    orig = holder.tqdm

    class Hook(orig):
        def __init__(self, *a, **k):
            k["disable"] = True  # no terminal bar, just the callbacks
            super().__init__(*a, **k)
            self._frames, self._total = 0, k.get("total")
            on_frames(0, self._total)

        def update(self, n=1):
            self._frames += n  # tqdm does not count while disabled
            on_frames(self._frames, self._total)
            return super().update(n)

    with _HOOK_LOCK:  # the swap is process-global
        holder.tqdm = Hook
        try:
            yield True
        finally:
            holder.tqdm = orig


def _mlx(audio: Path, model: str, progress, log):
    progress(None, None)  # setup (model download/load, silence and speech detection) has no measurable progress yet
    _need_ffmpeg()
    import mlx_whisper
    repo = _ensure_model(model, log)
    if model in ("large-v3", "medium") and 0 < _total_ram_gb() < 12:
        log(f"Warning: '{model}' needs several GB of memory and this Mac has {_total_ram_gb():.0f} GB; it may run out.")
    duration = audio_duration(audio)
    log("Detecting silences and speech...")
    silences = detect_silences(audio, duration)
    try:
        speech = detect_speech(audio)
    except Exception as e:  # VAD is an extra safeguard; fall back to silence detection alone
        log(f"Speech detection unavailable ({e}); relying on silence detection only.")
        speech = None
    cuts = plan_cuts(duration, silences)
    log(f"Transcribing with MLX '{model}' on the Apple GPU ({len(cuts) - 1} chunk{'s' if len(cuts) > 2 else ''})...")
    segments, language = [], None
    # `real` flips to True at the first measured update; until then the UI shows the indeterminate state.
    state = {"last": 0.0, "real": _mlx_tqdm_holder() is None}

    def report(seconds):  # monotonic, clamped to the audio length
        state["last"] = max(state["last"], min(max(seconds, 0.0), duration))
        progress(state["last"], duration)

    try:
        with tempfile.TemporaryDirectory(prefix="podscribe-chunks-") as tmp:
            for i, (lo, hi, start, end) in enumerate(chunk_bounds(cuts, silences, duration)):
                if coverage(lo, hi, silences) >= 0.98 or (speech is not None and not _speech_overlap(lo, hi, speech)):
                    # chunk has no speech: nothing to transcribe
                    state["real"] = True
                    report(hi)
                    continue
                if state["real"]:
                    report(lo)  # also lets a cancel take effect before the next chunk starts
                else:
                    progress(None, None)  # still setting up; also a cancel point

                def on_frames(done, total, lo=lo, hi=hi, start=start, end=end):
                    state["real"] = True
                    frac = min(done / total, 1.0) if total else 0.0
                    report(min(max(start + frac * (end - start), lo), hi))  # padding never moves the bar backwards

                wav = Path(tmp) / f"chunk{i}.wav"
                extract(audio, start, end, wav)
                opts = dict(path_or_hf_repo=repo, condition_on_previous_text=False, no_speech_threshold=NO_SPEECH_MAX,
                            compression_ratio_threshold=2.4, logprob_threshold=-1.0)
                if language:
                    opts["language"] = language
                with _frame_progress(on_frames) as hooked, contextlib.redirect_stdout(io.StringIO()):
                    # verbose=False enables mlx_whisper's per-window progress bar (hooked above) but also makes it
                    # print "Detected language", so stdout is swallowed to keep the CLI's transcript output clean
                    r = mlx_whisper.transcribe(str(wav), verbose=False if hooked else None, **opts)
                language = language or r.get("language")
                segments += shift_and_keep(r["segments"], start, lo, hi, last=(i == len(cuts) - 2))
                state["real"] = True
                report(hi)
    except PodscribeError:
        raise
    except Exception as e:
        msg = str(e)
        if re.search(r"out of memory|insufficient memory|kIOGPU", msg, re.I) or isinstance(e, MemoryError):
            raise PodscribeError(f"Ran out of memory with the '{model}' model. Close other apps or pick a smaller model (small).")
        raise PodscribeError(f"Transcription failed: {msg[:300]}")
    finally:
        _free_mlx()
    return clean_segments(segments, silences, speech), duration
