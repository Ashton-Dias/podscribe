from pathlib import Path

from tqdm import tqdm

from .engines import DEFAULT_MODEL, run, select_engine


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"[{h}:{m:02d}:{sec:02d}]" if h else f"[{m:02d}:{sec:02d}]"


def transcribe(mp3: Path, model: str = DEFAULT_MODEL, progress=None, log=print) -> dict:
    """Transcribe in memory; returns {"model", "engine", "duration", "segments"}. Nothing is written to disk.
    progress(done_seconds, total_seconds), or progress(None, None) while progress is not measurable yet;
    without it a tqdm bar is shown (on stderr)."""
    engine = select_engine()
    bar = None

    def report(done, total):
        nonlocal bar
        if progress:
            progress(done, total)
            return
        if done is None or not total:  # setup phase: nothing to show yet
            return
        if bar is None:
            bar = tqdm(total=round(total), unit="s", desc="Transcribing")
        bar.n = min(round(done), bar.total)
        bar.refresh()

    try:
        segments, duration = run(engine, mp3, model, report, log)
    finally:
        if bar:
            bar.close()
    return {"model": model, "engine": engine, "duration": duration, "segments": segments}
