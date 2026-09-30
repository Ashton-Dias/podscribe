import requests
from pathlib import Path
from tqdm import tqdm

from . import PodscribeError
from .net import safe_get
from .resolve import Episode

MAX_BYTES = 1_500_000_000


def download(ep: Episode, dest_dir: Path, progress=None) -> Path:
    """Download the episode audio into dest_dir (a temp dir the caller deletes).
    progress(done_bytes, total_bytes|None); without it a tqdm bar is shown (on stderr)."""
    mp3 = dest_dir / "episode.mp3"
    bar = None
    try:
        with safe_get(ep.mp3_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            if total and total > MAX_BYTES:
                raise PodscribeError(f"Audio file too large ({total / 1e6:.0f} MB).")
            if progress is None:
                bar = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")
            done = 0
            with open(mp3, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    done += len(chunk)
                    if done > MAX_BYTES:
                        raise PodscribeError("Audio file too large.")
                    f.write(chunk)
                    if progress:
                        progress(done, total)
                    else:
                        bar.update(len(chunk))
    except requests.RequestException as e:
        raise PodscribeError(f"Download failed: {e}")
    finally:
        if bar:
            bar.close()
    return mp3
