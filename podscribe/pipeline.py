"""download -> transcribe -> in-memory result. The audio lives in a temp dir that is always deleted."""
import glob
import shutil
import tempfile
from pathlib import Path

from .download import download
from .export import build_result
from .resolve import Episode
from .transcribe import transcribe

AUDIO_PREFIX = "podscribe-audio-"


def wipe_stale_audio() -> None:
    """Remove audio temp dirs left behind by a crashed run."""
    for d in glob.glob(str(Path(tempfile.gettempdir()) / (AUDIO_PREFIX + "*"))):
        shutil.rmtree(d, ignore_errors=True)


def transcribe_episode(ep: Episode, model: str, on_download=None, on_transcribe=None, log=print) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=AUDIO_PREFIX))
    try:
        mp3 = download(ep, tmp, on_download)
        data = transcribe(mp3, model, on_transcribe, log)
        return build_result(ep, data)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # success, failure or cancel: the audio never outlives the job
