"""Progress reporting for the MLX engine, against a fake mlx_whisper that mimics the real library's quirks:
`mlx_whisper.transcribe` is a FUNCTION (it shadows the submodule), the real submodule has `tqdm` (enabled only when
verbose=False), and verbose=False also prints "Detected language"."""
import sys
import tempfile
import types

import pytest

from podscribe import Cancelled, engines
from podscribe import transcribe as tr


class FakeTqdm:
    def __init__(self, *a, total=None, disable=False, unit=None, **k):
        self.total, self.disable, self.n = total, disable, 0

    def update(self, n=1):
        if not self.disable:  # like real tqdm: a disabled bar does not count
            self.n += n

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


@pytest.fixture
def fake_mlx(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = FakeTqdm
    sub = types.ModuleType("mlx_whisper.transcribe")
    sub.tqdm = tqdm_mod
    pkg = types.ModuleType("mlx_whisper")
    info = {"verbose": [], "has_bar": True, "lengths": {}, "duration": 120.0, "silences": []}

    def fake_transcribe(path, **kw):
        info["verbose"].append(kw.get("verbose", "missing"))
        if kw.get("verbose") is not None:
            print("Detected language: English")
        total = int(info["lengths"][str(path)] * 100)  # 100 mel frames per second
        if info["has_bar"]:
            bar = sub.tqdm.tqdm(total=total, unit="frames", disable=kw.get("verbose") is not False)
            with bar:
                done = 0
                while done < total:
                    step = min(3000, total - done)  # one update per 30 s window
                    done += step
                    bar.update(step)
        return {"language": "en", "segments": [{"start": 3.0, "end": 4.0, "text": "hello there", "no_speech_prob": 0.0, "avg_logprob": -0.2}]}

    pkg.transcribe = fake_transcribe  # the package attribute is the function, as in the real library
    monkeypatch.setitem(sys.modules, "mlx_whisper", pkg)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", sub)
    monkeypatch.setattr(engines, "_need_ffmpeg", lambda: None)
    monkeypatch.setattr(engines, "_ensure_model", lambda model, log: "repo")
    monkeypatch.setattr(engines, "_free_mlx", lambda: None)
    monkeypatch.setattr(engines, "_total_ram_gb", lambda: 16.0)
    monkeypatch.setattr(engines, "audio_duration", lambda a: info["duration"])
    monkeypatch.setattr(engines, "detect_silences", lambda a, d: info["silences"])
    monkeypatch.setattr(engines, "detect_speech", lambda a: [(0.0, info["duration"])])

    def fake_extract(audio, start, end, out):
        info["lengths"][str(out)] = end - start
        out.write_bytes(b"x")
    monkeypatch.setattr(engines, "extract", fake_extract)
    info["sub"], info["tmp"] = sub, tmp_path

    def run(duration, progress):
        info["duration"] = duration
        return engines._mlx("a.mp3", "large-v3-turbo", progress, lambda m: None)
    info["run"] = run
    return info


def collect(fake, duration):
    events = []
    fake["run"](duration, lambda d, t: events.append((d, t)))
    return events


def test_single_120s_chunk_reports_several_distinct_values(fake_mlx):
    """The bug: a 2-minute episode is one chunk and used to report only 0% then 100%."""
    ev = collect(fake_mlx, 120.0)
    vals = [d for d, _ in ev if d is not None]
    assert len(set(vals)) >= 4, vals
    assert vals == sorted(vals) and vals[-1] == 120.0
    assert any(0 < v < 120 for v in vals)
    assert fake_mlx["verbose"] == [False]  # the library's own progress bar was switched on


def test_setup_is_indeterminate_then_determinate_never_back(fake_mlx):
    ev = collect(fake_mlx, 120.0)
    assert ev[0] == (None, None)
    kinds = [d is None for d, _ in ev]
    assert kinds == sorted(kinds, reverse=True), "None must only appear before the first real value"
    assert kinds.count(True) >= 1 and kinds.count(False) >= 3
    assert all(t == 120.0 for d, t in ev if d is not None)


def test_stdout_stays_clean(fake_mlx, capsys):
    collect(fake_mlx, 120.0)
    assert capsys.readouterr().out == ""  # verbose=False would print "Detected language": swallowed


def test_multi_chunk_progress_is_monotonic_and_bounded(fake_mlx):
    ev = collect(fake_mlx, 600.0)  # 3 chunks; hard cuts so chunks are padded
    vals = [d for d, _ in ev if d is not None]
    assert vals == sorted(vals), "padding must never move the bar backwards"
    assert vals[-1] == 600.0 and max(vals) <= 600.0 and min(vals) >= 0.0
    assert len(set(vals)) >= 8
    # values inside the second chunk stay within its own span [240, 480] (+ the final report)
    assert any(240 < v < 480 for v in vals)


def test_hook_is_restored_after_success(fake_mlx):
    collect(fake_mlx, 120.0)
    assert fake_mlx["sub"].tqdm.tqdm is FakeTqdm


def test_cancel_from_the_hook_propagates_fast_and_cleans_up(fake_mlx):
    seen = []

    def progress(d, t):
        seen.append(d)
        if d is not None and d >= 30:
            raise Cancelled()
    with pytest.raises(Cancelled):
        fake_mlx["run"](600.0, progress)
    assert fake_mlx["sub"].tqdm.tqdm is FakeTqdm  # patch restored
    assert len([d for d in seen if d is not None]) <= 4  # stopped within the first window, not after the chunk
    assert [p.name for p in fake_mlx["tmp"].iterdir() if p.name.startswith("podscribe-chunks-")] == []


def test_fallback_when_the_library_has_no_progress_bar(fake_mlx):
    del fake_mlx["sub"].tqdm  # an mlx-whisper without that tqdm: must not crash
    fake_mlx["has_bar"] = False
    ev = collect(fake_mlx, 120.0)
    assert [d for d, _ in ev if d is not None] == [0.0, 120.0]  # per-chunk progress, as before
    assert fake_mlx["verbose"] == [None]


# ---- the transcribe() wrapper and the other engine
def test_cli_bar_is_not_created_during_setup(monkeypatch):
    created = []

    class Bar:
        def __init__(self, total=None, **k):
            created.append(total)
            self.total, self.n = total, 0

        def refresh(self):
            pass

        def close(self):
            pass
    monkeypatch.setattr(tr, "tqdm", Bar)
    monkeypatch.setattr(tr, "select_engine", lambda: "mlx")

    def fake_run(engine, audio, model, report, log):
        report(None, None)
        assert created == []  # nothing to draw yet
        report(0.0, 100.0)
        report(50.0, 100.0)
        return [], 100.0
    monkeypatch.setattr(tr, "run", fake_run)
    tr.transcribe("a.mp3", "small")  # no progress callback -> CLI bar
    assert created == [100]


def test_progress_callback_receives_none_then_numbers(monkeypatch):
    got = []
    monkeypatch.setattr(tr, "select_engine", lambda: "mlx")
    monkeypatch.setattr(tr, "run", lambda e, a, m, report, log: (report(None, None), report(10.0, 100.0), ([], 100.0))[-1])
    tr.transcribe("a.mp3", "small", progress=lambda d, t: got.append((d, t)))
    assert got == [(None, None), (10.0, 100.0)]


def test_faster_whisper_starts_indeterminate(monkeypatch):
    class Seg:
        start, end, text = 0.0, 5.0, " hi "

    class Info:
        duration = 10.0

    class WM:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, **k):
            return iter([Seg()]), Info()
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = WM
    utils = types.ModuleType("faster_whisper.utils")
    utils._MODELS = {"large-v3-turbo": 1}
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    ev = []
    segs, dur = engines._faster_whisper("a.mp3", "large-v3-turbo", lambda d, t: ev.append((d, t)), lambda m: None)
    assert ev[0] == (None, None) and ev[-1] == (10.0, 10.0) and segs[0]["text"] == "hi"


def test_ui_has_a_stall_safety_net():
    from pathlib import Path
    js = (Path(engines.__file__).parent / "web" / "static" / "app.js").read_text()
    assert "STALL_MS = 6000" in js and "Still working…" in js
    assert 'removeProperty("--p")' in js  # the indeterminate state must not be blocked by a stale inline progress value
