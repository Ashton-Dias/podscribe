import json

import pytest

import podscribe
from podscribe import engines, transcribe as tr


# ---- engine selection
def test_selects_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.delenv("PODSCRIBE_ENGINE", raising=False)
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
    assert engines.select_engine() == "mlx"


@pytest.mark.parametrize("system,machine", [("Darwin", "x86_64"), ("Linux", "x86_64"), ("Linux", "aarch64"), ("Windows", "AMD64")])
def test_falls_back_to_faster_whisper_elsewhere(monkeypatch, system, machine):
    monkeypatch.delenv("PODSCRIBE_ENGINE", raising=False)
    monkeypatch.setattr(engines.platform, "system", lambda: system)
    monkeypatch.setattr(engines.platform, "machine", lambda: machine)
    assert engines.select_engine() == "faster-whisper"


def test_env_override(monkeypatch):
    monkeypatch.setenv("PODSCRIBE_ENGINE", "faster-whisper")
    monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
    assert engines.select_engine() == "faster-whisper"
    monkeypatch.setenv("PODSCRIBE_ENGINE", "mlx")
    assert engines.select_engine() == "mlx"
    monkeypatch.setenv("PODSCRIBE_ENGINE", "bogus")
    with pytest.raises(podscribe.PodscribeError):
        engines.select_engine()


# ---- model mapping
def test_model_mapping_complete():
    assert set(engines.MLX_REPOS) == set(engines.MODELS)
    assert engines.MLX_REPOS["large-v3-turbo"] == "mlx-community/whisper-large-v3-turbo"
    assert engines.DEFAULT_MODEL == "large-v3-turbo"


def test_unknown_model_rejected():
    with pytest.raises(podscribe.PodscribeError):
        engines.run("mlx", "x.mp3", "huge", lambda *a: None)


def test_faster_whisper_turbo_fallback(monkeypatch):
    import faster_whisper.utils as u
    monkeypatch.setattr(u, "_MODELS", {"large-v3": "x"})
    msgs = []
    assert engines._fw_model_name("large-v3-turbo", msgs.append) == "large-v3"
    assert msgs and "large-v3" in msgs[0]


# ---- chunk planning / stitching
def test_short_audio_is_one_chunk():
    assert engines.plan_cuts(200, []) == [0.0, 200]


def test_cuts_snap_to_nearest_silence():
    cuts = engines.plan_cuts(1000, [(230, 236), (400, 403), (700, 703)])
    assert cuts[0] == 0.0 and cuts[-1] == 1000
    assert cuts[1] == 233  # silence midpoint nearest the 240s target
    assert all(b - a > 0 for a, b in zip(cuts, cuts[1:]))


def test_cuts_without_silence_use_target():
    cuts = engines.plan_cuts(1000, [])
    assert cuts[1] == 240 and cuts[-1] == 1000


def test_no_tiny_last_chunk():
    cuts = engines.plan_cuts(290, [])
    assert cuts == [0.0, 290]


def test_stitching_offsets_and_overlap_dedupe():
    # two chunks cut at 100s; each is padded by 2s, so the padded region is seen by both
    a = [{"start": 96, "end": 99, "text": "one"}, {"start": 99, "end": 102, "text": "boundary-left"}]      # chunk 0, offset 0
    b = [{"start": 0, "end": 1.5, "text": "boundary-left"}, {"start": 3, "end": 5, "text": "two"}]        # chunk 1, offset 98
    out = engines.shift_and_keep(a, 0, 0, 100, last=False) + engines.shift_and_keep(b, 98, 100, 200, last=True)
    assert [s["text"] for s in out] == ["one", "two"] or [s["text"] for s in out] == ["one", "boundary-left", "two"]
    texts = [s["text"] for s in out]
    assert texts.count("boundary-left") <= 1  # never duplicated
    assert out[-1]["start"] == 101 and out[-1]["end"] == 103


def test_last_chunk_keeps_tail():
    out = engines.shift_and_keep([{"start": 0, "end": 5, "text": "tail"}], 200, 200, 204, last=True)
    assert len(out) == 1


# ---- silence parsing and hallucination filtering
STDERR = """[silencedetect @ 0x1] silence_start: 10.5
[silencedetect @ 0x1] silence_end: 14.25 | silence_duration: 3.75
[silencedetect @ 0x1] silence_start: 95.0"""


def test_parse_silences_including_trailing():
    assert engines.parse_silences(STDERR, 100.0) == [(10.5, 14.25), (95.0, 100.0)]


def test_clean_drops_text_in_silence_and_repeats():
    silences = [(10.0, 30.0)]
    segs = [
        {"start": 0, "end": 5, "text": "Real speech."},
        {"start": 12, "end": 20, "text": "Thanks for watching!"},                               # inside a long silence
        {"start": 30, "end": 32, "text": "Hello", "no_speech_prob": 0.9, "avg_logprob": -1.5},   # no-speech + low confidence
        {"start": 33, "end": 34, "text": "yeah"}, {"start": 34, "end": 35, "text": "Yeah."},
        {"start": 35, "end": 36, "text": "yeah"}, {"start": 36, "end": 37, "text": "yeah"},      # 3rd+ repeat dropped
        {"start": 38, "end": 40, "text": "  "},
    ]
    assert [s["text"] for s in engines.clean_segments(segs, silences)] == ["Real speech.", "yeah", "Yeah."]


def test_coverage():
    assert engines.coverage(0, 10, [(0, 10)]) == 1.0
    assert engines.coverage(0, 10, [(0, 5)]) == 0.5
    assert engines.coverage(0, 10, [(0, 0.8)], min_len=1.0) == 0.0


# ---- in-memory transcribe(): same shape from both engines, nothing written to disk
@pytest.mark.parametrize("engine", ["mlx", "faster-whisper"])
def test_transcribe_returns_engine_independent_shape(tmp_path, monkeypatch, engine):
    segs = [{"start": 0.0, "end": 5.0, "text": "hello"}, {"start": 65.0, "end": 70.0, "text": "world"}]
    monkeypatch.setattr(tr, "select_engine", lambda: engine)
    monkeypatch.setattr(tr, "run", lambda eng, audio, model, report, log: (report(70, 70) or segs, 70.0))
    monkeypatch.chdir(tmp_path)
    out = tr.transcribe(tmp_path / "a.mp3", "small", progress=lambda *a: None)
    assert out == {"model": "small", "engine": engine, "duration": 70.0, "segments": segs}
    assert list(tmp_path.iterdir()) == []  # nothing was written


def test_progress_callback_can_cancel_and_is_not_swallowed(monkeypatch):
    from podscribe import Cancelled
    monkeypatch.setattr(tr, "select_engine", lambda: "faster-whisper")

    def fake_run(eng, audio, model, report, log):
        report(1, 10)
        raise AssertionError("should have been cancelled before this")

    def stop(done, total):
        raise Cancelled()
    monkeypatch.setattr(tr, "run", fake_run)
    with pytest.raises(Cancelled):
        tr.transcribe("x.mp3", "small", progress=stop)


def test_cancelled_is_not_caught_by_except_exception():
    from podscribe import Cancelled
    assert not issubclass(Cancelled, Exception)



def test_segment_starting_in_silence_is_trimmed():
    # Whisper attaches leading noise to the next segment; the start should move to where the silence ends
    silences = [(45, 91), (91, 93), (93, 100), (100, 135)]  # fragmented, contiguous
    out = engines.clean_segments([{"start": 115.0, "end": 140.0, "text": "real speech"}], silences)
    assert out == [{"start": 135, "end": 140.0, "text": "real speech"}]
    # a segment that starts in speech is untouched
    assert engines.clean_segments([{"start": 5.0, "end": 9.0, "text": "x"}], silences)[0]["start"] == 5.0


def test_merge_silences():
    assert engines.merge_silences([(10, 12), (0, 3), (3.2, 5)]) == [(0, 5), (10, 12)]


def test_vad_regions_drop_and_trim_segments():
    speech = [(0.0, 40.4), (129.8, 170.0)]
    segs = [
        {"start": 0, "end": 6, "text": "speech"},
        {"start": 55, "end": 85, "text": "Thank you."},          # loud music: outside every speech region
        {"start": 115, "end": 140, "text": "late speech"},        # noise before the speech: start trimmed
        {"start": 35, "end": 50, "text": "spans the end"},        # 5.4 of 15 s is speech (36%): kept, end trimmed
    ]
    out = engines.clean_segments(segs, [], speech)
    assert [(s["text"], s["start"], s["end"]) for s in out] == [
        ("speech", 0, 6), ("late speech", 129.8, 140), ("spans the end", 35, 40.4)]


def test_without_vad_falls_back_to_silence_rules():
    out = engines.clean_segments([{"start": 55, "end": 85, "text": "Thank you."}], [], None)
    assert len(out) == 1  # nothing to reject it on: documents that VAD is what catches loud music


def test_real_speech_stretched_over_noise_is_kept_and_trimmed():
    speech = [(129.8, 170.0)]
    out = engines.clean_segments([{"start": 115, "end": 135, "text": "you know, so, you know"}], [], speech)
    assert out == [{"start": 129.8, "end": 135, "text": "you know, so, you know"}]
    # ...but a tiny speech blip inside a long hallucinated segment is still dropped
    assert engines.clean_segments([{"start": 100, "end": 130.5, "text": "Thank you."}], [], speech) == []


def test_silence_cuts_are_not_padded_hard_cuts_are():
    silences = [(238, 242)]
    b = engines.chunk_bounds([0.0, 240.0, 480.0, 600.0], silences, 600.0)
    assert b[0] == (0.0, 240.0, 0.0, 240.0)                      # ends in silence: no pad; starts at 0
    assert b[1] == (240.0, 480.0, 240.0, 480.0 + engines.PAD)     # starts in silence; hard cut at 480 gets pad
    assert b[2] == (480.0, 600.0, 480.0 - engines.PAD, 600.0)    # hard cut start padded, end of file not


# ---- first-run download message and errors
def test_first_run_download_message(monkeypatch):
    import huggingface_hub
    got = {}
    monkeypatch.setattr(engines, "_hf_cached", lambda repo: False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo: got.setdefault("repo", repo))
    msgs = []
    assert engines._ensure_model("large-v3-turbo", msgs.append) == "mlx-community/whisper-large-v3-turbo"
    assert got["repo"] == "mlx-community/whisper-large-v3-turbo"
    assert msgs == ["Downloading model (one time, ~1.6 GB)..."]


def test_cached_model_prints_nothing(monkeypatch):
    monkeypatch.setattr(engines, "_hf_cached", lambda repo: True)
    msgs = []
    engines._ensure_model("small", msgs.append)
    assert msgs == []


def test_download_failure_is_friendly(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(engines, "_hf_cached", lambda repo: False)

    def boom(repo):
        raise OSError("no network")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    with pytest.raises(podscribe.PodscribeError) as e:
        engines._ensure_model("large-v3-turbo", lambda m: None)
    assert "download" in str(e.value).lower() and "no network" in str(e.value)


def test_missing_ffmpeg_message(monkeypatch):
    monkeypatch.setattr(engines.shutil, "which", lambda name: None)
    with pytest.raises(podscribe.PodscribeError) as e:
        engines.audio_duration("x.mp3")
    assert "brew install ffmpeg" in str(e.value)


def test_unreadable_audio_message(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio")
    with pytest.raises(podscribe.PodscribeError) as e:
        engines.audio_duration(bad)
    assert "bad.mp3" in str(e.value)
