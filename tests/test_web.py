import json
import tempfile
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import podscribe
from podscribe import Cancelled, export, net, pipeline
from podscribe.resolve import Episode, Match
from podscribe.web import app as app_mod, jobs as jobs_mod

REPO = Path(podscribe.__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", "podscribe.egg-info", ".git"}


def repo_snapshot():
    """(path, size, mtime) of every file in the repo (minus venv/caches), to prove a run creates or changes nothing."""
    import os
    out = set()
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = Path(root) / f
            st = p.stat()
            out.add((str(p), st.st_size, st.st_mtime_ns))
    return out


SPOTIFY = "https://open.spotify.com/episode/abc123"
EP = Episode("Episode 1 - Hello", "Test Show", "2024-01-02", "https://cdn.example.com/a.mp3", SPOTIFY)
SEGS = [{"start": 0, "end": 5, "text": "hello world"}, {"start": 65, "end": 70, "text": "second line"}]
DATA = {"model": "large-v3-turbo", "engine": "mlx", "duration": 70, "segments": SEGS}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A running app with the network and the engines faked, in an empty cwd and an empty temp dir."""
    tmp = tmp_path / "tmp"
    cwd = tmp_path / "cwd"
    tmp.mkdir(), cwd.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.chdir(cwd)
    state = {"match": Match("Episode 1 - Hello", [(100, EP)], True), "transcribe": None, "seen_dirs": []}

    def fake_download(ep, dest_dir, progress=None):
        state["seen_dirs"].append(dest_dir)
        mp3 = dest_dir / "episode.mp3"
        mp3.write_bytes(b"audio")
        progress and progress(1, 1)
        return mp3

    def fake_transcribe(mp3, model="large-v3-turbo", progress=None, log=print):
        assert mp3.exists()
        if state["transcribe"]:
            return state["transcribe"](progress)
        progress and progress(70, 70)
        return dict(DATA, model=model)

    monkeypatch.setattr(jobs_mod, "find_matches", lambda url: state["match"])
    monkeypatch.setattr(pipeline, "download", fake_download)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    state["tmp"], state["cwd"], state["repo_before"] = tmp, cwd, repo_snapshot()
    with TestClient(app_mod.create_app(), base_url="http://localhost") as client:
        client.state = state
        yield client


def wait(client, job_id, statuses=("done", "failed", "cancelled"), timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in statuses:
            return j
        time.sleep(0.03)
    raise AssertionError(f"job stuck: {client.get(f'/api/jobs/{job_id}').json()}")


def submit(client, **kw):
    r = client.post("/api/jobs", json={"url": SPOTIFY, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def assert_nothing_left(env):
    """No audio temp dirs, nothing in the working directory, and nothing new in the repo."""
    assert [p.name for p in env.state["tmp"].iterdir()] == []
    assert list(env.state["cwd"].iterdir()) == []
    assert repo_snapshot() == env.state["repo_before"]  # no files created, changed or removed in the repo


# ---- validation
def test_url_and_model_validation(env):
    for bad in ["https://example.com/x", "file:///etc/passwd", "https://open.spotify.com/show/abc", ""]:
        r = env.post("/api/jobs", json={"url": bad})
        assert r.status_code == 400 and "Spotify" in r.json()["error"]
    assert env.post("/api/jobs", json={"url": SPOTIFY, "model": "huge"}).status_code == 400


def test_only_one_job_at_a_time(env):
    gate = threading.Event()
    env.state["transcribe"] = lambda progress: (gate.wait(5), dict(DATA))[1]
    j = submit(env)
    wait(env, j["id"], statuses=("transcribing", "downloading"))
    r = env.post("/api/jobs", json={"url": SPOTIFY})
    assert r.status_code == 400 and "already running" in r.json()["error"]
    gate.set()
    wait(env, j["id"], statuses=("done",))


# ---- success: result is delivered once, nothing persists
def test_result_content_and_single_delivery(env):
    j = submit(env)
    wait(env, j["id"])
    r = env.get(f"/api/jobs/{j['id']}/result")
    assert r.status_code == 200
    res = r.json()
    assert res["title"] == EP.title and res["words"] == 4 and res["tokens"] >= 1
    assert res["filename_base"] == "test-show_2024-01-02_episode-1-hello"
    assert set(res) == {"title", "show", "date", "duration", "words", "tokens", "filename_base", "markdown"}
    md = res["markdown"]
    assert md.startswith("# Episode 1 - Hello\n") and "- **Show:** Test Show" in md and "- **Source:** " + SPOTIFY in md
    assert "Whisper large-v3-turbo (mlx)" in md and "## Transcript" in md and "[01:05] second line" in md
    # a second fetch (e.g. after a refresh) finds nothing: the server dropped it
    again = env.get(f"/api/jobs/{j['id']}/result")
    assert again.status_code == 400 and "no longer available" in again.json()["error"]
    assert env.get(f"/api/jobs/{j['id']}").status_code == 404
    assert env.get("/api/jobs/current").json() == {"job": None}
    assert_nothing_left(env)


def test_audio_is_deleted_even_on_success(env):
    wait(env, submit(env)["id"])
    assert env.state["seen_dirs"] and not any(d.exists() for d in env.state["seen_dirs"])


def test_failure_cleans_up_and_reports(env):
    def boom(progress):
        raise podscribe.PodscribeError("engine exploded")
    env.state["transcribe"] = boom
    j = wait(env, submit(env)["id"])
    assert j["status"] == "failed" and j["error"] == "engine exploded"
    assert not any(d.exists() for d in env.state["seen_dirs"])
    assert env.get(f"/api/jobs/{j['id']}/result").status_code == 400
    assert_nothing_left(env)


def test_unexpected_error_does_not_kill_the_worker(env):
    env.state["transcribe"] = lambda progress: 1 / 0
    j = wait(env, submit(env)["id"])
    assert j["status"] == "failed" and "Unexpected error" in j["error"]
    env.state["transcribe"] = None
    assert wait(env, submit(env)["id"])["status"] == "done"


# ---- cancel
def test_cancel_during_transcription(env):
    started = threading.Event()

    def slow(progress):
        started.set()
        for i in range(500):
            progress(i, 500)  # the job's callback raises Cancelled once cancel is requested
            time.sleep(0.01)
        return dict(DATA)
    env.state["transcribe"] = slow
    j = submit(env)
    assert started.wait(5)
    assert env.post(f"/api/jobs/{j['id']}/cancel").json() == {"ok": True}
    done = wait(env, j["id"])
    assert done["status"] == "cancelled"
    assert not any(d.exists() for d in env.state["seen_dirs"])
    assert env.get(f"/api/jobs/{j['id']}/result").status_code == 400
    assert_nothing_left(env)
    assert wait(env, submit(env)["id"])["status"] == "done"  # a new job can start right away


def test_cancel_while_waiting_for_a_choice(env):
    other = Episode("Episode 2", "Test Show", "2024-01-03", "https://cdn.example.com/b.mp3", SPOTIFY)
    env.state["match"] = Match("Ep", [(70, EP), (66, other)], False)
    j = submit(env)
    waiting = wait(env, j["id"], statuses=("needs_choice",))
    assert len(waiting["candidates"]) == 2
    env.post(f"/api/jobs/{j['id']}/cancel")
    assert wait(env, j["id"])["status"] == "cancelled"
    assert env.state["seen_dirs"] == []  # never downloaded anything
    assert_nothing_left(env)


# ---- uncertain match
def test_uncertain_match_choose_flow(env):
    other = Episode("Episode 2", "Test Show", "2024-01-03", "https://cdn.example.com/b.mp3", SPOTIFY)
    env.state["match"] = Match("Ep", [(70, EP), (66, other)], False)
    j = submit(env)
    w = wait(env, j["id"], statuses=("needs_choice",))
    assert [c["title"] for c in w["candidates"]] == ["Episode 1 - Hello", "Episode 2"]
    assert env.post(f"/api/jobs/{j['id']}/choose", json={"index": 7}).status_code == 400
    assert env.post(f"/api/jobs/{j['id']}/choose", json={"index": 1}).status_code == 200
    wait(env, j["id"], statuses=("done",))
    assert env.get(f"/api/jobs/{j['id']}/result").json()["title"] == "Episode 2"


def test_resume_after_refresh_shows_running_job(env):
    gate = threading.Event()
    env.state["transcribe"] = lambda progress: (gate.wait(5), dict(DATA))[1]
    j = submit(env)
    wait(env, j["id"], statuses=("transcribing", "downloading"))
    assert env.get("/api/jobs/current").json()["job"]["id"] == j["id"]
    gate.set()


# ---- expiry and startup/shutdown hygiene
def test_undelivered_result_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "find_matches", lambda url: Match("t", [(100, EP)], True))
    monkeypatch.setattr(jobs_mod, "transcribe_episode", lambda *a, **k: export.build_result(EP, DATA))
    m = jobs_mod.JobManager(ttl=0.2)
    m.start()
    try:
        job = m.create(SPOTIFY, "small")
        end = time.time() + 5
        while m.public(job["id"])["status"] != "done" and time.time() < end:
            time.sleep(0.02)
        time.sleep(0.3)
        m.create(SPOTIFY, "small")  # any new activity purges expired results
        assert m.public(job["id"]) is None
        with pytest.raises(podscribe.PodscribeError):
            m.take_result(job["id"])
    finally:
        m.shutdown()


def test_stale_audio_dirs_wiped_on_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stale = tmp_path / (pipeline.AUDIO_PREFIX + "crashed")
    other = tmp_path / "someone-elses-dir"
    stale.mkdir(), other.mkdir()
    (stale / "episode.mp3").write_bytes(b"x")
    m = jobs_mod.JobManager()
    m.start()
    m.shutdown()
    assert not stale.exists() and other.exists()


def test_shutdown_forgets_everything(env):
    gate = threading.Event()
    env.state["transcribe"] = lambda progress: (gate.wait(5), dict(DATA))[1]
    submit(env)
    m = env.app.state.manager
    gate.set()
    m.shutdown()
    assert m.jobs == {}


# ---- removed features stay removed
def test_removed_endpoints_are_gone(env):
    for path in ["/api/episodes", "/api/settings", "/api/jobs"]:
        assert env.get(path).status_code in (404, 405)
    assert env.delete("/api/episodes/a/b").status_code in (404, 405)


def test_page_is_a_single_screen():
    html = (Path(app_mod.__file__).parent / "static" / "index.html").read_text()
    assert "<nav" not in html and "Settings" not in html and "Transcripts" not in html


# ---- safety
def test_localhost_only(env):
    assert env.get("/api/health", headers={"host": "evil.example.com"}).status_code == 403
    assert env.post("/api/jobs", headers={"origin": "https://evil.example.com"}, json={"url": SPOTIFY}).status_code == 403
    assert env.get("/api/health").status_code == 200


def test_ssrf_guard():
    for u in ["http://127.0.0.1/a.mp3", "http://localhost/x", "http://169.254.169.254/latest", "file:///etc/passwd", "http://10.0.0.5/x"]:
        with pytest.raises(podscribe.PodscribeError):
            net.check_public_url(u)


# ---- export
def test_markdown_omits_missing_header_fields():
    md = export.build_markdown({"title": "T"}, {"segments": SEGS})
    assert md.startswith("# T\n") and "**Show:**" not in md and "**Duration:**" not in md and "[01:05] second line" in md


def test_stats():
    s = export.stats([{"text": "one two three"}, {"text": "four"}])
    assert s == {"words": 4, "tokens": round(len("one two three four") / 4)}


def test_done_screen_is_download_md_and_new_episode_only():
    js = (Path(app_mod.__file__).parent / "static" / "app.js").read_text()
    for gone in ["Search transcript", "type: \"search\"", "Download .txt", "Download .json", "Copy transcript",
                 "clipboard", "fmtTs", "segments", "lines\""]:
        assert gone not in js, gone
    assert "Download .md (for your LLM)" in js and "New episode" in js and "Nothing is saved" in js
    css = (Path(app_mod.__file__).parent / "static" / "style.css").read_text()
    assert ".preview" not in css and ".line{" not in css and ".ts{" not in css
    assert not hasattr(export, "build_txt")


# ---- theme: self-hosted assets, no external requests, palette tokens
STATIC = Path(app_mod.__file__).parent / "static"


def test_static_assets_make_no_external_requests():
    import re
    allowed = ("http://www.w3.org/2000/svg", "https://open.spotify.com/episode/...", "http://127.0.0.1", "http://localhost")
    for f in STATIC.rglob("*"):
        if f.suffix in {".html", ".css", ".js"}:
            text = f.read_text()
            for url in re.findall(r"https?://[^\s\"')<>]+", text):
                assert url.startswith(allowed), f"{f.name} references {url}"
            for cdn in ("googleapis", "gstatic", "cdn.", "unpkg", "jsdelivr", "cdnjs", "fonts.g", "@import"):
                assert cdn not in text, f"{f.name} mentions {cdn}"


def test_fonts_are_bundled_with_licenses():
    fonts = STATIC / "fonts"
    wanted = ["fraunces-400", "fraunces-400-italic", "fraunces-600", "instrument-sans-400", "instrument-sans-500",
              "instrument-sans-600", "ibm-plex-mono-400", "ibm-plex-mono-500"]
    for name in wanted:
        assert (fonts / f"{name}.woff2").stat().st_size > 5000
    for lic in ["fraunces", "instrument-sans", "ibm-plex-mono"]:
        assert "Open Font License" in (fonts / f"LICENSE-{lic}.txt").read_text()
    # the previous theme's fonts are gone, not left behind
    assert sorted(p.name for p in fonts.iterdir() if p.suffix == ".woff2") == sorted(f"{n}.woff2" for n in wanted)
    assert not [p for p in fonts.iterdir() if any(x in p.name for x in ("inter-", "playfair", "jetbrains"))]
    assert "static/fonts/*" in (REPO / "pyproject.toml").read_text()


def test_static_payload_budget():
    total = sum(f.stat().st_size for f in STATIC.rglob("*") if f.is_file())
    assert total < 450_000, f"static/ is {total} bytes"


def test_theme_tokens_and_accessibility_hooks():
    css = (STATIC / "style.css").read_text().lower()
    for token in ["#f3eee4", "#fbf8f2", "#1f2a24", "#5f7161", "#a64b2f",          # light: paper, surface, ink, moss, terracotta
                  "#101713", "#17211b", "#efe9dc", "#d98a6a"]:                      # dark: greenhouse at night
        assert token in css, token
    for family in ['"fraunces"', '"instrument sans"', '"ibm plex mono"']:
        assert family in css
    for old in ["playfair", "jetbrains", '"inter"', "#0d0d12", "#c9a84c"]:
        assert old not in css, old
    assert "prefers-color-scheme:dark" in css.replace(" ", "") and "prefers-reduced-motion" in css
    assert ":focus-visible" in css
    html = (STATIC / "index.html").read_text()
    assert 'name="theme-color"' in html and 'name="color-scheme"' in html and 'rel="icon"' in html
    js = (STATIC / "app.js").read_text()
    assert 'for: "url"' in js and 'for: "model"' in js  # labels are associated with their inputs
    assert 'role: notice && notice.error ? "alert"' in js or 'role: "alert"' in js


# ---- home screen copy and compact, fit-on-one-screen layout
def test_home_copy_and_compact_layout():
    js = (STATIC / "app.js").read_text()
    assert "For the episodes that caught your attention, but you'd prefer to " in js and 'h("em", {}, "read")' in js
    assert "Paste a Spotify episode link. A timestamped transcript is composed on your Mac, ready for LLM-powered summary and study." in js
    for old in ["Every voice", "observed", "Nothing is kept", "A careful, timestamped"]:
        assert old not in js, old
    css = (STATIC / "style.css").read_text().replace(" ", "")
    assert "100dvh" in css and ".intake{display:grid" in css           # fills the viewport, two-column intake on wide screens
    assert "@media(max-width:900px){.intake{display:block}.intake.fig{display:none}" in css  # stacks, figure hidden when narrow
    assert "min(3.7vw,6.4vh)" in css                                     # headline scales with viewport height too
    assert ".pick-list{display:grid;grid-template-columns:repeat(3" in css  # candidate plates side by side


# ---- progress: unknown during setup, then a real fraction
def test_job_progress_is_none_during_setup_then_a_fraction(env):
    gate = threading.Event()

    def stage(progress):
        progress(None, None)          # model load / detection: not measurable yet
        gate.wait(5)
        progress(35.0, 70.0)
        gate.wait(0.05)
        return dict(DATA)
    env.state["transcribe"] = stage
    j = submit(env)
    w = wait(env, j["id"], statuses=("transcribing",))
    assert w["status"] == "transcribing" and w["progress"] is None
    gate.set()
    assert wait(env, j["id"], statuses=("done",))["status"] == "done"
