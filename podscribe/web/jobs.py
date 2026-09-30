"""One transcription at a time, entirely in memory: nothing about a job or its result is written to disk.
A finished result is handed to the browser once, then dropped (or dropped after RESULT_TTL seconds)."""
import queue
import threading
import time
import uuid

from .. import Cancelled, PodscribeError
from ..pipeline import transcribe_episode, wipe_stale_audio
from ..resolve import find_matches

ACTIVE = ("queued", "finding", "needs_choice", "downloading", "transcribing")
RESULT_TTL = 30 * 60
CHOICE_TIMEOUT = 15 * 60
PUBLIC = ("id", "status", "progress", "message", "error", "title", "spotify_title", "candidates", "created", "finished")


class JobManager:
    def __init__(self, ttl: float = RESULT_TTL):
        self.ttl = ttl
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.ctl: dict[str, dict] = {}
        self.queue: queue.Queue = queue.Queue()
        self.thread: threading.Thread | None = None

    # ---- lifecycle
    def start(self):
        wipe_stale_audio()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="podscribe-worker")
        self.thread.start()

    def shutdown(self):
        with self.lock:
            for job_id in list(self.jobs):
                self.cancel(job_id)
        self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=30)  # lets the running job delete its temp audio
        with self.lock:
            self.jobs.clear()
            self.ctl.clear()

    # ---- api
    def _purge(self):
        now = time.time()
        for job_id, j in list(self.jobs.items()):
            if j["status"] not in ACTIVE and j["finished"] and now - j["finished"] > self.ttl:
                del self.jobs[job_id]
                self.ctl.pop(job_id, None)

    def create(self, url: str, model: str) -> dict:
        with self.lock:
            self._purge()
            if any(j["status"] in ACTIVE for j in self.jobs.values()):
                raise PodscribeError("A transcription is already running. Cancel it or wait for it to finish.")
            job = {"id": uuid.uuid4().hex[:10], "url": url, "model": model, "status": "queued", "progress": None,
                   "message": "", "error": None, "title": "", "spotify_title": None, "candidates": None,
                   "result": None, "created": time.time(), "finished": None}
            self.jobs[job["id"]] = job
            self.ctl[job["id"]] = {"cancel": threading.Event(), "choice": threading.Event(), "index": None}
        self.queue.put(job["id"])
        return self.public(job["id"])

    def update(self, job_id: str, **fields):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(fields)

    def public(self, job_id: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            return {k: j[k] for k in PUBLIC} if j else None

    def current(self) -> dict | None:
        with self.lock:
            active = [j for j in self.jobs.values() if j["status"] in ACTIVE]
            return self.public(active[-1]["id"]) if active else None

    def choose(self, job_id: str, index: int):
        with self.lock:
            j = self.jobs.get(job_id)
            if not j or j["status"] != "needs_choice":
                raise PodscribeError("Nothing to choose right now.")
            if not 0 <= index < len(j["candidates"]):
                raise PodscribeError("Invalid choice.")
            self.ctl[job_id]["index"] = index
            self.ctl[job_id]["choice"].set()

    def cancel(self, job_id: str):
        with self.lock:
            c = self.ctl.get(job_id)
            if c:
                c["cancel"].set()
                c["choice"].set()  # unblock a job that is waiting for a choice

    def take_result(self, job_id: str) -> dict:
        """Hand the finished transcript over exactly once, then forget the job."""
        with self.lock:
            j = self.jobs.get(job_id)
            if not j or j["status"] != "done" or j["result"] is None:
                raise PodscribeError("The transcript is no longer available. Nothing is saved; start a new transcription.")
            result = j["result"]
            del self.jobs[job_id]
            self.ctl.pop(job_id, None)
            return result

    # ---- worker
    def _loop(self):
        while True:
            job_id = self.queue.get()
            if job_id is None:
                return
            self._run(job_id)

    def _run(self, job_id: str):
        ctl = self.ctl[job_id]
        job = self.jobs[job_id]

        def check():
            if ctl["cancel"].is_set():
                raise Cancelled()

        def progress(fraction):
            check()
            self.update(job_id, progress=fraction)

        try:
            check()
            self.update(job_id, status="finding", message="Finding the podcast feed and episode")
            m = find_matches(job["url"])
            check()
            idx = 0
            if not m.confident:
                self.update(job_id, status="needs_choice", message="", spotify_title=m.spotify_title, candidates=[
                    {"score": round(s), "title": e.title, "show": e.show, "date": e.date, "feed_url": e.feed_url} for s, e in m.candidates])
                deadline = time.time() + CHOICE_TIMEOUT
                while not ctl["choice"].wait(0.5):
                    if time.time() > deadline:
                        raise PodscribeError("Timed out waiting for you to pick an episode.")
                check()
                idx = ctl["index"]
            ep = m.candidates[idx][1]

            def on_transcribe(d, t):  # d is None while progress is not measurable yet (model load, detection)
                self.update(job_id, status="transcribing", message=self.jobs[job_id]["message"]
                            if self.jobs[job_id]["status"] == "transcribing" else f"Transcribing with Whisper '{job['model']}'")
                progress(None if d is None or not t else min(d / t, 1))

            self.update(job_id, status="downloading", progress=0, message="Downloading audio", title=ep.title, candidates=None)
            result = transcribe_episode(
                ep, job["model"],
                on_download=lambda d, t: progress((d / t) if t else None),
                on_transcribe=on_transcribe,
                log=lambda msg: (check(), self.update(job_id, message=msg)))
            self.update(job_id, status="done", progress=1, message="", result=result, finished=time.time())
        except Cancelled:
            self.update(job_id, status="cancelled", message="", finished=time.time())
        except PodscribeError as e:
            self.update(job_id, status="failed", error=str(e), finished=time.time())
        except Exception as e:  # keep the worker alive whatever happens
            self.update(job_id, status="failed", error=f"Unexpected error: {e}", finished=time.time())
