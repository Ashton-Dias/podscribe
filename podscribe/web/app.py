"""Local-only web UI for podscribe. Bind to 127.0.0.1; every request must come from a localhost Host/Origin."""
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import PodscribeError
from ..engines import DEFAULT_MODEL, MODELS
from ..resolve import is_spotify_episode
from .jobs import JobManager

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class NewJob(BaseModel):
    url: str
    model: str = DEFAULT_MODEL


class Choice(BaseModel):
    index: int


def create_app(ttl: float | None = None) -> FastAPI:
    manager = JobManager(ttl) if ttl is not None else JobManager()

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        yield
        manager.shutdown()

    app = FastAPI(title="podscribe", lifespan=lifespan)
    app.state.manager = manager

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = urlparse("//" + request.headers.get("host", "")).hostname
        origin = request.headers.get("origin")
        if host not in LOCAL_HOSTS or (origin and urlparse(origin).hostname not in LOCAL_HOSTS):
            return JSONResponse({"error": "Forbidden: localhost only."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(PodscribeError)
    async def podscribe_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"models": MODELS, "default_model": DEFAULT_MODEL}

    @app.post("/api/jobs")
    def create_job(req: NewJob):
        if not is_spotify_episode(req.url):
            raise PodscribeError("Expected a Spotify episode URL (https://open.spotify.com/episode/...).")
        if req.model not in MODELS:
            raise PodscribeError(f"Model must be one of: {', '.join(MODELS)}.")
        return manager.create(req.url.strip(), req.model)

    @app.get("/api/jobs/current")
    def current_job():
        return {"job": manager.current()}

    def find(job_id):
        job = manager.public(job_id)
        if not job:
            raise HTTPException(404, "Job not found.")
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return find(job_id)

    @app.post("/api/jobs/{job_id}/choose")
    def choose(job_id: str, req: Choice):
        find(job_id)
        manager.choose(job_id, req.index)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        find(job_id)
        manager.cancel(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/result")
    def result(job_id: str):
        return manager.take_result(job_id)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
