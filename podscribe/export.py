"""In-memory transcript exports for pasting into an LLM (nothing is written to disk here)."""
import re

from .transcribe import fmt_ts


def slugify(s: str, max_len: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:max_len].strip("-") or "untitled"


def build_markdown(meta: dict, data: dict) -> str:
    dur = data.get("duration") or meta.get("duration")
    lines = [f"# {meta.get('title', 'Untitled episode')}", ""]
    for label, value in [("Show", meta.get("show")), ("Date", meta.get("date")),
                         ("Duration", f"{round(dur / 60)} min" if dur else None), ("Source", meta.get("source_url")),
                         ("Transcribed with", f"Whisper {data.get('model')} ({data.get('engine')})" if data.get("model") else None)]:
        if value:
            lines.append(f"- **{label}:** {value}")
    lines += ["", "## Transcript", ""]
    lines += [f"{fmt_ts(s['start'])} {s['text']}" for s in data["segments"]]
    return "\n".join(lines) + "\n"


def stats(segments: list[dict]) -> dict:
    """Word count and a rough token estimate (~4 characters per token) to judge LLM context fit."""
    text = " ".join(s["text"] for s in segments)
    return {"words": len(text.split()), "tokens": round(len(text) / 4)}


def build_result(ep, data: dict) -> dict:
    """Everything the UI/CLI needs, all in memory: header info, stats, and the Markdown file body."""
    meta = {"title": ep.title, "show": ep.show, "date": ep.date, "source_url": ep.source_url, "duration": data["duration"]}
    return {"title": ep.title, "show": ep.show, "date": ep.date, "duration": data["duration"],
            **stats(data["segments"]),
            "filename_base": f"{slugify(ep.show)}_{ep.date}_{slugify(ep.title)}",
            "markdown": build_markdown(meta, data)}
