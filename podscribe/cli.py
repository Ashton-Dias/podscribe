import argparse
import sys
from pathlib import Path

from . import PodscribeError
from .engines import DEFAULT_MODEL, MODELS
from .pipeline import transcribe_episode
from .resolve import resolve


def run(url, model, feed=None, out=None):
    """Transcribe and write the Markdown transcript to `out`, or print it. The audio is a temp file that is deleted."""
    log = lambda m: print(m, file=sys.stderr)
    ep = resolve(url, feed)
    log(f"Episode: {ep.title} ({ep.date}) — {ep.show}")
    result = transcribe_episode(ep, model, log=log)
    if out:
        Path(out).write_text(result["markdown"])
        log(f"Saved transcript to {out}")
    else:
        sys.stdout.write(result["markdown"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv and argv[0] == "web":
            p = argparse.ArgumentParser(prog="podscribe web", description="Run the local web UI (localhost only).")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--no-browser", action="store_true")
            a = p.parse_args(argv[1:])
            import threading
            import webbrowser

            import uvicorn

            from .web.app import create_app
            url = f"http://127.0.0.1:{a.port}"
            if not a.no_browser:
                threading.Timer(1.0, webbrowser.open, args=(url,)).start()
            print(f"podscribe web UI: {url}  (Ctrl+C to stop)")
            uvicorn.run(create_app(), host="127.0.0.1", port=a.port, log_level="warning")
            return 0
        p = argparse.ArgumentParser(prog="podscribe", description="Transcribe a Spotify podcast episode. Saves nothing "
                                    "unless you pass -o.", epilog="Web UI: podscribe web")
        p.add_argument("link", help="Spotify episode URL")
        p.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
        p.add_argument("--feed", help="RSS feed URL to search instead of auto-discovering the show's feed")
        p.add_argument("-o", "--output", help="write the Markdown transcript to this file (default: print to the terminal)")
        a = p.parse_args(argv)
        run(a.link, a.model, a.feed, a.output)
        return 0
    except PodscribeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
