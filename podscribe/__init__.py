class PodscribeError(Exception):
    """User-facing error: printed as one line, exit code 1."""


class Cancelled(BaseException):
    """Raised from progress callbacks to stop a job. Derives from BaseException (like KeyboardInterrupt) so the
    broad `except Exception` handlers around the transcription engines can't swallow it."""
