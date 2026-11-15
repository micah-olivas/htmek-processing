"""Worker processes that draw or fit off screen."""
import contextlib
import os


@contextlib.contextmanager
def headless_workers():
    """Within it, worker processes started (and inheriting the environment) use matplotlib's Agg backend: on macOS the
    default backend gives every worker its own Dock icon, though they only draw to files."""
    old = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = old
