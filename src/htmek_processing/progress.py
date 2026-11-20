"""Progress bars for the long steps (tqdm: a widget in a notebook, a text bar in a terminal or log)."""
from __future__ import annotations


def bar(total, desc, *, unit="it", min_total=1):
    """A progress bar for `total` items, labelled `desc`; a no-op object when tqdm is missing or `total` < `min_total`.
    Call .update(n) and .close() (or use it as a context manager)."""
    class _Null:
        def update(self, n=1): pass
        def set_postfix_str(self, s): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    if total < min_total:
        return _Null()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _Null()
    return tqdm(total=total, desc=desc, unit=unit, leave=True, dynamic_ncols=True, mininterval=0.5)


def track(iterable, desc, *, total=None, unit="it", min_total=1):
    """Iterate `iterable` with a progress bar (see bar)."""
    total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else 0)
    with bar(total, desc, unit=unit, min_total=min_total) as b:
        for x in iterable:
            yield x
            b.update(1)
