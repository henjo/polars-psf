import polars_psf as pp


class Recorder:
    """Proxy around the native ``ResultDir`` of a Dataset, recording what each scan asks the
    reader for (``calls``) and the batches it yields (``steps``, one list per parallel step)."""

    def __init__(self, inner):
        self.inner, self.calls, self.steps = inner, [], []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def scan(self, result, leaves, names=None, batch=None):
        self.calls.append(("wide", list(leaves), names))
        return self._record(self.inner.scan(result, leaves, names, batch))

    def scan_long(self, result, jobs, field=None, chunk_rows=1 << 22, batch=None):
        self.calls.append(("long", list(jobs)))
        return self._record(self.inner.scan_long(result, jobs, field, chunk_rows, batch))

    def _record(self, it):
        for step in it:
            self.steps.append(step)
            yield step

    def clear(self):
        self.calls.clear()
        self.steps.clear()


def recorded(path):
    """``(Dataset, Recorder)`` for ``path``."""
    d = pp.open(path)
    rec = d._r = Recorder(d._r)
    return d, rec
