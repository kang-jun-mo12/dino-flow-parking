"""Rectangle cover of an occupancy mask, used by the numeric scene tokens."""
import numpy as np


def occupancy_rectangles(mask):
    """Exactly partition occupied cells by merging identical horizontal runs."""
    active, rectangles = {}, []
    for r, line in enumerate(np.asarray(mask, dtype=bool)):
        switches = np.diff(np.r_[False, line, False].astype(int))
        runs = set(zip(np.flatnonzero(switches == 1), np.flatnonzero(switches == -1)))
        for run in list(active):
            if run not in runs:
                rectangles.append((active.pop(run), r, *run))
        for run in runs:
            active.setdefault(run, r)
    rectangles.extend((lo, len(mask), *run) for run, lo in active.items())
    return sorted(rectangles)
