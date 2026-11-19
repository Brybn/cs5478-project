"""Pre-close bob incidence (height peak >= 5 mm before the first close command) in demos and saved test rollouts.

Usage: bob_stats.py [test.json] [output.json]. Needs full per-step states/actions, so run it on a freshly
generated test.json (docs/workflow.md); the saved summary is results/bob_stats.json.
"""
import json
import sys

import h5py
import numpy as np
from scipy.signal import find_peaks


def bob(states, actions):
    closes = np.flatnonzero(np.asarray(actions)[:, 3] < -.5)
    stop = int(closes[0]) if len(closes) else len(actions)
    return len(find_peaks(np.asarray(states)[:stop + 1, 21], prominence=.005)[0])


out = {}
demo = [bob(f[k]["obs"][:], f[k]["actions"][:]) for n in ("train", "validation") for f in [h5py.File(f"data/split/{n}.h5")] for k in f]
print("official demos", len(demo), "bob>=1:", np.mean(np.array(demo) >= 1))
out["official_demos"] = {"episodes": len(demo), "bob_rate": float(np.mean(np.array(demo) >= 1))}
test = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/monitor_level/test.json"))
for key in ("variant", "force_n"):
    groups = {}
    for e in test["episodes"]:
        groups.setdefault(e[key], []).append(bob(e["states"], e["actions"]) >= 1)
    print(key, {k: round(float(np.mean(v)), 3) for k, v in sorted(groups.items())})
    out[f"test_bob_rate_by_{key}"] = {str(k): {"episodes": len(v), "bob_rate": float(np.mean(v))} for k, v in sorted(groups.items())}
if len(sys.argv) > 2:
    json.dump({"source": sys.argv[1], "definition": "height peak >= 5 mm (prominence) before the first close command", **out},
              open(sys.argv[2], "w"), indent=1)
