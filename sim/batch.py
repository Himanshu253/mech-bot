"""Parallel batch: python3 sim/batch.py N mode mod [workers]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from multiprocessing import Pool
import numpy as np

def job(args):
    seed, mode, mod = args
    import run
    from world import World
    from sim import base_config, KIT
    global _W
    try:
        _W
    except NameError:
        _W = World(os.path.join(KIT, "arena"), base_config()["mech"])
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        return run.run_match(seed, mode, mod, _W)

if __name__ == "__main__":
    n = int(sys.argv[1]); mode = sys.argv[2]; mod = sys.argv[3]; w = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    s0 = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    t0 = time.time()
    with Pool(w) as p:
        res = sum(p.map(job, [(s, mode, mod) for s in range(s0, s0 + n)]), [])
    sc = [r["score"] for r in res]
    print(f"{mod} {mode} n={len(res)} MEAN {np.mean(sc):.2f} ± {np.std(sc)/np.sqrt(len(sc)):.2f}  kills {np.mean([r['kills'] for r in res]):.2f} "
          f"death_t {np.mean([r['death_t'] or r['t_end'] for r in res]):.0f}  p99ms {max(r['ms_p99'] for r in res):.1f} max {max(r['ms_max'] for r in res):.1f}  [{time.time()-t0:.0f}s]")
    print(sorted(sc))
