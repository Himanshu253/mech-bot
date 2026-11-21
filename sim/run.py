"""Run local matches: python3 sim/run.py [n_matches] [practice|ffa] [seed0]"""
import os, sys, time, math, importlib, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from sim import Sim, base_config, KIT
from housebot import HouseBot, Squad
from world import World


class Silent:  # suppress mf.draw usage cost
    pass


def make_ours(cfg, slot, mod="bot"):
    B = importlib.import_module(mod)
    hello = {"type": "hello", "slot": slot, "seed": 1000 + slot, "map": {"dir": os.path.join(KIT, "arena")}, "config": cfg}
    b = B.Bot(hello); b.times = []
    return b


def run_match(seed, mode="practice", mod="bot", world=None, verbose=False, ours_n=None):
    cfg = base_config()
    rng = random.Random(seed)
    diffs = ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4 + ["expert"] * 3
    our_slot = rng.randrange(16)
    W = world
    brains = [None] * 16
    house_ids = [i for i in range(16) if i != our_slot]
    if mode == "practice":
        squad = Squad(W, list(zip(house_ids, diffs)))
        for hid, d in zip(house_ids, diffs):
            brains[hid] = (HouseBot(hid, d, squad, seed * 31 + hid), f"house-{d}")
        brains[our_slot] = (make_ours(cfg, our_slot, mod), "ours")
        ours = [our_slot]
    else:  # ffa: some copies of ours vs ffa house bots
        n = ours_n or 4
        ours = rng.sample(range(16), n)
        squad = Squad(W, [(i, rng.choice(diffs)) for i in range(16) if i not in ours], ffa=True)
        for hid, d in squad.members:
            brains[hid] = (HouseBot(hid, d, squad, seed * 31 + hid), f"house-{d}")
        for o in ours:
            brains[o] = (make_ours(cfg, o, mod), "ours")
    sim = Sim(brains, seed=seed, cfg=cfg, world=W)
    dt = 1.0 / cfg["match"]["tick_hz"]
    while sim.t < cfg["match"]["duration_s"] and sim.alive_count > 1 and any(sim.mechs[o].alive for o in ours):
        sim.step()
        for m in sim.mechs:
            if not m.alive and m.id not in ours:
                squad.dead.add(m.id)
    # remaining: if ours died, house ranks among themselves irrelevant
    res = []
    for o in ours:
        m = sim.mechs[o]
        if m.alive:
            alive = sorted([x for x in sim.mechs if x.alive], key=lambda x: -x.hp)
            place = 1 + alive.index(m)
        else:
            place = m.place
        tm = np.array(m.brain.times) if m.brain.times else np.zeros(1)
        res.append({"seed": seed, "slot": o, "place": place, "score": 16 - place, "kills": m.kills, "death_t": m.death_t,
                    "t_end": sim.t, "cubes": m.cubes, "weapons": [w["kind"] for w in m.weapons if w], "dmg": round(m.dmg_done),
                    "ms_p99": float(np.percentile(tm, 99)), "ms_max": float(tm.max())})
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mode = sys.argv[2] if len(sys.argv) > 2 else "practice"
    s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mod = sys.argv[4] if len(sys.argv) > 4 else "bot"
    W = World(os.path.join(KIT, "arena"), base_config()["mech"])
    allr = []
    for s in range(s0, s0 + n):
        t0 = time.time()
        r = run_match(s, mode, mod, W)
        allr += r
        for x in r:
            print(x, f"wall {time.time()-t0:.0f}s", flush=True)
    print("MEAN SCORE", np.mean([x["score"] for x in allr]), "kills", np.mean([x["kills"] for x in allr]))


def trace(seed, mode="practice", every=1.0, until=480):
    """Print our bot's state once per `every` seconds."""
    import mechfight as mf
    W = World(os.path.join(KIT, "arena"), base_config()["mech"])
    cfg = base_config()
    import sim as S
    orig = S.Sim.step
    state = {"next": 0}
    def step(self):
        orig(self)
        o = [m for m in self.mechs if m.name == "ours"][0]
        if self.t >= state["next"] and o.alive:
            state["next"] += every
            b = o.brain
            near = sorted([(round(math.hypot(m.pos[0]-o.pos[0], m.pos[1]-o.pos[1])), m.id, round(m.hp), m.name[6:]) for m in self.mechs if m.alive and m is not o])[:3]
            lab = (mf.draw._d.get("label") if hasattr(mf.draw, "_d") else "")
            print(f"t{self.t:5.1f} hp{o.hp:6.0f} e{o.energy:3.0f} pos({o.pos[0]:6.0f},{o.pos[1]:6.0f},{o.pos[2]:4.0f}) "
                  f"cub{o.cubes} k{o.kills} w{[w['kind'] for w in o.weapons if w]} mode={getattr(b,'mode','?')} "
                  f"goal={tuple(round(v) for v in b.goal) if b.goal else None} alive{self.alive_count} near{near}")
    S.Sim.step = step
    try:
        return run_match(seed, mode, "bot", W)
    finally:
        S.Sim.step = orig
