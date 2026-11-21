"""Approximate local simulator of mechfight/1 (protocol.md) for testing bots in-process.

Not the real server: physics, vision and AI are simplified. Good enough to catch bugs, compare
strategies, and check that the bot doesn't get stuck or wander out of the zone.
"""
import copy
import gzip
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from world import World  # noqa: E402

KIT = os.environ.get("MECHKIT", os.path.join(os.path.dirname(os.path.dirname(HERE)), "mechfight-kit"))
PI = math.pi


def angdiff(a, b):
    return (a - b + PI) % (2 * PI) - PI


def base_config():
    with gzip.open(os.path.join(KIT, "check/stream.jsonl.gz"), "rt") as f:
        return json.loads(f.readline())["config"]


class Mech:
    def __init__(self, mid, pos, cfg, brain, name):
        self.id = mid; self.brain = brain; self.name = name
        self.cfg = cfg
        self.stats = dict(cfg["mech"])
        self.wstats = copy.deepcopy(cfg["weapons"])
        self.pos = list(pos); self.vel = [0.0, 0.0, 0.0]
        self.aim = [random.uniform(-PI, PI), 0.0]
        self.cmd = {"move": [0, 0], "aim": list(self.aim), "boost": False, "fire": [False, False]}
        self.hp = self.max_hp = float(self.stats["hp"]); self.energy = float(self.stats["energy"])
        self.grounded = True; self.alive = True
        self.cubes = 0; self.upgrades = []
        self.weapons = [self.new_weapon("gun"), None]
        self.last_combat = -99; self.last_spend = -99; self.last_step = 0
        self.events = []; self.offer_q = []; self.offer = None; self.offer_id = 0
        self.seen_mem = {}  # id -> last seen t
        self.kills = 0; self.death_t = None; self.place = None
        self.coast = 0.0; self.short_until = -1
        self.thresholds_hit = 0
        self.dmg_done = 0.0

    def new_weapon(self, kind, ammo=None):
        w = {"kind": kind, "cooldown": 0.0}
        if kind == "gun":
            w["ammo"] = self.wstats["gun"]["mag"] if ammo is None else ammo; w["reload"] = 0.0
        if kind == "laser":
            w["charge"] = 0.0
        return w

    @property
    def scale(self):
        return 1 + self.cfg["cubes"]["size_per_cube"] * self.cubes

    def radius(self):
        return self.stats["radius"] * self.scale

    def height(self):
        return self.stats["height"] * self.scale

    def dmg_mult(self):
        c = self.cfg["cubes"]
        return 1 + c["damage_per_cube"] * min(self.cubes, c["stat_cap"])


class Sim:
    def __init__(self, brains, seed=0, cfg=None, squad_mode=True, verbose=False, mapdir=None, world=None):
        self.cfg = cfg or base_config()
        self.rng = random.Random(seed)
        random.seed(seed)
        self.W = world or World(mapdir or os.path.join(KIT, "arena"), self.cfg["mech"])
        W = self.W
        # solid height for physics (roofs walkable): H; terrain: G
        self.H = W.H
        self.t = 0.0; self.tick = 0
        self.verbose = verbose
        self.squad_mode = squad_mode
        zc = self.cfg["zone"]
        self.zone_c = list(zc["start_center"]); self.zone_r = zc["start_radius"]
        self.phase = 0
        self.phase_start = 0.0
        self._plan_zone()
        self.mechs = []
        self.npcs = []; self.crates = []; self.cubes = []; self.pickups = []; self.projs = []
        self.next_id = {"npc": 100, "crate": 1000, "cube": 5000, "pick": 8000}
        spawns = self._spawns(len(brains))
        for i, (brain, name) in enumerate(brains):
            self.mechs.append(Mech(i, spawns[i], self.cfg, brain, name))
        for _ in range(self.cfg["npc"]["count"]):
            p = self._rand_open(inzone=True)
            self.npcs.append({"id": self._nid("npc"), "pos": p, "home": list(p), "hp": self.cfg["npc"]["hp"],
                              "cool": 0.0, "target": None, "vel": [0, 0, 0], "aim": [0, 0]})
        for _ in range(self.cfg["crates"]["count"]):
            self.crates.append({"id": self._nid("crate"), "pos": self._rand_open(inzone=True), "hp": self.cfg["crates"]["hp"]})
        kinds = ["mortar", "laser", "lightning"]
        for _ in range(self.cfg["loot"]["floor_count"]):
            self.pickups.append({"id": self._nid("pick"), "kind": self.rng.choice(kinds), "pos": self._rand_open(inzone=True)})
        self.sounds = []  # (t, pos, kind, src)
        self.beams = []
        self.alive_count = len(self.mechs)
        self.deaths = []

    # ---------------------------------------------------------- setup
    def _nid(self, k):
        v = self.next_id[k]; self.next_id[k] += 1; return v

    def _rand_open(self, inzone=False):
        W = self.W
        for _ in range(2000):
            x = self.rng.uniform(W.bounds["minX"] + 40, W.bounds["maxX"] - 40)
            y = self.rng.uniform(W.bounds["minY"] + 40, W.bounds["maxY"] - 40)
            if inzone and math.hypot(x - self.zone_c[0], y - self.zone_c[1]) > self.zone_r * 0.85:
                continue
            if W.clearance(x, y) > 4 and not W.bldg[W.ij(x, y)]:
                return [x, y, W.ground(x, y)]
        return [0, 0, W.ground(0, 0)]

    def _spawns(self, n):
        out = []
        sep = self.cfg["match"]["spawn_min_separation"]
        for k in range(n):
            for tries in range(500):
                p = self._rand_open(inzone=True)
                if math.hypot(p[0] - self.zone_c[0], p[1] - self.zone_c[1]) > self.zone_r * self.cfg["match"]["spawn_zone_fraction"]:
                    continue
                if all(math.hypot(p[0] - q[0], p[1] - q[1]) > sep * (1 - tries / 600) for q in out):
                    out.append(p); break
            else:
                out.append(self._rand_open(True))
        return out

    def _plan_zone(self):
        phases = self.cfg["zone"]["phases"]
        self.zplan = []
        c = list(self.cfg["zone"]["start_center"]); r = self.cfg["zone"]["start_radius"]
        t = 0.0
        for ph in phases:
            nr = ph["radius"]
            # next circle entirely inside current
            for _ in range(100):
                a = self.rng.uniform(0, 2 * PI); d = self.rng.uniform(0, max(r - nr, 0))
                nc = [c[0] + math.cos(a) * d, c[1] + math.sin(a) * d]
                if self.W.clearance(nc[0], nc[1]) > 3 or nr > 30:
                    break
            self.zplan.append({"c0": list(c), "r0": r, "c1": nc, "r1": nr, "s0": t + ph["hold_s"],
                               "s1": t + ph["hold_s"] + ph["shrink_s"], "dps": ph["dps"], "t0": t})
            t += ph["hold_s"] + ph["shrink_s"]
            c, r = nc, nr

    def zone_now(self):
        t = self.t
        for z in self.zplan:
            if t < z["s1"]:
                if t <= z["s0"]:
                    return z["c0"], z["r0"], z
                f = (t - z["s0"]) / (z["s1"] - z["s0"])
                return [z["c0"][0] + (z["c1"][0] - z["c0"][0]) * f, z["c0"][1] + (z["c1"][1] - z["c0"][1]) * f], z["r0"] + (z["r1"] - z["r0"]) * f, z
        z = self.zplan[-1]
        return z["c1"], 0.0, z

    # ---------------------------------------------------------- geometry
    def solid(self, x, y):
        j, i = self.W.ij(x, y)
        return float(self.H[j, i])

    def floor_under(self, x, y, r, z):
        """highest surface under footprint that is <= z + step (what we stand on)"""
        best = -1e9
        for dx, dy in ((0, 0), (r * 0.7, 0), (-r * 0.7, 0), (0, r * 0.7), (0, -r * 0.7)):
            h = self.solid(x + dx, y + dy)
            if h <= z + self.cfg["mech"]["step_up"] + 0.01:
                best = max(best, h)
        return best

    def blocked(self, x, y, r, z):
        for a in range(8):
            dx = math.cos(a * PI / 4) * r; dy = math.sin(a * PI / 4) * r
            if self.solid(x + dx, y + dy) > z + self.cfg["mech"]["step_up"]:
                return True
        b = self.W.bounds
        return not (b["minX"] + 3 < x < b["maxX"] - 3 and b["minY"] + 3 < y < b["maxY"] - 3)

    def los(self, a, b):
        return self.W.los(a, b)

    # ---------------------------------------------------------- observation
    def obs_for(self, m):
        cfg = self.cfg
        vis = cfg["vision"]
        eye = [m.pos[0], m.pos[1], m.pos[2] + m.stats["eye_height"] * m.scale]
        near = vis["near_range"] + m.stats.get("near_range_bonus", 0)
        rng_ = vis["range"] + m.stats.get("vision_bonus", 0)
        half = math.radians(vis["fov_deg"]) / 2

        def sees(p, h):
            dx = p[0] - eye[0]; dy = p[1] - eye[1]; dist = math.hypot(dx, dy)
            if dist > rng_:
                return False
            if dist > near and abs(angdiff(math.atan2(dy, dx), m.aim[0])) > half:
                return False
            return self.los(eye, [p[0], p[1], p[2] + h * 0.5]) or self.los(eye, [p[0], p[1], p[2] + h])

        mechs = []
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            if sees(o.pos, o.height()):
                m.seen_mem[o.id] = self.t
            if self.t - m.seen_mem.get(o.id, -99) <= vis["memory_s"]:
                mechs.append({"id": o.id, "kind": "mech", "pos": [round(v, 2) for v in o.pos], "vel": list(o.vel),
                              "aim": list(o.aim), "hp": round(o.hp, 2), "max_hp": o.max_hp, "cubes": o.cubes,
                              "weapons": [w["kind"] for w in o.weapons if w],
                              "charging": any(w and w["kind"] == "laser" and w["charge"] > 0 for w in o.weapons)})
        for n in self.npcs:
            if sees(n["pos"], 3.5):
                m.seen_mem[n["id"]] = self.t
            if self.t - m.seen_mem.get(n["id"], -99) <= vis["memory_s"]:
                mechs.append({"id": n["id"], "kind": "npc", "pos": list(n["pos"]), "vel": list(n["vel"]), "aim": n["aim"],
                              "hp": n["hp"], "max_hp": cfg["npc"]["hp"]})
        projs = [{"id": p["id"], "kind": p["kind"], "pos": list(p["pos"]), "vel": list(p["vel"])}
                 for p in self.projs if p["owner"] != m.id and sees(p["pos"], 0.5)]
        crates = [{"id": c["id"], "pos": c["pos"], "hp": c["hp"]} for c in self.crates if sees(c["pos"], 2.4)]
        cubes = [{"id": c["id"], "pos": c["pos"]} for c in self.cubes if sees(c["pos"], 0.5)]
        picks = [{"id": p["id"], "kind": p["kind"], "pos": p["pos"]} for p in self.pickups if sees(p["pos"], 0.5)]
        sounds = []
        for (st, sp, kind, src) in self.sounds:
            if src == m.id or self.t - st > 0.1 + 1e-6:
                continue
            dx = sp[0] - m.pos[0]; dy = sp[1] - m.pos[1]; dist = math.hypot(dx, dy)
            if dist <= cfg["sound"]["range"].get(kind, 100):
                b = math.atan2(dy, dx) + self.rng.gauss(0, math.radians(cfg["sound"]["bearing_noise_deg"]) * m.stats.get("hearing", 1))
                band = "near" if dist < cfg["sound"]["near"] else ("mid" if dist < cfg["sound"]["mid"] else "far")
                sounds.append({"kind": kind, "bearing": b, "dist": band, "age": round(self.t - st, 2)})
        c, r, z = self.zone_now()
        nz = z
        zone = {"center": z["c0"], "radius": z["r0"], "next_center": z["c1"], "next_radius": z["r1"],
                "shrink_start": z["s0"], "shrink_end": z["s1"], "dps": z["dps"]}
        ws = []
        for w in m.weapons:
            if w is None:
                ws.append(None); continue
            d = dict(w); d["stats"] = dict(m.wstats[w["kind"]]); ws.append(d)
        st = dict(m.stats); st["eye_height"] = m.stats["eye_height"] * m.scale; st["muzzle_height"] = m.stats["muzzle_height"] * m.scale
        st["height"] = m.height(); st["radius"] = m.radius()
        if m.offer is None and m.offer_q:
            m.offer = m.offer_q.pop(0)
        obs = {"type": "obs", "tick": self.tick, "t": round(self.t, 2), "alive": self.alive_count,
               "self": {"id": m.id, "pos": list(m.pos), "vel": list(m.vel), "aim": list(m.aim), "hp": m.hp, "max_hp": m.max_hp,
                        "energy": m.energy, "max_energy": m.stats["energy"], "grounded": m.grounded, "cubes": m.cubes,
                        "upgrades": list(m.upgrades), "weapons": ws, "in_zone": math.hypot(m.pos[0] - c[0], m.pos[1] - c[1]) <= r,
                        "stats": st},
               "mechs": mechs, "projectiles": projs, "crates": crates, "cubes": cubes, "pickups": picks,
               "sounds": sounds, "beams": [], "events": m.events, "drops": [], "zone": zone,
               "offer": m.offer}
        m.events = []
        return obs

    # ---------------------------------------------------------- actions
    def apply(self, m, a):
        if a is None:
            m.coast += 0.1
            if m.coast > self.cfg["bot"]["coast_s"]:
                m.cmd["move"] = [0, 0]; m.cmd["boost"] = False
            m.cmd["fire"] = [False, False]
            return
        m.coast = 0.0
        mv = a.get("move", [0, 0]); n = math.hypot(*mv)
        if n > 1:
            mv = [mv[0] / n, mv[1] / n]
        m.cmd["move"] = mv
        m.cmd["boost"] = bool(a.get("boost", False))
        if "aim" in a:
            m.cmd["aim"] = list(a["aim"])
        m.cmd["fire"] = list(a.get("fire", [False, False]))
        m.cmd["jump"] = bool(a.get("jump", False))
        if a.get("reload"):
            g = m.weapons[0] if m.weapons[0] and m.weapons[0]["kind"] == "gun" else (m.weapons[1] if m.weapons[1] and m.weapons[1]["kind"] == "gun" else None)
            if g and g["reload"] == 0 and g["ammo"] < m.wstats["gun"]["mag"]:
                g["reload"] = m.wstats["gun"]["reload"]
        if "pick" in a and m.offer is not None:
            k = int(a["pick"])
            if 0 <= k < len(m.offer["choices"]):
                self._apply_upgrade(m, m.offer["choices"][k]["id"])
            m.offer = None
        if "take" in a:
            self._take(m, a["take"], a.get("drop"))
        elif "drop" in a:
            s = int(a["drop"])
            if m.weapons[s]:
                self._drop(m, s)

    def _drop(self, m, s):
        w = m.weapons[s]
        pid = self._nid("pick")
        self.pickups.append({"id": pid, "kind": w["kind"], "pos": list(m.pos), "ammo": w.get("ammo")})
        m.weapons[s] = None
        m.events.append({"type": "dropped", "id": pid, "kind": w["kind"], "slot": s})

    def _take(self, m, pid, drop):
        p = next((p for p in self.pickups if p["id"] == pid), None)
        if p is None:
            m.events.append({"type": "take_failed", "id": pid, "reason": "gone"}); return
        if math.hypot(p["pos"][0] - m.pos[0], p["pos"][1] - m.pos[1]) > self.cfg["loot"]["pickup_radius"] or abs(p["pos"][2] - m.pos[2]) > 4:
            m.events.append({"type": "take_failed", "id": pid, "reason": "range"}); return
        if any(w and w["kind"] == p["kind"] for w in m.weapons):
            m.events.append({"type": "take_failed", "id": pid, "reason": "have"}); return
        slot = None
        if m.weapons[0] is None:
            slot = 0
        elif m.weapons[1] is None:
            slot = 1
        elif drop is not None:
            slot = int(drop)
            self._drop(m, slot)
        if slot is None:
            m.events.append({"type": "take_failed", "id": pid, "reason": "full"}); return
        self.pickups.remove(p)
        w = m.new_weapon(p["kind"], p.get("ammo"))
        w["cooldown"] = max(self.cfg["loot"]["equip_s"], m.wstats[p["kind"]].get("cooldown", 0))
        m.weapons[slot] = w
        m.events.append({"type": "took", "id": pid, "kind": p["kind"], "slot": slot})

    def _apply_upgrade(self, m, uid):
        u = next(u for u in self.cfg["upgrades"] if u["id"] == uid)
        m.upgrades.append(uid)
        for e in u["effects"]:
            sec, key = e["path"].split(".", 1)
            if sec == "mech":
                d = m.stats
            else:
                d = m.wstats[key.split(".")[0]]; key = key.split(".")[1]
            if "mul" in e:
                d[key] = d.get(key, 0) * e["mul"]
            if "add" in e:
                d[key] = d.get(key, 0) + e["add"]
            if e["path"] == "mech.hp":
                old = m.max_hp; m.max_hp = self._maxhp(m); m.hp += m.max_hp - old
        m.events.append({"type": "upgrade", "id": uid})

    def _maxhp(self, m):
        c = self.cfg["cubes"]
        return m.stats["hp"] * (1 + c["hp_per_cube"] * min(m.cubes, c["stat_cap"]))

    def _gain_cubes(self, m, n):
        for _ in range(n):
            m.cubes += 1
            old = m.max_hp; m.max_hp = self._maxhp(m); m.hp += m.max_hp - old
            if m.cubes in self.cfg["cubes"]["upgrade_thresholds"]:
                ks = [w["kind"] for w in m.weapons if w]
                avail = [u for u in self.cfg["upgrades"] if m.upgrades.count(u["id"]) < u["max"] and (not u.get("requires") or u["requires"] in ks)]
                ch = self.rng.sample(avail, min(3, len(avail)))
                m.offer_id += 1
                m.offer_q.append({"id": m.offer_id, "expires": self.t + 5, "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in ch]})

    # ---------------------------------------------------------- damage
    def damage(self, src, tgt, amount, weapon, origin):
        """tgt: Mech, npc dict or crate dict"""
        if isinstance(tgt, Mech):
            if not tgt.alive:
                return
            rear = False
            if origin is not None:
                b = math.atan2(origin[1] - tgt.pos[1], origin[0] - tgt.pos[0])
                rear = abs(angdiff(b, tgt.aim[0])) > math.radians(self.cfg["mech"]["front_arc_deg"]) / 2
            if weapon != "zone":
                if rear:
                    amount *= self.cfg["mech"]["rear_mult"]
                amount *= (1 - tgt.stats.get("armor", 0))
            tgt.hp -= amount
            tgt.last_combat = self.t
            ev = {"type": "damaged", "amount": round(amount, 2), "weapon": weapon,
                  "bearing": None if origin is None else math.atan2(origin[1] - tgt.pos[1], origin[0] - tgt.pos[0]),
                  "by": None if src is None else src.id}
            if rear:
                ev["rear"] = True
            tgt.events.append(ev)
            if isinstance(src, Mech):
                src.last_combat = self.t; src.dmg_done += amount
                src.events.append({"type": "hit", "target": tgt.id, "amount": round(amount, 2), "weapon": weapon})
            if tgt.hp <= 0:
                self.kill(tgt, src)
        elif "home" in tgt:  # npc
            if tgt["hp"] <= 0:
                return
            tgt["hp"] -= amount
            if isinstance(src, Mech):
                tgt["target"] = src.id
                src.events.append({"type": "hit", "target": tgt["id"], "amount": amount, "weapon": weapon})
            if tgt["hp"] <= 0:
                self.npcs.remove(tgt)
                for _ in range(self.cfg["npc"]["cubes"]):
                    self.cubes.append({"id": self._nid("cube"), "pos": [tgt["pos"][0] + self.rng.uniform(-2, 2), tgt["pos"][1] + self.rng.uniform(-2, 2), tgt["pos"][2]]})
        else:  # crate
            if tgt["hp"] <= 0:
                return
            tgt["hp"] -= amount
            if tgt["hp"] <= 0:
                self.crates.remove(tgt)
                self.cubes.append({"id": self._nid("cube"), "pos": list(tgt["pos"])})
                if self.rng.random() < self.cfg["loot"]["crate_weapon_chance"]:
                    self.pickups.append({"id": self._nid("pick"), "kind": self.rng.choice(["mortar", "laser", "lightning"]),
                                         "pos": [tgt["pos"][0] + 1, tgt["pos"][1], tgt["pos"][2]]})
                self.sounds.append((self.t, tgt["pos"], "crate", None))

    def kill(self, m, src):
        m.alive = False; m.death_t = self.t
        self.deaths.append((self.t, m.id))
        self.sounds.append((self.t, m.pos, "death", m.id))
        for i, w in enumerate(m.weapons):
            if w and w["kind"] != "gun":
                self.pickups.append({"id": self._nid("pick"), "kind": w["kind"], "pos": [m.pos[0] + i * 2, m.pos[1], m.pos[2]]})
        for k in range(m.cubes + self.cfg["cubes"]["death_drop_extra"]):
            a = k * 2.4
            self.cubes.append({"id": self._nid("cube"), "pos": [m.pos[0] + math.cos(a) * 1.5, m.pos[1] + math.sin(a) * 1.5, m.pos[2]]})
        if isinstance(src, Mech):
            src.kills += 1
            src.events.append({"type": "kill", "target": m.id})

    # ---------------------------------------------------------- weapons
    def ray(self, m, origin, d, rng_, pierce=0):
        """hitscan: returns list of hit targets (mechs/npcs/crates) before a wall"""
        W = self.W
        step = 1.0
        n = int(rng_ / step)
        hit = []
        # vectorized wall test
        ts = np.arange(1, n + 1) * step
        X = origin[0] + d[0] * ts; Y = origin[1] + d[1] * ts; Z = origin[2] + d[2] * ts
        I = np.clip(np.rint(X - W.x0).astype(int), 0, W.nx - 1); J = np.clip(np.rint(Y - W.y0).astype(int), 0, W.ny - 1)
        blk = np.nonzero(Z < W.H[J, I])[0]
        wall = ts[blk[0]] if len(blk) else rng_
        cands = []
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            s = self._ray_cyl(origin, d, o.pos, o.radius(), o.height())
            if s is not None and s < wall:
                cands.append((s, o))
        for npc in self.npcs:
            s = self._ray_cyl(origin, d, npc["pos"], 1.8, 3.5)
            if s is not None and s < wall:
                cands.append((s, npc))
        for c in self.crates:
            s = self._ray_cyl(origin, d, c["pos"], 1.4, 2.4)
            if s is not None and s < wall:
                cands.append((s, c))
        cands.sort(key=lambda x: x[0])
        return [c for c in cands[:1 + pierce]], wall

    @staticmethod
    def _ray_cyl(o, d, p, r, h):
        # 2D closest approach
        dx, dy = d[0], d[1]
        rx, ry = p[0] - o[0], p[1] - o[1]
        dd = dx * dx + dy * dy
        if dd < 1e-9:
            return None
        s = (rx * dx + ry * dy) / dd
        if s < 0:
            return None
        cx = o[0] + dx * s - p[0]; cy = o[1] + dy * s - p[1]
        q = cx * cx + cy * cy
        if q > r * r:
            return None
        back = math.sqrt((r * r - q) / dd)
        s1 = max(0.0, s - back)
        z = o[2] + d[2] * s1
        if p[2] <= z <= p[2] + h:
            return s1 * math.sqrt(dd + d[2] * d[2]) if False else s1
        return None

    def fire_weapons(self, m, dt):
        for s, w in enumerate(m.weapons):
            if w is None:
                continue
            ws = m.wstats[w["kind"]]
            w["cooldown"] = max(0.0, w["cooldown"] - dt)
            trig = bool(m.cmd["fire"][s]) if s < len(m.cmd["fire"]) else False
            d = [math.cos(m.aim[1]) * math.cos(m.aim[0]), math.cos(m.aim[1]) * math.sin(m.aim[0]), math.sin(m.aim[1])]
            mz = m.pos[2] + m.stats["muzzle_height"] * m.scale
            off = m.radius() + 0.5
            org = [m.pos[0] + d[0] * off, m.pos[1] + d[1] * off, mz + d[2] * off]
            k = w["kind"]
            if k == "gun":
                if w["reload"] > 0:
                    w["reload"] -= dt
                    if w["reload"] <= 0:
                        w["reload"] = 0.0; w["ammo"] = ws["mag"]
                    continue
                if trig and w["cooldown"] <= 0 and w["ammo"] > 0:
                    sp = math.radians(ws["spread_deg"])
                    yaw = m.aim[0] + self.rng.gauss(0, sp); pit = m.aim[1] + self.rng.gauss(0, sp)
                    v = [math.cos(pit) * math.cos(yaw) * ws["speed"], math.cos(pit) * math.sin(yaw) * ws["speed"], math.sin(pit) * ws["speed"]]
                    self.projs.append({"id": self._nid("cube") + 900000, "kind": "gun", "owner": m.id, "pos": list(org), "vel": v,
                                       "dmg": ws["damage"] * m.dmg_mult(), "left": ws["range"], "src": m})
                    w["ammo"] -= 1; w["cooldown"] = ws["cooldown"]
                    self.sounds.append((self.t, list(m.pos), "gun", m.id))
                    if w["ammo"] == 0:
                        w["reload"] = ws["reload"]
            elif k == "mortar":
                if trig and w["cooldown"] <= 0:
                    oz = m.pos[2] + m.height() + 0.5
                    v = [d[0] * ws["speed"], d[1] * ws["speed"], d[2] * ws["speed"]]
                    self.projs.append({"id": self._nid("cube") + 900000, "kind": "mortar", "owner": m.id, "pos": [m.pos[0] + d[0] * off, m.pos[1] + d[1] * off, oz],
                                       "vel": v, "src": m, "ttl": ws["ttl"]})
                    w["cooldown"] = ws["cooldown"]
                    self.sounds.append((self.t, list(m.pos), "mortar", m.id))
            elif k == "laser":
                if self.t < m.short_until:
                    w["charge"] = 0.0; continue
                if w["charge"] > 0 or (trig and w["cooldown"] <= 0):
                    full = ws["charge"]
                    if w["charge"] == 0:
                        self.sounds.append((self.t, list(m.pos), "laser_charge", m.id))
                    w["charge"] += dt
                    release = (not trig and w["charge"] >= ws["min_charge"]) or w["charge"] >= full + ws["hold_s"]
                    if release:
                        frac = ws["min_frac"] + (1 - ws["min_frac"]) * min(1.0, w["charge"] / full)
                        hits, wall = self.ray(m, org, d, ws["range"], int(ws.get("pierce", 0)))
                        for _, tg in hits:
                            dmg = ws["damage"] * frac * m.dmg_mult()
                            if isinstance(tg, Mech) and any(x and x["kind"] == "mortar" for x in tg.weapons):
                                dmg *= 1 + ws["cookoff"]
                            self.damage(m, tg, dmg, "laser", m.pos)
                        w["charge"] = 0.0; w["cooldown"] = ws["cooldown"]
                        self.sounds.append((self.t, list(m.pos), "laser", m.id))
            elif k == "lightning":
                if trig and w["cooldown"] <= 0 and m.energy >= ws["energy"]:
                    m.energy -= ws["energy"]; m.last_spend = self.t
                    w["cooldown"] = ws["cooldown"]
                    self.sounds.append((self.t, list(m.pos), "lightning", m.id))
                    eye = [m.pos[0], m.pos[1], mz]
                    best = None; bd = 1e9
                    for tg in [o for o in self.mechs if o is not m and o.alive] + self.npcs + self.crates:
                        p = tg.pos if isinstance(tg, Mech) else tg["pos"]
                        dist = math.hypot(p[0] - m.pos[0], p[1] - m.pos[1])
                        if dist > ws["range"] or dist >= bd:
                            continue
                        if abs(angdiff(math.atan2(p[1] - m.pos[1], p[0] - m.pos[0]), m.aim[0])) > math.radians(ws["cone_deg"]) / 2 * 2:
                            continue
                        if not self.los(eye, [p[0], p[1], p[2] + 2]):
                            continue
                        best = tg; bd = dist
                    chain = []
                    if best is not None:
                        dmg = ws["damage"] * m.dmg_mult()
                        cur = best; done = set()
                        for j in range(int(ws["chain"]) + 1):
                            done.add(id(cur))
                            self.damage(m, cur, dmg, "lightning", m.pos)
                            if isinstance(cur, Mech) and any(x and x["kind"] == "laser" for x in cur.weapons):
                                cur.short_until = self.t + ws["short_s"]
                            dmg *= ws["chain_falloff"]
                            cp = cur.pos if isinstance(cur, Mech) else cur["pos"]
                            nxt = None; nd = ws["chain_range"]
                            for tg in [o for o in self.mechs if o is not m and o.alive] + self.npcs:
                                if id(tg) in done:
                                    continue
                                p = tg.pos if isinstance(tg, Mech) else tg["pos"]
                                dd = math.hypot(p[0] - cp[0], p[1] - cp[1])
                                if dd < nd:
                                    nd = dd; nxt = tg
                            if nxt is None:
                                break
                            cur = nxt

    def step_projectiles(self, dt):
        W = self.W
        keep = []
        for p in self.projs:
            alive = True
            if p["kind"] == "mortar":
                g = self.cfg["weapons"]["mortar"]["gravity"]
                p["ttl"] -= dt
                sub = 4
                for _ in range(sub):
                    h = dt / sub
                    p["vel"][2] -= g * h
                    np_ = [p["pos"][k] + p["vel"][k] * h for k in range(3)]
                    hitm = None
                    for o in self.mechs:
                        if o.alive and o.id != p["owner"] and math.hypot(np_[0] - o.pos[0], np_[1] - o.pos[1]) < o.radius() and o.pos[2] <= np_[2] <= o.pos[2] + o.height():
                            hitm = o
                    if hitm is not None or np_[2] < self.solid(np_[0], np_[1]) or p["ttl"] <= 0:
                        self.explode(p, np_, hitm); alive = False; break
                    p["pos"] = np_
            else:
                sp = math.sqrt(sum(v * v for v in p["vel"]))
                dist = sp * dt
                n = max(1, int(dist / 1.5))
                h = dt / n
                for _ in range(n):
                    np_ = [p["pos"][k] + p["vel"][k] * h for k in range(3)]
                    p["left"] -= sp * h
                    if p["left"] <= 0 or np_[2] < self.solid(np_[0], np_[1]):
                        alive = False; break
                    hit = None
                    for o in self.mechs:
                        if o.alive and o.id != p["owner"] and math.hypot(np_[0] - o.pos[0], np_[1] - o.pos[1]) < o.radius() + 0.3 and o.pos[2] <= np_[2] <= o.pos[2] + o.height():
                            hit = o; break
                    if hit is None and p["kind"] == "gun":
                        for c in self.npcs:
                            if math.hypot(np_[0] - c["pos"][0], np_[1] - c["pos"][1]) < 2.1 and c["pos"][2] <= np_[2] <= c["pos"][2] + 3.5:
                                hit = c; break
                        if hit is None:
                            for c in self.crates:
                                if math.hypot(np_[0] - c["pos"][0], np_[1] - c["pos"][1]) < 1.7 and c["pos"][2] <= np_[2] <= c["pos"][2] + 2.4:
                                    hit = c; break
                    if hit is not None:
                        src = p["src"] if p["kind"] == "gun" else None
                        org = [np_[0] - p["vel"][0] * 0.1, np_[1] - p["vel"][1] * 0.1]
                        self.damage(src, hit, p["dmg"], "gun" if p["kind"] == "gun" else "npc", org)
                        alive = False; break
                    p["pos"] = np_
            if alive:
                keep.append(p)
        self.projs = keep

    def explode(self, p, at, direct):
        ws = self.cfg["weapons"]["mortar"]; src = p["src"]
        m = src.dmg_mult()
        self.sounds.append((self.t, at, "explosion", None))
        if direct is not None:
            self.damage(src, direct, ws["damage"] * m, "mortar", at)
        R = src.wstats["mortar"]["splash_radius"]
        for o in self.mechs:
            if not o.alive or o is direct:
                continue
            dd = math.hypot(o.pos[0] - at[0], o.pos[1] - at[1])
            if dd < R + o.radius() and abs(o.pos[2] + 4 - at[2]) < R + 5:
                f = 1 - (1 - ws["splash_edge"]) * min(1, dd / R)
                dmg = ws["splash"] * f * m * (ws.get("self_mult", 0.5) if o is src else 1)
                self.damage(src if o is not src else None, o, dmg, "mortar", at)
                a = math.atan2(o.pos[1] - at[1], o.pos[0] - at[0])
                o.vel[0] += math.cos(a) * ws["knock"] * f; o.vel[1] += math.sin(a) * ws["knock"] * f; o.vel[2] += ws["knock_up"] * f
                o.energy = max(0, o.energy - ws["drain"] * f)
        for n in list(self.npcs) + list(self.crates):
            if math.hypot(n["pos"][0] - at[0], n["pos"][1] - at[1]) < R:
                self.damage(src, n, ws["splash"] * m, "mortar", at)

    # ---------------------------------------------------------- physics
    def move_mech(self, m, dt):
        st = m.stats
        mv = m.cmd["move"]
        charging = any(w and w["kind"] == "laser" and w["charge"] > 0 for w in m.weapons)
        boost = m.cmd.get("boost") and m.energy > 0
        if m.grounded:
            sp = st["boost_speed"] if (boost and (mv[0] or mv[1])) else st["walk_speed"]
            if charging:
                sp *= st["charge_move_mult"]
            tv = [mv[0] * sp, mv[1] * sp]
            acc = st["ground_accel"]
            if boost and (mv[0] or mv[1]):
                m.energy -= st["boost_drain"] * dt; m.last_spend = self.t
        else:
            tv = [mv[0] * st["walk_speed"], mv[1] * st["walk_speed"]]
            acc = st["air_accel"]
        for k in range(2):
            dv = tv[k] - m.vel[k]
            lim = acc * dt
            m.vel[k] += max(-lim, min(lim, dv))
        if m.cmd.get("jump") and m.grounded:
            m.vel[2] = st["jump_speed"]; m.grounded = False
        if not m.grounded:
            if boost:
                m.vel[2] = min(st["max_rise_speed"], m.vel[2] + st["jet_accel"] * dt)
                m.energy -= st["jet_drain"] * dt; m.last_spend = self.t
                self.sounds.append((self.t, list(m.pos), "jet", m.id)) if self.tick % 10 == 0 else None
            m.vel[2] -= st["gravity"] * dt
        r = m.radius()
        nx = m.pos[0] + m.vel[0] * dt; ny = m.pos[1] + m.vel[1] * dt
        z = m.pos[2]
        if not self.blocked(nx, ny, r, z):
            m.pos[0], m.pos[1] = nx, ny
        elif not self.blocked(nx, m.pos[1], r, z):
            m.pos[0] = nx; m.vel[1] *= 0.5
        elif not self.blocked(m.pos[0], ny, r, z):
            m.pos[1] = ny; m.vel[0] *= 0.5
        else:
            m.vel[0] = m.vel[1] = 0
        fl = self.floor_under(m.pos[0], m.pos[1], r, z)
        nz = z + m.vel[2] * dt
        if nz <= fl:
            nz = fl; m.vel[2] = 0; m.grounded = True
        else:
            if m.grounded and nz - fl < 1.0 and m.vel[2] <= 0:
                nz = fl
            else:
                m.grounded = False
        m.pos[2] = nz
        m.energy = min(st["energy"], m.energy + (st["energy_regen"] * dt if self.t - m.last_spend > st["energy_regen_delay"] else 0))
        sp2 = math.hypot(m.vel[0], m.vel[1])
        if m.grounded and sp2 > self.cfg["sound"]["step_speed"] and self.t - m.last_step > self.cfg["sound"]["step_every"]:
            m.last_step = self.t
            self.sounds.append((self.t, list(m.pos), "boost" if boost else "step", m.id))
        # aim slew
        tr = st["turn_rate"] * dt; pr = st["pitch_rate"] * dt
        dy = angdiff(m.cmd["aim"][0], m.aim[0])
        m.aim[0] = (m.aim[0] + max(-tr, min(tr, dy)) + PI) % (2 * PI) - PI
        tp = max(st["min_pitch"], min(st["max_pitch"], m.cmd["aim"][1]))
        m.aim[1] += max(-pr, min(pr, tp - m.aim[1]))

    def npc_step(self, dt):
        cfg = self.cfg["npc"]
        for n in self.npcs:
            n["cool"] = max(0, n["cool"] - dt)
            tgt = None
            if n["target"] is not None:
                tgt = next((o for o in self.mechs if o.id == n["target"] and o.alive), None)
                if tgt and (math.hypot(tgt.pos[0] - n["pos"][0], tgt.pos[1] - n["pos"][1]) > cfg["deaggro_range"]
                            or math.hypot(n["pos"][0] - n["home"][0], n["pos"][1] - n["home"][1]) > cfg["chase_leash"]):
                    tgt = None; n["target"] = None
            if tgt is None:
                for o in self.mechs:
                    if o.alive and math.hypot(o.pos[0] - n["pos"][0], o.pos[1] - n["pos"][1]) < cfg["aggro_range"]:
                        tgt = o; break
            if tgt is not None:
                eye = [n["pos"][0], n["pos"][1], n["pos"][2] + 3]
                c = [tgt.pos[0], tgt.pos[1], tgt.pos[2] + 4.5]
                if n["cool"] <= 0 and self.los(eye, c):
                    d = [c[k] - eye[k] for k in range(3)]; L = math.sqrt(sum(v * v for v in d))
                    yaw = math.atan2(d[1], d[0]) + self.rng.gauss(0, math.radians(cfg["spread_deg"]))
                    pit = math.asin(d[2] / L)
                    v = [math.cos(pit) * math.cos(yaw) * cfg["projectile_speed"], math.cos(pit) * math.sin(yaw) * cfg["projectile_speed"], math.sin(pit) * cfg["projectile_speed"]]
                    self.projs.append({"id": self._nid("cube") + 900000, "kind": "npc", "owner": n["id"], "pos": eye, "vel": v,
                                       "dmg": cfg["damage"], "left": 200, "src": None})
                    n["cool"] = cfg["cooldown"]
                    self.sounds.append((self.t, n["pos"], "npc_gun", None))

    # ---------------------------------------------------------- loop
    def step(self):
        dt = 1.0 / self.cfg["match"]["tick_hz"]
        if self.tick % self.cfg["match"]["decide_every"] == 0:
            for m in self.mechs:
                if not m.alive:
                    continue
                obs = self.obs_for(m)
                t0 = time.perf_counter()
                try:
                    a = m.brain.act(obs)
                except Exception as e:  # noqa
                    import traceback; traceback.print_exc()
                    a = None
                el = (time.perf_counter() - t0) * 1000
                if hasattr(m.brain, "times"):
                    m.brain.times.append(el)
                self.apply(m, a)
            self.sounds = [s for s in self.sounds if self.t - s[0] < 0.1 + 1e-6]
        c, r, z = self.zone_now()
        for m in self.mechs:
            if not m.alive:
                continue
            self.move_mech(m, dt)
            self.fire_weapons(m, dt)
            # zone
            if math.hypot(m.pos[0] - c[0], m.pos[1] - c[1]) > r:
                self.damage(None, m, z["dps"] * dt, "zone", None)
            elif self.t - m.last_combat > m.stats["regen_delay"] and m.hp < m.max_hp:
                m.hp = min(m.max_hp, m.hp + m.stats["regen_per_s"] * dt)
            if not m.alive:
                continue
            # cubes
            for cb in list(self.cubes):
                if math.hypot(cb["pos"][0] - m.pos[0], cb["pos"][1] - m.pos[1]) < self.cfg["cubes"]["pickup_radius"] and abs(cb["pos"][2] - m.pos[2]) < 4:
                    self.cubes.remove(cb); self._gain_cubes(m, 1)
                    m.events.append({"type": "pickup", "cubes": m.cubes})
            if m.offer is not None and self.t > m.offer["expires"]:
                self._apply_upgrade(m, m.offer["choices"][0]["id"]); m.offer = None
        self.step_projectiles(dt)
        self.npc_step(dt)
        # placements
        dead_now = [m for m in self.mechs if not m.alive and m.place is None]
        if dead_now:
            for m in dead_now:
                m.place = self.alive_count  # shared for same tick
            self.alive_count -= len(dead_now)
        self.t += dt; self.tick += 1

    def run(self, max_t=None):
        max_t = max_t or self.cfg["match"]["duration_s"]
        while self.t < max_t and self.alive_count > 1:
            self.step()
        alive = sorted([m for m in self.mechs if m.alive], key=lambda m: -m.hp)
        for k, m in enumerate(alive):
            m.place = k + 1
        return {m.id: (m.place, m.kills, m.death_t) for m in self.mechs}
