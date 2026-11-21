"""Mech Royale bot: cover-aware laser/lightning fighter with zone, loot and upgrade logic."""
import gc
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechfight as mf  # noqa: E402
from world import World  # noqa: E402

PI = math.pi
TAU = 2 * PI


def angdiff(a, b):
    return (a - b + PI) % TAU - PI


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# How much we like holding each weapon (pairs get a synergy bonus below).
WEAPON_VALUE = {"gun": 3.6, "laser": 5.0, "lightning": 3.4, "mortar": 2.4}
PAIR_BONUS = {frozenset(("laser", "lightning")): 0.0, frozenset(("gun", "laser")): 0.6}

UPGRADE_SCORE = {
    "plating": 10.0, "armor": 9.6, "laser_damage": 9.2, "laser_charge": 8.8, "nanites": 7.6,
    "lightning_efficiency": 7.4, "gun_damage": 7.2, "gun_rate": 7.0, "lightning_range": 6.8,
    "servos": 5.6, "reactor": 5.4, "sensors": 5.2, "gun_mag": 5.0, "capacitor": 4.8,
    "lightning_chain": 4.4, "gun_velocity": 4.6, "laser_pierce": 3.0, "gyros": 3.6, "jets": 3.4,
    "mortar_reload": 3.0, "mortar_velocity": 2.6, "mortar_splash": 2.8,
}


class Track:
    __slots__ = ("id", "pos", "vel", "aim", "hp", "max_hp", "weapons", "charging", "cubes", "t", "kind", "est")

    def __init__(self, m, t):
        self.id = m["id"]; self.kind = m.get("kind", "mech"); self.est = False
        self.update(m, t)

    def update(self, m, t):
        self.pos = list(m["pos"]); self.vel = list(m.get("vel", [0, 0, 0])); self.aim = list(m.get("aim", [0, 0]))
        self.hp = m.get("hp", 1400.0); self.max_hp = m.get("max_hp", 1400.0)
        self.weapons = m.get("weapons", []) or []; self.charging = bool(m.get("charging", False))
        self.cubes = m.get("cubes", 0) or 0; self.t = t; self.est = False

    def predict(self, now):
        dt = min(now - self.t, 3.0) if not self.est else 0.0
        return [self.pos[0] + self.vel[0] * dt * 0.6, self.pos[1] + self.vel[1] * dt * 0.6, self.pos[2]]


class Bot:
    name = "cover-and-arc"

    def __init__(self, hello):
        self.cfg = cfg = hello["config"]
        self.slot = hello["slot"]
        self.rng = random.Random(hello.get("seed", 1))
        mapdir = hello["map"]["dir"]
        try:
            self.W = World(mapdir, cfg["mech"], hello["map"].get("files"))
        except Exception as e:  # never leave the mech idle: fall back to a map-free policy
            mf.log(f"world load failed: {e!r}")
            self.W = None
        self.mech = cfg["mech"]
        self.wcfg = cfg["weapons"]
        self.reach = cfg["loot"]["pickup_radius"]
        self.cube_reach = cfg["cubes"].get("pickup_radius", 3.5)
        self.tick_dt = cfg["match"].get("decide_every", 2) / cfg["match"].get("tick_hz", 20)
        # memory
        self.tracks = {}         # id -> Track (mechs)
        self.npcs = {}           # id -> Track
        self.pickups = {}        # id -> (kind, pos, t)
        self.crates = {}         # id -> (pos, hp, t)
        self.cubes_seen = {}     # id -> (pos, t)
        self.sounds = []         # (t, bearing, distband, kind)
        self.last_damage_t = -99.0
        self.my_kills = 0
        self.others_deaths = 0
        self.last_alive = 16
        self.ffa = False
        # navigation state
        self.field = None; self.field_t = -9; self.field_pos = None
        self.goal = None; self.goal_kind = ""; self.path = None; self.path_goal = None
        self.cand = None; self.cand_t = -9
        self.stuck_t = 0.0; self.last_pos = None; self.unstick_until = -1; self.unstick_dir = (1, 0)
        self.jink = 1; self.jink_t = 0.0
        self.scan_phase = self.rng.random() * TAU
        self.laser_hold_start = None
        self.prev_t = 0.0
        self.target_id = None
        self.peek = False
        gc.collect(); gc.freeze(); gc.disable()  # collect manually on light ticks (see act)

    # ------------------------------------------------------------------ utils
    def zone_at(self, z, T):
        """Circle (cx, cy, r) at absolute time T, from the current obs zone block."""
        c0, r0 = z["center"], z["radius"]; c1, r1 = z["next_center"], z["next_radius"]
        s0, s1 = z["shrink_start"], z["shrink_end"]
        if T <= s0 or s1 <= s0:
            return (c0[0], c0[1], r0)
        f = min(1.0, (T - s0) / (s1 - s0))
        return (c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, r0 + (r1 - r0) * f)

    def weapon(self, kind):
        for i, w in enumerate(self.me["weapons"]):
            if w and w["kind"] == kind:
                return i, w
        return None, None

    def kinds(self):
        return [w["kind"] for w in self.me["weapons"] if w]

    def dmg_mult(self, cubes):
        c = self.cfg["cubes"]
        return 1 + c.get("damage_per_cube", 0.04) * min(cubes, c.get("stat_cap", 10))

    # ------------------------------------------------------------ dps model
    def our_dps(self, dist):
        tot = 0.0
        for w in self.me["weapons"]:
            if not w:
                continue
            k = w["kind"]; s = w["stats"]
            if k == "gun":
                if dist <= s["range"]:
                    acc = 0.92 if dist < 50 else (0.75 if dist < 120 else 0.5)
                    cyc = s["mag"] * s["cooldown"] + s["reload"]
                    tot += s["damage"] * s["mag"] / cyc * acc
            elif k == "laser":
                if dist <= s["range"]:
                    tot += s["damage"] / (s["charge"] + s["cooldown"]) * 0.85
            elif k == "lightning":
                if dist <= s["range"] + 15 and self.me["energy"] > 10:
                    tot += s["damage"] / s["cooldown"] * (1.0 if dist <= s["range"] else 0.5)
            elif k == "mortar":
                if 20 < dist < 230:
                    tot += (s["damage"] * 0.5 + s["splash"] * 0.4) / s["cooldown"]
        return tot * self.dmg_mult(self.me["cubes"])

    def their_dps(self, tr, dist):
        tot = 0.0
        w = self.wcfg
        for k in tr.weapons or ["gun"]:
            if k == "gun" and dist <= w["gun"]["range"]:
                acc = 0.8 if dist < 50 else (0.6 if dist < 120 else 0.4)
                tot += w["gun"]["damage"] * w["gun"]["mag"] / (w["gun"]["mag"] * w["gun"]["cooldown"] + w["gun"]["reload"]) * acc
            elif k == "laser" and dist <= w["laser"]["range"]:
                tot += w["laser"]["damage"] / (w["laser"]["charge"] + w["laser"]["cooldown"]) * 0.85
            elif k == "lightning" and dist <= w["lightning"]["range"] + 10:
                tot += w["lightning"]["damage"] / w["lightning"]["cooldown"]
            elif k == "mortar" and 20 < dist < 230:
                tot += 70.0
        mult = self.dmg_mult(tr.cubes)
        if "mortar" in self.kinds() and "laser" in (tr.weapons or []):
            tot *= 1.25
        return tot * mult * (1.0 - self.me["stats"].get("armor", 0.0))

    # ------------------------------------------------------------ perception
    def perceive(self, obs):
        t = obs["t"]; me = self.me
        seen_ids = set()
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                seen_ids.add(m["id"])
                if m["id"] in self.tracks:
                    self.tracks[m["id"]].update(m, t)
                else:
                    self.tracks[m["id"]] = Track(m, t)
            else:
                if m["id"] in self.npcs:
                    self.npcs[m["id"]].update(m, t)
                else:
                    self.npcs[m["id"]] = Track(m, t)
        for p in obs["pickups"]:
            self.pickups[p["id"]] = (p["kind"], p["pos"], t)
        vis_p = {p["id"] for p in obs["pickups"]}
        for c in obs["crates"]:
            self.crates[c["id"]] = (c["pos"], c.get("hp", 300), t)
        vis_c = {c["id"] for c in obs["crates"]}
        for c in obs["cubes"]:
            self.cubes_seen[c["id"]] = (c["pos"], t)
        vis_q = {c["id"] for c in obs["cubes"]}
        mp = me["pos"]
        # forget items that should be visible (close) but are not
        for store, vis in ((self.pickups, vis_p), (self.crates, vis_c), (self.cubes_seen, vis_q)):
            for k in list(store):
                v = store[k]
                pos = v[1] if store is self.pickups else v[0]
                if k not in vis and (d2(pos, mp) < 30 or t - v[-1] > 120):
                    del store[k]
        for k in list(self.npcs):
            if t - self.npcs[k].t > 60:
                del self.npcs[k]
        for e in obs["events"]:
            et = e["type"]
            if et == "damaged":
                if e.get("by") is not None:
                    self.last_damage_t = t
                    by = e["by"]
                    if by < 100 and (by not in self.tracks or t - self.tracks[by].t > 1.0):
                        dist = {"lightning": 35, "laser": 140, "gun": 110, "mortar": 140}.get(e["weapon"], 80)
                        b = e["bearing"] or 0.0
                        pos = [mp[0] + math.cos(b) * dist, mp[1] + math.sin(b) * dist, mp[2]]
                        fake = {"id": by, "pos": pos, "vel": [0, 0, 0], "weapons": [e["weapon"]] if e["weapon"] in WEAPON_VALUE else []}
                        if by in self.tracks:
                            tr = self.tracks[by]; tr.pos = pos; tr.t = t; tr.est = True
                            if e["weapon"] in WEAPON_VALUE and e["weapon"] not in tr.weapons:
                                tr.weapons = tr.weapons + [e["weapon"]]
                        else:
                            tr = Track(fake, t); tr.est = True; self.tracks[by] = tr
                    self.hurt_bearing = (t, e["bearing"])
                elif e.get("weapon") == "npc":
                    self.last_damage_t = t
            elif et == "kill":
                self.my_kills += 1
                self.tracks.pop(e["target"], None)
            elif et == "take_failed":
                self.pickups.pop(e["id"], None)
            elif et == "took":
                self.pickups.pop(e["id"], None)
            elif et == "dropped":
                self.pickups[e["id"]] = (e["kind"], list(mp), t)
        for s in obs["sounds"]:
            self.sounds.append((t - s["age"], s["bearing"], s["dist"], s["kind"]))
        self.sounds = [s for s in self.sounds if t - s[0] < 6.0][-40:]
        # deaths we did not cause => free-for-all (other mechs kill each other)
        alive = obs["alive"]
        if alive < self.last_alive:
            died = self.last_alive - alive
            ours = sum(1 for e in obs["events"] if e["type"] == "kill")
            self.others_deaths += max(0, died - ours)
            self.last_alive = alive
        if self.others_deaths >= 2:
            self.ffa = True
        # drop stale tracks
        for k in list(self.tracks):
            if t - self.tracks[k].t > 25:
                del self.tracks[k]

    def threats(self, t, horizon=8.0):
        out = []
        mp = self.me["pos"]
        for tr in self.tracks.values():
            age = t - tr.t
            if age <= horizon:
                p = tr.predict(t)
                dist = d2(p, mp)
                if dist < 300:
                    out.append((tr, p, dist, age))
        return out

    # -------------------------------------------------------------- planning
    def update_field(self, t, force=False):
        mp = self.me["pos"]
        if (not force and self.field is not None and t - self.field_t < 0.6
                and d2(mp, self.field_pos) < 6):
            return
        W = self.W
        z = self.zone
        if t - getattr(self, "_cost_t", -99) > 5.0:
            self._cost_t = t
            cx, cy, r = self.zone_at(z, t + 12)
            # penalize cells outside the soon-to-be zone (scaled by its damage)
            gx = W.x0 + np.arange(W.nnx) * 2.0; gy = W.y0 + np.arange(W.nny) * 2.0
            dd = np.hypot(gx[None, :] - cx, gy[:, None] - cy)
            extra = np.where(dd > r, 2.0 + z["dps"] / 15.0, 0.0).ravel()
            W.set_extra_cost(extra)
            return  # spread the work: search on a later tick
        self._heavy = True
        f = W.dijkstra_from(mp[0], mp[1], limit=320.0)
        if f is not None:
            self.field = f; self.field_t = t; self.field_pos = list(mp)
            self.path = None

    def path_dist(self, x, y):
        if self.field is None:
            return d2((x, y), self.me["pos"]) * 1.4
        j, i = self.W.nav_cell(x, y)
        v = self.field[1][j * self.W.nnx + i]
        if not np.isfinite(v):
            k = self.W.nearest_pass(x, y, 3)
            if k is not None and np.isfinite(self.field[1][k]):
                return float(self.field[1][k])
            return 1e9
        return float(v)

    def steer_to(self, goal):
        """Return a unit-ish 2D move vector toward goal, following the nav path."""
        mp = self.me["pos"]; W = self.W
        if goal is None:
            return (0.0, 0.0)
        dist = d2(mp, goal)
        if dist < 1.0:
            return (0.0, 0.0)
        if dist < 25 and W.walkable_line(mp, goal):
            return ((goal[0] - mp[0]) / dist, (goal[1] - mp[1]) / dist)
        if self.field is None:
            return ((goal[0] - mp[0]) / dist, (goal[1] - mp[1]) / dist)
        if self.path is None or self.path_goal is None or d2(self.path_goal, goal) > 4:
            k = W.nearest_pass(goal[0], goal[1], 4)
            d = self.field[1]
            if k is None or not np.isfinite(d[k]):
                # nearest reachable cell toward goal
                fin = np.isfinite(d)
                ks = np.nonzero(fin)[0]
                if len(ks):
                    jj, ii = np.divmod(ks, W.nnx)
                    xs = W.x0 + ii * 2.0; ys = W.y0 + jj * 2.0
                    sc = d[ks] * 0.3 + np.hypot(xs - goal[0], ys - goal[1])
                    k = int(ks[np.argmin(sc)])
                else:
                    k = None
            self.path = W.path_to(self.field, k) if k is not None else None
            self.path_goal = list(goal)
        if not self.path:
            return ((goal[0] - mp[0]) / dist, (goal[1] - mp[1]) / dist)
        # advance along path: pick the farthest node within reach that is straight-walkable
        best = None
        pts = [W.nav_xy(k) for k in self.path]
        # drop nodes already passed
        jbest = 0; bd = 1e9
        for j, p in enumerate(pts[:60]):
            dd = d2(p, mp)
            if dd < bd:
                bd = dd; jbest = j
        self.path = self.path[jbest:]
        pts = pts[jbest:]
        for j in range(min(len(pts) - 1, 14), 0, -1):
            if d2(pts[j], mp) < 22 and W.walkable_line(mp, pts[j]):
                best = pts[j]; break
        if best is None:
            best = pts[1] if len(pts) > 1 else pts[0]
        dd = d2(best, mp)
        if dd < 0.5:
            return ((goal[0] - mp[0]) / dist, (goal[1] - mp[1]) / dist)
        return ((best[0] - mp[0]) / dd, (best[1] - mp[1]) / dd)

    # ------------------------------------------------------------ positions
    def candidates(self, t, radius=42.0, n=90):
        """Reachable nav cells within path distance `radius`, subsampled, with their path distance."""
        f = self.field
        if f is None:
            return None
        d = f[1]
        ks = np.nonzero(d <= radius)[0]
        if len(ks) == 0:
            return None
        if len(ks) > n:
            ks = ks[np.linspace(0, len(ks) - 1, n).astype(int)]
        W = self.W
        jj, ii = np.divmod(ks, W.nnx)
        xs = W.x0 + ii * 2.0; ys = W.y0 + jj * 2.0
        zs = W.nS[jj, ii]
        # include current position
        mp = self.me["pos"]
        xs = np.append(xs, mp[0]); ys = np.append(ys, mp[1]); zs = np.append(zs, mp[2]); dd = np.append(d[ks], 0.0)
        return xs, ys, zs, dd

    def choose_position(self, t, mode, main, threats):
        """mode: 'engage', 'cover', 'retreat'. Returns goal (x, y)."""
        c = self.candidates(t, radius=45.0 if mode != "retreat" else 70.0)
        if c is None:
            return None
        xs, ys, zs, dd = c
        n = len(xs)
        ths = sorted(threats, key=lambda x: x[2])[:4]
        eh = self.me["stats"].get("eye_height", 8.0)
        vis = np.zeros((n, max(1, len(ths))), bool)
        if ths:
            top = self.me["stats"].get("height", 9.0) * 0.95
            A = np.repeat(np.stack([xs, ys, zs + top], 1), len(ths), 0)
            B = np.tile(np.array([[p[0], p[1], p[2] + 8.0] for (_, p, _, _) in ths]), (n, 1))
            vis = self.W.los_many(A, B, step=1.5).reshape(n, len(ths))
        cx, cy, r = self.zone_at(self.zone, t + 10)
        outz = np.hypot(xs - cx, ys - cy) > r - 4
        score = -0.07 * dd - outz * (6.0 + self.zone["dps"] / 8.0)
        if not ths:
            return None
        tdist = np.stack([np.hypot(xs - p[0], ys - p[1]) for (_, p, _, _) in ths], 1)
        main_idx = 0
        if main is not None:
            for k, (tr, _, _, _) in enumerate(ths):
                if tr.id == main.id:
                    main_idx = k
        pref_lo, pref_hi = self.pref_range()
        md = tdist[:, main_idx]
        band = np.where(md < pref_lo, pref_lo - md, np.where(md > pref_hi, md - pref_hi, 0.0))
        others = vis.sum(1) - vis[:, main_idx]
        if mode == "engage":
            score += 9.0 * vis[:, main_idx] - 3.5 * others - 0.12 * band
            # prefer some clearance nearby for jinking but near walls for quick cover
        elif mode == "cover":
            score += -9.0 * vis.any(1) - 0.05 * band - 0.6 * (tdist.min(1) < 25)
        else:  # retreat
            score += -9.0 * vis.any(1) + 0.1 * np.minimum(tdist.min(1), 140) - 0.03 * dd
        k = int(np.argmax(score))
        hit_here = t - self.last_damage_t < 0.8 and dd[k] < 6 and mode != "engage"
        if (mode == "retreat" and vis[k].any()) or hit_here:
            # no cover reachable: run directly away from the threats, toward the zone
            ax = sum(p[0] for (_, p, _, _) in ths) / len(ths); ay = sum(p[1] for (_, p, _, _) in ths) / len(ths)
            far = np.hypot(xs - ax, ys - ay) - 0.5 * np.hypot(xs - cx, ys - cy) * (outz * 3 + 0.2)
            k = int(np.argmax(far))
        self.dbg_vis = (xs, ys, vis.any(1))
        return (float(xs[k]), float(ys[k])), bool(vis[k].any()), bool(vis[-1].any())

    def pref_range(self):
        ks = self.kinds()
        if "lightning" in ks and self.me["energy"] > 25:
            return (8.0, 32.0)
        if "laser" in ks:
            return (45.0, 160.0)
        if "mortar" in ks and "gun" not in ks:
            return (50.0, 150.0)
        return (25.0, 90.0)

    # ------------------------------------------------------------- aiming
    def aim_at_point(self, p, z_off=None):
        me = self.me
        mz = me["pos"][2] + me["stats"].get("muzzle_height", 7.0)
        dx = p[0] - me["pos"][0]; dy = p[1] - me["pos"][1]
        tz = p[2] + (4.5 if z_off is None else z_off)
        h = math.hypot(dx, dy)
        return [math.atan2(dy, dx), math.atan2(tz - mz, max(h, 0.1))]

    def lead_point(self, tr, speed, now, extra=0.0):
        me = self.me
        P = tr.predict(now); V = tr.vel if not tr.est else [0, 0, 0]
        mz = me["pos"][2] + me["stats"].get("muzzle_height", 7.0)
        size = 1 + 0.05 * (tr.cubes or 0)
        tz = P[2] + 4.5 * size if tr.kind == "mech" else P[2] + 1.75
        rx, ry, rz = P[0] - me["pos"][0], P[1] - me["pos"][1], tz - mz
        vx, vy = V[0], V[1]
        vz = V[2] if abs(V[2]) < 3 else 0.0
        a = vx * vx + vy * vy + vz * vz - speed * speed
        b = 2 * (rx * vx + ry * vy + rz * vz)
        c = rx * rx + ry * ry + rz * rz
        tt = None
        if abs(a) < 1e-6:
            tt = -c / b if b else None
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                s = math.sqrt(disc)
                for cand in sorted(((-b - s) / (2 * a), (-b + s) / (2 * a))):
                    if cand > 0:
                        tt = cand; break
        if tt is None:
            tt = math.sqrt(c) / speed
        tt += extra
        return [P[0] + vx * tt, P[1] + vy * tt, tz + vz * tt - (4.5 * size if tr.kind == "mech" else 1.75)], tt

    def mortar_solution(self, target, prefer_high):
        me = self.me
        i, w = self.weapon("mortar")
        if w is None:
            return None
        v = w["stats"]["speed"]; g = w["stats"]["gravity"]
        ox, oy = me["pos"][0], me["pos"][1]
        oz = me["pos"][2] + me["stats"].get("height", 9.0) + 0.5
        dx = target[0] - ox; dy = target[1] - oy; D = math.hypot(dx, dy)
        h = target[2] + 1.0 - oz
        v2 = v * v
        disc = v2 * v2 - g * (g * D * D + 2 * h * v2)
        if disc < 0 or D < 1:
            return None
        s = math.sqrt(disc)
        hi = math.atan((v2 + s) / (g * D)); lo = math.atan((v2 - s) / (g * D))
        pitch = hi if prefer_high else lo
        if pitch > self.mech.get("max_pitch", 1.3) or pitch < self.mech.get("min_pitch", -0.7):
            pitch = lo
            if pitch > self.mech.get("max_pitch", 1.3):
                return None
        tof = D / (v * math.cos(pitch))
        return [math.atan2(dy, dx), pitch], tof

    # ------------------------------------------------------------ main step
    def fallback(self, obs):
        me = obs["self"]; x, y = me["pos"][0], me["pos"][1]
        g = obs["zone"]["next_center"]; yaw = math.atan2(g[1] - y, g[0] - x)
        act = {"move": [math.cos(yaw), math.sin(yaw)], "aim": [yaw, 0.0], "boost": not me["in_zone"],
               "jump": obs["tick"] % 40 == 0}
        if obs.get("offer"):
            act["pick"] = 0
        en = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if en:
            t = min(en, key=lambda m: d2(m["pos"], me["pos"]))["pos"]
            act["aim"] = [math.atan2(t[1] - y, t[0] - x), math.atan2(t[2] + 4.5 - me["pos"][2] - 7, max(d2(t, me["pos"]), 1))]
            act["fire"] = [True, True]
        for p in obs["pickups"]:
            if me["weapons"][1] is None and d2(p["pos"], me["pos"]) < 3.8:
                act["take"] = p["id"]
        return act

    def act(self, obs):
        if self.W is None:
            return self.fallback(obs)
        t0 = time.perf_counter()
        self._t0 = t0; self._heavy = False
        self.me = me = obs["self"]
        self.zone = obs["zone"]
        t = obs["t"]
        act = {}
        self.perceive(obs)
        mp = me["pos"]
        W = self.W
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"])
        self.update_field(t)

        threats = self.threats(t)
        hp_frac = me["hp"] / max(me["max_hp"], 1)
        # ---------------- target selection
        main = None; best = 1e18
        engaged = (t - self.last_damage_t < 3.0) or any(a < 3.0 and d < 230 for (_, _, d, a) in threats)
        for tr, p, dist, age in (threats if engaged else []):
            dps = self.our_dps(dist) + 1e-3
            ttk = tr.hp / dps
            score = ttk + dist / 60.0 + age * 2.0 + (3.0 if tr.est else 0.0)
            if tr.id == self.target_id:
                score -= 1.5
            if score < best:
                best = score; main = (tr, p, dist, age)
        self.target_id = main[0].id if main else None

        # ---------------- fight evaluation
        mode = "free"
        if main is not None:
            tr, p, dist, age = main
            active = [x for x in threats if x[3] < 5 and x[2] < 240]
            incoming = sum(self.their_dps(x[0], x[2]) for x in active) + 1e-3
            out = self.our_dps(max(dist, 10)) + 1e-3
            ttk = tr.hp / out
            ttd = me["hp"] / incoming
            margin = 1.0 if self.ffa else 1.8
            winning = ttk < ttd * margin
            n_act = len(active)
            low = hp_frac < (0.35 if self.ffa else 0.18)
            finishing = tr.hp < (380 if self.ffa else 500) and dist < 160 and n_act <= 2
            if low and not finishing:
                mode = "retreat"
            elif finishing or (winning and n_act <= 1):
                mode = "engage"
            elif winning:
                mode = "engage" if dist < (60 if self.ffa else 90) else "cover"   # several enemies: fight from cover / peeks
            else:
                mode = "cover" if (dist < 160 and hp_frac > 0.4) else "retreat"
            # in FFA, avoid starting long fights with healthy, distant mechs
            if self.ffa and mode == "engage" and age > 2.0 and dist > 120 and tr.hp > 700:
                mode = "cover"
        self.mode = mode

        # ---------------- movement goal
        move = (0.0, 0.0); boost = False; jump = False
        goal = None; label = mode
        zc = self.zone_at(self.zone, t + 6)
        dz = d2(mp, zc) - zc[2]
        zone_urgent = (not me["in_zone"]) or dz > -6
        # time to reach next circle
        nc = self.zone["next_center"]; nr = self.zone["next_radius"]
        dn = d2(mp, nc) - nr * 0.8
        travel = max(0.0, dn) / (me["stats"].get("walk_speed", 11.7) * 0.8)
        if self.zone["shrink_end"] > t and travel > 0 and self.zone["shrink_end"] - t < travel + 15:
            zone_urgent = True

        laser_i, laser = self.weapon("laser")
        grab = self.grab_nearby(t) if mode in ("engage", "cover") else None
        if grab is not None:
            goal, label = grab
            boost = False
        elif mode in ("engage", "cover", "retreat") and main is not None:
            tr, p, dist, age = main
            if mode == "engage" and laser is not None and "lightning" not in self.kinds():
                # peek cycle: hide while the laser recharges, expose once it can fire
                ready = laser["cooldown"] < 0.5
                if not ready and dist > 90 and tr.hp > 380:
                    mode = "cover"
            if mode == "cover" and laser is not None and laser.get("charge", 0) >= laser["stats"]["charge"] - 0.05 and hp_frac > 0.3:
                mode = "engage"   # charged behind cover: peek out and shoot
            self.mode = mode
            stale = t - self.cand_t > (0.25 if t - self.last_damage_t < 0.5 else 0.5)
            if self.cand is None or stale or self.cand[0] != mode:
                if self.cand is None or (not self._heavy and time.perf_counter() - t0 < 0.015) or (self.cand[0] != mode and time.perf_counter() - t0 < 0.02):
                    r = self.choose_position(t, mode, tr, threats)
                    if r is not None:
                        self.cand = (mode, r[0]); self.cand_t = t
            goal = self.cand[1] if self.cand else None
            if zone_urgent and self.zone["dps"] >= 35:
                goal = self.zone_goal(t)
            label = f"{mode} {tr.id} d{dist:.0f} hp{tr.hp:.0f}"
            boost = mode == "retreat" or (zone_urgent and not me["in_zone"])
        else:
            goal, label = self.free_goal(obs, t, zone_urgent)
            boost = zone_urgent and not me["in_zone"]

        self.goal = goal
        if goal is not None:
            move = self.steer_to(goal)
            if d2(mp, goal) < 2.5:
                move = (0.0, 0.0)

        # jink when exposed in a fight
        if mode == "engage" and main is not None:
            if t > self.jink_t:
                self.jink = -self.jink if self.rng.random() < 0.7 else self.jink
                self.jink_t = t + 0.35 + self.rng.random() * 0.6
            tr, p, dist, age = main
            by = math.atan2(p[1] - mp[1], p[0] - mp[0])
            px, py = -math.sin(by) * self.jink, math.cos(by) * self.jink
            if W.clearance(mp[0] + px * 4, mp[1] + py * 4) > W.radius + 0.3:
                k = 0.75 if (move[0] or move[1]) else 1.0
                move = (move[0] * (1 - k * 0.5) + px * k, move[1] * (1 - k * 0.5) + py * k)
        # dodge incoming projectiles
        dv = self.dodge(obs)
        if dv is not None:
            move = (move[0] + dv[0] * 1.5, move[1] + dv[1] * 1.5)
            if self.me["energy"] > 35:
                boost = True
        n = math.hypot(*move)
        if n > 1:
            move = (move[0] / n, move[1] / n)

        # stuck handling
        if self.last_pos is not None and n > 0.3 and me["grounded"]:
            sp = d2(mp, self.last_pos) / max(t - self.prev_t, 1e-3)
            if sp < 1.2:
                self.stuck_t += t - self.prev_t
            else:
                self.stuck_t = max(0.0, self.stuck_t - 0.2)
        else:
            self.stuck_t = max(0.0, self.stuck_t - 0.1)
        if self.stuck_t > 0.8 and t > self.unstick_until:
            self.unstick_until = t + 1.2
            a = math.atan2(move[1], move[0]) + self.rng.choice((-1, 1)) * (PI / 2 + self.rng.random())
            self.unstick_dir = (math.cos(a), math.sin(a))
            self.stuck_t = 0.0
            self.path = None
        if t < self.unstick_until:
            if t < self.unstick_until - 0.6:
                move = self.unstick_dir
            if me["energy"] > 20:
                jump = me["grounded"]
                boost = not me["grounded"] and me["energy"] > 15
        self.last_pos = list(mp); self.prev_t = t

        # don't sprint away energy needed for lightning in a fight
        _, lw = self.weapon("lightning")
        if lw is not None and mode == "engage" and me["energy"] < 40:
            boost = False
        if boost and me["energy"] < 12:
            boost = False

        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = bool(boost)
        act["jump"] = bool(jump)

        # ---------------- weapons & aim
        aim, fire, reload_ = self.weapons_and_aim(obs, t, main, threats, move)
        if self.mode == "retreat" and main is not None:
            # dealing damage resets regen: hold fire while healing unless it finishes them or they are on us
            tr, p, dist, age = main
            if not (tr.hp < 200 or dist < 35 or t - self.last_damage_t < 1.0):
                lw_i, lw_ = self.weapon("laser")
                fire = [bool(fire[i]) and lw_i == i and lw_.get("charge", 0) > 0 for i in range(2)]
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["fire"] = fire
        if reload_:
            act["reload"] = True

        # ---------------- loot handling
        tk = self.take_decision(obs)
        if tk:
            act.update(tk)
        if obs["tick"] % 100 == 0 and time.perf_counter() - t0 < 0.01:
            gc.collect()
        mf.draw.label(label[:70])
        if goal is not None:
            mf.draw.point([goal[0], goal[1], W.ground(goal[0], goal[1]) + 1], "#40c0ff", 5)
        return act

    def grab_nearby(self, t):
        """In a fight, still grab a cube or a better weapon that is a few steps away."""
        mp = self.me["pos"]
        best = None; bd = 1e9
        for cid, (pos, ts) in self.cubes_seen.items():
            if pos[2] > mp[2] + 3:
                continue
            pd = self.path_dist(pos[0], pos[1])
            if pd < 16 and pd < bd:
                bd = pd; best = ((pos[0], pos[1]), "grab cube")
        gain, pk = self.best_pickup(t)
        if pk is not None and gain >= 1.0:
            pd = self.path_dist(pk[1][0], pk[1][1])
            if pd < 22 + 8 * gain and pd < bd + 10:
                best = ((pk[1][0], pk[1][1]), f"grab {pk[0]}")
        return best

    # ---------------------------------------------------------- free roaming
    def zone_goal(self, t):
        mp = self.me["pos"]
        nc = self.zone["next_center"]; nr = self.zone["next_radius"]
        d = d2(mp, nc)
        target_r = nr * 0.55
        if d <= target_r:
            return (mp[0], mp[1])
        f = target_r / d
        return (nc[0] + (mp[0] - nc[0]) * f, nc[1] + (mp[1] - nc[1]) * f)

    def in_safe(self, p, t, margin=6.0):
        cx, cy, r = self.zone_at(self.zone, t + 20)
        return d2(p, (cx, cy)) < r - margin

    def free_goal(self, obs, t, zone_urgent):
        me = self.me; mp = me["pos"]
        if zone_urgent:
            return self.zone_goal(t), "zone"
        best = None; bs = -1e9; lab = "hold"
        # supply drops
        for dr in obs.get("drops", []):
            if not self.in_safe(dr["pos"], t, 0):
                continue
            pd = self.path_dist(dr["pos"][0], dr["pos"][1])
            if pd > 260:
                continue
            arrive = pd / 12.0
            if dr["lands_in"] > arrive + 15:
                continue
            contested = dr["holder"] is not None and dr["holder"] != me["id"]
            s = 60 - pd * 0.15 - (25 if contested else 0) + (1 - me["hp"] / me["max_hp"]) * 30
            if s > bs:
                bs = s; best = (dr["pos"][0], dr["pos"][1]); lab = "drop"
        def danger(pos):
            s = 0.0
            for tr in self.tracks.values():
                if t - tr.t < 10:
                    dd = d2(pos, tr.pos)
                    if dd < 120:
                        s += (120 - dd) * 0.25
            return s
        # cubes
        for cid, (pos, ts) in self.cubes_seen.items():
            if pos[2] > self.W.ground(pos[0], pos[1]) + 3:
                continue
            pd = self.path_dist(pos[0], pos[1])
            if pd > 200 or not self.in_safe(pos, t, 0):
                continue
            s = 40 - pd * 0.2 + (10 if me["cubes"] < 10 else 0) - danger(pos)
            if s > bs:
                bs = s; best = (pos[0], pos[1]); lab = "cube"
        # weapons
        gain, pk = self.best_pickup(t)
        if pk is not None:
            kind, pos = pk
            pd = self.path_dist(pos[0], pos[1])
            s = 25 + gain * 12 - pd * 0.15 - danger(pos)
            if s > bs:
                bs = s; best = (pos[0], pos[1]); lab = f"get {kind}"
        # crates / npcs for cubes (shooting handled in weapons logic)
        ft = self.farm_target(t)
        if ft is not None:
            pos, kind, dist = ft
            s = 18 - dist * 0.08
            if s > bs:
                bs = s
                # stand off at ~60% of the distance, staying out of npc aggro range
                if kind == "npc" and dist > 70:
                    f = (dist - 65) / dist
                    best = (mp[0] + (pos[0] - mp[0]) * f, mp[1] + (pos[1] - mp[1]) * f)
                elif kind == "crate" and dist > 60:
                    f = (dist - 40) / dist
                    best = (mp[0] + (pos[0] - mp[0]) * f, mp[1] + (pos[1] - mp[1]) * f)
                else:
                    best = (mp[0], mp[1])
                lab = f"farm {kind}"
        if best is None:
            best = self.hold_goal(t); lab = "hold"
        return best, lab

    def hold_goal(self, t):
        """A spot inside the next circle, near walls (cover), close to us."""
        mp = self.me["pos"]
        h = getattr(self, "_hold", None)
        if h and ((t - h[1] < 8 and self.in_safe(h[0], t, 10)) or self._heavy or time.perf_counter() - self._t0 > 0.015):
            return h[0]
        c = self.candidates(t, radius=140.0, n=400)
        if c is None:
            return self.zone_goal(t)
        xs, ys, zs, dd = c
        cx, cy, r = self.zone_at(self.zone, t + 30)
        nc = self.zone["next_center"]; nr = self.zone["next_radius"]
        W = self.W
        jj = np.clip(np.rint(ys - W.y0).astype(int), 0, W.ny - 1); ii = np.clip(np.rint(xs - W.x0).astype(int), 0, W.nx - 1)
        cl = W.clear[jj, ii]
        dn = np.hypot(xs - nc[0], ys - nc[1])
        score = -0.04 * dd - 3.0 * (cl > 7) - 2.0 * np.abs(dn / max(nr, 1) - 0.45) * 4 - 20 * (np.hypot(xs - cx, ys - cy) > r - 8)
        # stay away from known threats
        for tr in self.tracks.values():
            if t - tr.t < 20:
                score -= 12.0 * (np.hypot(xs - tr.pos[0], ys - tr.pos[1]) < 60)
        k = int(np.argmax(score))
        g = (float(xs[k]), float(ys[k]))
        self._hold = (g, t)
        return g

    def farm_target(self, t):
        if t > 300:
            return None
        mp = self.me["pos"]
        best = None; bd = 1e9
        for cid, (pos, hp, ts) in self.crates.items():
            if pos[2] > self.W.ground(pos[0], pos[1]) + 3 or not self.in_safe(pos, t, 0):
                continue
            dist = d2(pos, mp)
            if dist < 150 and dist < bd:
                bd = dist; best = (pos, "crate", dist)
        if self.weapon("laser")[1] is not None or self.weapon("gun")[1] is not None:
            for tr in self.npcs.values():
                if t - tr.t > 20 or not self.in_safe(tr.pos, t, 0):
                    continue
                dist = d2(tr.pos, mp)
                if dist < 160 and dist + 30 < bd and self.me["hp"] > 0.6 * self.me["max_hp"]:
                    bd = dist + 30; best = (tr.pos, "npc", dist)
        return best

    # ---------------------------------------------------------- loot
    def loadout_value(self, ks):
        v = sum(WEAPON_VALUE.get(k, 0) for k in ks)
        if len(ks) == 2:
            v += PAIR_BONUS.get(frozenset(ks), 0.0)
        return v

    def best_pickup(self, t):
        ks = self.kinds()
        cur = self.loadout_value(ks)
        best = None; bg = 0.4
        mp = self.me["pos"]
        for pid, (kind, pos, ts) in self.pickups.items():
            if kind in ks or pos[2] > self.W.ground(pos[0], pos[1]) + 3 or not self.in_safe(pos, t, 0):
                continue
            if d2(pos, mp) > 220:
                continue
            if len(ks) < 2:
                g = self.loadout_value(ks + [kind]) - cur
            else:
                g = max(self.loadout_value([ks[1 - i], kind]) for i in range(2)) - cur
            if g > bg:
                bg = g; best = (kind, pos)
        return (bg, best) if best else (0.0, None)

    def take_decision(self, obs):
        me = self.me; mp = me["pos"]
        ks = self.kinds()
        cur = self.loadout_value(ks)
        best = None; bg = 0.4
        for p in obs["pickups"]:
            if d2(p["pos"], mp) > self.reach - 0.2 or abs(p["pos"][2] - mp[2]) > 3.8:
                continue
            kind = p["kind"]
            if kind in ks:
                continue
            ws = me["weapons"]
            if ws[0] is None or ws[1] is None:
                g = self.loadout_value(ks + [kind]) - cur
                if g > bg:
                    bg = g; best = {"take": p["id"]}
            else:
                for i in (0, 1):
                    if ws[i]["kind"] == "laser" and ws[i].get("charge", 0) > 0:
                        continue
                    g = self.loadout_value([ws[1 - i]["kind"], kind]) - cur
                    if g > bg:
                        bg = g; best = {"take": p["id"], "drop": i}
        return best

    def pick_upgrade(self, offer):
        ks = self.kinds()
        owned = self.me.get("upgrades", [])
        best = 0; bs = -1e9
        for i, ch in enumerate(offer["choices"]):
            s = UPGRADE_SCORE.get(ch["id"], 4.0)
            for k in ("laser", "lightning", "gun", "mortar"):
                if ch["id"].startswith(k + "_") and k not in ks:
                    s -= 4
            s -= 0.3 * owned.count(ch["id"])
            if s > bs:
                bs = s; best = i
        return best

    # ---------------------------------------------------------- dodging
    def dodge(self, obs):
        me = self.me; mp = me["pos"]
        out = None; worst = 1e9
        for pr in obs["projectiles"]:
            p = pr["pos"]; v = pr["vel"]
            if pr["kind"] == "mortar":
                g = self.wcfg["mortar"]["gravity"]
                # landing time vs our height
                a = -0.5 * g; b = v[2]; c = p[2] - (mp[2] + 1)
                disc = b * b - 4 * a * c
                if disc < 0:
                    continue
                tl = (-b - math.sqrt(disc)) / (2 * a)
                if tl < 0 or tl > 4:
                    continue
                lx = p[0] + v[0] * tl; ly = p[1] + v[1] * tl
                dd = d2((lx, ly), mp)
                rad = self.wcfg["mortar"]["splash_radius"] + 4
                if dd < rad and tl < worst:
                    worst = tl
                    k = max(dd, 0.5)
                    out = ((mp[0] - lx) / k, (mp[1] - ly) / k)
                continue
            rx, ry, rz = mp[0] - p[0], mp[1] - p[1], (mp[2] + 4.5) - p[2]
            vv = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
            if vv < 1:
                continue
            tc = (rx * v[0] + ry * v[1] + rz * v[2]) / vv
            if tc <= 0 or tc > 0.8:
                continue
            cx = p[0] + v[0] * tc - mp[0]; cy = p[1] + v[1] * tc - mp[1]
            miss = math.hypot(cx, cy)
            if miss < 3.5 and tc < worst:
                worst = tc
                sp = math.hypot(v[0], v[1]) + 1e-6
                px, py = -v[1] / sp, v[0] / sp
                if px * (-cx) + py * (-cy) < 0:
                    px, py = -px, -py
                out = (px, py)
        return out

    # ---------------------------------------------------------- weapons
    def weapons_and_aim(self, obs, t, main, threats, move):
        me = self.me; mp = me["pos"]; W = self.W
        cur_aim = me["aim"]
        fire = [False, False]
        reload_ = False
        mh = me["stats"].get("muzzle_height", 7.0)
        muzzle = [mp[0], mp[1], mp[2] + mh]
        aim = None

        target = main[0] if main else None
        tpos = main[1] if main else None
        dist = main[2] if main else 1e9
        farm = None
        if target is None and self.mode == "free":
            farm = self.farm_shot(t)

        li, lw = self.weapon("laser")
        gi, gw = self.weapon("gun")
        zi, zw = self.weapon("lightning")
        mi, mw = self.weapon("mortar")

        def on_target(desired, tol_yaw, tol_pitch):
            return abs(angdiff(cur_aim[0], desired[0])) < tol_yaw and abs(cur_aim[1] - desired[1]) < tol_pitch

        if target is not None:
            age = main[3]
            size = 1 + 0.05 * (target.cubes or 0)
            center = [tpos[0], tpos[1], tpos[2] + 4.5 * size]
            visible = age < 0.25 and not target.est
            los_ok = visible and W.los(muzzle, center)
            rad_ang = math.atan2(2.2 * size, max(dist, 1))
            primary = None
            # lightning: point straight at them when close
            if zw is not None and dist < zw["stats"]["range"] + 25:
                primary = self.aim_at_point(tpos)
                if (dist <= zw["stats"]["range"] - 1 and zw["cooldown"] <= 0 and me["energy"] >= zw["stats"]["energy"]
                        and abs(angdiff(cur_aim[0], primary[0])) < math.radians(zw["stats"]["cone_deg"]) * 0.45
                        and W.los([mp[0], mp[1], mp[2] + mh], center)):
                    fire[zi] = True
            # laser
            if lw is not None:
                charging = lw.get("charge", 0) > 0
                full = lw["stats"]["charge"]
                lp = [tpos[0] + target.vel[0] * 0.1, tpos[1] + target.vel[1] * 0.1, tpos[2]] if visible else tpos
                desired = self.aim_at_point(lp, 4.5 * size)
                if primary is None or dist > 30:
                    primary = desired
                within = dist <= lw["stats"]["range"] + 5
                if charging:
                    held = lw["charge"]
                    ok = on_target(desired, rad_ang * 0.85, math.atan2(3.0 * size, max(dist, 1)))
                    if held >= full - 0.01 and los_ok and ok:
                        fire[li] = False  # release
                    elif held >= full + lw["stats"].get("hold_s", 2.0) - 0.25 and not los_ok:
                        fire[li] = True   # will fire anyway; keep aim on target
                    else:
                        fire[li] = True
                else:
                    if lw["cooldown"] <= 0.05 and within and (visible or age < 2.5) and not (zw is not None and dist < 25):
                        fire[li] = True   # start charging (also behind cover, before peeking)
            # gun
            if gw is not None:
                lead, tt = self.lead_point(target, gw["stats"]["speed"], t)
                gd = self.aim_at_point(lead, 4.5 * size if target.kind == "mech" else 1.75)
                if primary is None:
                    primary = gd
                if lw is not None and lw.get("charge", 0) > 0 and lw.get("charge", 0) >= lw["stats"]["charge"] - 0.15:
                    pass  # keep laser aim exact; gun fires only if close to on-target anyway
                if (visible and gw["reload"] == 0 and gw["ammo"] > 0 and gw["cooldown"] <= 0.1
                        and dist < gw["stats"]["range"] and los_ok
                        and on_target(gd, rad_ang * 1.3 + math.radians(gw["stats"]["spread_deg"]), math.atan2(4 * size, max(dist, 1)))):
                    fire[gi] = True
                if gw["ammo"] == 0 and gw["reload"] == 0:
                    reload_ = True
            # mortar
            if mw is not None and mw["cooldown"] <= 0 and 25 < dist < 225:
                sol = self.mortar_solution([tpos[0], tpos[1], tpos[2]], prefer_high=not los_ok)
                if sol is not None:
                    maim, tof = sol
                    if primary is None:
                        primary = maim
                    elif not los_ok and (lw is None or lw.get("charge", 0) == 0):
                        primary = maim
                    if on_target(maim, math.atan2(3.0, dist), 0.03) and (not visible or abs(target.vel[0]) + abs(target.vel[1]) < 8):
                        fire[mi] = True
            aim = primary
        elif farm is not None:
            fpos, fkind, fz = farm[:3]
            if gw is not None:
                tr = farm[3] if len(farm) > 3 else None
                if tr is not None:
                    lead, _ = self.lead_point(tr, gw["stats"]["speed"], t)
                    aim = self.aim_at_point(lead, fz)
                else:
                    aim = self.aim_at_point(fpos, fz)
                fd = d2(fpos, mp)
                if (gw["reload"] == 0 and gw["ammo"] > 0 and on_target(aim, math.atan2(1.0, max(fd, 1)) + 0.01, 0.05)
                        and W.los(muzzle, [fpos[0], fpos[1], fpos[2] + fz])):
                    fire[gi] = True
                if gw["ammo"] == 0 and gw["reload"] == 0:
                    reload_ = True
            elif lw is not None and fkind == "npc":
                aim = self.aim_at_point(fpos, fz)
                fire[li] = lw["cooldown"] <= 0.05 and not (lw.get("charge", 0) >= lw["stats"]["charge"] and on_target(aim, 0.02, 0.03))
        if aim is None:
            aim = self.scan_aim(t, move)
            # keep laser from discharging aimlessly
            if lw is not None and lw.get("charge", 0) > 0:
                fire[li] = True
            if gw is not None and gw["ammo"] < gw["stats"]["mag"] * 0.6 and gw["reload"] == 0:
                reload_ = True
        return aim, fire, reload_

    def farm_shot(self, t):
        mp = self.me["pos"]
        best = None; bd = 1e9
        for tr in self.npcs.values():
            if t - tr.t > 0.3:
                continue
            dd = d2(tr.pos, mp)
            if dd < 150 and dd < bd and self.me["hp"] > 0.5 * self.me["max_hp"]:
                bd = dd; best = (tr.pos, "npc", 1.75, tr)
        for cid, (pos, hp, ts) in self.crates.items():
            if t - ts > 0.3 or pos[2] > self.W.ground(pos[0], pos[1]) + 3:
                continue
            dd = d2(pos, mp)
            if dd < 120 and dd + 20 < bd:
                bd = dd + 20; best = (pos, "crate", 1.2)
        return best

    def scan_aim(self, t, move):
        me = self.me
        # face recent damage / sounds first
        hb = getattr(self, "hurt_bearing", None)
        if hb and t - hb[0] < 2.5 and hb[1] is not None:
            return [hb[1], 0.0]
        recent = [s for s in self.sounds if t - s[0] < 2.0 and s[3] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet")]
        if recent:
            s = min(recent, key=lambda s: {"near": 0, "mid": 1, "far": 2}[s[2]])
            return [s[1], 0.0]
        self.scan_phase += 0.12
        if abs(move[0]) + abs(move[1]) > 0.2:
            base = math.atan2(move[1], move[0])
            return [base + 0.9 * math.sin(self.scan_phase * 0.8), 0.0]
        return [self.scan_phase * 0.55 % TAU - PI, 0.0]


if __name__ == "__main__":
    mf.run(Bot)
