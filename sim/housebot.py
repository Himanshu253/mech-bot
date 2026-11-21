"""Approximate house bots: a squad that shares vision and hunts the prey (weakest-first escalation)."""
import math
import random

import numpy as np

PI = math.pi
DIFF = {  # aim noise (deg), lead quality, reaction (s)
    "easy": (4.0, 0.3, 0.6), "medium": (2.5, 0.6, 0.4), "hard": (1.3, 0.9, 0.25), "expert": (0.6, 1.0, 0.1)}


def angdiff(a, b):
    return (a - b + PI) % (2 * PI) - PI


class Squad:
    def __init__(self, world, members, ffa=False):
        self.W = world
        self.members = members  # list of (id, difficulty) in hunt order (weakest first)
        self.known = {}  # prey id -> (pos, vel, t)
        self.dead = set()
        self.field = None; self.field_t = -9; self.field_for = None
        self.ffa = ffa

    def hunters(self, t):
        alive = [m for m, _ in self.members if m not in self.dead]
        killed = len(self.dead)
        n = len(alive) if t > 300 else min(len(alive), 1 + killed)
        return set(alive[:n])

    def prey_field(self, t):
        if not self.known:
            return None
        pid, (pos, vel, ts) = max(self.known.items(), key=lambda kv: kv[1][2])
        if t - ts > 15:
            return None
        if self.field is None or t - self.field_t > 1.0 or self.field_for is None or math.hypot(pos[0] - self.field_for[0], pos[1] - self.field_for[1]) > 8:
            self.W.set_extra_cost(np.zeros(self.W.nny * self.W.nnx))
            f = self.W.dijkstra_from(pos[0], pos[1], limit=700)
            if f is not None:
                self.field = f; self.field_t = t; self.field_for = list(pos)
        return self.field


class HouseBot:
    def __init__(self, mid, diff, squad, seed):
        self.id = mid; self.diff = diff; self.squad = squad
        self.noise, self.lead, self.react = DIFF[diff]
        self.rng = random.Random(seed)
        self.first_seen = {}
        self.wander = None
        self.times = []

    def act(self, obs):
        me = obs["self"]; t = obs["t"]; mp = me["pos"]; W = self.squad.W
        sq = self.squad
        mids = {m for m, _ in sq.members}
        for e in obs["events"]:
            pass
        # share what we see
        prey_vis = []
        for m in obs["mechs"]:
            if m["kind"] != "mech":
                continue
            if not sq.ffa and m["id"] in mids:
                continue
            sq.known[m["id"]] = (m["pos"], m["vel"], t)
            prey_vis.append(m)
            self.first_seen.setdefault(m["id"], t)
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("by") is not None and e["by"] < 100 and (sq.ffa or e["by"] not in mids):
                b = e["bearing"] or 0
                if e["by"] not in sq.known or t - sq.known[e["by"]][2] > 2:
                    sq.known[e["by"]] = ([mp[0] + math.cos(b) * 80, mp[1] + math.sin(b) * 80, mp[2]], [0, 0, 0], t - 3)
        hunting = sq.ffa or self.id in sq.hunters(t)
        act = {"tick": obs["tick"], "fire": [False, False]}
        if obs.get("offer"):
            act["pick"] = 0
        # take weapons in reach
        ks = [w["kind"] for w in me["weapons"] if w]
        for p in obs["pickups"]:
            if math.hypot(p["pos"][0] - mp[0], p["pos"][1] - mp[1]) < 3.8 and p["kind"] not in ks and None in me["weapons"]:
                act["take"] = p["id"]; break
        zc = obs["zone"]["next_center"]; zr = obs["zone"]["next_radius"]
        cur_c = obs["zone"]["center"]
        # movement
        move = [0.0, 0.0]
        target = None
        if prey_vis:
            target = min(prey_vis, key=lambda m: math.hypot(m["pos"][0] - mp[0], m["pos"][1] - mp[1]))
        goal = None
        pref = 90 if "laser" in ks else 35
        if hunting and sq.known:
            f = sq.prey_field(t) if not sq.ffa else None
            pid, (ppos, pvel, pt) = max(sq.known.items(), key=lambda kv: kv[1][2]) if not sq.ffa else min(
                sq.known.items(), key=lambda kv: math.hypot(kv[1][0][0] - mp[0], kv[1][0][1] - mp[1]) + (t - kv[1][2]) * 20)
            dist = math.hypot(ppos[0] - mp[0], ppos[1] - mp[1])
            if t - pt < 20:
                if target is not None and dist < pref and W.los([mp[0], mp[1], mp[2] + 8], [ppos[0], ppos[1], ppos[2] + 5]):
                    a = math.atan2(ppos[1] - mp[1], ppos[0] - mp[0]) + PI / 2
                    s = 1 if (int(t / 1.5) + self.id) % 2 else -1
                    move = [math.cos(a) * s, math.sin(a) * s]
                elif f is not None:
                    move = self.descend(f, mp)
                else:
                    goal = ppos
        if move == [0.0, 0.0] and goal is None:
            # loot or go to zone
            if None in me["weapons"] and obs["pickups"]:
                p = min(obs["pickups"], key=lambda p: math.hypot(p["pos"][0] - mp[0], p["pos"][1] - mp[1]))
                if math.hypot(p["pos"][0] - mp[0], p["pos"][1] - mp[1]) < 80 and p["kind"] not in ks:
                    goal = p["pos"]
            if goal is None:
                if self.wander is None or math.hypot(self.wander[0] - mp[0], self.wander[1] - mp[1]) < 8 or math.hypot(self.wander[0] - zc[0], self.wander[1] - zc[1]) > zr * 0.8:
                    a = self.rng.uniform(0, 2 * PI); r = self.rng.uniform(0, zr * 0.6)
                    self.wander = [zc[0] + math.cos(a) * r, zc[1] + math.sin(a) * r]
                goal = self.wander if (obs["self"]["in_zone"] or True) else zc
        if goal is not None:
            dx = goal[0] - mp[0]; dy = goal[1] - mp[1]; d = math.hypot(dx, dy)
            if d > 2:
                move = [dx / d, dy / d]
        if not me["in_zone"]:
            dx = obs["zone"]["center"][0] - mp[0]; dy = obs["zone"]["center"][1] - mp[1]; d = math.hypot(dx, dy) + 1e-6
            move = [dx / d, dy / d]
        # simple stuck jump
        sp = math.hypot(me["vel"][0], me["vel"][1])
        act["jump"] = bool(sp < 1.0 and (move[0] or move[1]) and me["energy"] > 30 and self.rng.random() < 0.15)
        act["boost"] = bool(not me["grounded"] and me["energy"] > 20 and sp < 3)
        act["move"] = move
        # aim / fire
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("by") is not None and e["by"] < 100:
                self.provoked = t
        if target is not None and not hunting:
            dd = math.hypot(target["pos"][0] - mp[0], target["pos"][1] - mp[1])
            if dd > 70 and t - getattr(self, "provoked", -99) > 10:
                target = None
        if target is not None and t - self.first_seen.get(target["id"], t) >= self.react:
            tp = target["pos"]; tv = target["vel"]
            dist = math.hypot(tp[0] - mp[0], tp[1] - mp[1])
            for i, w in enumerate(me["weapons"]):
                if not w:
                    continue
                k = w["kind"]
                if k == "gun":
                    tt = dist / w["stats"]["speed"] * self.lead
                elif k == "laser":
                    tt = 0.05
                else:
                    tt = 0
                ax = tp[0] + tv[0] * tt; ay = tp[1] + tv[1] * tt
                mz = mp[2] + me["stats"]["muzzle_height"]
                n = math.radians(self.noise)
                yaw = math.atan2(ay - mp[1], ax - mp[0]) + self.rng.gauss(0, n)
                pitch = math.atan2(tp[2] + 4.5 - mz, max(dist, 1)) + self.rng.gauss(0, n * 0.5)
                if k == "mortar":
                    v = w["stats"]["speed"]; g = w["stats"]["gravity"]
                    s = v ** 4 - g * (g * dist * dist + 2 * (tp[2] - mp[2] - 9) * v * v)
                    if s < 0:
                        continue
                    pitch = math.atan((v * v - math.sqrt(s)) / (g * dist))
                if i == 0 or "aim" not in act:
                    act["aim"] = [yaw, pitch]
                yerr = abs(angdiff(me["aim"][0], yaw))
                if k == "gun":
                    act["fire"][i] = yerr < 0.08 and dist < 230
                elif k == "laser":
                    ch = w.get("charge", 0)
                    if ch > 0:
                        act["fire"][i] = not (ch >= w["stats"]["charge"] and yerr < math.radians(self.noise) + 0.02)
                    else:
                        act["fire"][i] = w["cooldown"] <= 0 and dist < 270
                elif k == "lightning":
                    act["fire"][i] = dist < 44 and yerr < 0.25 and me["energy"] > 5
                elif k == "mortar":
                    act["fire"][i] = yerr < 0.05 and 30 < dist < 220
        else:
            if move != [0.0, 0.0]:
                act["aim"] = [math.atan2(move[1], move[0]) + 0.6 * math.sin(t), 0.0]
            for i, w in enumerate(me["weapons"]):
                if w and w["kind"] == "laser" and w.get("charge", 0) > 0:
                    act["fire"][i] = True
        return act

    def descend(self, f, mp):
        W = self.squad.W
        k0, d, p = f
        j, i = W.nav_cell(mp[0], mp[1])
        best = None; bv = d[j * W.nnx + i] if np.isfinite(d[j * W.nnx + i]) else 1e18
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                jj, ii = j + dj, i + di
                if 0 <= jj < W.nny and 0 <= ii < W.nnx:
                    v = d[jj * W.nnx + ii]
                    if v < bv:
                        bv = v; best = (di, dj)
        if best is None:
            # off-grid: head to nearest passable
            k = W.nearest_pass(mp[0], mp[1], 5)
            if k is None:
                return [0.0, 0.0]
            x, y = W.nav_xy(k)
            dx, dy = x - mp[0], y - mp[1]
        else:
            dx, dy = best
        n = math.hypot(dx, dy) or 1
        return [dx / n, dy / n]
