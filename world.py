"""Static map model: rasterized height grids, line-of-sight, and grid navigation."""
import json, os, time, math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt, maximum_filter

NAV = 2  # nav grid step in meters (multiple of the 1 m terrain grid)


def _raster_poly(rings, x0, y0, nx, ny):
    """Even-odd rasterization of polygons (ring + holes) on the 1 m vertex grid.
    Returns (j0, i0, mask) for the bbox."""
    xs = np.concatenate([r[0::2] for r in rings]); ys = np.concatenate([r[1::2] for r in rings])
    i0 = max(int(math.floor(xs.min() - x0)) - 1, 0); i1 = min(int(math.ceil(xs.max() - x0)) + 2, nx)
    j0 = max(int(math.floor(ys.min() - y0)) - 1, 0); j1 = min(int(math.ceil(ys.max() - y0)) + 2, ny)
    if i1 <= i0 or j1 <= j0:
        return None
    W = i1 - i0; Hh = j1 - j0
    tog = np.zeros((Hh, W + 1), np.int32)
    for r in rings:
        px = np.asarray(r[0::2], float) - x0; py = np.asarray(r[1::2], float) - y0
        qx = np.roll(px, -1); qy = np.roll(py, -1)
        lo = np.minimum(py, qy); hi = np.maximum(py, qy)
        # rows j (cell center at y=j) crossing the edge: lo <= j < hi
        ja = np.ceil(lo).astype(int); jb = np.ceil(hi).astype(int)
        for k in np.nonzero(jb > ja)[0]:
            jr = np.arange(ja[k], jb[k])
            t = (jr - py[k]) / (qy[k] - py[k])
            xi = px[k] + t * (qx[k] - px[k])
            ci = np.clip(np.ceil(xi).astype(int) - i0, 0, W)
            rr = jr - j0
            ok = (rr >= 0) & (rr < Hh)
            np.add.at(tog, (rr[ok], ci[ok]), 1)
    inside = (np.cumsum(tog, axis=1)[:, :W] & 1).astype(bool)
    return j0, i0, inside


class World:
    def __init__(self, mapdir, mech_cfg, files=None):
        t0 = time.perf_counter()
        wj = next((f for f in (files or []) if f.endswith("world.json")), "v2/world.json")
        w = json.load(open(os.path.join(mapdir, wj)))
        tr = w["terrain"]
        self.x0, self.y0, self.nx, self.ny = tr["minX"], tr["minY"], tr["nx"], tr["ny"]
        self.bounds = w["bounds"]
        G = np.fromfile(os.path.join(mapdir, os.path.dirname(wj), tr["file"]), dtype="<i2").astype(np.float32)
        G = G.reshape(self.ny, self.nx) * float(tr.get("scale", 0.01))
        self.G = G
        H = G.copy()                                  # top of anything solid from the ground up
        BB = np.full_like(G, np.inf); BT = np.full_like(G, -np.inf)  # elevated (bridge) slab
        bldg = np.zeros(G.shape, bool)
        for p in w["prisms"]:
            rings = [np.asarray(p["ring"], float)] + [np.asarray(h, float) for h in p.get("holes", [])]
            r = _raster_poly(rings, self.x0, self.y0, self.nx, self.ny)
            if r is None:
                continue
            j0, i0, m = r
            sl = (slice(j0, j0 + m.shape[0]), slice(i0, i0 + m.shape[1]))
            g = G[sl]
            if not m.any():
                continue
            if p["kind"] == "bridge" or p["base"] > g[m].mean() + 3.0:
                bb = BB[sl]; bt = BT[sl]
                bb[m] = np.minimum(bb[m], p["base"]); bt[m] = np.maximum(bt[m], p["top"])
            else:
                h = H[sl]
                h[m] = np.maximum(h[m], p["top"])
                if p["top"] - np.median(g[m]) > 1.0:
                    bm = bldg[sl]; bm[m] = True
        self.H, self.BB, self.BT, self.bldg = H, BB, BT, bldg
        self.has_bridge = np.isfinite(BB)
        # walking surface at ground level (roofs are not used for navigation)
        S = G.copy()
        low_deck = self.has_bridge & (BB - G < 9.5)
        S[low_deck] = np.maximum(G[low_deck], BT[low_deck])
        S[bldg] = H[bldg]
        self.S = S
        # obstacles: building cells, and steps steeper than the mech can walk
        step = float(mech_cfg.get("step_up", 1.2))
        dh = np.zeros_like(S)
        for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = S[max(0, -dj):self.ny - max(0, dj), max(0, -di):self.nx - max(0, di)]
            b = S[max(0, dj):, max(0, di):][:a.shape[0], :a.shape[1]] if di >= 0 else S[max(0, dj):, :self.nx + di][:a.shape[0], :a.shape[1]]
            d = np.abs(a - b) / (1.414 if di and dj else 1.0)
            dh[max(0, -dj):max(0, -dj) + d.shape[0], max(0, -di):max(0, -di) + d.shape[1]] = np.maximum(
                dh[max(0, -dj):max(0, -dj) + d.shape[0], max(0, -di):max(0, -di) + d.shape[1]], d)
        obst = bldg | (dh > step * 1.05)
        m = 30  # map edge margin is a wall
        obst[:2, :] = obst[-2:, :] = True; obst[:, :2] = obst[:, -2:] = True
        self.obst = obst
        self.clear = distance_transform_edt(~obst).astype(np.float32)  # meters to nearest obstacle
        self.radius = float(mech_cfg.get("radius", 2.5))
        self._build_nav()
        self.build_s = time.perf_counter() - t0

    # ---------- grid helpers ----------
    def ij(self, x, y):
        i = int(round(x - self.x0)); j = int(round(y - self.y0))
        return min(max(j, 0), self.ny - 1), min(max(i, 0), self.nx - 1)

    def ground(self, x, y):
        j, i = self.ij(x, y)
        return float(self.G[j, i])

    def surface(self, x, y):
        j, i = self.ij(x, y)
        return float(self.S[j, i])

    def clearance(self, x, y):
        j, i = self.ij(x, y)
        return float(self.clear[j, i])

    # ---------- line of sight ----------
    def los(self, a, b, step=1.0):
        """True if the straight segment a->b (3D) clears buildings/terrain/bridges."""
        return bool(self.los_many(np.asarray([a], float), np.asarray([b], float), step)[0])

    def los_many(self, A, B, step=1.0, skip_end=2.0):
        """Vectorized LOS for segment pairs A[k]->B[k] (k x 3)."""
        A = np.asarray(A, float); B = np.asarray(B, float)
        d = B - A
        L = np.hypot(d[:, 0], d[:, 1])
        n = int(max(2, min(600, np.ceil(L.max() / step)))) if len(L) else 2
        t = (np.arange(1, n) / n)[None, :]  # exclude endpoints
        X = A[:, 0:1] + d[:, 0:1] * t; Y = A[:, 1:2] + d[:, 1:2] * t; Z = A[:, 2:3] + d[:, 2:3] * t
        # ignore samples within skip_end meters of either end (inside own/target cylinder)
        dist_a = t * L[:, None]; dist_b = (1 - t) * L[:, None]
        valid = (dist_a > skip_end) & (dist_b > skip_end)
        I = np.clip(np.rint(X - self.x0).astype(np.int32), 0, self.nx - 1)
        J = np.clip(np.rint(Y - self.y0).astype(np.int32), 0, self.ny - 1)
        blk = (Z < self.H[J, I]) | ((Z >= self.BB[J, I]) & (Z <= self.BT[J, I]))
        return ~np.any(blk & valid, axis=1)

    # ---------- navigation ----------
    def _build_nav(self):
        s = NAV
        cl = self.clear[::s, ::s]
        self.nny, self.nnx = cl.shape
        self.nS = self.S[::s, ::s]
        need = self.radius + 0.3
        self.npass = cl >= need
        # base cost: prefer some clearance
        self.ncost = (1.0 + 1.5 * np.clip((need + 3.0 - cl) / 3.0, 0, 1)).astype(np.float32)
        ny, nx = self.nny, self.nnx
        idx = np.arange(ny * nx).reshape(ny, nx)
        src = []; dst = []; ln = []
        S = self.nS
        step_ok = 1.2 * 1.05 * s * 1.5
        for di, dj, c in ((1, 0, 1), (0, 1, 1), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 0, 1), (0, -1, 1), (-1, -1, 1.4142), (-1, 1, 1.4142)):
            ys, xs = np.mgrid[0:ny, 0:nx]
            x2 = xs + di; y2 = ys + dj
            m = (x2 >= 0) & (x2 < nx) & (y2 >= 0) & (y2 < ny)
            a = idx[m]; b = (y2 * nx + x2)[m]
            pa = self.npass.ravel()[a] & self.npass.ravel()[b]
            dz = S.ravel()[b] - S.ravel()[a]
            ok = pa & (dz < step_ok)
            src.append(a[ok]); dst.append(b[ok]); ln.append(np.full(ok.sum(), c * s, np.float64))
        src = np.concatenate(src); dst = np.concatenate(dst); ln = np.concatenate(ln)
        G = csr_matrix((ln, (src, dst)), shape=(ny * nx, ny * nx))
        G.sort_indices()
        # recover per-entry destination and base length in CSR order
        self._nav_dst = G.indices.copy()
        self._nav_len = G.data.copy()
        self.navG = G
        self.cell_extra = np.zeros(ny * nx, np.float32)

    def nav_cell(self, x, y):
        i = int(round((x - self.x0) / NAV)); j = int(round((y - self.y0) / NAV))
        return min(max(j, 0), self.nny - 1), min(max(i, 0), self.nnx - 1)

    def nav_xy(self, k):
        j, i = divmod(int(k), self.nnx)
        return self.x0 + i * NAV, self.y0 + j * NAV

    def nearest_pass(self, x, y, rad=8):
        j, i = self.nav_cell(x, y)
        if self.npass[j, i]:
            return j * self.nnx + i
        best = None; bd = 1e9
        for dj in range(-rad, rad + 1):
            for di in range(-rad, rad + 1):
                jj, ii = j + dj, i + di
                if 0 <= jj < self.nny and 0 <= ii < self.nnx and self.npass[jj, ii]:
                    d = di * di + dj * dj
                    if d < bd:
                        bd = d; best = jj * self.nnx + ii
        return best

    def set_extra_cost(self, extra):
        """extra: flat array (nny*nnx) of multiplicative extra cost (>=0) per cell."""
        cost = self.ncost.ravel().astype(np.float64) + extra
        np.multiply(self._nav_len, cost[self._nav_dst], out=self.navG.data)

    def dijkstra_from(self, x, y, limit=400.0):
        k = self.nearest_pass(x, y)
        if k is None:
            return None
        d, p = dijkstra(self.navG, indices=k, limit=limit, return_predecessors=True)
        return k, d, p

    def path_to(self, field, goal_k):
        k0, d, p = field
        if goal_k is None or not np.isfinite(d[goal_k]):
            return None
        out = [goal_k]
        while out[-1] != k0 and p[out[-1]] >= 0 and len(out) < 2000:
            out.append(p[out[-1]])
        out.reverse()
        return out

    def walkable_line(self, a, b, need=None):
        """Can the mech center move straight from a to b (2D) with enough clearance?"""
        need = self.radius + 0.2 if need is None else need
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(L / 1.0))
        t = np.linspace(0, 1, n)
        X = a[0] + (b[0] - a[0]) * t; Y = a[1] + (b[1] - a[1]) * t
        I = np.clip(np.rint(X - self.x0).astype(int), 0, self.nx - 1)
        J = np.clip(np.rint(Y - self.y0).astype(int), 0, self.ny - 1)
        if np.any(self.clear[J, I] < need):
            return False
        s = self.S[J, I]
        return bool(np.all(np.diff(s) < 1.3))
