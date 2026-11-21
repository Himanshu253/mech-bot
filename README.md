# cover-and-arc: Mech Royale bot

A Python bot for the mechfight/1 battle royale. It fights from cover with a laser and gun, retreats to heal, and stays ahead of the zone.

## Files

| File | What it does |
| --- | --- |
| `Dockerfile` | `python:3.13.1-slim` plus pinned `numpy==2.2.6` and `scipy==1.15.3`, installed system-wide. |
| `bot.py` | The bot: perception, fight-or-retreat decisions, cover and peek positions, aiming, dodging, loot, upgrades, zone. |
| `world.py` | Map model built at startup from `/arena` (about 0.3 s): height grids, line-of-sight rays, 2 m path grid with Dijkstra. |
| `mechfight.py` | The kit's wrapper, unchanged. |
| `sim/` | Local simulator for testing (not copied into the image). |


## How it plays

- **Loadout:** keeps the gun and adds a laser (lightning or mortar if no laser is around). Weapon preferences are `WEAPON_VALUE` and `PAIR_BONUS` at the top of `bot.py`.
- **Fighting:** compares its time-to-kill with the time it would take the visible enemies to kill it, then engages, takes cover or retreats. With a laser it charges behind cover, steps out to fire, and ducks back while it recharges.
- **Cover:** a spot only counts as cover if no enemy eye can see the top of the mech.
- **Healing:** while retreating it holds fire, because dealing damage resets the regeneration timer.
- **Zone:** it plans against the circle the zone is shrinking to, not just the current one.
- **Loot:** when nothing is threatening, it picks up weapons, cubes, crates, drones and supply drops.
- **Upgrade order:** HP, armor, laser, then gun (`UPGRADE_SCORE`).
- **Mode detection:** if mechs die that it didn't kill, it assumes a free-for-all and plays more cautiously.

## Local simulator

The simulator approximates protocol.md: physics, weapons, vision, zone, NPCs, crates and loot. House bots hunt as a squad, weakest first, like in practice. It is **not** the real server. In it the kit's starter bot scores about 0.44 versus 1.15 on the real board, so its house bots are harsher than the real ones. Use it to compare versions, not to predict scores.

```sh
export MECHKIT=/path/to/mechfight-kit
python3 sim/batch.py 12 practice bot 1 100   # 12 matches (seeds 100-111), 1 worker
python3 sim/batch.py 12 practice starter      # reference
python3 sim/batch.py 3 ffa bot 1 300          # 4 copies of the bot in a free-for-all
python3 -c "import sys; sys.path.insert(0,'sim'); import run; run.trace(100, every=2)"   # per-second log of one match
```

To compare a variant, copy `bot.py` to `bot_x.py`, change it, and run `batch.py` with module `bot_x` on the same seeds. Expect about ±0.6 noise over 12 matches.
