# Forward Model Ball Tracking

A stationary robot watches a ball roam a 3D arena and runs a **forward model**:
it predicts where the ball will be after the unavoidable sense -> think -> move
latency (180 ms), compares each prediction against reality, and corrects. While
tracking it turns in place toward that prediction to keep the ball centred in
view. Once its predictions are consistently accurate *and* a charge could reach
the ball, it **strikes**, then begins tracking afresh from where the charge
ended.

> Built on **MuJoCo** (real 3D physics + offscreen rendering, installs cleanly on
> Windows with no compiler) and **OpenCV** for the HUD. The Kalman forward model
> is written from scratch.

![tracking](assets/demo_tracking.png)
![strike](assets/demo_strike.png)

## The core idea: predict -> act -> correct

Sensing, thinking, and acting all take time. If the robot aims at where it last
*saw* the ball, it always acts on stale information and misses. The forward model
extrapolates the ball's state forward by the full pipeline latency so the robot
acts on where the ball *will be*.

```
sense (70 ms) + think (60 ms) + move (50 ms) = 180 ms horizon
```

| Stage | What happens |
|-------|--------------|
| Sense | a camera frame arrives, already 70 ms old |
| Detect | the ball is located in the image |
| Back-project | the pixel is mapped onto the floor plane -> world (x, y) |
| Predict | a Kalman filter extrapolates 180 ms ahead |
| Correct | when reality arrives, the residual (px error) updates the filter |
| Act | the robot turns toward the prediction; once the residual stays small and the ball is reachable, it strikes |

### Strike rules

* **Reach gate.** The robot strikes only if, assuming the ball keeps its
  estimated velocity, it will still be within reach (`strike_speed x
  strike_duration_s` + contact distance) when the charge window ends. The ball
  is slower than the charge, so this is exactly "a straight-line intercept
  exists". It uses the filter's estimate, never ground truth. While confident
  but out of reach the robot holds (counted as `held_frames`).
* **Contact every step.** Contact is checked on every step of the charge. The
  first contact is a hit and ends the charge; a miss is a charge that never
  touched the ball in its whole window.
* **Respawn.** After a strike the robot turns back to the arena centre *first*,
  then the ball is placed where that camera can actually see it.
* **Turning.** `robot.turn_rate_deg_s` caps how fast the robot turns in place
  (0 = fixed facing, the old behaviour). Turning never moves the ball: with no
  strikes the ball's path still depends only on the seed.

The on-screen **residual bar** is the heart of the demo: it starts large (the
prediction overshoots) and shrinks as the velocity estimate locks on.

## Why world-space state estimation

Detections happen in image pixels, but pixel motion is wildly non-linear under
perspective (a ball near the camera moves hundreds of px; the same ball far away
barely moves). So the filter estimates state in **world coordinates** on the
ground plane, where constant-velocity motion is an accurate model, then projects
predictions back into the image for display. This is the standard monocular
ground-plane assumption and is what makes the residual actually converge.

## Architecture

```
main.py            # live window / preview PNGs: renders around a Simulation
simulation.py      # one episode, no rendering: world + controller + strike logic
controller.py      # the predict/act/correct controller
config.py          # loads config.yaml, applies dotted-key overrides
config.yaml        # all tunable parameters

sim/
  arena.py         # builds the MuJoCo model (empty box; pillars wired for Phase 2)
  ball.py          # smooth random motion, wall bounce, robot keep-out zone
  robot.py         # base position + rate-limited yaw; turns in place, charges on a strike
  camera.py        # render + project (3D->2D) + unproject (pixel->ground plane)
  world.py         # owns MjModel/MjData, syncs mocap bodies each frame
  rng.py           # independent random streams: ball motion, respawns, detector noise

perception/
  base.py          # Detector interface (Phase 2 YOLO drops in here)
  sim_detector.py  # Phase 1: ground-truth projection + pixel noise

forward_model/
  kalman.py        # constant-velocity Kalman filter (world meters), from scratch
  latency.py       # sense delay line + prediction/residual scheduling
  residual.py      # smoothed residual + confidence gating

viz/
  overlay.py       # match box, +180 ms box, overshoot vector, residual bar, HUD

tools/
  batch.py         # no-render batch harness (many seeds, config overrides)
  regression.py    # seed-7 regression check / re-record

tests/             # pytest suite; tests/golden/seed7.json holds expected results
```

## Setup

Requires Python 3.12 (pinned in `.python-version`; MuJoCo ships prebuilt wheels for it).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

### Dependencies

`requirements.txt` and `requirements-dev.txt` are **lock files**: every package,
including transitive ones, is pinned to an exact version and hash, so every
machine and every CI run gets the identical stack. Do not edit them by hand.
Edit `requirements.in` / `requirements-dev.in` (direct dependencies, loose
ranges) and re-lock:

```bash
pip install pip-tools
pip-compile --generate-hashes --strip-extras --allow-unsafe -o requirements.txt requirements.in
pip-compile --generate-hashes --strip-extras --allow-unsafe -o requirements-dev.txt requirements-dev.in
```

## Run

```bash
# live window (press q or ESC to quit)
.venv\Scripts\python.exe main.py

# overhead debug inset (shows the true ball + robot positions)
.venv\Scripts\python.exe main.py --top

# headless: render N frames, save preview PNGs + residual stats
.venv\Scripts\python.exe main.py --headless 1500 --top
```

Tune behaviour in `config.yaml` (latency, ball speed/turn rate, Kalman noise,
strike threshold). No code changes needed.

## Tests, regression and batch runs

```bash
# unit tests + the seed-7 regression check
.venv\Scripts\python.exe -m pytest

# just the seed-7 check, printing which numbers moved
.venv\Scripts\python.exe -m tools.regression

# no-render harness: 50 seeds x 1500 frames in a few seconds
.venv\Scripts\python.exe -m tools.batch
.venv\Scripts\python.exe -m tools.batch --set forward_model.residual_threshold_px=6
.venv\Scripts\python.exe -m tools.batch --standing-still --json results.json
```

**The regression rule.** `tests/golden/seed7.json` holds the strikes, hits,
misses and residual statistics of the seed-7, 1500-frame headless run. A change
that is *not* meant to alter results must leave it passing. A change that *is*
meant to alter results re-records it in the same commit, so the diff shows
exactly which numbers moved:

```bash
.venv\Scripts\python.exe -m tools.regression --record
```

**Independent randomness.** Ball motion, respawns and detector noise each draw
from their own stream (`sim/rng.py`), derived from the run seed and the stream
name. With the robot standing still, the ball's path depends only on the seed
and the ball/arena settings: filter, detector or camera changes cannot move it.
Once the robot acts differently (strikes at another moment), the runs
legitimately diverge from that point.

CI (`.github/workflows/ci.yml`) installs the lock with hash checking, then runs
the tests and the 50-run harness on Ubuntu and Windows with rendering disabled.

## Status

**Phase 1 complete:** empty arena, smooth random ball, robot with a single
camera that turns in place to follow the ball, simulated detector, world-space
Kalman forward model, latency pipeline, residual-driven confidence, reach-gated
strike with per-step contact, and the full HUD.

**Phase 2 (next):** add pillars (`sim/arena.py` already accepts them) and swap
`perception/sim_detector.py` for a real YOLO detector behind the same interface.
