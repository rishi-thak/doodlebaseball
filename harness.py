"""Play the 2019 July 4th Doodle batting game with a pluggable ball detector.

    python harness.py [--record runs/x] [--headless] [--max-seconds 20000]
"""
import argparse
import asyncio
import base64
import collections
import hashlib
import importlib
import json
import pathlib
import pickle
import sys
import time

import cv2
import numpy as np
from playwright.async_api import async_playwright

URL = "https://google.com/logos/2019/july4th19/r6/july4th19.html"
W, H = 960, 540

def board(frame):
    """Scoreboard box (x, y, w, h), or None when it isn't at batter-view size (camera panning
    after a hit). The board drifts a few px between pitches, so locate its dark panel each frame."""
    # < 40: at night the stands behind the board get dark enough to merge with it at < 60.
    dark = (frame[40:240, 250:710].max(axis=2) < 40).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return None
    x, y, w, h, _ = stats[1 + int(np.argmax(stats[1:, 4]))]
    return (x + 250, y + 40, w, h) if 185 <= w <= 215 and 33 <= h <= 45 else None


_yy, _xx = np.mgrid[-16:17, -16:17]
_RING = (np.hypot(_xx, _yy) >= 11) & (np.hypot(_xx, _yy) <= 14)


def lives(frame, box):
    """Lit ball icons in the board's left half. Judged by shape, not brightness: a lit icon has a
    red ring outline (ring score 0.24-0.62), an X sits on a filled disc with no ring (0.00-0.02).
    "Red" is relative to green, because the scene dims at dusk (~200 s into a game)."""
    x, y, w, h = box
    lit = 0
    for i in range(3):
        cx, cy = int(x + w * (0.1175 + 0.155 * i)), int(y + h / 2)
        p = frame[cy - 16:cy + 17, cx - 16:cx + 17].astype(int)
        if p.shape[:2] != _RING.shape:
            continue
        red = (p[..., 2] > 1.6 * p[..., 1] + 10) & (p[..., 2] > 90)
        lit += int(red[_RING].mean() > 0.12)
    return lit


# The score is 3 digits in a 5x7 LED dot font, read by nearest Hamming distance. All ten templates
# are copied from real frames (6 and 9 replaced earlier guesses that misread 9 as 3).
_FONT = {0: "01110100011000110001100011000101110", 1: "00100011000010000100001000010011111",
         2: "01110100010000100010001000100011111", 3: "01110100010000100110000011000101110",
         4: "00010001100101010010111110001000010", 5: "11111100001000001110000011000101110",
         6: "01110100011000010110100011000101110", 7: "11111000010001000100001000010000100",
         8: "01110100011000101110100011000101110", 9: "01110100011000100111000011000101110"}


def score(frame, box):
    """Runs on the scoreboard, or None if a digit is too far from every template."""
    x, y, w, h = box
    # LEDs dim at night (brightest ~186 vs ~255 by day), so "lit" is relative to the brightest LED.
    # Only blue pixels count: a sprite in front of the board can be pure white.
    d = frame[y:y + h, x + w // 2:x + w].astype(int)
    leds = d[..., 0][d[..., 0] > d[..., 2] + 40]
    lit_at = max(110, 0.65 * np.percentile(leds, 99)) if leds.size else 170
    total = 0
    for i in range(3):
        bits = ""
        for row in range(7):
            for col in range(5):  # dot centres, as fractions of the board box
                px = int(x + w * (0.5917 + 0.138 * i + 0.0221 * col))
                py = int(y + h * (0.172 + 0.105 * row))
                bits += "1" if frame[py - 1:py + 2, px - 1:px + 2, 0].max() > lit_at else "0"
        dist, digit = min((sum(a != b for a, b in zip(bits, t)), d) for d, t in _FONT.items())
        if dist > 2:
            return None
        total = total * 10 + digit
    return total


class Game:
    def __init__(self):
        self.latest = None  # (arrival_time, frame, jpeg_bytes)
        self.new_frame = asyncio.Event()

    async def start(self, pw, headless=False):
        self.browser = await pw.chromium.launch(headless=headless)
        self.page = await self.browser.new_page(viewport={"width": W, "height": H})
        await self.page.goto(URL, wait_until="networkidle")
        await asyncio.sleep(1.5)
        await self.page.mouse.click(W // 2, H // 2)  # play button -> intro + tutorial
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", self._on_frame)
        await self.cdp.send("Page.startScreencast",
                            {"format": "jpeg", "quality": 80, "maxWidth": W, "maxHeight": H})
        await asyncio.sleep(8)  # intro + tutorial; Space is ignored before the tap hint shows

    def _on_frame(self, ev):
        asyncio.ensure_future(self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]}))
        jpg = base64.b64decode(ev["data"])
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        self.latest = (time.perf_counter(), frame, jpg)
        self.new_frame.set()

    async def frame(self):
        # Chrome stops painting a covered headed window; fail loudly instead of hanging.
        await asyncio.wait_for(self.new_frame.wait(), 3)
        self.new_frame.clear()
        return self.latest

    async def swing(self):
        # Raw CDP keyDown returns as soon as the page has the event; keyUp needn't block the loop.
        key = {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32}
        await self.cdp.send("Input.dispatchKeyEvent", {"type": "keyDown", **key})
        asyncio.ensure_future(self.cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", **key}))

    async def close(self):
        await self.browser.close()


def shadow(frame):
    """The ball's shadow (x, y) in the pitch corridor, or None: reddish tablecloth darker than
    min(190, 0.8 x this frame's median red), in a wide ellipse. The 190 cap is the daytime rule (a
    lob's high ball casts a faint shadow that a pure 0.72x rule missed); the relative part is
    needed at night, when the whole cloth is as dark as a daytime shadow."""
    roi = frame[250:540, 400:560].astype(int)
    g, r = roi[..., 1], roi[..., 2]
    med = np.median(r)
    mask = ((r < min(190, 0.8 * med)) & (r > 0.35 * med) & (r - g > 15)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    blobs = [(stats[k][4], cent[k][0] + 400, cent[k][1] + 250) for k in range(1, n)
             if stats[k][4] >= 20 and stats[k][2] >= 1.4 * stats[k][3]]
    return max(blobs)[1:] if blobs else None


def hat_features(frame):
    """Pitcher's hat: hue histogram of saturated pixels plus white fraction (19 numbers). The hat
    changes color 1-2 s before a pitch and the color tells the pitch type; timing_model.py learns
    the mapping, so no colors are named here."""
    hsv = cv2.cvtColor(frame[155:180, 462:503], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    sat = (s > 100) & (v > 100)
    hist = np.bincount(h[sat] // 10, minlength=18)[:18] / len(hsv)
    return np.append(hist, ((s < 40) & (v > 215)).mean())


HAT_LOOKBACK = 0.5  # s before release; the hat already shows the coming pitch's color
EARLY = 0.08  # s of ball flight the timing model watches before committing (fastest target: 0.12 s)


# Fast pitches (swing due < FAST_WINDOW after release): aim FAST_MARGIN early. Their window is
# lopsided: on logged fastballs, swings 10-60 ms before the label never struck out, while +15..+22 ms
# late struck out ~1 in 3 deep in a game. Aim where a miss is cheap, not at the unbiased estimate.
FAST_WINDOW, FAST_MARGIN = 0.12, 0.015
HALF_FRAME = 0.008  # fire on the frame nearest the scheduled time, not the first one after it (+8 ms avg)
EARLY2 = 0.12  # second look: re-predict with more flight when the swing isn't due yet


def refine_features(hat, pts, t_game, k):
    """Refine model input, shared by training and play: hat + the first EARLY2 s of flight
    ([x, y, r, dy/dt, dr/dt, d2y/dt2], rates both time-normalized and raw) + game time.
    The longer window separates pitches whose first 0.08 s look identical but whose ideal swing
    differs by ~50 ms (seen on fastballs)."""
    dt, x, y, r = np.array(pts, float).T
    f = np.zeros(6)
    f[:3] = x[0], y[0], r[0]
    if len(pts) >= 3:
        f[3], f[4] = np.polyfit(dt, y, 1)[0], np.polyfit(dt, r, 1)[0]
    if len(pts) >= 4:
        f[5] = np.polyfit(dt, y, 2)[0]
    norm = f.copy()
    norm[3:6] *= np.exp(-k * t_game)
    return np.concatenate([hat, norm, f[3:6], [t_game]])


def flight_features(pts):
    """First EARLY s of the ball after release, [(dt, x, y, r)] -> release x, y, r and the
    rates dy/dt, dr/dt. The hat says the pitch type; these say how fast this one is."""
    dt, x, y, r = np.array(pts, float).T
    if len(pts) < 3:
        return np.array([x[0], y[0], r[0], 0.0, 0.0])
    return np.array([x[0], y[0], r[0], np.polyfit(dt, y, 1)[0], np.polyfit(dt, r, 1)[0]])


# The shadow is the ball's ground position, so its y is pure depth (ball y mixes in height).
# Fit from 115 contact swings: contact when (PLATE_Y - shadow_y) / shadow_speed is about 0.09-0.25 s,
# late misses at 0.07-0.09 s.
PLATE_Y, TTC_SWING = 520, 0.12


class Policy:
    """model given ("learned"): EARLY s after release, predict the swing time from the hat color
    seen HAT_LOOKBACK s before release, the ball's first EARLY s of flight and the game time (the
    game speeds up), then swing at that time whether or not the ball is still visible.
    No model ("rules"): ball visible -> swing when its predicted y, `lead` s ahead, reaches y_hit;
    ball lost mid-pitch (ghost pitch) -> swing when the shadow's time to the plate drops to
    TTC_SWING, extrapolating its last speed if it drops out too."""

    def __init__(self, y_hit, lead, model=None, t_start=0.0):
        self.y_hit, self.lead, self.model, self.t_start = y_hit, lead, model, t_start
        # Per-model switches, so an older version can be replayed exactly (v12 had neither).
        self.half_frame = (model or {}).get("half_frame", HALF_FRAME)
        self.fast_margin = (model or {}).get("fast_margin", FAST_MARGIN)
        self.hats = collections.deque(maxlen=300)  # (t, hat features), ~5 s
        self.swing_at = None  # learned mode: predicted swing time for the current pitch
        # The scheduled swing lives outside the track: a stray detection that resets the track
        # right before swing time (loopy late-game breaking balls) must not cancel it.
        self.scheduled = None
        self.refine_at = None  # time of the second look (EARLY2), when one is pending
        self.first_checked = False  # fast-pitch path consulted for this pitch
        self.decisions = []  # (t decided, release t0, path, swing time) - for logs and the replay video
        self.last_swing = -9.0
        self.track = []  # [(t, x, y, r)] ball, current pitch
        self.shadows = []  # [(t, y)] shadow, current pitch
        self.swung = False
        self.ghost = False  # this pitch's swing came from the shadow while the ball was unseen
        self.pitches = []  # (swung, ghost) for each finished pitch

    def _pitch(self):
        # A pitch moves down the screen; batted balls go up, and r<5 blobs are runners/fielders.
        # Not "starts small": high lobs enter from the top already at r~17.
        tr = self.track
        # 10 px: on ghost pitches the color detector loses the ball after only ~16 px.
        return len(tr) >= 3 and tr[0][3] >= 5 and tr[-1][2] - tr[0][2] >= 10

    def _end(self):
        reached = self.track and (max(p[3] for p in self.track) >= 15 or any(y >= 450 for _, y in self.shadows))
        if self._pitch() and reached:
            self.pitches.append((self.swung, self.ghost))
        self.track, self.shadows, self.swung, self.ghost, self.swing_at = [], [], False, False, None
        self.refine_at = None
        self.first_checked = False

    def _speed_t(self, t):
        """Game time for the speedup: held at the model's t_sat (the game stops speeding up at
        ~750 s, issue #62). Models without t_sat (v12d and older) replay unchanged."""
        return min(t - self.t_start, self.model.get("t_sat") or float("inf"))

    def _clock(self, t):
        """Game-time feature, if the model was trained with one."""
        return [t - self.t_start] if self.model.get("time_feature") else []

    def _shadow_ttc(self, t):
        if len(self.shadows) < 3:
            return float("inf")
        (t0, y0), (t1, y1) = self.shadows[-3], self.shadows[-1]
        v = (y1 - y0) / max(t1 - t0, 1e-3)
        if v < 150:  # not moving toward the plate
            return float("inf")
        return (PLATE_Y - (y1 + v * (t - t1))) / v  # dead-reckon across shadow dropouts

    def update(self, t, det, shd=None, hat=None):
        fired = self._update(t, det, shd, hat)
        if fired:
            self.last_swing = t
        return fired

    def _update(self, t, det, shd, hat):
        if hat is not None:
            self.hats.append((t, hat))
        if self.scheduled is not None and t >= self.scheduled - self.half_frame:
            self.scheduled = None
            if t - self.last_swing >= 1.2:
                # A ghost pitch's track can end (0.4 s with no ball or shadow) before its swing is
                # due. Mark "swung" only on a live track: on an empty one the flag outlived the pitch
                # and blocked every swing on the next pitch (4 of 29 strikes in v12r/d/e games).
                if self.track:
                    self.swung, self.ghost = True, det is None
                return True
        if det is not None:
            x, y, r = det[:3]
            if self.track and (y < self.track[-1][2] - 5  # moved back up: different ball
                               or t - self.track[-1][0] > 0.3):  # long gap: e.g. a lob seen rising, then dropping
                self._end()
            self.track.append((t, x, y, r))
        if self._pitch() and shd is not None and (
                not self.shadows or 0 <= shd[1] - self.shadows[-1][1] < 60):  # forward, no jumps
            self.shadows.append((t, shd[1]))
        if not self.track:
            return False
        if t - max(self.track[-1][0], self.shadows[-1][0] if self.shadows else 0) > 0.4:
            self._end()
            return False
        # After a hit, the batted ball can look like a new pitch: no second swing within 1.2 s.
        if self.swung or t - self.last_swing < 1.2:
            return False
        if self.model is not None:
            t0 = self.track[0][0]
            (ta, xa, ya, ra) = self.track[0]
            # Lob fast path: decide at the first detection (confirmed moving down by the second).
            if (self.swing_at is None and self.scheduled is None and "lob" in self.model and self.hats
                    and len(self.track) >= 2 and ya < 240 and ra >= 14 and self.track[1][2] > ya):
                hat = min(self.hats, key=lambda h: abs(h[0] - (ta - HAT_LOOKBACK)))[1]
                speedup = np.exp(-self.model["k"] * self._speed_t(ta))
                pred = float(self.model["lob"].predict([np.concatenate([hat, [xa, ya, ra], self._clock(ta)])])[0])
                self.swing_at = self.scheduled = ta + pred * speedup
                self.decisions.append((t, ta, "lob", self.swing_at))
                if t >= self.scheduled - self.half_frame:
                    self.scheduled = None
                    self.swung, self.ghost = True, det is None
                    return True
            # Fast-pitch path: late in a game a fastball's ideal swing comes before EARLY. Ask the
            # first-detection model once, at the second detection; commit only if it says "soon".
            # Checked once, at the first detection (2nd or 3rd) that confirms downward motion: right
            # after release the ball can sit still for a frame, and a 2nd-detection-only check
            # then never ran (late fastballs fell through to the main model, too late).
            if (self.swing_at is None and self.scheduled is None and "first" in self.model and self.hats
                    and not self.first_checked and 2 <= len(self.track) <= 3 and ra >= 5
                    and self.track[-1][2] > ya):
                self.first_checked = True
                hat = min(self.hats, key=lambda h: abs(h[0] - (ta - HAT_LOOKBACK)))[1]
                speedup = np.exp(-self.model["k"] * self._speed_t(ta))
                feats = np.concatenate([hat, [xa, ya, ra], self._clock(ta)])
                if self.model.get("first_speed"):  # ball speed from release to this detection (issue #70)
                    tb, _, yb, rb = self.track[-1]
                    feats = np.concatenate([feats, [(yb - ya) / (tb - ta) * speedup, (rb - ra) / (tb - ta) * speedup,
                                                    (tb - ta) / speedup]])
                pred = float(self.model["first"].predict([feats])[0]) * speedup
                if "first_cal" in self.model:  # isotonic correction fit on cross-validated predictions
                    pred = float(self.model["first_cal"].predict([pred])[0])
                if pred < EARLY + 0.02:
                    self.swing_at = self.scheduled = ta + pred - (self.fast_margin if pred < FAST_WINDOW else 0)
                    self.decisions.append((t, ta, "fast", self.swing_at))
                    if t >= self.scheduled - self.half_frame:
                        self.scheduled = None
                        self.swung, self.ghost = True, det is None
                        return True
            if (self.swing_at is None and self.scheduled is None and self.refine_at is None and self._pitch()
                    and self.hats and t - t0 >= EARLY):
                hat = min(self.hats, key=lambda h: abs(h[0] - (t0 - HAT_LOOKBACK)))[1]
                pts = [(p[0] - t0, *p[1:]) for p in self.track if p[0] - t0 <= EARLY]
                feats = np.concatenate([hat, flight_features(pts)])
                # The game speeds up over time: rates in, swing time out, all time-normalized.
                speedup = np.exp(-self.model["k"] * self._speed_t(t0))
                feats[22:24] *= speedup  # timing_model.RATES
                feats = np.concatenate([feats, self._clock(t0)])
                pred = t0 + float(self.model["model"].predict([feats])[0]) * speedup
                if pred - t0 < FAST_WINDOW:
                    pred -= self.fast_margin
                if "refine" in self.model and pred > t0 + EARLY2 + 0.03:
                    self.refine_at = t0 + EARLY2  # not due yet: look again with more flight
                else:
                    self.swing_at = pred
                    self.decisions.append((t, t0, "main", self.swing_at))
            if self.swing_at is None and self.refine_at is not None and t >= self.refine_at:
                self.refine_at = None
                hat = min(self.hats, key=lambda h: abs(h[0] - (t0 - HAT_LOOKBACK)))[1]
                pts = [(p[0] - t0, *p[1:]) for p in self.track if p[0] - t0 <= EARLY2]
                feats = refine_features(hat, pts, self._speed_t(t0), self.model["k"])
                speedup = np.exp(-self.model["k"] * self._speed_t(t0))
                self.swing_at = t0 + float(self.model["refine"].predict([feats])[0]) * speedup
                self.decisions.append((t, t0, "refine", self.swing_at))
            if self.swing_at is not None and self.scheduled is None and not self.swung:
                self.scheduled = self.swing_at
                if t >= self.scheduled - self.half_frame:  # already due
                    self.scheduled = None
                    self.swung, self.ghost = True, det is None
                    return True
            return False
        if det is not None:
            if not self._pitch() or det[2] < 10:
                return False
            (t0, _, y0, _), (t1, _, y1, _) = self.track[-3], self.track[-1]
            vy = (y1 - y0) / max(t1 - t0, 1e-3)
            self.swung = bool(det[1] + vy * self.lead >= self.y_hit)
            return self.swung
        # Ball gone mid-flight (ghost pitch): time the swing off the shadow.
        if self._pitch() and t - self.track[-1][0] > 0.08 and self._shadow_ttc(t) <= TTC_SWING:
            self.swung = self.ghost = True
        return self.swung


async def main(args):
    detect = importlib.import_module(f"detect_{args.detector}").detect
    model_bytes = pathlib.Path(args.model).read_bytes() if args.policy == "learned" else b""
    model = pickle.loads(model_bytes) if model_bytes else None
    model_id = hashlib.sha1(model_bytes).hexdigest()[:8] if model_bytes else ""  # the model actually played
    policy = Policy(args.y_hit, args.lead, model)
    rec = pathlib.Path(args.record) if args.record else None
    if rec:
        (rec / "frames").mkdir(parents=True, exist_ok=True)
        log = open(rec / "log.jsonl", "w")
        (rec / "config.json").write_text(json.dumps(  # written up front: a crashed game has no summary
            {"detector": args.detector, "policy": args.policy, "argv": sys.argv[1:]}))
    game = Game()
    latencies, strikes, score_now = [], 0, 0
    async with async_playwright() as pw:
        await game.start(pw, headless=args.headless)
        t_start, n, zero_since = time.perf_counter(), 0, None
        policy.t_start = t_start
        last_pitch = t_start
        lives_now, started, last_start_press, cand, cand_t = 3, False, -99, None, 0
        ended_by = "max_seconds"
        while time.perf_counter() - t_start < args.max_seconds:
            try:
                t_arr, frame, jpg = await game.frame()
            except asyncio.TimeoutError:  # screen went static: the game-over card
                ended_by = "static_screen"
                break
            dets = detect(frame)
            det = dets[0] if dets else None
            started = started or det is not None
            if not started and t_arr - last_start_press > 8:  # Space leaves the tutorial
                await game.swing()
                last_start_press = t_arr
            shd = shadow(frame)
            swing = policy.update(t_arr, det, shd, hat_features(frame))
            if swing:
                await game.swing()
                latencies.append(time.perf_counter() - t_arr)
            box = board(frame)
            sc = score(frame, box) if box else None
            reading = (lives(frame, box), sc) if sc is not None else None
            if reading and reading != cand:
                cand, cand_t = reading, t_arr
            # Trust a reading once it has held 0.5 s: sprites passing over the board cause brief
            # misreads. Lives only go down; a "lit" read after an X is a red sprite in front of it.
            if reading and reading == cand and t_arr - cand_t >= 0.5:
                lv, sc = reading
                if lv < lives_now:
                    strikes += lives_now - lv
                    lives_now = lv
                    print(f"strike -> lives {lv}  (pitches so far {len(policy.pitches)})")
                if sc > score_now:  # runs only go up; a lower read is a misread
                    print(f"runs {score_now} -> {sc}")
                    score_now = sc
            if rec:
                name = f"{n:06d}.jpg"
                (rec / "frames" / name).write_bytes(jpg)
                log.write(json.dumps({"i": n, "t": round(t_arr - t_start, 4), "frame": name,
                                      "dets": [list(map(float, d)) for d in dets],
                                      "shadow": [float(v) for v in shd] if shd else None,
                                      "swing": swing, "ghost": swing and policy.ghost,
                                      "board": reading}) + "\n")
            n += 1
            zero_since = (zero_since or t_arr) if lives_now == 0 else None
            last_pitch = t_arr if policy.track else last_pitch
            # Game over: lives 0 and no pitch for 8 s. Lives alone once misread at dusk and ended
            # games that were still going.
            if zero_since and t_arr - zero_since > 3 and t_arr - last_pitch > 8:
                ended_by = "game_over"
                break
        shot = (rec or pathlib.Path(".")) / "final.png"
        await game.page.screenshot(path=str(shot))
        await game.close()
    elapsed = time.perf_counter() - t_start
    pitches = len(policy.pitches)
    summary = {"detector": args.detector, "policy": args.policy, "model": model_id, "ended_by": ended_by,
               "duration_s": round(elapsed, 1), "y_hit": args.y_hit, "lead": args.lead,
               "runs": score_now, "pitches": pitches, "swung": sum(s for s, _ in policy.pitches),
               "ghost_swings": sum(g for _, g in policy.pitches), "strikes": strikes,
               "contact_rate": round((pitches - strikes) / pitches, 3) if pitches else None,
               "fps": round(n / elapsed, 1),
               "swing_latency_ms_p50": round(1000 * float(np.median(latencies)), 1) if latencies else None,
               "final_screenshot": str(shot)}
    print(json.dumps(summary))
    if rec:
        (rec / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--record")
    ap.add_argument("--y-hit", type=float, default=380)
    ap.add_argument("--lead", type=float, default=0.06)
    ap.add_argument("--max-seconds", type=float, default=20000)
    ap.add_argument("--policy", choices=["learned", "rules"], default="learned")
    ap.add_argument("--model", default="models/v12.7.pkl")
    ap.add_argument("--headless", action="store_true")
    asyncio.run(main(ap.parse_args()))
