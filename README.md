# Doodle Baseball Bot

A vision-only bot for the Google Doodle 2019 July 4th baseball game. It watches the screen, finds the ball with a small YOLO model, and times each swing with a learned model (v12.7). Best game: 3,878 runs, played over about 3.6 hours. The on-screen scoreboard stops at 999.

## Setup

Python 3.14 is what it was built on.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Play

```bash
python harness.py
```

This opens Chromium, starts the game, and plays until game over. It prints a JSON summary at the end.

Options:

- `--headless`: run without a visible window.
- `--record runs/game1`: save every frame and a per-frame log.
- `--max-seconds N`: stop after N seconds. The default is 20000.

The YOLO model runs on Apple Silicon (`mps`) by default. On other machines set `BALL_DEVICE=cpu` or `BALL_DEVICE=cuda`.

## Files

- `harness.py`: browser control, screen readers (ball shadow, pitcher's hat color, scoreboard), and the swing policy.
- `detect_yolo.py`: ball detector.
- `weights/ball.pt`: YOLO11n ball weights.
- `models/v12.7.pkl`: swing-timing model.
