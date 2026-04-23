#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# -----------------------
# Trainer: deps + build
# -----------------------
cd "$ROOT/trainer"
source env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

python src/build/run_pyinstaller_trainer.py
deactivate

# -----------------------
# Painter: deps + build
# -----------------------
cd "$ROOT/painter"
source env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

python src/build/run_pyinstaller_workstation.py
deactivate

# -----------------------
# Bundle trainer into app
# -----------------------
APP="$ROOT/painter/dist/RetinaPainter.app"
APP_MACOS="$APP/Contents/MacOS"

# Sanity checks (fail fast, no mystery)
test -d "$APP" || { echo "ERROR: app not found at: $APP"; exit 1; }
test -d "$ROOT/trainer/src/dist/RetinaPainterTrainer" || { echo "ERROR: trainer onedir not found at: $ROOT/trainer/src/dist/RetinaPainterTrainer"; exit 1; }
test -f "$ROOT/trainer/src/dist/RetinaPainterTrainer/RetinaPainterTrainer" || { echo "ERROR: trainer executable missing inside onedir folder"; exit 1; }

# Ensure we don't accidentally launch a stale single-file helper
rm -f "$APP_MACOS/RetinaPainterTrainer"

# Copy the full onedir folder into the app as a bundle folder
rm -rf "$APP_MACOS/RetinaPainterTrainerBundle"
cp -R "$ROOT/trainer/src/dist/RetinaPainterTrainer" "$APP_MACOS/RetinaPainterTrainerBundle"

# Make sure main trainer binary is executable
chmod +x "$APP_MACOS/RetinaPainterTrainerBundle/RetinaPainterTrainer"

echo "OK: built workstation app at: $APP"

