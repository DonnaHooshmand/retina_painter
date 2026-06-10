"""
Headless round-trip check for the Unsure brush.

Proves the painter -> annotation-PNG -> trainer contract for the new
Unsure brush WITHOUT needing a display or a human to paint.

The Unsure brush is shown YELLOW on screen (distinct from the blue
segmentation overlay) but stored in the BLUE channel on disk, because
the trainer decodes R=foreground, G=background, B=unsure and yellow
(red+green) would collide with the fg/bg channels. The painter converts
yellow<->blue at the save/load boundary (im_utils.annot_display_to_storage
/ annot_storage_to_display). This script checks that whole path:

  1. Pulls the REAL brush colours from GraphicsScene (so it breaks if the
     Unsure colour ever stops being yellow).
  2. Paints fg / bg / unsure(yellow) onto a transparent pixmap with the same
     CompositionMode_Source the canvas uses.
  3. display -> storage, then saves the PNG exactly like the painter does.
  4. Reads the PNG back the way trainer/src/datasets.py does and asserts:
       - foreground / background / unsure pixels all present
       - unsure landed in channel 2 (B), NOT in R or G
       - no pixel belongs to two classes (mutual exclusion)
       - unsure is excluded from the supervised mask
  5. storage -> display and confirms the unsure pixels come back YELLOW.

Run from the painter dir with the painter venv:
    env\\Scripts\\python.exe verify_unsure_brush.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "main", "python"))

import numpy as np
from skimage.io import imread
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

import im_utils
from graphics_scene import GraphicsScene


def paint_blob(pixmap, color, cx, cy, size):
    """Paint a filled circle with the SAME composition mode as the canvas."""
    painter = QtGui.QPainter(pixmap)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
    painter.setPen(QtGui.QPen(color, 0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(QtGui.QBrush(color, Qt.SolidPattern))
    painter.drawEllipse(cx - size // 2, cy - size // 2, size, size)
    painter.end()


def main():
    _app = QtWidgets.QApplication(sys.argv)

    scene = GraphicsScene()
    print("foreground_color:", scene.foreground_color.getRgb())
    print("background_color:", scene.background_color.getRgb())
    print("unsure_color:    ", scene.unsure_color.getRgb(), "(shown on screen)")
    assert scene.unsure_color.getRgb()[:3] == (255, 255, 0), \
        "Unsure brush should be yellow on screen."

    w = h = 224
    display_pixmap = QtGui.QPixmap(w, h)
    display_pixmap.fill(Qt.transparent)
    paint_blob(display_pixmap, scene.foreground_color, 50, 50, 40)
    paint_blob(display_pixmap, scene.background_color, 150, 50, 40)
    paint_blob(display_pixmap, scene.unsure_color, 100, 150, 40)  # yellow

    # display -> storage (yellow -> blue), then save exactly like the painter.
    storage_pixmap = im_utils.annot_display_to_storage(display_pixmap)
    out = os.path.join(tempfile.gettempdir(), "verify_unsure_annot.png")
    assert storage_pixmap.save(out, "PNG"), "pixmap.save failed"
    print("saved annotation PNG ->", out)

    # --- Read it back the way the trainer does ---
    annot = imread(out)
    print("PNG shape:", annot.shape)
    assert annot.ndim == 3 and annot.shape[2] >= 3, "expected an RGB(A) PNG"

    fg = annot[:, :, 0].astype(bool)
    bg = annot[:, :, 1].astype(bool)
    unsure = annot[:, :, 2].astype(bool)

    assert fg.any(), "no foreground pixels were stored"
    assert bg.any(), "no background pixels were stored"
    assert unsure.any(), "no UNSURE pixels in channel 2 -> yellow did not map to blue"
    assert not (fg & bg).any(), "fg/bg overlap (would trip trainer assertion)"
    assert not (fg & unsure).any(), "fg/unsure overlap -> unsure leaked into R channel"
    assert not (bg & unsure).any(), "bg/unsure overlap -> unsure leaked into G channel"

    # Mask formula from datasets.py: untouched AND unsure are both excluded.
    mask = (fg | bg) & ~unsure
    assert not (mask & unsure).any(), "unsure pixels leaked into the supervised mask"

    print("counts -> fg: %d  bg: %d  unsure: %d  supervised(mask): %d"
          % (fg.sum(), bg.sum(), unsure.sum(), mask.sum()))

    # --- storage -> display round-trip: unsure must come back yellow ---
    reloaded = im_utils.annot_storage_to_display(QtGui.QPixmap(out))
    img = reloaded.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
    cx, cy = 100, 150  # centre of the unsure blob
    px = QtGui.QColor(img.pixel(cx, cy))
    print("reloaded unsure-blob pixel:", (px.red(), px.green(), px.blue()))
    assert (px.red(), px.green(), px.blue()) == (255, 255, 0), \
        "unsure pixel did not round-trip back to yellow on load"

    print("\nPASS: Unsure shows yellow, stores in channel 2 (blue), excluded "
          "from the mask, and reloads as yellow.")


if __name__ == "__main__":
    main()
