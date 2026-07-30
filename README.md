# Inter-Hand Tracking & Optical Filter Lens Engine

[English](README.md) | [Tiếng Việt](README_VN.md)

Real-time dual-hand tracking application built with OpenCV and MediaPipe Tasks Vision (`HandLandmarker`). Supports selective touch-and-pull fingertip gating, minimalist string lines, and interactive optical camera filters inside the inter-hand shape.

---

## Features

- **Dual-Hand Tracking**: Real-time 3D tracking of 21 keypoints per hand.
- **Selective Touch-and-Pull Gating**: Connects only finger pairs that touch ($<55\text{px}$) and pull apart ($>85\text{px}$).
- **Minimalist String Lines**: Crisp thin string lines without bulky halos or node circles.
- **5 Interior Camera Filters**: Live optical lens inside the shape (`1`: X-Ray, `2`: Thermal, `3`: Pixelate, `4`: Edge Sketch, `5`: Glitch Shift).
- **Aspect-Ratio Preserving Window**: Drag borders to resize or press `f` for Fullscreen without video distortion.

---

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run application**:
   ```bash
   python main.py
   ```

---

## Keyboard Controls

| Key | Action |
| :--- | :--- |
| **`f`** | Toggle Fullscreen Mode ON / OFF |
| **`s`** | Toggle Hand Skeleton Overlay ON / OFF |
| **`t`** | Cycle Interior Camera Filter Mode |
| **`1` - `5`** | Jump to Filter Mode 1-5 |
| **`d`** | Toggle Telemetry HUD ON / OFF |
| **`q`** / **`ESC`** | Exit Application |
