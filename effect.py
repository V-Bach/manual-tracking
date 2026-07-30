from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np
import cv2
from tracker import HandLandmarkData

# MediaPipe Fingertip Landmark Indices:
# Index 0: Thumb Tip (4)
# Index 1: Index Tip (8)
# Index 2: Middle Tip (12)
# Index 3: Ring Tip (16)
# Index 4: Pinky Tip (20)
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGERTIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Distance Thresholds (in pixels) for Per-Finger Touch-and-Pull Gating
TOUCH_THRESHOLD = 55.0        # Distance between Left & Right fingertip k to ARM interaction
PULL_APART_THRESHOLD = 85.0   # Distance required after arming to ACTIVATE connection for finger k
CLOSE_THRESHOLD = 50.0        # Distance to DEACTIVATE connection when finger k touches together again

# Theme Colors (BGR format) - Clean Minimalist Lines
COLOR_STRING_LINE = (255, 255, 255)   # Crisp Clean White String Line
COLOR_STRING_CYAN = (255, 255, 0)     # Neon Cyan String Line
COLOR_STRING_MAGENTA = (255, 0, 255)  # Neon Magenta String Line
COLOR_DOT_MARKER = (0, 255, 255)     # Bright Dot Marker for Fingertips

# Filter Modes
FILTER_NAMES = {
    1: "X-Ray / Negative Invert",
    2: "Thermal Heatmap (Inferno)",
    3: "Retro 8-Bit Pixelate",
    4: "Edge Line Art (Sketch)",
    5: "Cyberpunk RGB Glitch Shift",
}


class FingertipEffect:
    """
    Real-time inter-hand creative coding effect.
    Minimalist String Lines & Interactive Interior Camera Filter Engine:
      - Renders clean thin string lines without heavy neon halos or node circles.
      - Applies interactive video filters inside the shape bounded between hands.
    """

    def __init__(self, alpha_smooth: float = 0.7, show_debug: bool = True):
        self.alpha_smooth = alpha_smooth
        self.show_debug = show_debug
        self.filter_mode = 1  # Default: Mode 1 (X-Ray / Negative Invert, as in image.png)

        # Per-Finger-Pair Boolean Flags (for 5 finger pairs: Thumb, Index, Middle, Ring, Pinky)
        self.active_pairs: List[bool] = [False] * 5
        self.armed_pairs: List[bool] = [False] * 5
        self.pair_distances: List[float] = [0.0] * 5

        # History map storing smoothed fingertip coordinates: {"Left": array, "Right": array}
        self.smoothed_pts_map: Dict[str, np.ndarray] = {}

    def cycle_filter(self):
        """Cycle to the next interior camera filter."""
        self.filter_mode = (self.filter_mode % 5) + 1
        print(f"Switched to Filter Mode [{self.filter_mode}]: {FILTER_NAMES[self.filter_mode]}")

    def set_filter_mode(self, mode: int):
        """Set specific filter mode (1 to 5)."""
        if mode in FILTER_NAMES:
            self.filter_mode = mode
            print(f"Switched to Filter Mode [{self.filter_mode}]: {FILTER_NAMES[self.filter_mode]}")

    def _extract_raw_fingertips(self, hand_data: HandLandmarkData) -> List[Tuple[float, float]]:
        """Extract raw (px, py) coordinates for the 5 fingertip landmarks."""
        lm_map = {lm["id"]: (float(lm["px"]), float(lm["py"])) for lm in hand_data.landmarks}
        raw_pts = []
        for fid in FINGERTIP_IDS:
            if fid in lm_map:
                raw_pts.append(lm_map[fid])
            else:
                raw_pts.append((0.0, 0.0))
        return raw_pts

    def _smooth_points(self, hand_label: str, raw_pts: List[Tuple[float, float]]) -> np.ndarray:
        """Apply Exponential Moving Average (EMA) smoothing to fingertip coordinates."""
        current_arr = np.array(raw_pts, dtype=np.float32)

        if hand_label not in self.smoothed_pts_map:
            self.smoothed_pts_map[hand_label] = current_arr
        else:
            prev_arr = self.smoothed_pts_map[hand_label]
            smoothed_arr = self.alpha_smooth * current_arr + (1.0 - self.alpha_smooth) * prev_arr
            self.smoothed_pts_map[hand_label] = smoothed_arr

        return self.smoothed_pts_map[hand_label]

    def _apply_interior_filter(self, frame: cv2.Mat) -> cv2.Mat:
        """Apply the selected visual filter to the camera frame."""
        h, w, c = frame.shape

        if self.filter_mode == 1:
            # Mode 1: X-Ray / Negative Invert (matches image.png reference!)
            inverted = cv2.bitwise_not(frame)
            # Tint with cyan/magenta hue
            tint = np.zeros_like(frame)
            tint[:, :, 0] = 80   # Blue
            tint[:, :, 2] = 120  # Red
            return cv2.addWeighted(inverted, 0.85, tint, 0.15, 0)

        elif self.filter_mode == 2:
            # Mode 2: Thermal Heatmap (Inferno Colormap)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

        elif self.filter_mode == 3:
            # Mode 3: Retro 8-Bit Pixelate (Mosaic Lens)
            small_w = max(16, w // 12)
            small_h = max(12, h // 12)
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif self.filter_mode == 4:
            # Mode 4: Edge Line Art (Sketch Filter)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # Glowing cyan edge lines over dark background
            edges_bgr[:, :, 0] = cv2.add(edges_bgr[:, :, 0], 255)
            edges_bgr[:, :, 1] = cv2.add(edges_bgr[:, :, 1], 200)
            return cv2.addWeighted(frame, 0.3, edges_bgr, 0.7, 0)

        elif self.filter_mode == 5:
            # Mode 5: Cyberpunk RGB Glitch Shift
            glitched = frame.copy()
            # Shift Blue channel left
            glitched[:, :, 0] = np.roll(frame[:, :, 0], 12, axis=1)
            # Shift Red channel right
            glitched[:, :, 2] = np.roll(frame[:, :, 2], -12, axis=1)
            return glitched

        return frame

    def _render_mesh(self, frame: cv2.Mat, left_pts_int: np.ndarray, right_pts_int: np.ndarray):
        """
        Render Minimalist String Lines and Composite Interior Camera Filter.
        """
        active_pairs_data = []
        for k in range(5):
            if self.active_pairs[k]:
                l_pt = left_pts_int[k]
                r_pt = right_pts_int[k]
                avg_y = (l_pt[1] + r_pt[1]) / 2.0
                active_pairs_data.append((avg_y, k, l_pt, r_pt))

        if not active_pairs_data:
            return

        # Sort active pairs vertically from top to bottom
        active_pairs_data.sort(key=lambda item: item[0])

        active_ks = [item[1] for item in active_pairs_data]
        active_l_pts = [item[2] for item in active_pairs_data]
        active_r_pts = [item[3] for item in active_pairs_data]

        # Case A: Exactly 1 finger pair active (e.g. Index only)
        if len(active_pairs_data) == 1:
            l_pt = (int(active_l_pts[0][0]), int(active_l_pts[0][1]))
            r_pt = (int(active_r_pts[0][0]), int(active_r_pts[0][1]))

            # Clean thin string line
            cv2.line(frame, l_pt, r_pt, COLOR_STRING_LINE, 2, cv2.LINE_AA)

            # Minimalist fingertip dot markers (3px)
            cv2.circle(frame, l_pt, 4, COLOR_STRING_MAGENTA, -1, cv2.LINE_AA)
            cv2.circle(frame, r_pt, 4, COLOR_STRING_CYAN, -1, cv2.LINE_AA)
            return

        # Case B: Multi-finger active pairs (2, 3, 4, or 5 active fingers)
        # 1. Build Perimeter Polygon Loop
        perimeter_loop = active_l_pts + active_r_pts[::-1]
        perimeter_pts = np.array(perimeter_loop, dtype=np.int32)

        # 2. Composite Interior Camera Filter inside the polygon mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [perimeter_pts], 255)

        filtered_frame = self._apply_interior_filter(frame)

        # Composite filtered interior onto current frame
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        np.copyto(frame, filtered_frame, where=(mask_3d > 0))

        # 3. Draw Clean Minimalist String Boundary Lines (No heavy glow or halos!)
        # Perimeter polygon border string
        cv2.polylines(frame, [perimeter_pts], isClosed=True, color=COLOR_STRING_LINE, thickness=2, lineType=cv2.LINE_AA)

        # Inner cross-hand string lines
        for l_pt, r_pt in zip(active_l_pts, active_r_pts):
            l_tuple = (int(l_pt[0]), int(l_pt[1]))
            r_tuple = (int(r_pt[0]), int(r_pt[1]))
            cv2.line(frame, l_tuple, r_tuple, (220, 220, 220), 1, cv2.LINE_AA)

            # Subtle minimalist dot markers at active fingertips
            cv2.circle(frame, l_tuple, 4, COLOR_STRING_MAGENTA, -1, cv2.LINE_AA)
            cv2.circle(frame, r_tuple, 4, COLOR_STRING_CYAN, -1, cv2.LINE_AA)

    def _render_debug_overlay(self, frame: cv2.Mat):
        """Render debug telemetry HUD on screen."""
        if not self.show_debug:
            return

        active_finger_names = [FINGERTIP_NAMES[k] for k in range(5) if self.active_pairs[k]]
        filter_info = f"Filter Mode [{self.filter_mode}]: {FILTER_NAMES[self.filter_mode]}"

        lines = [
            f"Active Connections: {active_finger_names if active_finger_names else 'None'}",
            filter_info + "  (Press 't' or 1-5 to switch)",
            f"Distances: T:{int(self.pair_distances[0])} I:{int(self.pair_distances[1])} M:{int(self.pair_distances[2])} R:{int(self.pair_distances[3])} P:{int(self.pair_distances[4])}",
        ]

        y = frame.shape[0] - 80
        for idx, line in enumerate(lines):
            color = (0, 255, 0) if idx == 0 else ((255, 255, 0) if idx == 1 else (255, 255, 255))
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y += 22

    def render(self, frame: cv2.Mat, detected_hands: List[HandLandmarkData]) -> cv2.Mat:
        """
        Main render entry point. Gated selectively per finger pair with Minimalist Lines & Interior Filter.
        """
        if not detected_hands:
            self.active_pairs = [False] * 5
            self.armed_pairs = [False] * 5
            self.pair_distances = [0.0] * 5
            self.smoothed_pts_map.clear()
            self._render_debug_overlay(frame)
            return frame

        # Separate detected hands spatially across screen X
        left_hand_data: Optional[HandLandmarkData] = None
        right_hand_data: Optional[HandLandmarkData] = None

        if len(detected_hands) >= 2:
            sorted_hands = sorted(
                detected_hands,
                key=lambda h: np.mean([lm["px"] for lm in h.landmarks]) if h.landmarks else 0
            )
            left_hand_data = sorted_hands[0]
            right_hand_data = sorted_hands[1]
        elif len(detected_hands) == 1:
            hand = detected_hands[0]
            avg_x = np.mean([lm["px"] for lm in hand.landmarks]) if hand.landmarks else frame.shape[1] / 2
            if avg_x < frame.shape[1] / 2:
                left_hand_data = hand
            else:
                right_hand_data = hand

        active_labels = set()

        # Smooth and extract Left Hand fingertips
        left_pts_int: Optional[np.ndarray] = None
        if left_hand_data:
            active_labels.add("Left")
            raw_l = self._extract_raw_fingertips(left_hand_data)
            smoothed_l = self._smooth_points("Left", raw_l)
            left_pts_int = np.int32(np.round(smoothed_l))

        # Smooth and extract Right Hand fingertips
        right_pts_int: Optional[np.ndarray] = None
        if right_hand_data:
            active_labels.add("Right")
            raw_r = self._extract_raw_fingertips(right_hand_data)
            smoothed_r = self._smooth_points("Right", raw_r)
            right_pts_int = np.int32(np.round(smoothed_r))

        # Update Per-Finger Pair Boolean Logic
        if left_pts_int is not None and right_pts_int is not None:
            for k in range(5):
                l_pt = left_pts_int[k]
                r_pt = right_pts_int[k]
                dist = float(math.hypot(l_pt[0] - r_pt[0], l_pt[1] - r_pt[1]))
                self.pair_distances[k] = dist

                if not self.active_pairs[k]:
                    if dist < TOUCH_THRESHOLD:
                        self.armed_pairs[k] = True
                    if self.armed_pairs[k] and dist > PULL_APART_THRESHOLD:
                        self.active_pairs[k] = True
                else:
                    if dist < CLOSE_THRESHOLD:
                        self.active_pairs[k] = False
                        self.armed_pairs[k] = False
        else:
            self.active_pairs = [False] * 5
            self.armed_pairs = [False] * 5
            self.pair_distances = [0.0] * 5

        # Render connections ONLY for active finger pairs
        if any(self.active_pairs) and left_pts_int is not None and right_pts_int is not None:
            self._render_mesh(frame, left_pts_int, right_pts_int)

        # Clean up stale smoothing states
        stale_keys = [k for k in self.smoothed_pts_map if k not in active_labels]
        for k in stale_keys:
            del self.smoothed_pts_map[k]

        self._render_debug_overlay(frame)

        return frame
