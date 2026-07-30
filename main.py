import time
import numpy as np
import cv2
from camera import Camera
from tracker import HandTracker
from effect import FingertipEffect

WINDOW_TITLE = "Inter-Hand Stretched Visual Mesh"


def main():
    print("=" * 60)
    print("  MediaPipe Inter-Hand Minimalist String Engine")
    print("  Shortcuts:")
    print("    'f'   : Toggle Fullscreen mode ON / OFF")
    print("    's'   : Toggle hand Skeleton overlay ON / OFF")
    print("    't'   : Cycle interior camera filter (X-Ray, Thermal, Pixelate, Edge, Glitch)")
    print("    '1'-'5': Jump to specific camera filter mode")
    print("    'd'   : Toggle debug Telemetry HUD ON / OFF")
    print("    'q' / ESC : Exit application")
    print("=" * 60)

    # Initialize camera stream (native 640x480 resolution)
    camera = Camera(device_id=0, width=640, height=480)

    # Initialize hand tracker for multi-hand tracking (2 hands)
    tracker = HandTracker(
        model_path="hand_landmarker.task",
        max_num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize Inter-Hand Fingertip Visual Effect Engine
    effect = FingertipEffect(alpha_smooth=0.7, show_debug=True)

    # Setup resizable OpenCV window
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 720)

    show_skeleton = True  # Toggle with 's' key
    is_fullscreen = False  # Toggle with 'f' key

    prev_time = time.time()
    start_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Error: Unable to capture video frame.")
                break

            # Flip frame horizontally for intuitive mirrored self-view
            frame = cv2.flip(frame, 1)

            # Monotonic timestamp in milliseconds
            timestamp_ms = int((time.time() - start_time) * 1000)

            # Process frame through MediaPipe Hand Tracker
            detected_hands, result = tracker.process(
                frame, timestamp_ms=timestamp_ms
            )

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # 1. Render Inter-Hand Dual-Handle Stretched Visual Mesh & Filter
            frame = effect.render(frame, detected_hands)

            # 2. Draw debug hand landmark skeleton overlay (Toggleable via 's' key)
            if show_skeleton:
                frame = tracker.draw_landmarks(frame, result)

            # 3. Draw FPS overlay
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Aspect-Ratio Preserving Display Resizing
            try:
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user.")
                    break
                rect = cv2.getWindowImageRect(WINDOW_TITLE)
                win_w, win_h = rect[2], rect[3]
            except Exception:
                win_w, win_h = 0, 0

            if win_w > 0 and win_h > 0:
                frame_h, frame_w = frame.shape[:2]
                scale = min(win_w / frame_w, win_h / frame_h)
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)

                if new_w > 0 and new_h > 0:
                    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    x_off = (win_w - new_w) // 2
                    y_off = (win_h - new_h) // 2
                    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized_frame
                    cv2.imshow(WINDOW_TITLE, canvas)
                else:
                    cv2.imshow(WINDOW_TITLE, frame)
            else:
                cv2.imshow(WINDOW_TITLE, frame)

            # Keyboard shortcut listeners ('q', ESC, 'f', 's', 't', '1'-'5', 'd')
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Exiting application...")
                break
            elif key in (ord("f"), ord("F")):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, prop)
                print(f"Fullscreen Mode: {'ON' if is_fullscreen else 'OFF'}")
            elif key in (ord("s"), ord("S")):
                show_skeleton = not show_skeleton
                print(f"Skeleton Overlay: {'ON' if show_skeleton else 'OFF'}")
            elif key in (ord("t"), ord("T")):
                effect.cycle_filter()
            elif ord("1") <= key <= ord("5"):
                mode = key - ord("0")
                effect.set_filter_mode(mode)
            elif key in (ord("d"), ord("D")):
                effect.show_debug = not effect.show_debug

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        tracker.close()
        camera.release()
        print("Application shutdown clean.")


if __name__ == "__main__":
    main()
