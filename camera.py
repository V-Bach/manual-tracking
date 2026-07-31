import threading
import cv2


class Camera:
    """
    High-performance multithreaded VideoCapture wrapper.
    Uses DirectShow (cv2.CAP_DSHOW) on Windows to eliminate MSMF frame errors and runs frame capture
    in a dedicated background thread for 0ms read latency and maximum FPS.
    """

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.device_id = device_id
        self.width = width
        self.height = height

        # Attempt DirectShow backend on Windows to fix MSMF errors & boost FPS
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera with device ID {self.device_id}"
            )

        # Set resolution & buffer properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initial frame read
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        # Start dedicated background thread for 0ms capture latency
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Background thread loop capturing frames continuously."""
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        """Read the latest captured frame instantly without CPU blocking."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        """Release camera resources cleanly."""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
