import cv2


class Camera:
    """Wrapper around OpenCV VideoCapture for webcam video streaming."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera with device ID {self.device_id}"
            )

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Read a frame from the webcam. Returns (success_flag, frame_matrix)."""
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """Release camera resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
