import threading
from threading import Lock

import cv2

from logger import get_logger

logger = get_logger(__name__)


class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()

    def __init__(self, rtsp_link):
        capture = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(
            target=self.rtsp_cam_buffer, args=(capture,), name="rtsp_read_thread"
        )
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self, capture):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()

    def getFrame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None


class CamReader:
    def __init__(self,
                 username, password, ip_address, channel):
        rtsp_url = f"rtsp://{username}:{password}@{ip_address}/{channel}"

        self.camera = Camera(rtsp_url)
        logger.info("Camera initialized")
        self.prev_time = 0

    def capture(self):
        logger.info("Capturing frame")
        # Capture single frame, from current stream
        frame = self.camera.getFrame()
        return frame
