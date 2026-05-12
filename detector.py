import cv2
import threading
import time
import atexit

from rdk_infer import RdkYoloV8
from config import MODEL_BIN, WIDTH, HEIGHT


_RDK_ENGINE = None
_RDK_ENGINE_LOCK = threading.Lock()


def _get_rdk_engine():
    global _RDK_ENGINE
    with _RDK_ENGINE_LOCK:
        if _RDK_ENGINE is None:
            _RDK_ENGINE = RdkYoloV8(MODEL_BIN)
        return _RDK_ENGINE


class AIDetector:
    def __init__(self, device: str = "/dev/video8"):
        self.device = device
        self.latest_frame = None
        self.latest_results = []
        self.running = True
        self._cleaned = False

        self.infer_engine = _get_rdk_engine()

        threading.Thread(target=self._read_stream, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()

        atexit.register(self.cleanup)

    def _read_stream(self):
        cap = None

        def _open():
            c = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return c
            c.release()
            return None

        while self.running and cap is None:
            cap = _open()
            if cap is None:
                time.sleep(0.2)

        while self.running and cap is not None:
            ret, frame = cap.read()
            if ret:
                self.latest_frame = frame
            else:
                time.sleep(0.05)
                cap.release()
                cap = None
                while self.running and cap is None:
                    cap = _open()
                    if cap is None:
                        time.sleep(0.2)

        if cap is not None:
            cap.release()

    def _infer_loop(self):
        last_ts = 0.0
        while self.running:
            frame = self.latest_frame
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_ts < 0.05:  # ~20fps
                time.sleep(0.005)
                continue
            last_ts = now

            try:
                dets = self.infer_engine.infer(frame, only_person=False, conf_th=0.25)
                self.latest_results = self.infer_engine.to_view_objects(dets)
            except Exception as e:
                if int(time.time()) % 5 == 0:
                    print(f"[RdkInfer] error: {e}")
                time.sleep(0.05)

    def get_data(self):
        return self.latest_frame, self.latest_results

    def cleanup(self):
        if getattr(self, "_cleaned", False):
            return
        self._cleaned = True
        print("正在清理资源...")
        self.running = False
        print("DEBUG: 资源清理完成.")
