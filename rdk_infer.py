import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from hobot_dnn import pyeasy_dnn as dnn


COCO_PERSON_CLASS_ID = 0
COCO_NUM_CLASSES = 80


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # 归一化 bbox: [x, y, w, h]，x/y 为左上角
    bbox: Tuple[float, float, float, float]
    class_id: int


class RdkYoloV8:
    """YOLOv8 RDK X5 BPU 推理封装：输入 BGR frame，输出归一化 bbox detections。

    依赖 hobot_dnn (pyeasy_dnn)。
    模型要求：已通过 D-Robotics 工具链编译为 .bin，输出带片上 NMS。
    输出格式预期：[batch, num_det, 6]，每行 [x1, y1, x2, y2, score, class_id]（归一化 0-1）。
    若模型输出格式不同，只需调整 _parse_output 即可，其余不动。
    """

    def __init__(self, bin_path: str):
        self.bin_path = bin_path
        self._lock = threading.Lock()
        self._models = None
        self._input_h = 640
        self._input_w = 640
        self._ready = False
        self._last_error_ts = 0.0
        self._init_lock = threading.Lock()

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True

        now = time.time()
        if self._last_error_ts != 0.0 and now - self._last_error_ts < 5.0:
            return False

        with self._init_lock:
            if self._ready:
                return True
            now = time.time()
            if self._last_error_ts != 0.0 and now - self._last_error_ts < 5.0:
                return False
            try:
                self._models = dnn.load([self.bin_path])
                # 从模型属性读取输入尺寸（如果能拿到）
                props = self._models[0].inputs[0].properties
                if hasattr(props, 'valid_shape'):
                    shape = props.valid_shape
                    # shape 通常为 [1, H, W, C] 或 [1, C, H, W]
                    if len(shape) == 4:
                        self._input_h = int(shape[1]) if shape[1] > 3 else int(shape[2])
                        self._input_w = int(shape[2]) if shape[1] > 3 else int(shape[3])
                self._ready = True
                return True
            except Exception as e:
                self._last_error_ts = time.time()
                self._ready = False
                self._models = None
                print(f"[RdkInfer] 初始化失败: {e}")
                return False

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)
        # bayese NV12 模型要求 YUV NV12 输入
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
        # I420 → NV12：U/V 平面交错
        h, w = self._input_h, self._input_w
        y = yuv[:h, :]
        uv_i420 = yuv[h:, :].reshape(-1, 2)
        uv_nv12 = uv_i420[:, [0, 1]].reshape(h // 2, w)
        nv12 = np.vstack([y, uv_nv12])
        return nv12.astype(np.uint8)[np.newaxis, ...]

    def _parse_output(self, outputs, conf_th: float, only_person: bool) -> List[Detection]:
        """解析 BPU 输出张量。

        预期格式：outputs[0].buffer shape = [1, num_det, 6]
        每行：[x1, y1, x2, y2, score, class_id]，坐标已归一化到 0-1。
        若模型输出顺序不同，在此处调整字段索引即可。
        """
        dets: List[Detection] = []

        buf = np.array(outputs[0].buffer).squeeze()  # (num_det, 6) 或 (6,)
        if buf.ndim == 1:
            buf = buf[np.newaxis, :]

        for row in buf:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, cls = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5])
            if score < conf_th:
                continue
            if only_person and cls != COCO_PERSON_CLASS_ID:
                continue

            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            label = "person" if cls == COCO_PERSON_CLASS_ID else f"class_{cls}"
            dets.append(Detection(label=label, confidence=score, bbox=(x1, y1, w, h), class_id=cls))

        return dets

    def infer(self, frame_bgr: np.ndarray, only_person: bool = False, conf_th: float = 0.0) -> List[Detection]:
        with self._lock:
            if not self._ensure_ready() or self._models is None:
                return []
            inp = self._preprocess(frame_bgr)
            try:
                outputs = self._models[0].forward(inp)
            except Exception as e:
                print(f"[RdkInfer] forward 失败: {e}")
                return []

        return self._parse_output(outputs, conf_th, only_person)

    @staticmethod
    def to_view_objects(dets: List[Detection]) -> List[Dict[str, Any]]:
        return [
            {
                "label": d.label,
                "confidence": d.confidence,
                "bbox": [float(d.bbox[0]), float(d.bbox[1]), float(d.bbox[2]), float(d.bbox[3])],
                "class_id": d.class_id,
            }
            for d in dets
        ]
