import threading
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from hailo_platform.pyhailort import pyhailort as ph


# COCO 类别 id：person = 0
COCO_PERSON_CLASS_ID = 0
COCO_NUM_CLASSES = 80


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # 归一化 bbox: [x, y, w, h]，x/y 为左上角
    bbox: Tuple[float, float, float, float]
    class_id: int


class HailoYoloV8:
    """YOLOv8 Hailo 推理封装：输入 BGR frame，输出归一化 bbox detections。

    关键点：
    - 必须用 `with configured_model:` + `activate()`，否则会 segfault / stream 未激活
    - 输出是 NMS postprocess，需要用 HailoRTTransformUtils 解码成 per-class detections
    """

    def __init__(self, hef_path: str = "/usr/share/hailo-models/yolov8s_h8.hef"):
        self.hef_path = hef_path

        # Hailo 运行时对象（线程内共享，外部用锁保护）
        self._lock = threading.Lock()

        self._vdevice = None
        self._model = None
        self._configured = None
        self._input_name = None
        self._output_name = None
        self._output_info = None
        self._ready = False
        self._last_init_error_ts = 0.0
        self._init_lock = threading.Lock()
        self._service_restart_attempted = False

    def _ensure_ready(self) -> bool:
        """延迟初始化 Hailo 资源。

        返回 True 表示可用，否则 False（例如 device 被占用）。
        """
        if self._ready:
            return True

        now = time.time()
        # 避免疯狂重试刷屏（更保守一点）
        if now - self._last_init_error_ts < 5.0 and self._last_init_error_ts != 0.0:
            return False

        with self._init_lock:
            if self._ready:
                return True

            now = time.time()
            if now - self._last_init_error_ts < 5.0 and self._last_init_error_ts != 0.0:
                return False

            try:
                vdevice = ph.VDevice()
                model = vdevice.create_infer_model(self.hef_path)
                configured = model.configure()

                input_name = model.input_names[0]
                output_name = model.output_names[0]
                output_info = model.output(output_name)

                # 只有全部成功后再写入成员，避免半初始化状态
                self._vdevice = vdevice
                self._model = model
                self._configured = configured
                self._input_name = input_name
                self._output_name = output_name
                self._output_info = output_info

                self._ready = True
                return True
            except Exception as e:
                self._last_init_error_ts = time.time()
                self._ready = False
                self._vdevice = None
                self._model = None
                self._configured = None
                self._input_name = None
                self._output_name = None
                self._output_info = None

                # 常见：设备被“卡住”导致 out_of_physical_devices。
                # 我们只尝试一次重启 hailort.service，然后等待下一轮重试。
                msg = str(e)
                if ("OUT_OF_PHYSICAL_DEVICES" in msg or "HAILO_OUT_OF_PHYSICAL_DEVICES" in msg) and not self._service_restart_attempted:
                    self._service_restart_attempted = True
                    try:
                        subprocess.run(["sudo", "systemctl", "restart", "hailort.service"], check=False)
                    except Exception:
                        pass

                return False

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # 模型输入是 640x640x3，dtype=uint8（从后处理配置和 stream info 得到）
        resized = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        # Hailo YOLOv8 输入一般是 RGB 或 BGR？rpicam 示例通常用 RGB。
        # 为确保正确，我们转 RGB。若后续检测始终为 0，可切回 BGR。
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    def infer(self, frame_bgr: np.ndarray, only_person: bool = False, conf_th: float = 0.0) -> List[Detection]:
        """对单帧推理，返回 Detection 列表（归一化 bbox）。"""
        # Hailo 不可用（设备被占用/服务异常）时，直接返回空，不让上层崩溃
        # 重要：ensure_ready 也放在 _lock 下，避免多线程同时 init/run 造成 _configured 竞态
        with self._lock:
            if not self._ensure_ready() or self._configured is None:
                return []

            inp = self._preprocess(frame_bgr)

            with self._configured:
                self._configured.activate()
                bindings = self._configured.create_bindings()

                bindings.input(self._input_name).set_buffer(inp)

                # 输出是 float32 + NMS_BY_CLASS
                out = np.zeros((self._output_info.shape[0],), dtype=np.float32)
                bindings.output(self._output_name).set_buffer(out)

                # timeout 5s
                self._configured.run([bindings], 5000)
                self._configured.deactivate()

        # 解码为 per-class NMS 格式（list，长度=classes，每项是 ndarray）
        per_class = ph.HailoRTTransformUtils.output_raw_buffer_to_nms_format_single_frame(out, COCO_NUM_CLASSES)

        dets: List[Detection] = []

        # per_class[i] 的每一行代表一个 detection，字段顺序需要确认。
        # 常见格式为 [y_min, x_min, y_max, x_max, score] 或 [x_min, y_min, x_max, y_max, score]
        # 我们先做“自适应”判断：看前4个值是否像坐标（0-1），第5个像 score。
        for class_id, arr in enumerate(per_class):
            if only_person and class_id != COCO_PERSON_CLASS_ID:
                continue
            if arr is None or len(arr) == 0:
                continue

            for row in arr:
                if row is None or len(row) < 5:
                    continue

                a, b, c, d, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                if score < conf_th:
                    continue

                # 判断坐标顺序。
                # 如果 a,b,c,d 都在 0..1 且 c>=a d>=b：可能是 [y1,x1,y2,x2]
                # 反之如果 c>=a d>=b 也可能是 [x1,y1,x2,y2]
                # 我们用“谁更像 y”：y 通常在 [0,1]，但都一样；这里用约定优先 y,x,y,x。
                if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and 0.0 <= c <= 1.0 and 0.0 <= d <= 1.0:
                    # 默认按 [y1, x1, y2, x2]
                    y1, x1, y2, x2 = a, b, c, d
                else:
                    # 如果不是归一化（不太可能），跳过
                    continue

                # 转换为 [x, y, w, h]
                x = max(0.0, min(1.0, x1))
                y = max(0.0, min(1.0, y1))
                w = max(0.0, min(1.0, x2) - x)
                h = max(0.0, min(1.0, y2) - y)

                label = "person" if class_id == COCO_PERSON_CLASS_ID else f"class_{class_id}"
                dets.append(Detection(label=label, confidence=float(score), bbox=(x, y, w, h), class_id=class_id))

        return dets

    @staticmethod
    def to_view_objects(dets: List[Detection]) -> List[Dict[str, Any]]:
        """转换成 view.py 现有逻辑可直接消费的结构。"""
        out: List[Dict[str, Any]] = []
        for d in dets:
            out.append({
                "label": d.label,
                "confidence": d.confidence,
                "bbox": [float(d.bbox[0]), float(d.bbox[1]), float(d.bbox[2]), float(d.bbox[3])],
                "class_id": d.class_id,
            })
        return out
