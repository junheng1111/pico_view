from flask import Flask, Response
import time
import cv2
from detector import AIDetector

app = Flask(__name__)
ai_engine = AIDetector()

# 调试开关：需要打印 person 边框信息时设为 True
DEBUG_PERSON_BBOX = True

# 虚拟运镜（数字裁剪）开关：不依赖物理云台/电机
ENABLE_VIRTUAL_PTZ = True

# 运镜参数（都基于归一化坐标 0..1）
DEADBAND_X = 0.08  # 人中心点在画面中心±8% 以内不动，避免抖动
DEADBAND_Y = 0.10
SMOOTHING = 0.15   # 越小越“黏”，越大越“跟手”

# 虚拟镜头裁剪比例（<1 表示放大/裁剪）
CROP_W = 0.70
CROP_H = 0.70

_vp_center_x = 0.5
_vp_center_y = 0.5


def bbox_to_pixels(bbox, frame_shape):
    """将 bbox 转成像素坐标 (x1, y1, x2, y2)。

    兼容常见格式：
    - [x, y, w, h] (归一化，左上角+宽高)
    - [x1, y1, x2, y2] (像素)
    - [x, y, w, h] (像素，左上角+宽高)
    """
    if not bbox or len(bbox) != 4:
        return None

    fh, fw = frame_shape[0], frame_shape[1]
    a, b, c, d = bbox

    # 1) 归一化 [x,y,w,h]
    if max(a, b, c, d) <= 1.0 and (a + c) <= 1.0 and (b + d) <= 1.0:
        x1 = int(a * fw)
        y1 = int(b * fh)
        x2 = int((a + c) * fw)
        y2 = int((b + d) * fh)
        return x1, y1, x2, y2

    # 2) 像素 [x1,y1,x2,y2]
    if c > a and d > b:
        return int(a), int(b), int(c), int(d)

    # 3) 像素 [x,y,w,h]
    return int(a), int(b), int(a + c), int(b + d)

def generate():
    last_print_ts = 0.0
    printed_sample = False
    global _vp_center_x, _vp_center_y
    while True:
        frame, detections = ai_engine.get_data()
        
        if frame is None:
            # 如果没数据，给一个极短的等待，不要卡死连接
            time.sleep(0.05)
            continue

        person_count = 0
        best_person = None
        best_person_score = -1.0
        if (not printed_sample) and detections:
            print("DEBUG: detections sample:", detections[0])
            # 打印一下所有 label，看看 person 的名字到底是什么
            labels = []
            for o in detections[:10]:
                if isinstance(o, dict):
                    labels.append(o.get('label') or o.get('class'))
            print("DEBUG: first labels:", labels)
            printed_sample = True
        #print(f"[Web Server] 检测到 {len(detections)} 个目标, ")
        for obj in detections:
            #print(f"  - 目标: {obj.get('label')} 置信度: {obj.get('confidence'):.2f}")
            # 关键过滤：只处理 label 为 person 的目标
            if obj.get('label') == 'person':
                person_count += 1

                # 选择一个“主目标”用于运镜：优先最高置信度
                try:
                    score = float(obj.get('confidence', 0.0))
                except Exception:
                    score = 0.0
                if score > best_person_score:
                    best_person_score = score
                    best_person = obj

                bbox = obj.get('bbox')
                coords = bbox_to_pixels(bbox, frame.shape)
                if not coords:
                    continue
                x1, y1, x2, y2 = coords

                # 计算边框大小(像素)
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)

                # 可选：打印人框信息（每秒最多打印 1 次，避免刷屏）
                if DEBUG_PERSON_BBOX:
                    now = time.time()
                    if now - last_print_ts >= 1.0:
                        conf = obj.get('confidence', 0)
                        print(f"[Person] conf={conf:.2f} bbox_px=({x1},{y1})-({x2},{y2}) size=({bw}x{bh})")
                        last_print_ts = now

                # 画绿色的框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 写上标签
                conf = obj.get('confidence', 0)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. 可选：在左上角实时显示人数
        cv2.putText(frame, f"Count: {person_count}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 4. 虚拟运镜：根据主目标(person)中心点偏移来移动裁剪窗口
        if ENABLE_VIRTUAL_PTZ and best_person is not None:
            coords = bbox_to_pixels(best_person.get('bbox'), frame.shape)
            if coords:
                x1, y1, x2, y2 = coords
                fh, fw = frame.shape[0], frame.shape[1]

                # person 中心点归一化
                cx = ((x1 + x2) * 0.5) / fw
                cy = ((y1 + y2) * 0.5) / fh

                # deadband：小幅移动不跟随
                dx = cx - 0.5
                dy = cy - 0.5
                if abs(dx) < DEADBAND_X:
                    cx = 0.5
                if abs(dy) < DEADBAND_Y:
                    cy = 0.5

                # smoothing：指数平滑，避免抖动
                _vp_center_x = (1.0 - SMOOTHING) * _vp_center_x + SMOOTHING * cx
                _vp_center_y = (1.0 - SMOOTHING) * _vp_center_y + SMOOTHING * cy

                # 计算裁剪窗口（保持在 0..1 内）
                half_w = CROP_W * 0.5
                half_h = CROP_H * 0.5
                min_cx = half_w
                max_cx = 1.0 - half_w
                min_cy = half_h
                max_cy = 1.0 - half_h
                _vp_center_x = max(min_cx, min(max_cx, _vp_center_x))
                _vp_center_y = max(min_cy, min(max_cy, _vp_center_y))

                left = int((_vp_center_x - half_w) * fw)
                right = int((_vp_center_x + half_w) * fw)
                top = int((_vp_center_y - half_h) * fh)
                bottom = int((_vp_center_y + half_h) * fh)

                # 保护边界
                left = max(0, min(fw - 2, left))
                right = max(left + 1, min(fw, right))
                top = max(0, min(fh - 2, top))
                bottom = max(top + 1, min(fh, bottom))

                cropped = frame[top:bottom, left:right]
                # 给用户一点反馈：画出当前裁剪窗口（可注释）
                # cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

                if cropped.size > 0:
                    frame = cv2.resize(cropped, (fw, fh), interpolation=cv2.INTER_LINEAR)

        # 将 720P 压缩为 480P 传输，大幅降低局域网带宽压力
        small_frame = cv2.resize(frame, (640, 360))
        ret, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        
        if not ret:
            continue

        #print("[Web Server] 发送一帧数据") # 如果能打印这个，说明网页必出画面
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<html><body style="background:#000; display:flex; justify-content:center;"><img src="/video_feed" style="max-width:100%;"></body></html>'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)