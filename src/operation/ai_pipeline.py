"""
ai_pipeline.py

AI tracking pipeline -- ổn định target, hiển thị tất cả khuôn mặt đã đăng ký.
"""
import os
import sys
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import face_database
from operation.event_logger import EventLogger
from operation.face_recognizer import FaceRecognizer
from operation.pose_estimator import PoseEstimator
from operation.body_features import BodyFeatureExtractor, upper_body_box
from operation.servo_controller import ServoController
from operation.exercise_manager import exercise_manager
from operation.behavior_manager import behavior_manager


def _bbox_iou(a, b):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict_and_correct(self, cx, cy):
        if not self.initialized:
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.initialized = True
        meas = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
        self.kf.predict()
        corrected = self.kf.correct(meas)
        return int(corrected[0][0]), int(corrected[1][0])

    def predict_only(self):
        """Predict next position WITHOUT a measurement correction."""
        if not self.initialized:
            return None, None
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

    def reset(self):
        self.initialized = False


def _crop_safe(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class AIPipeline(threading.Thread):
    def __init__(self, camera_thread, event_logger=None, room_name="camera", room_id=None,
                 door_ws=None):
        super().__init__()
        self.camera_thread = camera_thread
        self.running = True
        self.daemon = True
        self.room_name = room_name
        self.room_id = room_id or room_name
        self.door_ws = door_ws
        self.logger = event_logger or EventLogger()

        self.device = config.get_torch_device()
        print(f"[AI:{self.room_name}] YOLO running on {self.device.upper()}.")
        self.yolo_face = YOLO(config.YOLO_FACE_MODEL_PATH)
        if not os.path.exists(config.YOLO_FACE_MODEL_PATH):
            print(f"[AI:{self.room_name}] WARNING: face model not found, using person model.")
            self.yolo_face = YOLO(config.YOLO_PERSON_MODEL_PATH)

        self.yolo_person = YOLO(config.YOLO_PERSON_MODEL_PATH)

        self.recognizer = FaceRecognizer()
        self.recognizer.load_database()
        if self.recognizer.is_empty():
            print(f"[AI:{self.room_name}] WARNING: face_db empty.")

        self.pose_estimator = PoseEstimator()
        
        # Chỉ giữ lại model trích xuất đặc trưng để DÙNG CHO VIỆC SO KHỚP (nhận diện)
        # Đã XÓA hoàn toàn logic tự động quét rải rác để cập nhật (_update_body_profiles)
        self.body_extractor = BodyFeatureExtractor()

        self._last_db_check = 0.0
        self.DB_RELOAD_CHECK_INTERVAL_SEC = getattr(config, "FACE_DB_RELOAD_CHECK_INTERVAL_SEC", 2.0)

        # Frame-skip counters
        self._frame_counter = 0
        self.SEARCHING_SKIP_FRAMES = getattr(config, "SEARCHING_SKIP_FRAMES", 0)
        self.TRACKING_SKIP_FRAMES = getattr(config, "TRACKING_SKIP_FRAMES", 0)
        self.POSE_EVERY_N_FRAMES = getattr(config, "POSE_EVERY_N_FRAMES", 1)

        self.kalman = KalmanFilter2D()

        # Servo
        self.servo = None
        self._servo_lost_counter = 0
        self._servo_was_tracking = False
        self._last_frame_shape = None
        self.SERVO_RETURN_TO_CENTER_AFTER_LOST_FRAMES = getattr(config, "SERVO_RETURN_TO_CENTER_AFTER_LOST_FRAMES", 15)
        
        if self.room_id in getattr(config, "SERVO_ENABLED_ROOMS", []):
            try:
                self.servo = ServoController(config.SERVO_CONFIG, door_ws=self.door_ws)
                if self.door_ws is None:
                    print(f"[AI:{self.room_name}] Pan/tilt servo ENABLED (room id='{self.room_id}') but no door_ws -- SIMULATE mode.")
                else:
                    print(f"[AI:{self.room_name}] Pan/tilt servo ENABLED (room id='{self.room_id}').")
            except Exception as e:
                print(f"[AI:{self.room_name}] Servo init failed ({e}) -- continuing WITHOUT servo.")
                self.servo = None

        # FSM
        self.STATE_SEARCHING = "SEARCHING"
        self.STATE_TRACKING = "TRACKING"
        self.STATE_LOST = "LOST"
        self.current_state = self.STATE_SEARCHING

        # Target
        self.locked_identity = None
        self.target_bbox = None
        self.target_center = None
        self._last_locked_identity = None
        self.lock_mode = None

        self.all_faces = []
        self.lost_counter = 0
        self.LOST_THRESHOLD = 10

        # Handoff
        self.on_target_lost_callback = None
        self._boundary_exit_streak = 0

        # Output lock
        self._lock = threading.Lock()
        self.has_target = False
        self.raw_bbox = None
        self.smoothed_bbox = None
        self.smoothed_center = None
        self.last_step_latency_ms = 0.0
        self.last_detect_ms = 0.0
        self.last_recognize_ms = 0.0

    def _set_output(self, has_target, raw_bbox, smoothed_bbox, center, drive_servo=True):
        with self._lock:
            self.has_target = has_target
            self.raw_bbox = raw_bbox
            self.smoothed_bbox = smoothed_bbox
            self.smoothed_center = center
        # drive_servo=False: dung cho cac frame "doan" (Kalman predict_only,
        # xem _step_tracking) khong co bbox moi thuc su -- xem ghi chu trong
        # _drive_servo() ve ly do KHONG duoc lai servo bang diem doan.
        if drive_servo:
            self._drive_servo(has_target, center, smoothed_bbox)

    def _drive_servo(self, has_target, center, box=None):
        if self.servo is None:
            return

        if has_target and box is not None and self._last_frame_shape is not None:
            self._servo_lost_counter = 0
            self._servo_was_tracking = True
            h, w = self._last_frame_shape
            cx, cy = w / 2.0, h / 2.0

            # QUAN TRONG: tinh sai so tu CHINH TAM O XANH (box) dang ve tren
            # dashboard, KHONG dung diem Kalman (center) duoc truyen vao.
            # Ly do: o cac frame "doan" (predict_only, xem _step_tracking
            # khi TRACKING_SKIP_FRAMES > 0), diem Kalman se NGOAI SUY chay
            # vuot len phia truoc trong khi self.target_bbox (o xanh that)
            # dung yen -- neu dung diem do de tinh error thi servo se DUOI
            # THEO MOT DIEM AO chay xa hon ca o xanh that, gay ra chay lo
            # y het trieu chung "van bi" da gap. Dung tam box + gioi han
            # duoi tinh servo NGAY tai day dam bao error va ranh gioi dung
            # (dead-zone) LUON cung mot nguon du lieu, khong bao gio lech
            # pha voi nhau.
            bx1, by1, bx2, by2 = box
            box_cx = (bx1 + bx2) / 2.0
            box_cy = (by1 + by2) / 2.0
            error_x = box_cx - cx
            error_y = box_cy - cy
            box_half_w = (bx2 - bx1) / 2.0
            box_half_h = (by2 - by1) / 2.0

            self.servo.update(error_x, error_y, box_half_w, box_half_h)
            return

        if self._servo_was_tracking:
            self.servo.reset_integral()
            self._servo_was_tracking = False

        self._servo_lost_counter += 1
        if (self._servo_lost_counter == self.SERVO_RETURN_TO_CENTER_AFTER_LOST_FRAMES
                and not self.servo.scanning):
            self.servo.go_to_center(config.SERVO_CONFIG)

    def get_servo_status(self):
        if self.servo is None:
            return False, None, None
        return True, self.servo.pan_angle, self.servo.tilt_angle

    def _clear_target(self):
        if self.locked_identity is not None:
            self._last_locked_identity = self.locked_identity
        self.locked_identity = None
        self.target_bbox = None
        self.target_center = None
        self.lock_mode = None
        self.kalman.reset()
        self._set_output(False, None, None, None)

    def force_clear_target(self):
        self.current_state = self.STATE_SEARCHING
        self._clear_target()

    def get_lock_mode(self):
        return self.lock_mode

    def set_on_target_lost_callback(self, fn):
        self.on_target_lost_callback = fn

    def _estimate_exit_direction(self):
        cfg = config.HANDOFF_CONFIG.get(self.room_id)
        if cfg is None:
            return "UNKNOWN", {}
        if cfg["type"] == "static":
            cx = self.target_center[0] if self.target_center is not None else None
            return "UNKNOWN", {"center_x": cx}
        if self.servo is None:
            return "UNKNOWN", {}
        pan = self.servo.pan_angle
        if pan >= cfg["pan_right_boundary"]:
            return "RIGHT", {"pan_angle": pan}
        if pan <= cfg["pan_left_boundary"]:
            return "LEFT", {"pan_angle": pan}
        return "UNKNOWN", {"pan_angle": pan}

    def _check_dynamic_boundary_exit(self):
        cfg = config.HANDOFF_CONFIG.get(self.room_id)
        if cfg is None or cfg.get("type") != "dynamic" or self.servo is None:
            self._boundary_exit_streak = 0
            return False
        pan = self.servo.pan_angle
        beyond = pan <= cfg["pan_left_boundary"] or pan >= cfg["pan_right_boundary"]
        self._boundary_exit_streak = self._boundary_exit_streak + 1 if beyond else 0
        return self._boundary_exit_streak >= cfg.get("boundary_confirm_frames", 8)

    def _declare_lost_with_handoff(self, reason_event):
        name = self.locked_identity
        direction, meta = self._estimate_exit_direction()
        self.current_state = self.STATE_LOST
        self.logger.log(reason_event, name=name, room=self.room_name, direction=direction, **meta)
        print(f"[FSM:{self.room_name}] Lost '{name}' ({reason_event}) -> LOST, direction={direction}")
        exercise_manager.set_online(name, False)
        if self.on_target_lost_callback:
            try:
                self.on_target_lost_callback(name, self.room_id, direction, self.target_center)
            except Exception as e:
                print(f"[AI:{self.room_name}] on_target_lost_callback error: {e}")
        self._clear_target()

    def run(self):
        print(f"[AI:{self.room_name}] FSM started.")
        self.logger.log("PIPELINE_STARTED", room=self.room_name)

        while self.running:
            ret, frame = self.camera_thread.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now_check = time.time()
            if now_check - self._last_db_check >= self.DB_RELOAD_CHECK_INTERVAL_SEC:
                self._last_db_check = now_check
                self.recognizer.reload_if_changed()

            step_start = time.time()
            try:
                self._step(frame)
            except Exception as e:
                print(f"[AI:{self.room_name}] Error: {e}, resetting.")
                self.logger.log("PIPELINE_ERROR", error=str(e), room=self.room_name)
                self.current_state = self.STATE_SEARCHING
                self._clear_target()

            with self._lock:
                self.last_step_latency_ms = (time.time() - step_start) * 1000.0

            # Smart sleep
            step_ms = self.last_step_latency_ms
            max_step_ms = getattr(config, "STEP_SLEEP_SKIP_IF_STEP_TOOK_LONGER_THAN_SEC", 0.03) * 1000.0
            if step_ms < max_step_ms:
                time.sleep(getattr(config, "STEP_SLEEP_SEC", 0.01))

    def _step(self, frame):
        self._last_frame_shape = frame.shape[:2]
        self._frame_counter += 1
        if self.current_state == self.STATE_SEARCHING:
            self._step_searching(frame)
        elif self.current_state == self.STATE_TRACKING:
            self._step_tracking(frame)
        elif self.current_state == self.STATE_LOST:
            self._step_lost(frame)

    # =============================================
    # SEARCHING
    # =============================================
    def _step_searching(self, frame):
        if self.SEARCHING_SKIP_FRAMES > 0 and self._frame_counter % (self.SEARCHING_SKIP_FRAMES + 1) != 0:
            return

        detect_start = time.time()
        results = self.yolo_face(frame, verbose=False, device=self.device)
        detect_ms = (time.time() - detect_start) * 1000.0

        candidates = []
        all_faces = []

        recognize_start = time.time()
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                if crop is None:
                    continue
                name, score = self.recognizer.identify(crop)
                if name is not None:
                    cx = (xyxy[0] + xyxy[2]) // 2
                    cy = (xyxy[1] + xyxy[3]) // 2
                    center = (cx, cy)
                    candidates.append((xyxy, name, score, center))
                    all_faces.append({
                        "name": name, "bbox": xyxy.tolist(), "score": score, "center": center
                    })
        recognize_ms = (time.time() - recognize_start) * 1000.0
        self.last_detect_ms = detect_ms
        self.last_recognize_ms = recognize_ms

        with self._lock:
            self.all_faces = all_faces

        print(f"[DEBUG] SEARCHING: found {len(all_faces)} registered faces")

        if candidates:
            # Ưu tiên 1: Thấy mặt -> khóa bằng FACE
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            best = max(candidates, key=lambda c: c[2] / (abs(c[3][0]-center_x) + abs(c[3][1]-center_y) + 1))
            bbox, name, score, center = best
            self.locked_identity = name
            self.target_bbox = list(bbox)
            self.target_center = center
            self.lock_mode = "FACE"
            self.kalman.reset()
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self.current_state = self.STATE_TRACKING
            self.lost_counter = 0
            self.logger.log("TARGET_ACQUIRED", name=name, score=f"{score:.2f}", room=self.room_name)
            print(f"[FSM:{self.room_name}] Acquired '{name}' -> TRACKING")
        else:
            # Ưu tiên 2: KHÔNG thấy mặt -> thử nhận diện bằng hình dáng thân người
            shape_bbox, shape_name, shape_score = self._find_identity_by_body_shape(
                frame, candidate_names=None
            )
            if shape_bbox is not None:
                cx = (shape_bbox[0] + shape_bbox[2]) // 2
                cy = (shape_bbox[1] + shape_bbox[3]) // 2
                self.locked_identity = shape_name
                self.target_bbox = shape_bbox
                self.target_center = (cx, cy)
                self.lock_mode = "BODY_SHAPE"
                self.kalman.reset()
                kx, ky = self.kalman.predict_and_correct(cx, cy)
                self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
                self.current_state = self.STATE_TRACKING
                self.lost_counter = 0
                self.logger.log(
                    "TARGET_ACQUIRED_BY_BODY", name=shape_name,
                    similarity=f"{shape_score:.2f}", room=self.room_name,
                )
                print(f"[FSM:{self.room_name}] Acquired '{shape_name}' by BODY SHAPE "
                      f"(similarity={shape_score:.2f}) -> TRACKING")
            else:
                self._clear_target()

    # =============================================
    # TRACKING -- ổn định, không nhảy lung tung
    # =============================================
    def _step_tracking(self, frame):
        # --- Skip-frame fast path ---
        if self.TRACKING_SKIP_FRAMES > 0 and self._frame_counter % (self.TRACKING_SKIP_FRAMES + 1) != 0:
            if self.target_center is not None and self.target_bbox is not None:
                pred = self.kalman.predict_only()
                if pred[0] is not None:
                    # drive_servo=False: day la diem NGOAI SUY (chua co
                    # detect that o frame nay), self.target_bbox van la o
                    # xanh CU. Neu lai servo bang diem ngoai suy nay trong
                    # khi dead-zone lai tinh tren o xanh cu -> lech pha,
                    # servo chay lo qua o xanh that. Chi cap nhat hien thi,
                    # cho servo dung yen cho toi frame co detect that.
                    self._set_output(True, self.target_bbox, self.target_bbox, pred, drive_servo=False)
            if self.target_bbox is not None:
                self._run_exercise_tracking(frame, self.target_bbox, center=self.target_center)
            return

        # --- Normal (full-detection) path below ---
        detect_start = time.time()
        results = self.yolo_face(frame, verbose=False, device=self.device)
        detect_ms = (time.time() - detect_start) * 1000.0

        candidates = []
        all_faces = []

        recognize_start = time.time()
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                if crop is None:
                    continue
                name, score = self.recognizer.identify(crop)
                if name is not None:
                    cx = (xyxy[0] + xyxy[2]) // 2
                    cy = (xyxy[1] + xyxy[3]) // 2
                    center = (cx, cy)
                    candidates.append((xyxy, name, score, center))
                    all_faces.append({
                        "name": name, "bbox": xyxy.tolist(), "score": score, "center": center
                    })
        recognize_ms = (time.time() - recognize_start) * 1000.0
        self.last_detect_ms = detect_ms
        self.last_recognize_ms = recognize_ms

        with self._lock:
            self.all_faces = all_faces

        print(f"[DEBUG] TRACKING: found {len(all_faces)} registered faces, locked='{self.locked_identity}'")

        current_found = None
        for cand in candidates:
            if cand[1] == self.locked_identity:
                current_found = cand
                break

        if current_found is not None:
            self.lost_counter = 0
            self.lock_mode = "FACE"
            bbox, name, score, center = current_found
            self.target_bbox = list(bbox)
            self.target_center = center
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self._run_exercise_tracking(frame, self.target_bbox, center=(kx, ky))
            if self._check_dynamic_boundary_exit():
                self._declare_lost_with_handoff("TARGET_LOST_BOUNDARY")
                return
            return

        body_bbox = self._find_body_continuity(frame)
        if body_bbox is not None:
            self.lost_counter = 0
            self.lock_mode = "BODY"
            cx = (body_bbox[0] + body_bbox[2]) // 2
            cy = (body_bbox[1] + body_bbox[3]) // 2
            self.target_bbox = list(body_bbox)
            self.target_center = (cx, cy)
            kx, ky = self.kalman.predict_and_correct(cx, cy)
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self._run_exercise_tracking(frame, self.target_bbox, center=(kx, ky))
            if self._check_dynamic_boundary_exit():
                self._declare_lost_with_handoff("TARGET_LOST_BOUNDARY")
                return
            return

        shape_bbox, shape_name, shape_score = self._find_identity_by_body_shape(
            frame, candidate_names=[self.locked_identity]
        )
        if shape_bbox is not None:
            self.lost_counter = 0
            self.lock_mode = "BODY_SHAPE"
            cx = (shape_bbox[0] + shape_bbox[2]) // 2
            cy = (shape_bbox[1] + shape_bbox[3]) // 2
            self.target_bbox = shape_bbox
            self.target_center = (cx, cy)
            kx, ky = self.kalman.predict_and_correct(cx, cy)
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self._run_exercise_tracking(frame, self.target_bbox, center=(kx, ky))
            self.logger.log("BODY_SHAPE_MATCH", name=self.locked_identity, similarity=f"{shape_score:.2f}", room=self.room_name)
            if self._check_dynamic_boundary_exit():
                self._declare_lost_with_handoff("TARGET_LOST_BOUNDARY")
                return
            return

        self.lost_counter += 1
        if self.lost_counter >= self.LOST_THRESHOLD:
            self._declare_lost_with_handoff("TARGET_LOST")

    def _find_identity_by_body_shape(self, frame, candidate_names=None, min_similarity=None):
        try:
            person_results = self.yolo_person(frame, verbose=False, device=self.device, classes=[0])
        except Exception as e:
            print(f"[AI:{self.room_name}] body-shape match yolo_person error: {e}")
            return None, None, 0.0

        if min_similarity is None:
            min_similarity = (
                config.BODY_MATCH_MIN_SIMILARITY if candidate_names
                else config.BODY_MATCH_MIN_SIMILARITY_COLD
            )
        min_samples = getattr(config, "BODY_MATCH_MIN_SAMPLES", 5)

        best_box, best_name, best_score = None, None, -1.0
        for r in person_results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, upper_body_box(xyxy, config.UPPER_BODY_CROP_RATIO))
                features = self.body_extractor.extract(crop)
                if features is None:
                    continue
                name, score = face_database.match_body_profile(
                    features, candidate_names=candidate_names, min_sample_count=min_samples
                )
                if name is not None and score > best_score:
                    best_box, best_name, best_score = xyxy, name, score

        if best_box is not None and best_score >= min_similarity:
            return list(best_box), best_name, best_score
        return None, None, 0.0

    def _find_body_continuity(self, frame):
        if self.target_bbox is None:
            return None
        try:
            results = self.yolo_person(frame, verbose=False, device=self.device, classes=[0])
        except Exception as e:
            print(f"[AI:{self.room_name}] yolo_person error: {e}")
            return None

        best_iou = 0.0
        best_box = None
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                iou = _bbox_iou(xyxy, self.target_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_box = xyxy

        if best_box is not None and best_iou >= 0.25:
            return list(best_box)
        return None

    def _run_exercise_tracking(self, frame, bbox, center=None):
        if self.POSE_EVERY_N_FRAMES > 1 and self._frame_counter % self.POSE_EVERY_N_FRAMES != 0:
            return

        name = self.locked_identity
        if name is None:
            return

        is_assigned = exercise_manager.is_assigned(name)
        if is_assigned:
            exercise_manager.set_online(name, True)

        crop = _crop_safe(frame, bbox)
        result = self.pose_estimator.process(name, crop, bbox_center=center)

        if is_assigned and result["rep_completed"]:
            exercise_manager.register_rep(name, result["rep_completed"])
            self.logger.log("EXERCISE_REP", name=name, exercise=result["rep_completed"], room=self.room_name)
            print(f"[Exercise:{self.room_name}] '{name}' completed a {result['rep_completed']} rep")

        if result["behavior_events"]:
            behavior_manager.record_many(name, result["behavior_events"])
            self.logger.log("BEHAVIOR_DETECTED", name=name, events=",".join(result["behavior_events"]), room=self.room_name)

    # =============================================
    # LOST -- tìm lại bất kỳ ai đã đăng ký
    # =============================================
    def _step_lost(self, frame):
        results = self.yolo_face(frame, verbose=False, device=self.device)
        all_faces = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                if crop is None:
                    continue
                name, score = self.recognizer.identify(crop)
                if name is not None:
                    cx = (xyxy[0] + xyxy[2]) // 2
                    cy = (xyxy[1] + xyxy[3]) // 2
                    all_faces.append({
                        "name": name, "bbox": xyxy.tolist(), "score": score, "center": (cx, cy)
                    })
        with self._lock:
            self.all_faces = all_faces

        print(f"[DEBUG] LOST: found {len(all_faces)} registered faces")

        if all_faces:
            chosen = None
            for face in all_faces:
                if face["name"] == self.locked_identity:
                    chosen = face
                    break
            if chosen is None:
                chosen = all_faces[0]
            name = chosen["name"]
            bbox = chosen["bbox"]
            center = chosen["center"]
            self.locked_identity = name
            self.target_bbox = bbox
            self.target_center = center
            self.lock_mode = "FACE"
            self.kalman.reset()
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self.current_state = self.STATE_TRACKING
            self.lost_counter = 0
            self.logger.log("TARGET_REACQUIRED", name=name, room=self.room_name)
            print(f"[FSM:{self.room_name}] Reacquired '{name}' -> TRACKING")
            return

        shape_bbox = shape_name = None
        shape_score = 0.0
        if self._last_locked_identity is not None:
            shape_bbox, shape_name, shape_score = self._find_identity_by_body_shape(
                frame, candidate_names=[self._last_locked_identity]
            )
        if shape_bbox is None:
            shape_bbox, shape_name, shape_score = self._find_identity_by_body_shape(
                frame, candidate_names=None
            )

        if shape_bbox is not None:
            cx = (shape_bbox[0] + shape_bbox[2]) // 2
            cy = (shape_bbox[1] + shape_bbox[3]) // 2
            self.locked_identity = shape_name
            self.target_bbox = shape_bbox
            self.target_center = (cx, cy)
            self.lock_mode = "BODY_SHAPE"
            self.kalman.reset()
            kx, ky = self.kalman.predict_and_correct(cx, cy)
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self.current_state = self.STATE_TRACKING
            self.lost_counter = 0
            self.logger.log("TARGET_REACQUIRED_BY_BODY", name=shape_name, similarity=f"{shape_score:.2f}", room=self.room_name)
            print(f"[FSM:{self.room_name}] Reacquired '{shape_name}' by body shape (similarity={shape_score:.2f}) -> TRACKING")
        else:
            self._clear_target()

    # =============================================
    # Public accessors
    # =============================================
    def get_ai_result(self):
        with self._lock:
            return self.has_target, self.raw_bbox, self.smoothed_bbox, self.smoothed_center, self.current_state

    def get_all_faces(self):
        with self._lock:
            return self.all_faces.copy()

    def get_locked_identity(self):
        return self.locked_identity

    def get_last_latency_ms(self):
        with self._lock:
            return self.last_step_latency_ms

    def get_latency_breakdown(self):
        with self._lock:
            return self.last_detect_ms, self.last_recognize_ms

    def stop(self):
        self.running = False
        try:
            self.body_extractor.close()
        except Exception:
            pass
        if self.servo is not None:
            try:
                self.servo.close()
            except Exception:
                pass