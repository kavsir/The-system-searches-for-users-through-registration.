"""
Shared configuration for both the registration (Flask) app and the
operation (camera + AI recognition) app.
"""

import os

# ---------------------------------------------------------------------------
# Device selection (NVIDIA GPU if available, otherwise CPU)
# ---------------------------------------------------------------------------

# USE_GPU=True means "prefer GPU". Every model loader still falls back to
# CPU automatically if no usable GPU/CUDA setup is found at runtime, so
# this never crashes the app on a machine without a GPU.
USE_GPU = True


def get_torch_device():
    """
    Return "cuda" if PyTorch can see a usable NVIDIA GPU, otherwise "cpu".
    Used by Ultralytics/YOLO, which is built on PyTorch.
    """
    if not USE_GPU:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_insightface_ctx_id():
    """
    Return the ctx_id expected by InsightFace's FaceAnalysis.prepare():
    a GPU index (0, 1, ...) to use CUDA, or -1 to force CPU.
    InsightFace uses onnxruntime under the hood. If onnxruntime-gpu (with a
    working CUDA setup) isn't installed, onnxruntime silently falls back to
    its CPU provider on its own -- but checking torch.cuda.is_available()
    first lets us pick the GPU ctx_id only when a GPU genuinely looks usable,
    and avoids the "Specified provider 'CUDAExecutionProvider' is not in
    available provider names" warning when there's clearly no GPU at all.
    """
    if not USE_GPU:
        return -1
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

# src/config.py -> project root is one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Legacy "labeled folders" dataset layout. No longer written to -- kept
# only so migrate_to_sqlite.py can read the old data once and import it
# into FACE_DB_PATH below. Safe to delete these folders after migrating.
FACE_DB_DIR = os.path.join(DATA_DIR, "face_db")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Current dataset storage: a single SQLite file holding every registered
# person's embeddings + raw/processed images. See face_database.py.
FACE_DB_PATH = os.path.join(DATA_DIR, "face_dataset.db")

# ---------------------------------------------------------------------------
# Door servo ESP32 (Dev Module) -- WebSocket
# ---------------------------------------------------------------------------
# ONE physical ESP32 Dev Module drives BOTH door servos (Phong 1 + Phong 2)
# and connects IN to this server as a single WebSocket *client* (see
# operation/door_ws_server.py and esp32_servo.ino). Each door/room is
# addressed by the same id as its entry in CAMERAS (e.g. "cam1", "cam2"),
# multiplexed over that one connection -- we don't need to know the
# ESP32's IP, we just bind and listen here.
# ---------------------------------------------------------------------------

# Cross-app links
# ---------------------------------------------------------------------------
# Port app_registration.py listens on (see its `app.run(..., port=...)`).
# app_dashboard.py exposes this via /api/config so dashboard.html can build
# a working "Dang ky khuon mat" link regardless of which host/IP the
# dashboard is being viewed from.
REGISTRATION_APP_PORT = 5000

DOOR_WS_HOST = "0.0.0.0"   # interface the door WebSocket server binds to
DOOR_WS_PORT = 8765        # must match `ws_port` in esp32_servo.ino

# Door lockdown model (see app_dashboard.py's _lockdown_all_doors):
#   - With nobody registered detected anywhere, every door button works
#     normally -- open/close on request, no auto behavior.
#   - The instant a registered person is detected in ANY room, every door
#     in the system is force-closed automatically. Doors never reopen by
#     themselves afterwards -- opening always requires a manual dashboard
#     button press, regardless of who's present.
# There is no "close after N seconds of absence" timer anymore: closing is
# triggered by presence being detected, not by absence.
CAMERAS = [
    {
        "id": "cam1",
        "room_name": "Phong 1",
        "url": "http://10.208.229.179/stream",
    },
    {
        "id": "cam2",
        "room_name": "Phong 2",
        "url": "http://10.208.229.227/stream",
    },
]

# Kept for backward compatibility with any code (e.g. registration app)
# that still imports a single CAMERA_URL directly.
CAMERA_URL = CAMERAS[0]["url"]

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 30

# How long (seconds) CameraReader's watchdog waits with NO new frame,
# despite the connection still looking "open", before it force-releases
# the stream and reconnects. Fixes the case where the ESP32-CAM's stream
# dies mid-connection (Wi-Fi drop, camera reboot, momentary wrong IP...)
# and cv2.VideoCapture.read() ends up blocked forever instead of actually
# returning an error -- without this, the camera never automatically
# recovers even after its signal comes back. See operation/camera_reader.py.
CAMERA_STALL_TIMEOUT_SEC = 5.0

# ---------------------------------------------------------------------------
# YOLO / face detection
# ---------------------------------------------------------------------------

YOLO_FACE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")
YOLO_PERSON_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

# Minimum confidence required to accept a detected face/person box.
AI_CONF_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Face recognition (InsightFace)
# ---------------------------------------------------------------------------

# Minimum cosine similarity (normed embeddings -> plain dot product) for a
# detected face to be considered a match for a registered person. Lower this
# (e.g. 0.4) if recognition feels too strict; raise it if strangers are
# getting matched to a registered name.
FACE_RECOGNITION_THRESHOLD = 0.35

# How many AI-pipeline steps to wait between identity re-checks while a face
# is already locked and tracked by CSRT. Identity doesn't need to be
# re-verified every frame -- this just confirms we're still tracking the
# right person.
IDENTITY_RECHECK_INTERVAL = 10

# How often (seconds) each AIPipeline thread checks face_database's
# face_db_version counter to see if app_registration.py (a SEPARATE
# process) just registered/deleted someone, and hot-reloads its embeddings
# if so. This is what removes the "must restart app_dashboard.py after
# registering someone" requirement. Cheap (one small SELECT) -- 1-3s is a
# good default; lower it if you want new registrations picked up faster.
FACE_DB_RELOAD_CHECK_INTERVAL_SEC = 2.0

# ---------------------------------------------------------------------------
# Body recognition (long-term, clothing-invariant body-shape profile)
# ---------------------------------------------------------------------------

# How often (seconds), PER PERSON, each AIPipeline re-measures a
# registered face's body-shape ratios (see operation/body_features.py).
# Runs for EVERY registered face seen this step, not just the locked
# target -- an extra YOLO-person pass + a mediapipe Pose call per person
# is real CPU cost on top of the existing pipeline, so this is throttled
# rather than run every frame. Raise it (e.g. 5.0) if CPU is tight with
# multiple rooms running at once.
BODY_PROFILE_UPDATE_INTERVAL_SEC = 2.0

# Chi lay NUA THAN TREN (vai/nguc/khuyu tay/co tay) cua bbox than nguoi
# truoc khi dua vao BodyFeatureExtractor -- xem operation/body_features.py
# upper_body_box(). Dung CHUNG mot gia tri nay o ca luc xay ho so
# (_update_body_profiles) lan luc so khop (_find_identity_by_body_shape)
# de body_aspect_ratio con so sanh duoc giua 2 lan do.
UPPER_BODY_CROP_RATIO = 0.55

# "Du gan de nua than tren hien ro" -- chi XAY ho so than nguoi
# (_update_body_profiles) khi bbox than nguoi (YOLO-person, full-body)
# cao it nhat % nay so voi chieu cao khung hinh. Nguoi dung xa hon cho
# ra bbox nho hon -> landmark nua than tren (vai/khuyu tay/co tay) de bi
# lan/nhieu -> bo qua, doi ho toi gan hon (vd vua buoc vao phong) moi do.
BODY_PROFILE_MIN_HEIGHT_RATIO = 0.45

# ---------------------------------------------------------------------------
# Body-shape fallback identification (used when face recognition can't see
# a face at all -- turned away, too far, bad angle). See ai_pipeline.py's
# _find_identity_by_body_shape(). Deliberately a SEPARATE, much higher bar
# than face recognition's own FACE_RECOGNITION_THRESHOLD, since body shape
# alone is a far weaker biometric signal than a face embedding.
# ---------------------------------------------------------------------------
# Targeted re-check: "is this probably the specific person we JUST lost
# track of" -- a lower bar is acceptable because context already narrows
# it down to one candidate.
BODY_MATCH_MIN_SIMILARITY = 0.85
# Cold match: "does this unrecognized body belong to ANY registered
# person" -- stricter, since there's no context narrowing the candidates.
BODY_MATCH_MIN_SIMILARITY_COLD = 0.92
# A profile needs at least this many real sightings before it's trusted
# enough to be matched against at all.
BODY_MATCH_MIN_SAMPLES = 2

# ---------------------------------------------------------------------------
# Pan/tilt servo (CAM2 ONLY -- add-on, not a replacement)
# ---------------------------------------------------------------------------
# Physically steers cam2's camera mount to keep the currently-locked,
# REGISTERED person's face/body centered in frame. Every existing
# mechanism keeps working exactly as before, for every room INCLUDING
# cam2: face recognition, the SEARCHING/TRACKING/LOST FSM, the body-shape
# fallback, exercise/behavior tracking, door lockdown, tracking-photo
# snapshots, movement notifications. The servo only reacts to WHERE the
# pipeline already decided the target is (whatever lock_mode found them);
# it never changes WHO is tracked or HOW.
#
# Hardware: ALL FOUR servos (pan, tilt, cua cam1, cua cam2) now live on ONE
# ESP32 Dev Module + ONE PCA9685 board (channels 0/1/2/3 respectively),
# talking to the server over the SAME WebSocket connection used for the
# doors -- see esp32_servo.ino + operation/door_ws_server.py's
# send_pan_tilt() + operation/servo_controller.py. No Serial/USB link and
# no separate esp32_pca9685_controller.ino needed anymore -- that firmware
# was merged into esp32_servo.ino.
SERVO_ENABLED_ROOMS = ["cam2"]

SERVO_CONFIG = {
    "control": {
        # --- Buoc di chuyen + nghi (khong con PID/tang toc nua) ---
        # Buoc CO DINH moi lan servo di chuyen (do). Giam xuong (vd 1-2)
        # neu muon cham/chac hon nua; tang len (vd 5) neu muon nhanh hon
        # (nhung de mat on dinh hon).
        "max_step_per_frame": 5,
        # Buoc toi da RIENG cho tilt (ngan len/xuong). Theo yeu cau: ca
        # pan va tilt deu quay CO DINH 5 do moi lan, khong phu thuoc
        # khoang cach con lai toi o xanh.
        "max_step_per_frame_tilt": 5,
        # Thoi gian (giay) servo "khung" lai sau MOI lan di chuyen thuc su,
        # cho camera vat ly xoay xong + AI kip xu ly ra frame moi phan anh
        # dung vi tri hien tai truoc khi danh gia tiep. Day la phan chinh
        # giup servo "cham ma chac", khong dao dong qua lai. Tang len (vd
        # 0.3-0.5) neu van con dao dong; giam xuong (vd 0.15) neu thay
        # phan hoi qua cham.
        "post_move_settle_sec": 0.25,

        # Ranh gioi dung: mac dinh DONG theo chinh o vuong xanh (bbox
        # nguoi dang khoa) do ai_pipeline.py truyen len moi frame -- servo
        # dung ngay khi tam khung hinh lot vao ben trong o xanh do.
        # dead_zone_px chi con dung lam PHUONG AN DU PHONG khi khong co
        # o xanh (vd dang scan/preempt).
        "dead_zone_px": 70,
        # San toi thieu cho ranh gioi dung DONG o tren -- tranh o xanh qua
        # nho (nguoi o xa) khien servo rung/khong bao gio thuc su on dinh.
        "min_dead_zone_px": 25,
        # Bien do "cham" (px) cong them vao ranh gioi o xanh -- chi can tam
        # khung hinh CHAM TOI MEP o xanh (hoac gan mep, trong khoang nay)
        # la dung, KHONG can di vao han ben trong o xanh moi dung. Tang len
        # neu muon dung tu xa hon nua; giam ve 0 neu muon dung dung luc lot
        # han vao ben trong.
        "touch_margin_px": 20,
        # Nguong NHA KHOA (px) -- CONG THEM vao dead_zone khi truc DANG
        # dung (settled) de quyet dinh khi nao moi coi la "roi khoi o
        # xanh that su" va bat dau di chuyen lai. Cang lon thi servo cang
        # "cung", it bi nhich theo nhung xe dich/rung nho sau khi da dung
        # -- chi thuc su di chuyen lai khi muc tieu doi cho ro rang.
        "release_margin_px": 90,

        "pan_min": 0,
        "pan_max": 180,
        "tilt_min": 30,
        "tilt_max": 150,
        "pan_center": 90,
        "tilt_center": 90,
        "invert_pan": True,
        "invert_tilt": False,
        # Toi thieu bao nhieu giay giua 2 lan gui lenh pan/tilt qua WebSocket
        # (tranh spam qua nhieu goi tin/giay toi ESP32).
        "send_interval_sec": 0.05,
    },
}

# Sau bao nhieu step lien tiep KHONG co muc tieu (has_target=False) thi
# servo tu tra ve vi tri trung tam an toan (pan_center/tilt_center) --
# tuong duong safety.return_to_center_on_lost + lost_target_frames_threshold
# o project pan-tilt goc.
SERVO_RETURN_TO_CENTER_AFTER_LOST_FRAMES = 15

# ---------------------------------------------------------------------------
# Target-loss safety behavior
# ---------------------------------------------------------------------------
# Seconds of consecutive "no target" frames tolerated before the FSM
# actually declares the target lost. Prevents 1-2 frame detection hiccups
# from flapping the state back and forth.
LOST_GRACE_PERIOD_SEC = 1.0

# ---------------------------------------------------------------------------
# CPU load / latency tuning
# ---------------------------------------------------------------------------
# Running multiple AIPipeline threads (one per camera) on CPU means they
# compete for the same cores -- YOLO-face/YOLO-person/InsightFace are all
# heavy enough that running 2+ at once noticeably slows each one down.
# These knobs reduce wasted CPU time without changing input resolution or
# detection accuracy.

# In SEARCHING, how many pipeline steps to skip between YOLO-face calls.
# 0 = run every step (old behavior). 2 means: run on step 0, skip steps 1
# and 2, run again on step 3, etc. -- roughly a 1/3 reduction in CPU spent
# on face detection while still scanning often enough that a newly-arrived
# registered person is found within a fraction of a second, not noticeably
# slower from a user's point of view.
SEARCHING_SKIP_FRAMES = 2
# In TRACKING, how many pipeline steps to skip between YOLO-face +
# InsightFace calls while the locked target's face is still visible.
# The Kalman filter bridges the gap so the smoothed center (crosshair +
# servo) keeps moving smoothly. 0 = run every step. 1 means: run on step
# 0, skip step 1, run again on step 2, etc. -- the single biggest CPU
# saving in the whole pipeline because YOLO-face + InsightFace +
# body-profile updates are all skipped at once.
TRACKING_SKIP_FRAMES = 1

# MediaPipe Pose (used for exercise rep counting + behavior recognition)
# is one of the heaviest per-step costs. Run it every Nth frame instead
# of every frame -- still catches every rep/behavior while cutting CPU
# significantly, especially with multiple rooms running at once.
# Applies only inside _run_exercise_tracking() when a registered person
# is locked (SEARCHING/LOST states don't run pose at all).
POSE_EVERY_N_FRAMES = 2
# Fixed delay (seconds) at the end of every pipeline step, applied only
# when the step itself was fast. If a step already took a while (because
# the CPU was busy with the other camera's thread), this sleep is skipped
# entirely instead of stacking on top of an already-slow frame. This
# replaces a flat time.sleep(0.01) that fired unconditionally even when
# the CPU had no time to spare.
STEP_SLEEP_SEC = 0.01
STEP_SLEEP_SKIP_IF_STEP_TOOK_LONGER_THAN_SEC = 0.03

# ---------------------------------------------------------------------------
# Behavior recognition (Dung / Di chuyen / Nhay / Gio tay / Nam)
# ---------------------------------------------------------------------------
BEHAVIOR_SNAPSHOT_INTERVAL_SEC = 5

# ---------------------------------------------------------------------------
# Floor plan & inferred presence
# ---------------------------------------------------------------------------
# How long (seconds) a "inferred presence" highlight stays on a no-cam room
# before being automatically cleared if no camera confirms the target.
# Can also be cleared manually via the dashboard Reset button.
# Set to 0 to disable auto-clear (keep until cam sees target again).
INFERRED_PRESENCE_TIMEOUT_SEC = 60

# Floor plan room definitions.
# Each room has:
#   id         -- must match a CAMERAS[*]["id"] if the room has a camera,
#                or any unique string for cam-less rooms.
#   name       -- display label on the floor map.
#   cam_id     -- set to the matching camera id if this room has a cam,
#                or None for cam-less rooms.
#   neighbors  -- list of room ids directly reachable from this room.
#                Used by the inference engine to decide which no-cam rooms
#                to highlight when a target disappears from a cam room.
FLOOR_PLAN = [
    {
        "id": "p1",
        "name": "Phong 1",
        "cam_id": "cam1",
        "neighbors": ["p2"],
    },
    {
        "id": "p2",
        "name": "Phong 2",
        "cam_id": "cam2",
        "neighbors": ["p1", "p3", "p4", "p5"],
    },
    {
        "id": "p3",
        "name": "Phong 3",
        "cam_id": None,
        "neighbors": ["p2"],
    },
    {
        "id": "p4",
        "name": "Phong 4",
        "cam_id": None,
        "neighbors": ["p2"],
    },
    {
        "id": "p5",
        "name": "Phong 5",
        "cam_id": None,
        "neighbors": ["p2"],
    },
]

# ---------------------------------------------------------------------------
# Handoff doi tuong Phong 1 (tinh) <-> Phong 2 (dong, servo pan/tilt)
# ---------------------------------------------------------------------------
# "type": toa do nao dang tin de suy luan huong mat dau.
#   - "static":  cam khong xoay -- KHONG chia bien trai/phai nua. He Phong 1
#                mat dau (bat ke ly do/huong gi) la coi nhu doi tuong co the
#                da sang Phong 2, chu dong quay don dau servo cam2 luon.
#   - "dynamic": dung goc Pan servo, vi cam luon xoay bam muc tieu nen
#                pixel khong phan anh vi tri that trong phong.
HANDOFF_CONFIG = {
    "cam1": {  # Phong 1 -- cam tinh
        "type": "static",
        "handoff_to_room": "cam2",
        "handoff_to_pan": 130,
    },
    "cam2": {  # Phong 2 -- cam dong (pan/tilt)
        "type": "dynamic",
        "pan_left_boundary": 40,
        "pan_right_boundary": 160,
        "boundary_confirm_frames": 8,
        "exit_towards_room": {
            "LEFT": "cam1",
            "RIGHT": None,
        },
        # BAT KE mat tich vi ly do/huong gi (ke ca direction="UNKNOWN" --
        # tuc mat dau binh thuong, khong phai do cham bien pan) deu roi
        # ve phong nay de bao truoc + bat dau dem gio cho quet.
        "default_target_room": "cam1",
    },
}
# Luu y: SERVO_CONFIG["control"]["pan_min"/"pan_max"] (0-180 do) van giu
# nguyen la gioi han co khi cua servo. HANDOFF_CONFIG["cam2"]["pan_left/
# right_boundary"] (40/160 do) la bien logic cua can phong, hep hon -- servo
# van co the quay xa hon ve mat co khi, nhung he thong coi viec vuot bien
# logic la "doi tuong da roi phong".
#
# Neu trong thuc te bien "ra khoi phong ve huong Phong 1" lai la BIEN PHAI
# (vi du do invert_pan=True lam nguoc chieu truc giac), chi can doi lai:
#   "exit_towards_room": {"LEFT": None, "RIGHT": "cam1"}
# -- khong can sua logic o servo_controller.py / ai_pipeline.py /
# app_dashboard.py, vi tat ca deu doc gia tri tu day.

# ---------------------------------------------------------------------------
# Ban giao Phong 2 (dong) -> Phong 1 (tinh): cho + tu quet tim
# ---------------------------------------------------------------------------
# Thoi gian (giay) cho phong DICH (vd Phong 1) tu quet ra nguoi vua duoc
# bao "co the dang quay lai" truoc khi phong NGUON (vd Phong 2, cam dong)
# chu dong xoay servo di tim thay vi ngoi cho thu dong.
HANDOFF_WAIT_BEFORE_SCAN_SEC = 10

# Gioi han TILT khi servo dang o che do QUET CHU DONG tim nguoi vua ban giao
# -- khac voi tilt_min/tilt_max trong SERVO_CONFIG (30-150, dung khi bam
# muc tieu binh thuong bang PID). Quet chu dong dung dai rong hon theo yeu
# cau: 40 -> 160.
HANDOFF_SCAN_TILT_MIN = 40
HANDOFF_SCAN_TILT_MAX = 160

# Buoc xoay (do) moi lan tick khi dang quet chu dong.
HANDOFF_SCAN_STEP_DEG = 3

# Chu ky (giay) giua 2 lan tick quet chu dong -- khong can nhanh nhu PID
# (send_interval_sec=0.05s) vi day chi la quet "mo" khong bam theo error do
# duoc, quet cham vua de mat nguoi van kip nhin thay khi camera luot qua.
HANDOFF_SCAN_TICK_INTERVAL_SEC = 0.3