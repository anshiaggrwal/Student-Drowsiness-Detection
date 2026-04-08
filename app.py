import streamlit as st
import cv2
import numpy as np
import time
import math
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import urllib.request
import json

# ─────────────────────────────────────────
#  Safe mediapipe import
# ─────────────────────────────────────────
try:
    import mediapipe as mp
    _test = mp.solutions.face_mesh
except AttributeError:
    st.error("❌ mediapipe version incompatible. Set mediapipe==0.10.13 in requirements.txt", icon="🚨")
    st.stop()

mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose           = mp.solutions.pose
mp_holistic       = mp.solutions.holistic

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Drowsiness & Posture Monitor",
    page_icon="😴",
    layout="wide",
)

st.markdown("""
<style>
    .alert-box {
        background: #cc0000; color: white; padding: 14px 20px;
        border-radius: 8px; font-size: 1.2rem; font-weight: bold;
        text-align: center; animation: pulse 0.6s infinite alternate;
    }
    @keyframes pulse { from {opacity:1;} to {opacity:0.55;} }
    .ok-box {
        background: #155215; color: #aaffaa; padding: 12px 20px;
        border-radius: 8px; font-size: 1rem; text-align: center;
    }
    .wait-box {
        background: #333; color: #aaa; padding: 12px 20px;
        border-radius: 8px; font-size: 1rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TURN / ICE server config
#  Using metered.ca free TURN — works on
#  Render, Railway, Heroku, etc.
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_ice_servers():
    """
    Fetch free TURN credentials from metered.ca.
    Falls back to Google STUN only if the fetch fails.
    metered.ca free tier: 500MB/month — enough for demos.
    """
    try:
        url = "https://drowsiness.metered.live/api/v1/turn/credentials?apiKey=placeholder"
        # Use open.metered.ca public demo endpoint (no key needed for basic STUN+TURN)
        resp = urllib.request.urlopen(
            "https://openrelay.metered.ca/api/v1/turn/credentials?apiKey=openrelayproject",
            timeout=5
        )
        servers = json.loads(resp.read().decode())
        return servers
    except Exception:
        # Fallback: multiple public STUN servers
        return [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            {"urls": "stun:stun2.l.google.com:19302"},
            {"urls": "stun:stun3.l.google.com:19302"},
            {"urls": "stun:stun4.l.google.com:19302"},
            {"urls": "stun:openrelay.metered.ca:80"},
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443?transport=tcp",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]

# ─────────────────────────────────────────
#  Cached MediaPipe models
# ─────────────────────────────────────────
@st.cache_resource
def load_face_mesh():
    return mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

face_mesh = load_face_mesh()

# ─────────────────────────────────────────
#  Detection constants
# ─────────────────────────────────────────
EYE_AR_THRESH          = 0.27
EYE_AR_CONSEC_FRAMES   = 20
MOUTH_AR_THRESH        = 0.60
MOUTH_AR_CONSEC_FRAMES = 15

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [13,  14,  78,  308,  82,  312, 87, 317]

class Config:
    HEAD_DOWN_ANGLE       = -60
    HEAD_BACK_ANGLE       = -20
    HEAD_LEFT_ANGLE       =  20
    HEAD_RIGHT_ANGLE      = -20
    HAND_PROXIMITY_FACTOR =  0.8

def aspect_ratio(landmarks, indices):
    try:
        A = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
        B = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
        C = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
        return (A + B) / (2.0 * C)
    except IndexError:
        return 0.0

# ─────────────────────────────────────────
#  PostureDetector
# ─────────────────────────────────────────
class PostureDetector:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _dist(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def detect_head_position(self, results):
        if not results.face_landmarks:
            return "Unknown"
        fl    = results.face_landmarks.landmark
        nose  = np.array([fl[1].x,   fl[1].y,   fl[1].z])
        l_eye = np.array([fl[33].x,  fl[33].y,  fl[33].z])
        r_eye = np.array([fl[263].x, fl[263].y, fl[263].z])
        mid   = (l_eye + r_eye) / 2
        fwd   = nose - mid
        n     = np.linalg.norm(fwd)
        if n < 1e-6:
            return "Upright"
        fwd  /= n
        cam_up = np.array([0.0, -1.0, 0.0])
        pitch  = math.degrees(math.asin(np.clip(np.dot(fwd, cam_up), -1.0, 1.0)))
        fwd_h  = fwd - np.dot(fwd, cam_up) * cam_up
        hn     = np.linalg.norm(fwd_h)
        if hn < 1e-6:
            return "Upright"
        fwd_h /= hn
        yaw = math.degrees(math.atan2(fwd_h[0], -fwd_h[2]))
        if   yaw   >  Config.HEAD_LEFT_ANGLE:  return "Left"
        elif yaw   <  Config.HEAD_RIGHT_ANGLE: return "Right"
        elif pitch <  Config.HEAD_DOWN_ANGLE:  return "Down"
        elif pitch > -Config.HEAD_BACK_ANGLE:  return "Back"
        return "Upright"

    def detect_hand_position(self, results):
        if not results.pose_landmarks or not results.face_landmarks:
            return "Unknown"
        pl   = results.pose_landmarks.landmark
        fl   = results.face_landmarks.landmark
        nose = [pl[mp_pose.PoseLandmark.NOSE.value].x,
                pl[mp_pose.PoseLandmark.NOSE.value].y]
        chin = [fl[152].x, fl[152].y]
        lw   = [pl[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                pl[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        rw   = [pl[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                pl[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        le   = [pl[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                pl[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        re   = [pl[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                pl[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        thr  = self._dist(le, re) * Config.HAND_PROXIMITY_FACTOR
        if self._dist(lw, chin) < thr or self._dist(rw, chin) < thr:
            return "Hands on Chin"
        if self._dist(lw, rw) < thr and lw[1] < nose[1] and rw[1] < nose[1]:
            return "Hands Crossed"
        if self._dist(lw, le) < thr or self._dist(rw, re) < thr:
            return ("Head on Hands"
                    if self.detect_head_position(results) == "Down"
                    else "Hands on Head")
        return "Neutral"

    def analyze_posture(self, results):
        head = self.detect_head_position(results)
        hand = self.detect_hand_position(results)
        if head == "Upright" and hand in ("Neutral", "Unknown"):
            return "Attentive: Upright posture"
        parts = []
        if head != "Upright":
            parts.append(f"{head} head")
        if hand not in ("Neutral", "Unknown"):
            parts.append(hand)
        return f"Posture: {', '.join(parts)}" if parts else "Neutral posture"

    def close(self):
        self.holistic.close()

# ─────────────────────────────────────────
#  Thread-safe state
# ─────────────────────────────────────────
class DetectionState:
    def __init__(self):
        self._lock            = threading.Lock()
        self.eye_counter      = 0
        self.yawn_counter     = 0
        self.yawn_sequence    = 0
        self.alarm_on         = False
        self.last_yawn_time   = 0.0
        self.alarm_start_time = 0.0
        self.eye_ar           = 0.0
        self.mouth_ar         = 0.0
        self.posture          = "Waiting…"

    def update(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snap(self):
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

@st.cache_resource
def get_resources():
    return DetectionState(), PostureDetector()

detection_state, posture_detector = get_resources()

# ─────────────────────────────────────────
#  Video callback
# ─────────────────────────────────────────
def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hol = posture_detector.holistic.process(rgb)
    if hol.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, hol.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    for hand_lm in [hol.left_hand_landmarks, hol.right_hand_landmarks]:
        if hand_lm:
            mp_drawing.draw_landmarks(img, hand_lm, mp_holistic.HAND_CONNECTIONS)

    posture = posture_detector.analyze_posture(hol)
    fm      = face_mesh.process(rgb)

    s             = detection_state.snap()
    eye_counter   = s["eye_counter"]
    yawn_counter  = s["yawn_counter"]
    yawn_sequence = s["yawn_sequence"]
    alarm_on      = s["alarm_on"]
    last_yawn     = s["last_yawn_time"]
    alarm_start   = s["alarm_start_time"]
    ear_val       = s["eye_ar"]
    mar_val       = s["mouth_ar"]
    now           = time.time()

    if fm and fm.multi_face_landmarks:
        for face_lm in fm.multi_face_landmarks:
            lm_arr = np.array([(lm.x * w, lm.y * h) for lm in face_lm.landmark])
            mp_drawing.draw_landmarks(
                img, face_lm, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            if len(lm_arr) < max(max(LEFT_EYE), max(RIGHT_EYE), max(MOUTH)):
                continue

            ear_val = (aspect_ratio(lm_arr, LEFT_EYE) +
                       aspect_ratio(lm_arr, RIGHT_EYE)) / 2.0
            mar_val  = aspect_ratio(lm_arr, MOUTH)

            if ear_val < EYE_AR_THRESH:
                eye_counter += 1
                if eye_counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(img, "EYES CLOSED!", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if not alarm_on:
                        alarm_on    = True
                        alarm_start = now
            else:
                eye_counter = 0

            if mar_val > MOUTH_AR_THRESH:
                yawn_counter += 1
                if yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
                    cv2.putText(img, "YAWNING!", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    yawn_sequence = (yawn_sequence + 1) if now - last_yawn < 5 else 1
                    last_yawn     = now
                    if yawn_sequence >= 4 and not alarm_on:
                        alarm_on    = True
                        alarm_start = now
                    yawn_counter = 0
            else:
                yawn_counter = 0
                if now - last_yawn >= 4 and yawn_sequence > 0:
                    yawn_sequence = 0

            if alarm_on and eye_counter == 0 and yawn_counter == 0:
                if now - alarm_start > 1:
                    alarm_on = False

    cv2.putText(img, f"EAR: {ear_val:.2f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"MAR: {mar_val:.2f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, posture,               (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
    if yawn_sequence > 0:
        cv2.putText(img, f"Yawns: {yawn_sequence}/4", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    if alarm_on:
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)
        cv2.putText(img, "!! DROWSINESS ALERT !!", (w // 2 - 210, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3)

    detection_state.update(
        eye_counter=eye_counter, yawn_counter=yawn_counter,
        yawn_sequence=yawn_sequence, alarm_on=alarm_on,
        last_yawn_time=last_yawn, alarm_start_time=alarm_start,
        eye_ar=ear_val, mouth_ar=mar_val, posture=posture,
    )
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    EYE_AR_THRESH          = st.slider("Eye AR Threshold",          0.10, 0.40, 0.27, 0.01)
    MOUTH_AR_THRESH        = st.slider("Mouth AR Threshold (Yawn)", 0.40, 0.90, 0.60, 0.01)
    EYE_AR_CONSEC_FRAMES   = st.slider("Eye Closed Frames",         5,    60,   20)
    MOUTH_AR_CONSEC_FRAMES = st.slider("Yawn Frames",               5,    40,   15)
    st.markdown("---")
    st.markdown("**Alert Conditions**")
    st.markdown("🔴 Eyes closed > N frames")
    st.markdown("🔴 4 yawns within 5 s")
    st.markdown("🟠 Non-upright head posture")

# ─────────────────────────────────────────
#  Title
# ─────────────────────────────────────────
st.title("😴 Drowsiness & Posture Detection")
st.caption("Real-time driver / student monitoring · Sound alerts play in your browser automatically.")

# ─────────────────────────────────────────
#  Browser beep (Web Audio API)
# ─────────────────────────────────────────
st.components.v1.html("""
<script>
(function () {
    var audioCtx = null, active = false;
    function initAudio() {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
    }
    function beepOnce() {
        if (!active || !audioCtx) return;
        var osc = audioCtx.createOscillator();
        var gain = audioCtx.createGain();
        osc.type = 'square'; osc.frequency.value = 880;
        gain.gain.value = 0.5;
        osc.connect(gain); gain.connect(audioCtx.destination);
        osc.start(); osc.stop(audioCtx.currentTime + 0.2);
        setTimeout(beepOnce, 500);
    }
    function startAlarm() { if (active) return; initAudio(); active = true; beepOnce(); }
    function stopAlarm()  { active = false; }
    window.addEventListener('click',   initAudio);
    window.addEventListener('keydown', initAudio);
    setInterval(function () {
        var el = parent.document.getElementById('alarm_flag');
        if (!el) return;
        el.innerText.trim() === '1' ? startAlarm() : stopAlarm();
    }, 300);
})();
</script>
""", height=0)

# ─────────────────────────────────────────
#  Main layout
# ─────────────────────────────────────────
col_cam, col_stats = st.columns([3, 1])

with col_cam:
    st.info("📷 Click **START** → allow camera → detection begins automatically.", icon="ℹ️")

    # Get ICE servers (with TURN fallback)
    ice_servers = get_ice_servers()

    ctx = webrtc_streamer(
        key="drowsiness-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": ice_servers}),
        video_frame_callback=video_callback,
        media_stream_constraints={
            "video": {
                "width":     {"ideal": 480},
                "height":    {"ideal": 360},
                "frameRate": {"ideal": 10},   # lowered to reduce server load
            },
            "audio": False,
        },
        async_processing=True,
    )

with col_stats:
    st.subheader("📊 Live Stats")
    ph_ear     = st.empty()
    ph_mar     = st.empty()
    ph_posture = st.empty()
    ph_alarm   = st.empty()
    ph_yawn    = st.empty()
    ph_flag    = st.empty()

# ─────────────────────────────────────────
#  Live stats loop
# ─────────────────────────────────────────
if ctx.state.playing:
    while ctx.state.playing:
        snap = detection_state.snap()

        ph_ear.metric("👁️ EAR (eyes)", f"{snap['eye_ar']:.3f}",
                      delta="⚠ closed"  if snap['eye_ar']   < EYE_AR_THRESH  else "open")
        ph_mar.metric("👄 MAR (yawn)", f"{snap['mouth_ar']:.3f}",
                      delta="⚠ yawning" if snap['mouth_ar'] > MOUTH_AR_THRESH else "normal")

        icon = "🟢" if "Attentive" in snap["posture"] else "🟠"
        ph_posture.info(f"{icon} {snap['posture']}")

        if snap["alarm_on"]:
            ph_alarm.markdown('<div class="alert-box">🚨 DROWSINESS ALERT!</div>',
                              unsafe_allow_html=True)
        else:
            ph_alarm.markdown('<div class="ok-box">✅ Alert & Awake</div>',
                              unsafe_allow_html=True)

        if snap["yawn_sequence"] > 0:
            ph_yawn.warning(f"😮 Yawn count: {snap['yawn_sequence']}/4")
        else:
            ph_yawn.empty()

        flag = "1" if snap["alarm_on"] else "0"
        ph_flag.markdown(f'<div id="alarm_flag" style="display:none">{flag}</div>',
                         unsafe_allow_html=True)
        time.sleep(0.25)
else:
    ph_ear.metric("👁️ EAR", "—")
    ph_mar.metric("👄 MAR", "—")
    ph_posture.info("⏸ Camera not started")
    ph_alarm.markdown('<div class="wait-box">⏸ Waiting for stream…</div>',
                      unsafe_allow_html=True)

with st.expander("ℹ️ How it works"):
    st.markdown("""
| Detection | Trigger | Alert |
|-----------|---------|-------|
| **Eyes closed** | EAR < threshold for N frames | 🔴 Red border + browser beep |
| **Yawning** | MAR > threshold × N frames, 4× within 5 s | 🔴 Red border + browser beep |
| **Head tilt** | Left / Right / Down / Back | 🟠 Label on video |
| **Hands on face** | Wrist near chin / ears | 🟠 Label on video |

- **EAR** = Eye Aspect Ratio (lower → more closed)
- **MAR** = Mouth Aspect Ratio (higher → more open / yawning)
""")