import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit.components.v1 as components

import av
import cv2
import mediapipe as mp
import math
import os
import base64


# ======================
# CONFIG
# ======================

ALARM_FILE = "assets/alarm.wav"


def play_alarm():

    if os.path.exists(ALARM_FILE):

        with open(ALARM_FILE, "rb") as f:
            sound = f.read()

        encoded = base64.b64encode(sound).decode()


        components.html(
            f"""
            <audio id="alarmSound" autoplay loop>
                <source
                src="data:audio/wav;base64,{encoded}"
                type="audio/wav">
            </audio>

            <script>

            let audio =
            document.getElementById("alarmSound");

            audio.volume = 1.0;

            audio.play();

            </script>
            """,
            height=0,
        )



# ======================
# PAGE
# ======================

st.set_page_config(
    page_title="AI Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)


st.markdown(
"""
<h1 style='text-align:center;color:#00ffaa'>
🚗 AI Driver Drowsiness Detection System
</h1>
<hr>
""",
unsafe_allow_html=True
)



# ======================
# SIDEBAR
# ======================

st.sidebar.header(
    "⚙️ Detection Controls"
)


EAR_THRESHOLD = st.sidebar.slider(
    "👁 Eye Aspect Ratio Threshold",
    0.10,
    0.40,
    0.23,
    0.01
)


FRAME_LIMIT = st.sidebar.slider(
    "😴 Drowsy Frame Limit",
    5,
    50,
    15
)



# ======================
# STATE
# ======================

if "ear" not in st.session_state:
    st.session_state.ear = 0


if "frames" not in st.session_state:
    st.session_state.frames = 0


if "status" not in st.session_state:
    st.session_state.status = "🟢 ACTIVE"



# ======================
# MEDIAPIPE
# ======================

mp_face_mesh = mp.solutions.face_mesh


LEFT_EYE=[
    362,385,387,
    263,373,380
]


RIGHT_EYE=[
    33,160,158,
    133,153,144
]


def distance(a,b):

    return math.dist(a,b)



def EAR(points):

    A=distance(
        points[1],
        points[5]
    )

    B=distance(
        points[2],
        points[4]
    )

    C=distance(
        points[0],
        points[3]
    )

    return (A+B)/(2*C)



# ======================
# VIDEO PROCESSOR
# ======================


class Detector(VideoProcessorBase):


    def __init__(self):

        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

        self.counter = 0



    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        result = self.mesh.process(rgb)


        h,w,_ = img.shape


        if result.multi_face_landmarks:


            face=result.multi_face_landmarks[0]


            left=[]
            right=[]


            for i in LEFT_EYE:

                lm=face.landmark[i]

                point=(
                    int(lm.x*w),
                    int(lm.y*h)
                )


                left.append(point)


                cv2.circle(
                    img,
                    point,
                    3,
                    (0,255,0),
                    -1
                )



            for i in RIGHT_EYE:

                lm=face.landmark[i]

                point=(
                    int(lm.x*w),
                    int(lm.y*h)
                )


                right.append(point)


                cv2.circle(
                    img,
                    point,
                    3,
                    (0,255,0),
                    -1
                )



            ear = (
                EAR(left)
                +
                EAR(right)
            ) / 2



            if ear < EAR_THRESHOLD:

                self.counter += 1

            else:

                self.counter = 0



            if self.counter >= FRAME_LIMIT:

                status="🚨 DROWSY"

                color=(0,0,255)


            else:

                status="🟢 ACTIVE"

                color=(0,255,0)



            st.session_state.ear = round(
                ear,
                3
            )


            st.session_state.frames = self.counter


            st.session_state.status = status



            cv2.putText(
                img,
                status,
                (30,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )


            cv2.putText(
                img,
                f"EAR: {ear:.2f}",
                (30,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2
            )



        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )



# ======================
# DASHBOARD
# ======================

camera,panel = st.columns(
    [3,1]
)


with camera:

    st.subheader(
        "📹 Live Driver Camera"
    )


    webrtc_streamer(

        key="camera",

        video_processor_factory=Detector,

        media_stream_constraints={
            "video":True,
            "audio":False
        }
    )



with panel:


    st.subheader(
        "📊 Monitoring Panel"
    )


    st.metric(
        "👁 Eye Aspect Ratio",
        st.session_state.ear
    )


    st.metric(
        "😴 Drowsy Frames",
        st.session_state.frames
    )


    st.metric(
        "🚨 Status",
        st.session_state.status
    )



    if st.session_state.status == "🚨 DROWSY":


        st.error(
            "🚨 WAKE UP DRIVER"
        )


        play_alarm()


    else:


        st.success(
            "Driver Awake 🟢"
        )



st.markdown("---")

st.caption(
    "MediaPipe FaceMesh + EAR Algorithm + Streamlit WebRTC + Docker 🐳"
)