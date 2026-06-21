import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import base64


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="AI Driver Monitoring",
    page_icon="🚗",
    layout="wide"
)


# ===============================
# ALARM
# ===============================

def play_alarm():

    try:
        with open("assets/alarm.wav", "rb") as f:
            sound = f.read()

        encoded = base64.b64encode(sound).decode()

        st.markdown(
            f"""
            <audio autoplay>
            <source 
            src="data:audio/wav;base64,{encoded}" 
            type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True
        )

    except:
        pass



# ===============================
# UI
# ===============================

st.title(
    "🚗 AI Driver Drowsiness Detection System"
)

st.caption(
    "MediaPipe + Eye Aspect Ratio (EAR) + OpenCV"
)



st.sidebar.header(
    "⚙️ Configuration"
)


EAR_THRESHOLD = st.sidebar.slider(
    "Eye Aspect Ratio Threshold",
    0.10,
    0.40,
    0.25
)


FRAME_LIMIT = st.sidebar.slider(
    "Drowsy Frames",
    5,
    60,
    20
)



col1,col2,col3 = st.columns(3)

status_box = col1.empty()
ear_box = col2.empty()
alert_box = col3.empty()


camera_window = st.empty()



# ===============================
# MEDIAPIPE
# ===============================

mp_face_mesh = mp.solutions.face_mesh


face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)



LEFT_EYE=[
    33,160,158,133,153,144
]


RIGHT_EYE=[
    362,385,387,263,373,380
]



def calculate_EAR(points):

    points=np.array(points)

    A=np.linalg.norm(
        points[1]-points[5]
    )

    B=np.linalg.norm(
        points[2]-points[4]
    )

    C=np.linalg.norm(
        points[0]-points[3]
    )


    return (A+B)/(2*C)




counter=0


start = st.button(
    "▶ Start Monitoring"
)



# ===============================
# CAMERA LOOP
# ===============================


if start:


    cap=cv2.VideoCapture(0)


    while True:


        success,frame=cap.read()


        if not success:

            st.error(
                "Camera not available"
            )

            break



        rgb=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result=face_mesh.process(
            rgb
        )



        status="NO FACE"

        ear_value=0



        if result.multi_face_landmarks:


            face=result.multi_face_landmarks[0]


            h,w,_=frame.shape



            left=[
                [
                int(face.landmark[i].x*w),
                int(face.landmark[i].y*h)
                ]

                for i in LEFT_EYE
            ]


            right=[
                [
                int(face.landmark[i].x*w),
                int(face.landmark[i].y*h)
                ]

                for i in RIGHT_EYE
            ]



            ear_value=(

                calculate_EAR(left)
                +
                calculate_EAR(right)

            )/2



            for p in left+right:

                cv2.circle(
                    frame,
                    tuple(p),
                    3,
                    (0,255,0),
                    -1
                )



            if ear_value < EAR_THRESHOLD:


                counter += 1


                if counter > FRAME_LIMIT:

                    status="DROWSY 🚨"

                    play_alarm()


            else:

                counter=0

                status="AWAKE 😊"





        status_box.metric(
            "Driver Status",
            status
        )


        ear_box.metric(
            "👁 EAR Value",
            round(
                ear_value,
                3
            )
        )



        if "DROWSY" in status:

            alert_box.error(
                "WAKE UP!"
            )

        else:

            alert_box.success(
                "Safe"
            )




        camera_window.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )



    cap.release()