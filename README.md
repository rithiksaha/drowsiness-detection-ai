\# 🚗 AI Driver Drowsiness Detection System



!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

!\[Docker](https://img.shields.io/badge/Docker-Ready-blue)



\---



\## 📌 Project Overview



AI Driver Drowsiness Detection System is a real-time computer vision application designed to monitor driver alertness using facial landmarks.



The system calculates the \*\*Eye Aspect Ratio (EAR)\*\* from live webcam input to determine whether the driver's eyes are open or closed. If drowsiness is detected continuously, an alarm is triggered.



This project demonstrates:



\- Real-Time Computer Vision

\- Facial Landmark Detection

\- Driver Monitoring System

\- AI Safety Application

\- Dockerized ML Deployment



\---



\## 🎯 Features



✔ Real-time webcam monitoring  

✔ Face landmark detection  

✔ Eye Aspect Ratio (EAR) calculation  

✔ Drowsiness detection  

✔ Alarm alert system  

✔ Interactive Streamlit dashboard  

✔ Adjustable EAR threshold  

✔ Adjustable drowsiness sensitivity  

✔ Docker container support  





\---



\## 🧠 How It Works



Camera Input



⬇



MediaPipe Face Mesh



⬇



Eye Landmark Extraction



⬇



EAR Calculation



⬇



Eye State Detection



⬇



Drowsiness Alert





\---



\## 👁 Eye Aspect Ratio (EAR)



EAR measures eye openness using eye landmark distances.



Formula:





EAR = (A + B) / (2.0 \* C)





Where:



\- A = vertical eye distance

\- B = vertical eye distance

\- C = horizontal eye width





Decision:



```

EAR >= Threshold

→ Driver Awake 😊



EAR < Threshold

→ Eye Closed 😴

```



If eyes remain closed for multiple frames:



```

Trigger Alarm 🚨

```



\---



\## 🖥 Dashboard Preview



Features:



\- Live camera feed

\- Driver status

\- EAR value monitoring

\- Alert panel

\- Sensitivity controls





\---



\## 🛠 Tech Stack



| Technology | Purpose |

|----------|---------|

| Python | Programming |

| OpenCV | Image Processing |

| MediaPipe | Face Landmark Detection |

| NumPy | Numerical Processing |

| Streamlit | Web Dashboard |

| Docker | Container Deployment |



\---



\## 📂 Project Structure



```

drowsiness-detection/



│

├── assets/

│   └── alarm.wav

│

├── src/

│

├── streamlit\_app.py

├── app.py

├── train\_model.py

│

├── requirements.txt

├── Dockerfile

├── README.md

├── .gitignore

└── .dockerignore



```



\---



\## ⚙️ Installation



Clone repository:



```bash

git clone https://github.com/YOUR\_USERNAME/drowsiness-detection.git

```



Go inside project:



```bash

cd drowsiness-detection

```



Create virtual environment:



```bash

python -m venv venv

```



Activate:



Windows:



```bash

venv\\Scripts\\activate

```



Linux:



```bash

source venv/bin/activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## 🚀 Run Application



Start dashboard:



```bash

streamlit run streamlit\_app.py

```



Open browser:



```

http://localhost:8501

```



\---



\# 🐳 Docker Deployment





Build Docker Image:



```bash

docker build -t drowsiness-ai .

```





Run Container:



```bash

docker run -p 8501:8501 drowsiness-ai

```





Open:



```

http://localhost:8501

```





\---



\## 📊 Algorithm



```

Start Camera



&#x20;     ↓



Detect Face



&#x20;     ↓



Detect Eye Landmarks



&#x20;     ↓



Calculate EAR



&#x20;     ↓



EAR < Threshold?



&#x20;     ↓



Count Frames



&#x20;     ↓



Trigger Alarm

```





\---



\## Future Improvements



\- Cloud deployment

\- Mobile camera support

\- Driver fatigue prediction

\- Head pose estimation

\- Yawning detection

\- Analytics dashboard





\---



\## Author



Developed by \*\*Rithik Saha\*\*



AI | Computer Vision | MLOps | Cloud





⭐ If you like this project, give it a star!



