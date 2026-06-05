import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from transformers import pipeline
# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Aircraft Detection System",
    page_icon="✈",
    layout="wide"
)

st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(
        180deg,
        #020617,
        #0f172a,
        #1e293b
    );
}

/* Stars Effect */
.stApp::before{
    content:"";
    position:fixed;
    width:100%;
    height:100%;
    top:0;
    left:0;
    background-image:
        radial-gradient(white 1px, transparent 1px),
        radial-gradient(#38bdf8 1px, transparent 1px);
    background-size:50px 50px;
    opacity:0.15;
    z-index:-1;
}

/* Neon Glow Cards */
[data-testid="stMetric"]{
    background: rgba(0,255,255,0.08);
    border:1px solid cyan;
    border-radius:15px;
    padding:20px;
    box-shadow:0 0 20px cyan;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#0f172a;
}

/* Headings */
h1,h2,h3{
    color:#38bdf8;
    text-shadow:0px 0px 15px cyan;
}

/* Radar Animation */
.radar{
    width:300px;
    height:300px;
    border-radius:50%;
    border:2px solid cyan;
    margin:auto;
    position:relative;
    overflow:hidden;
    box-shadow:0px 0px 30px cyan;
}

.radar::before{
    content:"";
    position:absolute;
    width:50%;
    height:50%;
    background:linear-gradient(
        90deg,
        transparent,
        rgba(0,255,255,0.8)
    );
    transform-origin:bottom right;
    animation:spin 3s linear infinite;
}

@keyframes spin{
    from{
        transform:rotate(0deg);
    }
    to{
        transform:rotate(360deg);
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# BANNER
# ==================================================



# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("✈ Aircraft Detection")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Image Detection", "Video Detection", "About Project"]
)
# ==================================================
# LOAD MODEL
# ==================================================

model = YOLO("yolov8n.pt")
text_gen = pipeline("text-generation", model="gpt2")

st.title("✈ Aircraft Detection + AI Explanation")

uploaded_file = st.file_uploader("Upload Aircraft Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # YOLO Detection
    results = yolo_model(image)
    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_container_width=True)

    aircraft_count = len(results[0].boxes)
    st.metric("Detected Objects", aircraft_count)

    # GPT Explanation
    prompt = f"{aircraft_count} aircraft detected. Explain why this is important for aerospace surveillance."
    explanation = text_gen(prompt, max_length=80, do_sample=True)[0]['generated_text']

    st.markdown("## 🧠 AI Explanation")
    st.write(explanation)

# ==================================================
# HOME PAGE
# ==================================================

if menu == "Home":
    
    
    st.markdown("""
    <div style="
    text-align:center;
    padding:30px;
    border-radius:20px;
    background:rgba(0,255,255,0.05);
    box-shadow:0 0 30px cyan;
    ">

    <h1>🛰 SPACE COMMAND CENTER</h1>

    <h3>Aerospace Surveillance AI</h3>

    <p>
    Real-Time Aircraft Monitoring & Detection System
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="radar"></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#001f3f,#00509e,#00b4d8);
    padding:40px;
    border-radius:20px;
    text-align:center;
    ">
    
    <h1 style="color:white;">
    ✈ Aircraft Detection System
    </h1>

    <h3 style="color:white;">
    AI Powered Aerospace Surveillance using YOLOv8
    </h3>

    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.image(
        "aircraftlogo.webp",
        use_container_width=True
    )

    # Aircraft Video Section

    st.markdown("## 🎥 Aircraft Detection Demo Video")
    
    video_file = open("aircraft_video/Battleefield.mp4", "rb")
    video_bytes = video_file.read()
    
    st.video(video_bytes)
    
    st.caption("✈ Aircraft Detection using YOLOv8 and Computer Vision")

    # Hero Banner
    
    
    st.write("")

    # Statistics

    st.markdown("## 📊 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", "YOLOv8")

    with col2:
        st.metric("Framework", "PyTorch")

    with col3:
        st.metric("Accuracy", "95%+")

    with col4:
        st.metric("Detection", "Real-Time")

    
    st.markdown("""
    <div style="
    background:gray;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.2);
    ">
    
    <h2 style="color:#003366;">
    🚀 Project Overview
    </h2>
    
    <p style="font-size:18px;color:black;">
    
    Aircraft Detection using Image and Videos is an AI-powered
    aerospace surveillance system.
    
    The system automatically detects aircraft using
    Computer Vision, Deep Learning and YOLOv8.
    
    It improves aviation safety, airport monitoring,
    and airspace surveillance.
    
    </p>
    
    </div>
    """, unsafe_allow_html=True)

    # Overview

   
    st.markdown("---")

    # Gallery

    st.markdown("## 🛩 Aircraft Gallery")

    image_folder = "images"

    if os.path.exists(image_folder):

        image_files = os.listdir(image_folder)

        cols = st.columns(3)

        for i, image_name in enumerate(image_files):

            image_path = os.path.join(
                image_folder,
                image_name
            )

            with cols[i % 3]:

                st.image(
                    image_path,
                    caption=image_name,
                    use_container_width=True
                )

    st.markdown("---")

    # Technologies

    st.markdown("## ⚙ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.success("Python")

        st.success("OpenCV")

        st.success("NumPy")

        st.success("Matplotlib")

    with col2:

        st.success("YOLOv8")

        st.success("PyTorch")

        st.success("Streamlit")

        st.success("Deep Learning")

    st.markdown("---")

    # Applications

    st.markdown("## 🌍 Real World Applications")

    st.success("✈ Airport Monitoring")

    st.success("🛰 Satellite Image Analysis")

    st.success("🛡 Military Surveillance")

    st.success("🚁 Drone Monitoring")

    st.success("📡 Airspace Security")

    st.info(
        "Open Image Detection page from sidebar to test aircraft images."
    )

# ==================================================
# IMAGE DETECTION PAGE
# ==================================================
   
elif menu == "Image Detection":
    
    st.markdown("""
    <div style="
    padding:25px;
    border-radius:15px;
    background:linear-gradient(90deg,#003366,#00509e);
    text-align:center;
    ">
    <h1 style="color:white;">
    🛫 Aircraft Detection Dashboard
    </h1>

    <h4 style="color:white;">
    AI Powered Aircraft Detection using YOLOv8
    </h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    st.image(
        "imagelogo.jpg",
        use_container_width=True
    )

    # Background CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #0a192f,
            #112240,
            #233554
        );
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Banner
    

    st.write("")

    uploaded_file = st.file_uploader(
        "📤 Upload Aircraft Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div style="
            background:white;
            padding:10px;
            border-radius:10px;
            text-align:center;
            ">
            <h3 style="color:#003366;">
            📷 Original Image
            </h3>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                image,
                use_container_width=True
            )

        results = model(image)

        result_img = results[0].plot()

        with col2:

            st.markdown("""
            <div style="
            background:white;
            padding:10px;
            border-radius:10px;
            text-align:center;
            ">
            <h3 style="color:#003366;">
            🎯 Detection Result
            </h3>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                result_img,
                use_container_width=True
            )

        aircraft_count = len(results[0].boxes)

        st.metric(
            "Detected Objects",
            aircraft_count
        )

        names = model.names

        st.markdown("## 📋 Detection Details")

        for box in results[0].boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            object_name = names[class_id]

        st.markdown(f"""
        <div style="
        background:white;
        padding:15px;
        border-radius:10px;
        margin-bottom:10px;
        box-shadow:0px 2px 8px rgba(0,0,0,0.2);
        ">
        
        <h4 style="
        color:#003366;
        font-weight:bold;
        ">
        ✈ Object : {object_name}
        </h4>
        
        <h3 style="
        color:#000000;
        font-weight:bold;
        ">
        📊 Confidence : {confidence:.2%}
        </h3>
        
        </div>
        """, unsafe_allow_html=True)
#video detection
# ==================================================
# VIDEO DETECTION PAGE
# ==================================================

elif menu == "Video Detection":

    st.title("🎥 Aircraft Video Detection")

    uploaded_video = st.file_uploader(
        "Upload Aircraft Video",
        type=["mp4", "avi", "mov"]
    )
    if uploaded_video is not None:

        video_path = "aircraft video.mp4"
    
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
    
        st.video(video_path)
    
        if st.button("Start Detection"):
    
            cap = cv2.VideoCapture(video_path)
    
            stframe = st.empty()
    
            while cap.isOpened():
    
                ret, frame = cap.read()
    
                if not ret:
                    break
    
                results = model(frame)
    
                annotated_frame = results[0].plot()
    
                stframe.image(
                    annotated_frame,
                    channels="BGR",
                    use_container_width=True
                )
    
            cap.release()
    
            # Detection Summary yahi hona chahiye
            st.markdown("## 📊 Detection Summary")
            detected_objects = {}
    
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
            
                ret, frame = cap.read()
            
                if not ret:
                    break
            
                results = model(frame)

                for box in results[0].boxes:
                    class_id = int(box.cls[0])

                    confidence = float(box.conf[0])
                
                    object_name = model.names[class_id]

    

                    if object_name not in detected_objects:
                
                        detected_objects[object_name] = {
                            "count": 1,
                            "confidence": confidence
                        }
                
                    else:
                
                        detected_objects[object_name]["count"] += 1
                
                        if confidence > detected_objects[object_name]["confidence"]:
                
                            detected_objects[object_name]["confidence"] = confidence
                    

   
                
                    
            cap.release()
            
            total_aircraft = sum(
                item["count"]
                for item in detected_objects.values()
            )
            
            st.metric(
                "✈ Total Aircraft Detected",
                total_aircraft
            )
            
            st.markdown("## 📋 Aircraft Details")
            
            for name, info in detected_objects.items():
            
                st.markdown(f"""
                <div style="
                background:#0f172a;
                padding:15px;
                border-radius:10px;
                margin-bottom:10px;
                border:1px solid cyan;
                ">
            
                <h3 style="color:cyan;">
                ✈ Aircraft : {name}
                </h3>
            
                <h4 style="color:white;">
                Count : {info['count']}
                </h4>
            
                <h4 style="color:white;">
                Confidence : {info['confidence']:.2%}
                </h4>
            
                </div>
                """, unsafe_allow_html=True)
            
            st.success("✅ Video Detection Completed")

            
     
        

    
    
    

   
# ==================================================
# ABOUT PAGE
# ==================================================

elif menu == "About Project":
    
    st.markdown("""
    <div style="
    background:linear-gradient(90deg,#001F3F,#00509E);
    padding:20px;
    border-radius:15px;
    text-align:center;
    ">
    <h1 style="color:white;">
    ✈ About Aircraft Detection Project
    </h1>
    <h4 style="color:white;">
    AI Powered Aerospace Surveillance System
    </h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
<div style="height:40px;"></div>
""", unsafe_allow_html=True)

    # About Banner
    st.image("pagelogo.jpg", use_container_width=True)

    
    st.markdown("---")

    # Problem Statement
    st.markdown("## 🎯 Problem Statement")

    st.info("""
    Monitoring aircraft manually in aerial or satellite images
    is time-consuming and prone to human errors.

    This project automates aircraft detection using
    Computer Vision and Deep Learning.
    """)

    # Objectives
    st.markdown("## 🎯 Project Objectives")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✈ Detect Aircraft Automatically")
        st.success("✈ Reduce Manual Monitoring")
        st.success("✈ Improve Airspace Security")

    with col2:
        st.success("✈ Increase Detection Accuracy")
        st.success("✈ Enhance Aviation Safety")
        st.success("✈ Enable Real-Time Monitoring")

    st.markdown("---")

    # Technologies
    st.markdown("## ⚙ Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Language", "Python")

    with col2:
        st.metric("Framework", "PyTorch")

    with col3:
        st.metric("Model", "YOLOv8")

    st.markdown("---")

    # Future Scope
    st.markdown("## 🚀 Future Scope")

    st.success("🛰 Satellite Image Aircraft Detection")
    st.success("🚁 Drone Monitoring System")
    st.success("🛡 Military Surveillance")
    st.success("🏢 Airport Security Systems")
    st.success("📡 Real-Time Video Analytics")
    st.success("🌍 Smart Aerospace Monitoring")

    st.markdown("---")

    # Team / Developer Section
    st.markdown("## 👩‍💻 Developer")

    st.info("""
    Project Name : Aircraft Detection using Image and Videos

    Theme : Aerospace

    Technologies : Python, OpenCV, PyTorch, YOLOv8,
    NumPy, Matplotlib, Streamlit

    Developed as an AI & Computer Vision Project.
    """)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <center>
    ✈ Aircraft Detection System |
    Developed using YOLOv8, OpenCV & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)

