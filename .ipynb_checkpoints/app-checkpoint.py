import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import matplotlib.pyplot as plt
# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Aircraft Detection projecty",
    page_icon="✈",
    layout="wide"
)
st.error("✈ AIRCRAFT_DETECTION FOLDER APP")
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
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.title("✈ Aircraft Detection")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Image Detection", "Video Detection", "About Project"]
)
# ==================================================
# LOAD MODEL
# ==================================================

model = YOLO("yolov8n.pt")


# ==================================================
# HOME PAGE
# ==================================================
if menu == "Home":

    # ================= MAIN TITLE =================
    st.markdown("""
    <div style="
        text-align:center;
        padding:30px;
        border-radius:20px;
        background:rgba(0,255,255,0.05);
        border:1px solid #00f5ff;
        box-shadow:0 0 25px #00f5ff;
    ">
        <h1 style="color:#00f5ff; font-size:40px;">
        ✈ AIRCRAFT DETECTION SYSTEM
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= LOGO =================
    st.image("aircraftlogo.webp", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= SPACE TITLE =================
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#38bdf8; text-shadow:0 0 15px #00f5ff;">
        🛰 SPACE COMMAND CENTER
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # ================= RADAR =================
    st.markdown("""
    <div class="radar"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#00f5ff;">
        🛰 SCANNING SKY...
    </div>
    """, unsafe_allow_html=True)

    # ================= STATUS CARDS =================
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "YOLOv8")
    col2.metric("Framework", "PyTorch")
    col3.metric("Accuracy", "95%+")
    col4.metric("Mode", "LIVE AI")

    # ================= GALLERY SECTION =================
    st.markdown("## 🛩 AIRCRAFT GALLERY")

    image_folder = "images"

    if os.path.exists(image_folder):

        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        image_files.sort()

        cols = st.columns(3)

        for i, img_name in enumerate(image_files):

            img_path = os.path.join(image_folder, img_name)

            with cols[i % 3]:
                st.image(img_path, use_container_width=True)

    else:
        st.warning("❌ Images folder not found")

    # ================= OVERVIEW =================
    st.markdown("""
    <div style="
        background:rgba(0,255,255,0.06);
        border:1px solid #00f5ff;
        padding:25px;
        border-radius:20px;
        box-shadow:0 0 25px #00f5ff;
        backdrop-filter: blur(8px);
        text-align:center;
    ">

    <h2 style="color:#00f5ff;">
    🛰 SYSTEM OVERVIEW
    </h2>

    <p style="color:white; font-size:17px; line-height:1.6;">
    This AI system monitors airspace using deep learning.<br>
    It detects aircraft in real-time using YOLOv8 and computer vision.<br>
    Designed for aviation safety, military surveillance, and smart airports.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ================= METRICS AGAIN =================
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## 📊 SYSTEM DASHBOARD")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "YOLOv8")
    col2.metric("Framework", "PyTorch")
    col3.metric("Accuracy", "95%+")
    col4.metric("Mode", "LIVE AI")

# ==================================================
# IMAGE DETECTION PAGE
# ==================================================
elif menu == "Image Detection":

    # ================= HEADER =================
    st.markdown("""
    <div style="
        text-align:center;
        padding:25px;
        border-radius:20px;
        background:rgba(0,255,255,0.05);
        border:1px solid #00f5ff;
        box-shadow:0 0 25px #00f5ff;
    ">
        <h1 style="color:#00f5ff;">
        🛫 AIRCRAFT DETECTION DASHBOARD
        </h1>

        
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= INFO =================
    st.markdown("""
    <div style="text-align:center; color:white; opacity:0.8;">
        📡 Upload image to detect aircraft using AI model
    </div>
    """, unsafe_allow_html=True)

    # ================= UPLOAD IMAGE =================
    uploaded_file = st.file_uploader(
        "📤 Upload Aircraft Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Load image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        # ================= ORIGINAL IMAGE =================
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(image, use_container_width=True)

        # ================= AI DETECTION =================
        results = model(image)
        result_img = results[0].plot()

        with col2:
            st.markdown("### 🎯 Detection Result")
            st.image(result_img, use_container_width=True)

        # ================= METRICS =================
        st.markdown("---")
        st.metric("✈ Detected Objects", len(results[0].boxes))

        # ================= DETAILS =================
        st.markdown("## 📋 Detection Details")

        names = model.names

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = names[class_id]

            st.markdown(f"""
            <div style="
                background:rgba(0,255,255,0.05);
                border:1px solid #00f5ff;
                padding:15px;
                border-radius:15px;
                margin-bottom:10px;
            ">
                <h3 style="color:#00f5ff;">
                    ✈ Object: {object_name}
                </h3>
                <h4 style="color:white;">
                    Confidence: {confidence:.2%}
                </h4>
            </div>
            """, unsafe_allow_html=True)
#video detection
# ==================================================
# VIDEO DETECTION PAGE
# ==================================================
elif menu == "Video Detection":

    # ================= HEADER =================
    st.markdown("""
    <div style="
        text-align:center;
        padding:25px;
        border-radius:20px;
        background:rgba(0,255,255,0.05);
        border:1px solid #00f5ff;
        box-shadow:0 0 25px #00f5ff;
    ">
        <h1 style="
            color:#00f5ff;
            text-shadow:0 0 15px #00f5ff;
        ">
        🎥 SKY AIRCRAFT LIVE MONITOR
        </h1>

       
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "📤 Upload Aircraft Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        video_path = "aircraft_video.mp4"

        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(video_path)

        # ================= START BUTTON =================
        if st.button("🚀 START SKY SCANNING"):

            cap = cv2.VideoCapture(video_path)

            stframe = st.empty()
            graph_area = st.empty()
            status_box = st.empty()

            frame_list = []
            aircraft_list = []

            frame_no = 0

            # ================= LIVE DETECTION LOOP =================
            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_no += 1

                results = model(frame)

                annotated_frame = results[0].plot()

                # ================= LIVE FRAME =================
                stframe.image(
                    annotated_frame,
                    channels="BGR",
                    use_container_width=True
                )

                # ================= COUNT =================
                aircraft_count = len(results[0].boxes)

                frame_list.append(frame_no)
                aircraft_list.append(aircraft_count)

                # ================= STATUS BOX =================
                status_box.markdown(f"""
                <div style="
                    text-align:center;
                    padding:15px;
                    border-radius:15px;
                    background:rgba(0,255,255,0.08);
                    border:1px solid #00f5ff;
                    box-shadow:0 0 20px #00f5ff;
                ">
                    <h2 style="color:#00f5ff;">
                    ✈ SKY STATUS: {aircraft_count} AIRCRAFT DETECTED
                    </h2>
                </div>
                """, unsafe_allow_html=True)

                # ================= LIVE GRAPH =================
                fig, ax = plt.subplots()

                ax.plot(frame_list, aircraft_list,
                        color='#00f5ff',
                        linewidth=2)

                ax.set_title("📊 Live Sky Traffic Graph", color='#00f5ff')
                ax.set_xlabel("Frame")
                ax.set_ylabel("Aircraft Count")
                ax.grid(True)

                graph_area.pyplot(fig)

            cap.release()

            # ================= FINAL SUMMARY =================
            st.markdown("## 📊 FINAL AIRSPACE REPORT")

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

            total_aircraft = sum(item["count"] for item in detected_objects.values())

            st.markdown(f"""
            <div style="
                text-align:center;
                padding:20px;
                border-radius:20px;
                background:rgba(0,255,255,0.08);
                border:1px solid #00f5ff;
                box-shadow:0 0 25px #00f5ff;
            ">
                <h1 style="color:#00f5ff;">
                ✈ TOTAL SKY TRAFFIC: {total_aircraft}
                </h1>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("## 📋 AIRCRAFT DETAILS")

            for name, info in detected_objects.items():

                st.markdown(f"""
                <div style="
                    background:rgba(0,255,255,0.05);
                    border:1px solid #00f5ff;
                    padding:15px;
                    border-radius:15px;
                    margin-bottom:10px;
                    box-shadow:0 0 15px #00f5ff;
                ">

                <h3 style="color:#00f5ff;">
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

            st.success("✅ SKY SCANNING COMPLETED")
   
# ==================================================
# ABOUT PAGE
# ==================================================
elif menu == "About Project":

    # ================= HEADER =================
    st.markdown("""
    <div style="
        text-align:center;
        padding:30px;
        border-radius:20px;
        background:rgba(0,255,255,0.05);
        border:1px solid #00f5ff;
        box-shadow:0 0 30px #00f5ff;
    ">
        <h1 style="color:#00f5ff; text-shadow:0 0 15px #00f5ff;">
        ✈ ABOUT AIRCRAFT DETECTION PROJECT
        </h1>

       
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= BANNER =================
    st.image("pagelogo.jpg", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= SECTION CARD STYLE =================
    st.markdown("""
    <style>
    .card {
        background: rgba(0,255,255,0.05);
        border: 1px solid rgba(0,255,255,0.2);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(0,255,255,0.2);
    }

    .title {
        color:#00f5ff;
        font-size:22px;
        margin-bottom:10px;
    }

    .text {
        color:white;
        opacity:0.85;
        font-size:15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================= PROBLEM =================
    st.markdown("""
    <div class="card">
        <div class="title">🎯 Problem Statement</div>
        <div class="text">
        Monitoring aircraft manually in aerial or satellite images is time-consuming and prone to human error.
        This system automates detection using AI and Deep Learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= OBJECTIVES =================
    st.markdown("""
    <div class="card">
        <div class="title">🚀 Project Objectives</div>
        <div class="text">
        ✈ Detect aircraft automatically<br>
        ✈ Reduce manual monitoring<br>
        ✈ Improve airspace security<br>
        ✈ Increase detection accuracy<br>
        ✈ Enable real-time monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= TECHNOLOGY =================
    st.markdown("""
    <div class="card">
        <div class="title">⚙ Technologies Used</div>
        <div class="text">
        Python, OpenCV, PyTorch, YOLOv8, NumPy, Matplotlib, Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= FUTURE =================
    st.markdown("""
    <div class="card">
        <div class="title">🚀 Future Scope</div>
        <div class="text">
        🛰 Satellite aircraft detection<br>
        🚁 Drone monitoring system<br>
        🛡 Military surveillance<br>
        🏢 Airport security systems<br>
        📡 Real-time video analytics<br>
        🌍 Smart aerospace monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= DEVELOPER =================
    st.markdown("""
    <div class="card">
        <div class="title">👩‍💻 Developer</div>
        <div class="text">
        Project: Aircraft Detection System<br>
        Theme: Aerospace AI<br>
        Built using YOLOv8 + OpenCV + Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= FOOTER =================
    st.markdown("---")

    st.markdown("""
    <div style="text-align:center; color:#00f5ff;">
    ✈ Aircraft Detection System | YOLOv8 • OpenCV • Streamlit
    </div>
    """, unsafe_allow_html=True)
