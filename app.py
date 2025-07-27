import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Or yolov8s.pt for better accuracy

# Streamlit setup
st.set_page_config(page_title="Real-Time Pedestrian Detection", layout="wide")
st.title("🚗 Real-Time Pedestrian Detection using YOLOv8")

# Sidebar options
mode = st.sidebar.radio("Choose Mode:", ["Live Webcam", "Upload Video", "Night / Thermal Video"])
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.2, max_value=0.9, value=0.4, step=0.05)

FRAME_WINDOW = st.empty()


# Live Webcam Mode
if mode == "Live Webcam":
    if 'running' not in st.session_state:
        st.session_state.running = False

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Webcam"):
        st.session_state.running = True
    if col2.button("⏹ Stop Webcam"):
        st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        prev_time = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))
            results = model(frame, conf=conf_thresh)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > conf_thresh:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        FRAME_WINDOW.image([])


# Upload Video Mode
elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        # Output setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 360))
                results = model(frame, conf=conf_thresh)[0]

                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > conf_thresh:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                out.write(frame)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        out.release()
        os.unlink(video_path)
        st.success("✅ Video processed successfully!")

        with open(out_path, 'rb') as f:
            st.download_button("📥 Download Annotated Video", f, file_name="annotated_output.mp4")


# Thermal/Night Mode
elif mode == "Night / Thermal Video":
    st.markdown("🌙 Upload Night Vision or Thermal Video")
    uploaded_file = st.file_uploader("Upload thermal/night video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        # Output setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        with st.spinner("Processing thermal/night video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to pseudo-color
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

                results = model(thermal, conf=conf_thresh)[0]

                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > conf_thresh:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(thermal, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(thermal, f"Person {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                out.write(thermal)
                stframe.image(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        out.release()
        os.unlink(video_path)
        st.success("✅ Thermal video processed!")

        with open(out_path, 'rb') as f:
            st.download_button("📥 Download Annotated Thermal Video", f, file_name="thermal_output.mp4")



