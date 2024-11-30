import cv2 as cv
import face_recognition
import dlib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from imutils import face_utils
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import streamlit as st
from collections import defaultdict
import time
from scipy.interpolate import make_interp_spline

# [Previous imports remain the same]

def setup_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            font-family: 'Inter', sans-serif;
            color: #E2E8F0;
            line-height: 1.6;
            padding: 0;
            margin: 0;
        }
        
        .title-container {
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
            padding: 2rem 0;
            border-bottom: 1px solid rgba(59, 130, 246, 0.3);
            text-align: center;
            margin: 0;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 2.75rem;
            background: linear-gradient(45deg, #60A5FA, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin: 0;
            padding: 0;
            text-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .start-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            padding: 2rem;
            text-align: center;
        }
        
        .start-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3.75rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            background: linear-gradient(45deg, #60A5FA, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 2.5px;
            text-shadow: 0 6px 10px rgba(0,0,0,0.15);
        }
        
        .start-subtitle {
            color: #94A3B8;
            font-family: 'Inter', sans-serif;
            font-size: 1.3rem;
            margin-bottom: 2.5rem;
            max-width: 700px;
            letter-spacing: 0.5px;
            line-height: 1.7;
            text-align: center;
            width: 100%;
            display: block;
            opacity: 0.8;
        }
        
        .video-container, .analytics-container {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 20px;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 1.5rem;
            margin: 0;
            box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .video-container:hover, .analytics-container:hover {
            box-shadow: 0 40px 80px -20px rgba(0, 0, 0, 0.4);
        }
        
        .section-header {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 1.4rem;
            color: #60A5FA;
            margin: 0 0 1.5rem 0;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            border-bottom: 3px solid #60A5FA;
            padding-bottom: 0.75rem;
        }
        
        .status-container {
            background: rgba(30, 41, 59, 0.9);
            border-radius: 20px;
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 15px 30px -10px rgba(0, 0, 0, 0.2);
        }
        
        .status-badge {
            padding: 0.5rem 1.2rem;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            font-family: 'Inter', sans-serif;
        }
        
        .attentive {
            background: rgba(34, 197, 94, 0.15);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        
        .distracted {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .alert-container {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            color: #F87171;
        }
        
        .block-container {padding: 1.5rem !important; max-width: 100% !important;}
        .element-container {margin: 0 !important;}
        div[data-testid="stVerticalBlock"] {padding: 0 !important; gap: 1.5rem !important;}
        
        /* Other styles remain the same */
        </style>
    """, unsafe_allow_html=True)

def show_start_screen():
    """Display the start screen"""
    st.markdown("""
        <style>
        .header-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 2rem auto;
            width: 100%;
        }
        .start-title {
            text-align: center !important;
            width: 100%;
        }
        .centered-subtitle {
            text-align: center;
            width: 100%;
        }
        .start-subtitle {
            text-align: center !important;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with title and subtitle
    st.markdown("""
        <div class="header-container">
            <h1 class="start-title">Attention Monitoring System</h1>
            <div class="centered-subtitle">
                <p class="start-subtitle">Intelligent AI-Powered Student Engagement Tracking</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Content container for image and button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        image_path = "C:/Users/harsh/OneDrive/Desktop/pyinfo/face_recognition/tanya_landmark/Screenshot 2024-11-30 171824.png"
        st.image(image_path, use_container_width=True, caption="Smart Classroom Analytics")
        
        return st.button("Start Monitoring", key="start_button", help="Begin real-time student attention tracking")

def load_known_faces(faces_directory):
    for image_file in os.listdir(faces_directory):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(image_file)[0]
            image_path = os.path.join(faces_directory, image_file)
            known_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(known_image)[0]
            st.session_state.known_faces_dict[name] = face_encoding

def calculate_attention_score(yaw, pitch):
    base_score = 100
    yaw_degrees = abs(np.degrees(yaw[0]))
    pitch_degrees = np.degrees(pitch[0])
    
    # Yaw (left-right) penalties
    yaw_penalty = 0
    if yaw_degrees > 45:
        yaw_penalty = min(30, (yaw_degrees - 45))
    
    # Pitch (up-down) penalties
    pitch_penalty = 0
    if pitch_degrees > 30:
        pitch_penalty = min(30, (pitch_degrees - 30))
    elif pitch_degrees < -30:
        pitch_penalty = min(30, (abs(pitch_degrees) - 30))
    
    score = base_score - yaw_penalty - pitch_penalty
    
    # Extreme movement penalties
    if yaw_degrees > 70 or abs(pitch_degrees) > 50:
        score = max(0, score - 20)
    
    return max(0, min(100, score))

def update_attention_data(name, score, current_time):
    st.session_state.attention_scores[name].append(score)
    st.session_state.attention_times[name].append(current_time)
    
    if len(st.session_state.attention_scores[name]) > st.session_state.MAX_HISTORY:
        st.session_state.attention_scores[name] = st.session_state.attention_scores[name][-st.session_state.MAX_HISTORY:]
        st.session_state.attention_times[name] = st.session_state.attention_times[name][-st.session_state.MAX_HISTORY:]

def update_dashboard(attention_chart):
    fig = plt.figure(figsize=(10, 6), facecolor='#1E293B')
    ax = fig.add_subplot(111)
    
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 10)
    
    for name in st.session_state.attention_scores:
        if not st.session_state.attention_scores[name] or not st.session_state.attention_times[name]:
            continue
            
        scores = st.session_state.attention_scores[name][-100:]
        times = []
        
        if st.session_state.attention_times[name]:
            min_time = min(st.session_state.attention_times[name][-100:])
            times = [t - min_time for t in st.session_state.attention_times[name][-100:]]
        
        if times and scores:
            if len(times) > 3:
                try:
                    spline = make_interp_spline(times, scores, k=min(3, len(times)-1))
                    smooth_times = np.linspace(min(times), max(times), 300)
                    smooth_scores = spline(smooth_times)
                    
                    gradient = np.linspace(0, 1, len(smooth_times))
                    points = np.array([smooth_times, smooth_scores]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    norm = plt.Normalize(0, 100)
                    lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.6, linewidth=2)
                    lc.set_array(np.array(smooth_scores))
                    ax.add_collection(lc)
                    
                    ax.fill_between(smooth_times, smooth_scores, alpha=0.15, color='#3B82F6')
                    ax.scatter(times[::5], scores[::5], color='#3B82F6', s=30, alpha=0.6)
                    
                    ax.set_xlim(min(times), max(times))
                    
                except Exception:
                    ax.plot(times, scores, color='#3B82F6', linewidth=2, alpha=0.8)
                    ax.fill_between(times, scores, alpha=0.15, color='#3B82F6')
            else:
                ax.plot(times, scores, color='#3B82F6', linewidth=2, alpha=0.8)
                ax.fill_between(times, scores, alpha=0.15, color='#3B82F6')
    
    ax.set_facecolor('#1E293B')
    ax.grid(True, linestyle='--', alpha=0.1, color='white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color((1, 1, 1, 0.3))
    ax.spines['bottom'].set_color((1, 1, 1, 0.3))
    
    ax.set_xlabel('Time (seconds)', color='white', fontsize=10, fontfamily='Inter')
    ax.set_ylabel('Attention Score', color='white', fontsize=10, fontfamily='Inter')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    attention_chart.pyplot(fig)
    plt.close()

def run_monitoring_system(video_placeholder, attention_chart, alert_placeholder, status_placeholder, stop_button):
    cam = None
    try:
        faces_directory = "known_faces"
        if not os.path.exists(faces_directory):
            st.error(f"Error: Directory '{faces_directory}' not found. Please create it and add student images.")
            return

        if not os.listdir(faces_directory):
            st.error("Error: No face images found in the known_faces directory.")
            return

        load_known_faces(faces_directory)

        cam = cv.VideoCapture(0)
        if not cam.isOpened():
            st.error("Error: Could not access camera")
            return

        while True:
            ret, frame = cam.read()
            if not ret:
                st.error("Error: Could not read from camera")
                break

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            current_time = time.time()
            active_students = set()
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(list(st.session_state.known_faces_dict.values()), face_encoding, tolerance=0.6)
                
                if True in matches:
                    name = list(st.session_state.known_faces_dict.keys())[matches.index(True)]
                    active_students.add(name)
                    st.session_state.last_seen[name] = current_time

                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    rect = dlib.rectangle(left, top, right, bottom)
                    landmarks = st.session_state.landmark_predictor(gray_frame, rect)
                    landmarks = face_utils.shape_to_np(landmarks)

                    for (x, y) in landmarks:
                        cv.circle(frame, (x, y), 1, (0, 255, 255), -1)

                    image_points = np.array([
                        landmarks[30], landmarks[8], landmarks[36],
                        landmarks[45], landmarks[48], landmarks[54]
                    ], dtype=np.float32)

                    success, rotation_vector, translation_vector = cv.solvePnP(
                        st.session_state.MODEL_POINTS, image_points, st.session_state.CAMERA_MATRIX, st.session_state.DIST_COEFFS, flags=cv.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        yaw, pitch = rotation_vector[:2]
                        score = calculate_attention_score(yaw, pitch)
                        update_attention_data(name, score, current_time)

                        status = "Attentive" if score > 40 else "Distracted"
                        color = (46, 197, 94) if status == "Attentive" else (239, 68, 68)
                        
                        cv.rectangle(frame, (left, top), (right, bottom), color, 1)
                        text_overlay = frame.copy()
                        cv.rectangle(text_overlay, (left, top - 30), (right, top), color, cv.FILLED)
                        cv.addWeighted(text_overlay, 0.3, frame, 0.7, 0, frame)
                        cv.putText(frame, f"{name} ({score:.0f}%)", 
                                 (left + 6, top - 10),
                                 cv.FONT_HERSHEY_DUPLEX, 
                                 0.5,
                                 (255, 255, 255),
                                 1)

                        status_placeholder.markdown(f"""
                            <div class="status-container">
                                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: #3B82F6;">
                                    Student Status
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <span style="color: #94A3B8;">Student Name</span>
                                    <span style="color: #F8FAFC; font-weight: 500;">{name}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <span style="color: #94A3B8;">Attention Score</span>
                                    <span style="color: #F8FAFC; font-weight: 500;">{score:.1f}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <span style="color: #94A3B8;">Status</span>
                                    <span class="status-badge {status.lower()}">{status}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-top: 1rem; font-size: 0.9rem;">
                                    <span style="color: #94A3B8;">Last Updated</span>
                                    <span style="color: #F8FAFC;">{datetime.now().strftime('%H:%M:%S')}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            alert_messages = []
            for name in st.session_state.known_faces_dict.keys():
                last_seen_time = st.session_state.last_seen.get(name)
                if (last_seen_time and 
                    name not in active_students and 
                    current_time - last_seen_time > 3 and 
                    current_time - last_seen_time < 30):
                    alert_messages.append(name)
            
            if alert_messages:
                alert_html = """
                    <div class="alert-container">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">⚠️ Attention Required</div>
                """
                for name in alert_messages:
                    alert_html += f'<div style="margin-top: 0.3rem;">Student "{name}" is not visible</div>'
                alert_html += "</div>"
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            update_dashboard(attention_chart)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB", use_container_width=True)

            if stop_button:
                break

            time.sleep(0.1)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if cam is not None:
            cam.release()
        cv.destroyAllWindows()

def run_dashboard():
    st.markdown(
        '<div class="title-container"><h1 class="main-title">Smart Student Attention Monitoring Dashboard</h1></div>', 
        unsafe_allow_html=True
    )

    left_column, right_column = st.columns([0.6, 0.4], gap="small")
    
    with left_column:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        stop_button = st.button('Stop Session')

    with right_column:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Attention Analytics</h3>', unsafe_allow_html=True)
        attention_chart = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        alert_placeholder = st.empty()
        status_placeholder = st.empty()

    run_monitoring_system(video_placeholder, attention_chart, alert_placeholder, status_placeholder, stop_button)

def main():
    st.set_page_config(layout="wide", page_title="Smart Student Attention Monitoring Dashboard")
    
    # Initialize session state variables if they don't exist
    if 'initialized' not in st.session_state:
        st.session_state.known_faces_dict = {}
        st.session_state.attention_scores = defaultdict(list)
        st.session_state.attention_times = defaultdict(list)
        st.session_state.last_seen = {}
        st.session_state.MAX_HISTORY = 100
        st.session_state.initialized = True
        
        # Model points and camera matrices
        st.session_state.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        st.session_state.CAMERA_MATRIX = np.array([
            [1000, 0, 500],
            [0, 1000, 300],
            [0, 0, 1]
        ], dtype=np.float32)

        st.session_state.DIST_COEFFS = np.zeros((4,1))
        st.session_state.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    setup_styles()

    if 'started' not in st.session_state:
        st.session_state.started = False

    if not st.session_state.started:
        start_clicked = show_start_screen()
        if start_clicked:
            st.session_state.started = True
            st.rerun()
    else:
        run_dashboard()

if __name__ == "__main__":
    main()