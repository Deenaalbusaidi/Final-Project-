import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import hashlib
from datetime import datetime
import os
import tempfile
from pathlib import Path
import base64

# --- Page Setup with Custom Theme ---
st.set_page_config(
    page_title="FreshQuality AI - Produce Assessment",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Red, Gray, White Theme ---
def set_custom_theme():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #2d2d2d !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #d32f2f !important;
        font-family: 'Arial', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #d32f2f !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #b71c1c !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #d32f2f !important;
    }
    
    /* Cards and containers */
    .main-special-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #d32f2f;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Radio buttons in sidebar */
    .st-eb, .st-ec {
        background-color: #404040 !important;
    }
    
    /* File uploader */
    .st-emotion-cache-1r6slb0 {
        border: 2px dashed #d32f2f !important;
        border-radius: 10px;
        background-color: rgba(211, 47, 47, 0.05);
    }
    
    /* Success messages */
    .st-emotion-cache-1r6slb0 {
        background-color: rgba(76, 175, 80, 0.1) !important;
    }
    
    /* Video container */
    .video-container {
        position: relative;
        width: 100%;
        height: 300px;
        border-radius: 15px;
        overflow: hidden;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom metric styling */
    .custom-metric {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .custom-metric .value {
        font-size: 2rem;
        font-weight: bold;
        color: #d32f2f;
        margin-bottom: 0.5rem;
    }
    
    .custom-metric .label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #d32f2f;
    }
    
    </style>
    """, unsafe_allow_html=True)

# --- Video Background Function ---
def add_video_background():
    # You can replace this with your actual video file path
    video_html = """
    <div class="video-container">
        <video autoplay muted loop playsinline style="width: 100%; height: 100%; object-fit: cover;">
            <source src="https://cdn.pixabay.com/video/2023/04/12/160932_800853311_tiny.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                   background: linear-gradient(45deg, rgba(211, 47, 47, 0.7), rgba(45, 45, 45, 0.7)); 
                   display: flex; align-items: center; justify-content: center;">
            <h2 style="color: white; font-size: 2.5rem; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                🍎 FreshQuality AI 🥒<br>
                <span style="font-size: 1.2rem;">Intelligent Produce Assessment System</span>
            </h2>
        </div>
    </div>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# --- Configuration ---
IMG_SIZE = (224, 224)

# Class mappings (must match your training)
quality_class_names = {
    0: 'unripe',
    1: 'ripe', 
    2: 'overripe',
    3: 'bruised'
}

# --- Image Processing Functions ---
def enhanced_bg_mask_rgb(arr):
    """Improved background removal using color thresholding"""
    try:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        lower_white1 = np.array([0, 0, 200])
        upper_white1 = np.array([180, 55, 255])
        lower_white2 = np.array([0, 0, 150])
        upper_white2 = np.array([180, 80, 255])
        
        mask1 = cv2.inRange(hsv, lower_white1, upper_white1)
        mask2 = cv2.inRange(hsv, lower_white2, upper_white2)
        background_mask = cv2.bitwise_or(mask1, mask2)
        foreground_mask = cv2.bitwise_not(background_mask)
        
        kernel = np.ones((5,5), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        masked = cv2.bitwise_and(arr, arr, mask=foreground_mask)
        
        return masked, foreground_mask
    except Exception as e:
        return arr, np.ones(arr.shape[:2], dtype=np.uint8) * 255

def extract_color_features_lab(arr, mask):
    """Extract color statistics in LAB space"""
    try:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        masked_lab = lab[mask > 0]
        
        if len(masked_lab) == 0:
            return np.zeros(12)
        
        l_channel = masked_lab[:, 0]
        a_channel = masked_lab[:, 1]
        b_channel = masked_lab[:, 2]
        
        l_mean, a_mean, b_mean = np.mean(masked_lab, axis=0)
        l_std, a_std, b_std = np.std(masked_lab, axis=0)
        l_median = np.median(l_channel)
        color_variance = np.var(masked_lab, axis=0)
        hist_l = np.histogram(l_channel, bins=8, range=(0, 255))[0]
        hist_l = hist_l / (np.sum(hist_l) + 1e-6)
        
        features = np.array([
            l_mean, a_mean, b_mean, l_std, a_std, b_std,
            l_median, color_variance[0], np.mean(color_variance[1:])
        ] + hist_l[:3].tolist())
        
        if len(features) != 12:
            if len(features) > 12:
                features = features[:12]
            else:
                features = np.pad(features, (0, 12 - len(features)), mode='constant')
        
        return features.astype(np.float32)
    except Exception as e:
        return np.zeros(12, dtype=np.float32)

def preprocess_image_for_prediction(image_path):
    """Complete preprocessing that matches training"""
    try:
        img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
        arr = np.array(img, dtype=np.uint8)
        masked, mask = enhanced_bg_mask_rgb(arr)
        stats = extract_color_features_lab(arr, mask)
        img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        return img_processed, stats
    except Exception as e:
        img_processed = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_processed)
        stats = np.zeros(12, dtype=np.float32)
        return img_processed, stats

# --- Database Functions ---
def init_database():
    db_path = "produce_fruit_veg_quality.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Produce_Samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            item_name TEXT,
            scan_date TIMESTAMP,
            image_path TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Quality_Results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            quality_class TEXT,
            confidence REAL,
            freshness_index REAL,
            FOREIGN KEY (sample_id) REFERENCES Produce_Samples (sample_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Shelf_Life_Metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            predicted_storage_days INTEGER,
            optimal_temp_C INTEGER,
            mock_decay_rate REAL,
            FOREIGN KEY (sample_id) REFERENCES Produce_Samples (sample_id)
        )
    ''')

    conn.commit()
    conn.close()

class ProduceDatabase:
    def __init__(self, db_path="produce_fruit_veg_quality.db"):
        self.db_path = db_path

    def get_all_samples(self):
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT ps.sample_id, ps.item_name, ps.scan_date, ps.image_path,
                   qr.quality_class, qr.confidence, qr.freshness_index,
                   slm.predicted_storage_days, slm.optimal_temp_C, slm.mock_decay_rate
            FROM Produce_Samples ps
            JOIN Quality_Results qr ON ps.sample_id = qr.sample_id
            JOIN Shelf_Life_Metrics slm ON ps.sample_id = slm.sample_id
            ORDER BY ps.scan_date DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_quality_distribution(self):
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT quality_class, COUNT(*) as count,
                   AVG(freshness_index) as avg_freshness
            FROM Quality_Results
            GROUP BY quality_class
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def calculate_shelf_life(self, item_name, quality_class, freshness_index):
        base_shelf_life = {
            'apple': 14, 'banana': 7, 'mango': 5, 'tomato': 10, 'cucumber': 12
        }
        quality_multipliers = {
            'unripe': 1.3, 'ripe': 1.0, 'overripe': 0.5, 'bruised': 0.3
        }

        item_key = next((key for key in base_shelf_life if key in item_name.lower()), 'apple')
        base_days = base_shelf_life[item_key]
        quality_mult = quality_multipliers.get(quality_class.lower(), 1.0)
        freshness_mult = freshness_index / 10.0

        predicted_days = int(base_days * quality_mult * freshness_mult)
        optimal_temps = {'apple': 4, 'banana': 13, 'mango': 10, 'tomato': 12, 'cucumber': 10}
        optimal_temp = optimal_temps.get(item_key, 5)
        decay_rate = round(1.0 / max(predicted_days, 1), 3)

        return {
            'predicted_days': max(1, predicted_days),
            'optimal_temp': optimal_temp,
            'decay_rate': decay_rate
        }

    def insert_sample(self, image_path, item_name, quality_class, confidence, freshness_index):
        file_hash = hashlib.md5(image_path.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT sample_id FROM Produce_Samples WHERE file_hash=?', (file_hash,))
            existing = cursor.fetchone()
            if existing:
                return existing[0]

            cursor.execute('''
                INSERT INTO Produce_Samples (file_hash, item_name, scan_date, image_path)
                VALUES (?, ?, ?, ?)
            ''', (file_hash, item_name, datetime.now(), image_path))
            sample_id = cursor.lastrowid

            cursor.execute('''
                INSERT INTO Quality_Results (sample_id, quality_class, confidence, freshness_index)
                VALUES (?, ?, ?, ?)
            ''', (sample_id, quality_class, confidence, freshness_index))

            shelf = self.calculate_shelf_life(item_name, quality_class, freshness_index)
            cursor.execute('''
                INSERT INTO Shelf_Life_Metrics (sample_id, predicted_storage_days, optimal_temp_C, mock_decay_rate)
                VALUES (?, ?, ?, ?)
            ''', (sample_id, shelf['predicted_days'], shelf['optimal_temp'], shelf['decay_rate']))

            conn.commit()
            return sample_id
        finally:
            conn.close()

# --- Model Prediction Class ---
class ProduceQualitySystem:
    def __init__(self, model_path="improved_produce_quality_model.h5"):
        try:
            self.model = tf.keras.models.load_model(model_path)
            st.success("✅ AI Model Loaded Successfully!")
            self.using_real_model = True
        except Exception as e:
            st.warning(f"⚠️ Using demo mode (model not found: {e})")
            self.model = None
            self.using_real_model = False
        self.class_names = ['unripe', 'ripe', 'overripe', 'bruised']

    def predict_quality(self, image_path):
        if self.using_real_model:
            try:
                img_processed, stats = preprocess_image_for_prediction(image_path)
                img_batch = np.expand_dims(img_processed, axis=0)
                stats_batch = np.expand_dims(stats, axis=0)
                class_pred, freshness_pred = self.model.predict([img_batch, stats_batch], verbose=0)
                predicted_class_idx = np.argmax(class_pred[0])
                confidence = np.max(class_pred[0])
                quality = self.class_names[predicted_class_idx]
                freshness_score = freshness_pred[0][0]
                return quality, float(confidence), float(freshness_score)
            except Exception as e:
                return self.mock_predict(image_path)
        else:
            return self.mock_predict(image_path)

    def mock_predict(self, image_path):
        try:
            image = Image.open(image_path)
            arr = np.array(image)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            hue, sat, val = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
            
            if sat > 100 and hue < 30:
                return 'unripe', 0.85, 7.0
            elif sat > 80 and (hue < 15 or hue > 150):
                return 'ripe', 0.9, 9.0
            elif val < 80:
                return 'overripe', 0.8, 4.0
            else:
                return 'bruised', 0.75, 3.0
        except:
            return 'ripe', 0.8, 7.0

    def extract_item_name(self, filename):
        f = filename.lower()
        for fruit in ['apple', 'banana', 'mango', 'tomato', 'cucumber']:
            if fruit in f:
                return fruit.capitalize()
        return "Fresh Produce"

# --- Custom UI Components ---
def create_custom_metric(value, label, icon="📊"):
    st.markdown(f"""
    <div class="custom-metric">
        <div class="value">{icon} {value}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def create_special_card(content, title="", icon="🔍"):
    st.markdown(f"""
    <div class="main-special-card">
        <h3>{icon} {title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# --- UI Pages ---
def main():
    # Apply custom theme
    set_custom_theme()
    
    # Header with video background
    add_video_background()
    
    # Initialize systems
    init_database()
    db = ProduceDatabase()
    system = ProduceQualitySystem()

    # Sidebar with custom styling
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #d32f2f; border-radius: 10px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0;'>🧭 NAVIGATION</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio("", [
            "📷 SCAN PRODUCE", 
            "📸 WEBCAM CAPTURE", 
            "📊 DASHBOARD", 
            "📈 QUALITY ANALYTICS", 
            "📅 SHELF LIFE", 
            "🧾 SCAN HISTORY"
        ], label_visibility="collapsed")

    try:
        df = db.get_all_samples()
        quality_df = db.get_quality_distribution()
    except Exception:
        df, quality_df = pd.DataFrame(), pd.DataFrame()

    if page == "📷 SCAN PRODUCE":
        scan_produce_page(system, db)
    elif page == "📸 WEBCAM CAPTURE":
        webcam_page(system, db)
    elif page == "📊 DASHBOARD":
        dashboard_page(df)
    elif page == "📈 QUALITY ANALYTICS":
        quality_page(quality_df)
    elif page == "📅 SHELF LIFE":
        shelf_life_page(df)
    elif page == "🧾 SCAN HISTORY":
        recent_scans_page(df)

def scan_produce_page(system, db):
    st.markdown("""
    <div class="main-special-card">
        <h2>📷 UPLOAD & ANALYZE</h2>
        <p>Upload an image of fruits or vegetables for AI-powered quality assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'], 
                                  help="Supported formats: JPG, JPEG, PNG")
    
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded.getvalue())
            temp_path = tmp_file.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🚀 ANALYZE WITH AI", use_container_width=True):
                with st.spinner("🔬 AI is analyzing produce quality..."):
                    quality, conf, fresh = system.predict_quality(temp_path)
                    item = system.extract_item_name(uploaded.name)
                    sid = db.insert_sample(uploaded.name, item, quality, conf, fresh)
                    show_results(item, quality, conf, fresh, db)
                
                os.unlink(temp_path)

def webcam_page(system, db):
    st.markdown("""
    <div class="main-special-card">
        <h2>📸 LIVE CAPTURE</h2>
        <p>Use your camera to capture real-time produce images for instant analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    picture = st.camera_input("Capture produce image")
    
    if picture:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(picture.getvalue())
            temp_path = tmp_file.name
            
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(picture)
            st.image(image, caption="📸 Captured Image", use_column_width=True)
        
        with col2:
            if st.button("🚀 ANALYZE CAPTURE", use_container_width=True):
                with st.spinner("🔬 Analyzing captured image..."):
                    quality, conf, fresh = system.predict_quality(temp_path)
                    item = "Fresh Produce"
                    sid = db.insert_sample("webcam_capture.jpg", item, quality, conf, fresh)
                    show_results(item, quality, conf, fresh, db)
                
                os.unlink(temp_path)

def show_results(item, quality, conf, fresh, db):
    st.markdown("""
    <div class="main-special-card">
        <h2>✅ ANALYSIS COMPLETE</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Quality indicator with color coding
    quality_colors = {
        'unripe': '#ff9800',
        'ripe': '#4caf50', 
        'overripe': '#ff5722',
        'bruised': '#f44336'
    }
    
    quality_color = quality_colors.get(quality, '#666666')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_custom_metric(item, "PRODUCE ITEM", "🍎")
    
    with col2:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="value" style="color: {quality_color};">{quality.upper()}</div>
            <div class="label">QUALITY STATUS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        create_custom_metric(f"{conf*100:.1f}%", "CONFIDENCE LEVEL", "🎯")
    
    with col4:
        create_custom_metric(f"{fresh:.1f}/10", "FRESHNESS SCORE", "⭐")
    
    # Shelf life information
    shelf = db.calculate_shelf_life(item, quality, fresh)
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_special_card(
            f"**{shelf['predicted_days']} days** remaining\n\n"
            f"Optimal storage: **{shelf['optimal_temp']}°C**",
            "📅 SHELF LIFE PREDICTION"
        )
    
    with col2:
        # Freshness gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fresh,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': quality_color},
                'steps': [
                    {'range': [0, 4], 'color': "#ff5252"},
                    {'range': [4, 7], 'color': "#ffb74d"},
                    {'range': [7, 10], 'color': "#66bb6a"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': fresh
                }
            },
            title={'text': "FRESHNESS INDEX", 'font': {'color': '#d32f2f'}}
        ))
        fig.update_layout(height=300, font={'color': "#2d2d2d"})
        st.plotly_chart(fig, use_container_width=True)

def dashboard_page(df):
    st.markdown("""
    <div class="main-special-card">
        <h2>📊 SYSTEM DASHBOARD</h2>
        <p>Comprehensive overview of produce quality assessments</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available. Start by scanning some produce!")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_custom_metric(len(df), "TOTAL SCANS", "📈")
    
    with col2:
        create_custom_metric(f"{df['freshness_index'].mean():.1f}/10", "AVG FRESHNESS", "⭐")
    
    with col3:
        create_custom_metric(f"{df['predicted_storage_days'].mean():.1f} days", "AVG SHELF LIFE", "📅")
    
    with col4:
        create_custom_metric(f"{df['confidence'].mean()*100:.1f}%", "AVG CONFIDENCE", "🎯")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(df, x='freshness_index', nbins=10, 
                           title="Freshness Distribution",
                           color_discrete_sequence=['#d32f2f'])
        fig1.update_layout(template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        quality_counts = df['quality_class'].value_counts()
        fig2 = px.pie(values=quality_counts.values, names=quality_counts.index,
                     title="Quality Class Distribution",
                     color_discrete_sequence=['#d32f2f', '#ff5252', '#ff867c', '#ffab91'])
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

def quality_page(quality_df):
    st.markdown("""
    <div class="main-special-card">
        <h2>📈 QUALITY ANALYTICS</h2>
        <p>Detailed analysis of produce quality patterns and trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    if quality_df.empty:
        st.warning("No quality data available yet.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(quality_df, x='quality_class', y='count',
                    title="Quality Distribution",
                    color='quality_class',
                    color_discrete_sequence=['#d32f2f', '#ff5252', '#ff867c', '#ffab91'])
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.bar(quality_df, x='quality_class', y='avg_freshness',
                     title="Average Freshness by Quality",
                     color='quality_class',
                     color_discrete_sequence=['#66bb6a', '#4caf50', '#ff9800', '#f44336'])
        fig2.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

def shelf_life_page(df):
    st.markdown("""
    <div class="main-special-card">
        <h2>📅 SHELF LIFE ANALYSIS</h2>
        <p>Predictive analytics for produce storage and freshness duration</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No shelf life data available yet.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        grouped = df.groupby('item_name')['predicted_storage_days'].mean().reset_index()
        fig1 = px.bar(grouped, x='item_name', y='predicted_storage_days', 
                     title="Average Shelf Life by Item",
                     color_discrete_sequence=['#d32f2f'])
        fig1.update_layout(template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(df, x='freshness_index', y='predicted_storage_days',
                         color='quality_class', 
                         title="Freshness vs Shelf Life Correlation",
                         hover_data=['item_name'],
                         color_discrete_sequence=['#d32f2f', '#ff5252', '#ff867c', '#ffab91'])
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

def recent_scans_page(df):
    st.markdown("""
    <div class="main-special-card">
        <h2>🧾 SCAN HISTORY</h2>
        <p>Complete record of all produce quality assessments</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No scan history available.")
        return
    
    df['scan_date'] = pd.to_datetime(df['scan_date']).dt.strftime('%Y-%m-%d %H:%M')
    
    display_df = df[['item_name', 'scan_date', 'quality_class', 'freshness_index', 'confidence', 'predicted_storage_days']].copy()
    display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
    display_df['freshness_index'] = display_df['freshness_index'].round(1)
    display_df.columns = ['Item', 'Scan Date', 'Quality', 'Freshness', 'Confidence', 'Shelf Life (days)']
    
    st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()