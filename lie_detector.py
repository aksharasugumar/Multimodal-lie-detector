import cv2
import numpy as np
import sounddevice as sd
import librosa
import threading
import time
from datetime import datetime
from tensorflow.keras.models import model_from_json # type: ignore
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# =====================================================
# CONFIG
# =====================================================

FACE_JSON = 'emotiondetector.json'
FACE_H5 = 'emotiondetector.h5'
VOICE_JSON = 'enhanced_emotion.json'
VOICE_H5 = 'enhanced_emotion.h5'

face_labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

voice_emotions = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

truth_lie_map = {
    'happy': 'Truth', 'neutral': 'Truth', 'surprise': 'Truth', 'calm': 'Truth',
    'angry': 'Lie', 'disgust': 'Lie', 'fear': 'Lie', 'sad': 'Lie'
}

colors = {'Truth': 'green', 'Lie': 'red'}

# =====================================================
# LOAD MODELS
# =====================================================

with open(FACE_JSON, "r") as json_file:
    face_json = json_file.read()
face_model = model_from_json(face_json)
face_model.load_weights(FACE_H5)

with open(VOICE_JSON, "r") as json_file:
    voice_json = json_file.read()
voice_model = model_from_json(voice_json)
voice_model.load_weights(VOICE_H5)

le = LabelEncoder()
le.fit(voice_emotions)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =====================================================
# FUNCTIONS
# =====================================================

def preprocess_face(gray_frame):
    feature = np.array(gray_frame, dtype="float32")
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def extract_features_live(audio, sr, max_len=216):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    combined = np.vstack((mfcc, delta, delta2, chroma, spec_contrast))
    if combined.shape[1] < max_len:
        pad_width = max_len - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :max_len]
    combined = combined.reshape(1, combined.shape[0], max_len, 1)
    combined = (combined - np.mean(combined)) / np.std(combined)
    return combined

def record_voice(duration=3, sr=16000):
    """Record voice for given seconds."""
    start_time_str = datetime.now().strftime("%H:%M:%S")
    status_label.config(text=f"🎙️ Mic ON – Started at {start_time_str}", foreground="purple")
    root.update()
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    end_time_str = datetime.now().strftime("%H:%M:%S")
    status_label.config(text=f"🎙️ Mic OFF – Ended at {end_time_str}", foreground="gray")
    root.update()
    return audio.flatten()

def predict_voice():
    audio = record_voice()
    features = extract_features_live(audio, 16000)
    preds = voice_model.predict(features, verbose=0)
    idx = np.argmax(preds)
    emotion = le.inverse_transform([idx])[0]
    confidence = np.max(preds)
    return emotion, confidence

def predict_face_live():
    start_time = time.time()
    captured_emotions = []
    while time.time() - start_time < 3:
        if not cap or not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            features = preprocess_face(face_img)
            preds = face_model.predict(features, verbose=0)
            idx = np.argmax(preds)
            captured_emotions.append((face_labels[idx], np.max(preds)))
    if captured_emotions:
        emotion = max(set([e[0] for e in captured_emotions]), key=[e[0] for e in captured_emotions].count)
        confidence = np.mean([e[1] for e in captured_emotions if e[0] == emotion])
        return emotion, confidence
    return None, None

# =====================================================
# CAMERA & GUI CONTROL
# =====================================================

cap = None
camera_running = False
frame_rgb = None

def start_camera():
    global cap, camera_running
    if camera_running:
        messagebox.showinfo("Info", "Camera already running.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible!")
        return
    camera_running = True
    status_label.config(text="📷 Camera started. Ready to detect.", foreground="blue")
    update_video()

def stop_camera():
    global cap, camera_running
    camera_running = False
    if cap and cap.isOpened():
        cap.release()
    video_label.config(image="")
    status_label.config(text="🛑 Camera stopped.", foreground="red")

def update_video():
    global frame_rgb
    if camera_running and cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((400, 300))
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
    if camera_running:
        root.after(15, update_video)

def analyze_combined():
    if not camera_running:
        messagebox.showwarning("Warning", "Start the camera first!")
        return
    status_label.config(text="🎥 Recording face & voice for 3 seconds...", foreground="blue")
    root.update()

    def task():
        face_emotion, face_conf = predict_face_live()
        voice_emotion, voice_conf = predict_voice()

        if not face_emotion or not voice_emotion:
            status_label.config(text="❌ Could not detect both inputs.", foreground="red")
            return

        face_truth = truth_lie_map.get(face_emotion, "Truth")
        voice_truth = truth_lie_map.get(voice_emotion, "Truth")

        combined_truth = "Truth" if (face_truth == "Truth" and voice_truth == "Truth") else "Lie"
        combined_conf = (face_conf + voice_conf) / 2

        result_text = (
            f"🧠 Face Emotion: {face_emotion} → {face_truth}\n"
            f"🎙️ Voice Emotion: {voice_emotion} → {voice_truth}\n"
            f"📊 Combined Result: {combined_truth} ({combined_conf*100:.2f}%)"
        )
        status_label.config(text=result_text, foreground=colors[combined_truth])

    threading.Thread(target=task, daemon=True).start()

# =====================================================
# TKINTER GUI
# =====================================================

root = tk.Tk()
root.title("Multimodal Lie Detection System")
root.geometry("700x600")
root.resizable(False, False)

ttk.Label(root, text="🤖 Multimodal Lie Detection using Face & Voice", font=("Arial", 16, "bold")).pack(pady=10)

video_label = ttk.Label(root)
video_label.pack(pady=10)

# Camera Control Buttons
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)
ttk.Button(btn_frame, text="Start Camera", command=start_camera).grid(row=0, column=0, padx=10)
ttk.Button(btn_frame, text="Stop Camera", command=stop_camera).grid(row=0, column=1, padx=10)

# Detection Button
ttk.Button(root, text="Start Detection (3 sec)", command=analyze_combined).pack(pady=15)

status_label = ttk.Label(root, text="Press 'Start Camera' to begin.", font=("Arial", 12))
status_label.pack(pady=15)

ttk.Button(root, text="Exit", command=lambda: (stop_camera(), root.destroy())).pack(side="bottom", pady=20)

root.mainloop()
