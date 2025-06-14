import sys
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import os
from PIL import Image as PILImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QLabel, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import mediapipe as mp

# Load Model and Labels
model = load_model("C:/Users/onkar/Desktop/Onkar personal/Programming/projects/SignFlow/Frontend/sign_language_modelv19_phase1.h5")
# Correct label list
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space'
]
label_map = {i: label for i, label in enumerate(labels)}


# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# ====== Main Menu ======
class MainMenu(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.setStyleSheet(""" QWidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0f7fa, stop:1 #80deea); } """)
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 20)
        layout.setSpacing(20)

        title = QLabel("📱 SignFlow - Communication App")
        title.setStyleSheet("font-size: 30px; font-weight: bold; color: #006064;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        subtitle = QLabel("Bridge the gap between sign and spoken language")
        subtitle.setStyleSheet("font-size: 16px; font-style: italic; color: #004d40;")
        layout.addWidget(subtitle, alignment=Qt.AlignCenter)

        layout.addSpacing(20)
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(20)

        modes = [
            ("✋ Sign to Speech", "Convert hand signs into spoken words", "sign"),
            ("🗣️ Speech to Sign", "Turn your voice into visual sign cues", "speech"),
            ("🔁 Bidirectional", "Two-way communication made simple", "both")
        ]

        for title, desc, mode in modes:
            btn_widget = QWidget()
            btn_widget.setStyleSheet("""
                QWidget { background-color: white; border: 2px solid #4dd0e1; border-radius: 10px; }
                QWidget:hover { background-color: #e0f7fa; }
            """)
            btn_layout_inner = QVBoxLayout(btn_widget)
            btn_layout_inner.setContentsMargins(15, 10, 15, 10)

            btn_title = QPushButton(title)
            btn_title.setStyleSheet("""
                QPushButton { font-size: 18px; font-weight: bold; color: #00796b; background-color: transparent; border: none; text-align: left; }
                QPushButton:hover { color: #004d40; }
            """)
            btn_title.setCursor(Qt.PointingHandCursor)
            btn_title.clicked.connect(lambda _, m=mode: self.switch_callback(m))

            btn_desc = QLabel(desc)
            btn_desc.setStyleSheet("font-size: 13px; color: #555;")

            btn_layout_inner.addWidget(btn_title)
            btn_layout_inner.addWidget(btn_desc)
            btn_layout.addWidget(btn_widget)

        layout.addLayout(btn_layout)
        layout.addStretch()

        footer = QLabel("© 2025 SignFlow • Created by SY1507")
        footer.setStyleSheet("font-size: 12px; color: #555;")
        layout.addWidget(footer, alignment=Qt.AlignCenter)

        self.setLayout(layout)


# ====== Sign to Speech ======
class SignToSpeech(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.engine = pyttsx3.init()
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(200)  # Slower frame rate (200ms = 5 FPS)

        self.sentence = ""
        self.last_predicted_label = ""
        self.frame_count = 0
        self.threshold = 10

        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        layout = QVBoxLayout()
        title = QLabel("✋ Sign to Speech Mode")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        self.video_label = QLabel("📷 Camera Feed")
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Detected sentence will appear here...")
        layout.addWidget(self.result_text)

        btn_layout = QHBoxLayout()
        buttons = {
            "▶️ Start Camera": self.start_camera,
            "⏸️ Pause": self.toggle_camera,
            "🧹 Clear": self.clear_text,
            "🗣️ Speak": self.speak_text,
            "🔙 Back": self.go_back
        }
        for text, action in buttons.items():
            btn = QPushButton(text)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(action)
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start()

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()

    def clear_text(self):
        self.sentence = ""
        self.result_text.clear()

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        x1, y1, x2, y2 = 100, 100, 300, 300  # Default ROI
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = rgb_frame.shape
            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]
            x1, y1, x2, y2 = max(min(xs) - 20, 0), max(min(ys) - 20, 0), min(max(xs) + 20, w), min(max(ys) + 20, h)

        roi = rgb_frame[y1:y2, x1:x2]
        if roi.size > 0:
            processed = cv2.resize(roi, (128, 128)) / 255.0
            prediction = model.predict(np.expand_dims(processed, axis=0))[0]
            confidence = np.max(prediction)
            predicted_label = label_map.get(np.argmax(prediction), "")

            if predicted_label == self.last_predicted_label:
                self.frame_count += 1
            else:
                self.last_predicted_label = predicted_label
                self.frame_count = 0

            if self.frame_count == self.threshold:
                self.sentence += predicted_label
                self.result_text.setPlainText(self.sentence.strip())
                self.frame_count = 0

            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{predicted_label} ({confidence*100:.1f}%)"
            cv2.putText(rgb_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        img = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def speak_text(self):
        text = self.result_text.toPlainText()
        if text.strip():
            self.engine.say(text)
            self.engine.runAndWait()

    def go_back(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
        self.stack.setCurrentIndex(0)


# ====== Speech to Sign ======
class SpeechToSign(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.recognizer = sr.Recognizer()
        self.timer = QTimer()
        self.image_index = 0
        self.images = []
        self.dataset_path = "C:/Users/onkar/Desktop/Onkar personal/Programming/projects/SignFlow/dataset 2/asl_dataset/"

        layout = QVBoxLayout()
        title = QLabel("🗣️ Speech to Sign Mode")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        self.image_label = QLabel("Sign Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        self.transcript_label = QLabel("Transcribed text will appear here...")
        self.transcript_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.transcript_label)

        btn_layout = QHBoxLayout()
        self.btn_speak = QPushButton("🎙️ Speak")
        self.btn_speak.clicked.connect(self.capture_speech)
        btn_layout.addWidget(self.btn_speak)

        self.btn_back = QPushButton("🔙 Back")
        self.btn_back.clicked.connect(self.go_back)
        btn_layout.addWidget(self.btn_back)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def capture_speech(self):
        try:
            with sr.Microphone() as source:
                self.transcript_label.setText("Listening...")
                QApplication.processEvents()
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                self.transcript_label.setText(f"Recognized: {text}")
                self.display_sign_images(text)

        except sr.UnknownValueError:
            self.transcript_label.setText("Sorry, I didn't catch that.")
        except sr.RequestError:
            self.transcript_label.setText("Speech recognition service error.")

    def display_sign_images(self, text):
        self.images = []
        for char in text.upper():
            if char == " ":
                continue
            folder_path = os.path.join(self.dataset_path, char)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    img_path = os.path.join(folder_path, image_files[0])
                    self.images.append(img_path)

        if self.images:
            self.image_index = 0
            self.show_next_image()
        else:
            self.image_label.setText("No sign images available.")

    def show_next_image(self):
        if self.image_index < len(self.images):
            img_path = self.images[self.image_index]
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_index += 1
            QTimer.singleShot(1000, self.show_next_image)
        else:
            self.image_label.setText("")

    def go_back(self):
        self.stack.setCurrentIndex(0)


# ====== Bidirectional Mode ======
class Bidirectional(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        title = QLabel("🔁 Bidirectional Mode: Sign <-> Speech")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        self.sign_mode = SignToSpeech(stack)
        self.speech_mode = SpeechToSign(stack)

        dual_layout = QHBoxLayout()
        dual_layout.addWidget(self.sign_mode, 1)
        dual_layout.addWidget(self.speech_mode, 1)

        layout.addLayout(dual_layout)
        self.setLayout(layout)


# ====== Main Window ======
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignFlow")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.menu = MainMenu(self.switch_screen)
        self.sign = SignToSpeech(self.stack)
        self.speech = SpeechToSign(self.stack)
        self.both = Bidirectional(self.stack)

        self.stack.addWidget(self.menu)
        self.stack.addWidget(self.sign)
        self.stack.addWidget(self.speech)
        self.stack.addWidget(self.both)
        self.stack.setCurrentIndex(0)

    def switch_screen(self, mode):
        self.stack.setCurrentIndex({"sign": 1, "speech": 2, "both": 3}[mode])


# ====== Run App ======
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
