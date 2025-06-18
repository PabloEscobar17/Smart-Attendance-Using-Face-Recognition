# 🧠 Smart Attendance Using Face Recognition

A secure, AI-driven attendance system that uses facial recognition for marking presence in real-time. Designed for schools, colleges, or offices, this project enables seamless, contactless, and tamper-proof attendance logging.

---

## ✅ Features

- 🎭 Face Registration using DeepFace + RetinaFace
- 🧠 Real-time face authentication with `cosine_similarity`
- 📊 Live metrics tracking (inference time, confidence, accuracy)
- 👨‍🏫 Admin dashboard with complete user & attendance control
- 🔒 Secure login system with hashed passwords
- 💾 SQLite-based backend using SQLAlchemy
- 📸 Attendance via webcam, verified using facial embeddings
- 🌐 Web app built with Flask

---

## 🚀 Tech Stack

| Layer       | Tools & Libraries                    |
|-------------|--------------------------------------|
| Frontend    | HTML, CSS, Jinja2 (Flask Templates)  |
| Backend     | Python, Flask, Flask-Login           |
| AI Model    | DeepFace (Facenet + RetinaFace)      |
| DB & ORM    | SQLite + SQLAlchemy                  |
| Auth        | werkzeug.security + Flask-Login      |
| Metrics     | Python `deque`, `cosine_similarity`  |

---

## 🧰 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/smart-attendance-face-recognition.git
cd smart-attendance-face-recognition
