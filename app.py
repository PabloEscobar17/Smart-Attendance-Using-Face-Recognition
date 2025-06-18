import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only errors

from flask_migrate import Migrate
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faceauth.db'  # Update this as per your project
db = SQLAlchemy(app)

migrate = Migrate(app, db)


# Add these with your other imports at the top
import pickle
import time
from threading import Thread
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from deepface import DeepFace
import os
import cv2
import numpy as np
import base64
from datetime import datetime
import sqlite3
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faceauth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'user_data'
app.config['ATTENDANCE_FOLDER'] = 'attendance_records'

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ATTENDANCE_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    face_encoding = db.Column(db.LargeBinary)
    is_admin = db.Column(db.Boolean, default=False)  # Add this line
    registered_on = db.Column(db.DateTime, default=datetime.utcnow)


# Attendance model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    method = db.Column(db.String(20))  # 'face' or 'manual'

    user = db.relationship('User', backref=db.backref('attendances', lazy=True))


# Initialize database
with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Helper functions
def save_face_encoding(user_id, embedding):
    """Save face encoding to database"""
    try:
        # Convert numpy array to bytes
        embedding_bytes = pickle.dumps(np.array(embedding))
        user = User.query.get(user_id)
        user.face_encoding = embedding_bytes
        db.session.commit()
        return True
    except Exception as e:
        print(f"Error saving encoding: {e}")
        return False


def verify_face_live(img_path, user_encoding_bytes):
    """Enhanced verification with metrics tracking"""
    start_time = time.time()

    try:
        stored_encoding = pickle.loads(user_encoding_bytes)
        current_embedding = DeepFace.represent(
            img_path=img_path,
            model_name='Facenet',
            enforce_detection=True,
            detector_backend='opencv'
        )[0]['embedding']

        similarity = cosine_similarity([stored_encoding], [current_embedding])[0][0]
        inference_time = time.time() - start_time

        # Update live metrics
        is_match = similarity > 0.6
        update_live_metrics(is_match, similarity, inference_time)

        return is_match, similarity, inference_time

    except Exception as e:
        inference_time = time.time() - start_time
        update_live_metrics(False, 0, inference_time)
        return False, 0, inference_time

live_metrics = {
    "inference_time": deque(maxlen=20),
    "confidence": deque(maxlen=20),
    "frame_count": 0,
    "correct_matches": 0
}

def update_live_metrics(success, confidence, inference_time):
    """Update metrics in real-time"""
    live_metrics["confidence"].append(confidence)
    live_metrics["inference_time"].append(inference_time)
    live_metrics["frame_count"] += 1
    if success:
        live_metrics["correct_matches"] += 1


def create_admin_account():
    """Create admin account if it doesn't exist"""
    admin_username = "admin"
    admin_password = "secureadmin123"  # Change this to your desired password

    if not User.query.filter_by(username=admin_username).first():
        admin = User(
            username=admin_username,
            password=generate_password_hash(admin_password),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin account created successfully!")

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('Admin access required!', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()

            login_user(new_user)
            flash('Account created successfully! Please register your face.', 'success')
            return redirect(url_for('face_register'))

    return render_template('register.html')


@app.route('/face-register', methods=['GET', 'POST'])
@login_required
def face_register():
    if request.method == 'POST':
        face_data = request.form.get('face_data')

        if not face_data or ',' not in face_data:
            flash('Invalid image data received', 'danger')
            return redirect(url_for('face_register'))

        try:
            img_data = face_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)

            if not img_bytes:
                flash('Empty image data after decoding', 'danger')
                return redirect(url_for('face_register'))

            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                flash('Failed to decode image. Please try again.', 'danger')
                return redirect(url_for('face_register'))

            # Save temp image for DeepFace
            temp_path = f'temp_face_{current_user.id}.jpg'
            cv2.imwrite(temp_path, img)

            try:
                embedding = DeepFace.represent(
                    img_path=temp_path,
                    model_name='Facenet',
                    enforce_detection=True,
                    detector_backend='retinaface'
                )[0]['embedding']

                if save_face_encoding(current_user.id, embedding):
                    flash('Face registered successfully!', 'success')
                else:
                    flash('Failed to save face data', 'danger')

            except ValueError as e:
                flash(f'No face detected: {str(e)}', 'danger')
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f'Unexpected error while processing image: {str(e)}', 'danger')
            return redirect(url_for('face_register'))

    return render_template('face_register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next') or url_for('dashboard')
            return redirect(next_page)
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/mark-attendance', methods=['GET', 'POST'])
@login_required
def mark_attendance():
    if request.method == 'POST':
        img_data = request.form.get('face_data').split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        temp_path = f'att_temp_{current_user.id}.jpg'
        cv2.imwrite(temp_path, img)

        # Use the enhanced verification
        is_match, confidence, inference_time = verify_face_live(temp_path, current_user.face_encoding)

        if is_match:
            new_att = Attendance(user_id=current_user.id, method='face')
            db.session.add(new_att)
            db.session.commit()
            flash(f'Verified! Confidence: {confidence:.2f}', 'success')
        else:
            flash(f'Verification failed (Confidence: {confidence:.2f})', 'danger')

        os.remove(temp_path)

    return render_template('mark_attendance.html')


# Add this with your other route definitions
@app.route('/metrics')
@login_required
def metrics():
    """Render the live metrics dashboard"""
    return render_template('metrics.html')

# Make sure you have the metrics endpoint in your get_metrics function too
@app.route('/api/metrics')
@login_required
def api_metrics():
    """Endpoint for live metrics data (JSON)"""
    return {
        "accuracy": f"{(live_metrics['correct_matches']/live_metrics['frame_count'])*100:.2f}%" if live_metrics['frame_count'] > 0 else "0%",
        "avg_inference_ms": f"{sum(live_metrics['inference_time'])/len(live_metrics['inference_time'])*1000:.2f}" if live_metrics['inference_time'] else "0",
        "avg_confidence": f"{sum(live_metrics['confidence'])/len(live_metrics['confidence']):.2f}" if live_metrics['confidence'] else "0",
        "total_frames": live_metrics["frame_count"]
    }

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    # Get last 5 attendance records
    attendances = Attendance.query.filter_by(user_id=current_user.id) \
        .order_by(Attendance.timestamp.desc()) \
        .limit(5) \
        .all()

    return render_template('dashboard.html',
                           user=current_user,
                           attendances=attendances)


@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Admin access required!', 'danger')
        return redirect(url_for('dashboard'))

    # Get all attendance records
    attendances = Attendance.query.order_by(Attendance.timestamp.desc()).all()

    # Get all users
    users = User.query.all()

    return render_template('admin_dashboard.html',
                           attendances=attendances,
                           users=users)


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('home'))


if __name__ == '__main__':
    with app.app_context():
        create_admin_account()
    app.run(host="127.0.0.1", port=5050, debug=True)



# Add to your existing imports


# Global variables for live metrics

@app.route('/metrics')
@login_required
def get_metrics():
    """Endpoint for live metrics"""
    return {
        "accuracy": f"{(live_metrics['correct_matches']/live_metrics['frame_count'])*100:.2f}%" if live_metrics['frame_count'] > 0 else "0%",
        "avg_inference_ms": f"{sum(live_metrics['inference_time'])/len(live_metrics['inference_time'])*1000:.2f}" if live_metrics['inference_time'] else "0",
        "avg_confidence": f"{sum(live_metrics['confidence'])/len(live_metrics['confidence']):.2f}" if live_metrics['confidence'] else "0",
        "total_frames": live_metrics["frame_count"]
    }