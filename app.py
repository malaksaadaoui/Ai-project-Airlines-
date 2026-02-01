from flask import Flask, render_template, request, redirect, url_for ,jsonify, session
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sqlalchemy import func
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask import abort

import torch

from transformers import BertTokenizer
from model import BertMultiTask
from services.llm_service import call_llm, update_state
from flask import flash, request, redirect, url_for, render_template
from flask_login import login_required
from PIL import Image
import pytesseract
import re
from datetime import datetime
import os
import uuid
import pandas as pd
import joblib
# =========================
# APP CONFIG
# =========================
app = Flask(__name__)
app.secret_key = "cle_secrete_session"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = "login_register"
login_manager.init_app(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
category_encoder = joblib.load("category_encoder.pkl")
sentiment_map = joblib.load("sentiment_map.pkl")
inv_sentiment_map = {v: k for k, v in sentiment_map.items()} 
model = BertMultiTask(
    num_sentiments=3,
    num_categories=len(category_encoder.classes_)
)
model.load_state_dict(torch.load("bert_multitask.pt", map_location=device))
model.to(device)
model.eval()

# =========================
# USER MODEL
# =========================
@app.route("/")
def home():
    if not current_user.is_authenticated:
        return redirect(url_for("login_register"))

    if current_user.role == "admin":
        return redirect(url_for("admin"))

    return redirect(url_for("comment"))


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="client") 

class CustomerComment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    flight_id = db.Column(db.String(20))
    text = db.Column(db.Text)

    sentiment = db.Column(db.String(20))
    sentiment_score = db.Column(db.Float)
    category = db.Column(db.String(50))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =========================
# LOAD MODELS & METRICS
# =========================
clv_pipeline = joblib.load("clv_pipeline.pkl")
cluster_pipeline = joblib.load("flight_cluster_pipeline.pkl")
plane_cluster_pipeline = joblib.load("cluster_pipeline.pkl")
pipeline = joblib.load("rul_pipeline.pkl")
features = joblib.load("features_all.pkl") 
feature_labels = {
    'setting1': 'Operating Setting 1',
    'setting2': 'Operating Setting 2',
    's2': 'Compressor Inlet Temperature',#
    's3': 'Fan Speed',#
    's4': 'Oil Pressure',#
    's6': 'Turbine Temperature',#
    's7': 'Fuel Flow Rate',
    's8': 'Vibration Sensor 1',
    's9': 'Vibration Sensor 2',
    's11': 'Exhaust Gas Temperature',#
    's12': 'Pressure Ratio',#
    's13': 'Throttle Position',
    's14': 'Intake Pressure',
    's15': 'Rotor Speed',
    's17': 'Cooling Airflow',
    's20': 'Oil Temperature',#
    's21': 'Vibration Sensor 3'
}
cluster_description = {
    0: "⚠️ Highly Degraded Engine (Critical – Immediate Maintenance)",
    1: "✅ Normal Operating Condition",
    2: "🟡 Progressive Degradation (Monitor Closely)",
    3: "🔴 Advanced Degradation (High Risk)"
}
print("🔹 Loading Churn pipeline...")
churn_model_package = joblib.load("churn_pipeline.pkl")
churn_pipeline = churn_model_package['pipeline']
churn_transformers = churn_model_package['transformers']
print("✅ Churn pipeline loaded.")



print("🔹 Loading Cluster Activity pipeline...")
cluster_activity_pipeline = joblib.load("cluster_activity_pipeline.pkl")
cluster_activity_info = joblib.load("cluster_activity_info.pkl")
print("✅ Cluster Activity pipeline loaded (k=3).")
with open("metrics.json", "r") as f:
    MODEL_METRICS = json.load(f)


churn_metrics_data = joblib.load("churn_metrics.pkl")
CHURN_METRICS = churn_metrics_data['all_models_comparison']
print("✅ Churn metrics loaded.")

print("✅ All models loaded successfully!")
# =========================
# LOGIN / REGISTER SINGLE PAGE
# =========================
from flask import send_from_directory
import os

@app.route('/assets/<path:filename>')
def custom_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'assets'), filename)

@app.route("/auth", methods=["GET", "POST"])
def login_register():
    error = None

    if request.method == "POST":

        # LOGIN
        if "login-submit" in request.form:
            email = request.form["email"]
            password = request.form["password"]

            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                if user.role == "admin":
                    return redirect(url_for("admin"))
                return redirect(url_for("comment"))
            error = "Invalid email or password"

        # REGISTER (STAFF ONLY)
        elif "register-submit" in request.form:
            nom = request.form["nom"]
            email = request.form["email"]
            password = generate_password_hash(request.form["password"])

            if User.query.filter_by(email=email).first():
                error = "Email already registered"
            else:
                user = User(
                    nom=nom,
                    email=email,
                    password=password,
                    role="client"
                )
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return redirect(url_for("comment"))

    return render_template("login_register.html", error=error)
# =========================
# LOGOUT
# =========================
@app.route("/logout")
@login_required
def logout():
    user_role = current_user.role  # save role before logout
    logout_user()                  # log out user
    
    if user_role == "admin":
        return redirect(url_for("login_register"))
    else:  # client
        return redirect(url_for("comment"))

@app.route("/index")
@login_required
@admin_required
def index():
    return render_template("index.html")
# =========================
# PREDICTION PAGE (BOTH)
# =========================

@app.route("/predict/rul", methods=["GET", "POST"])
@login_required
@admin_required
def predict_rul():
    prediction = None
    error_msg = None

    if request.method == "POST":
        try:
            # Collect numeric inputs from the form
            input_data = {}
            for f in features:
                value = request.form.get(f, None)
                if value is None or value.strip() == "":
                    raise ValueError(f"Missing input for {feature_labels[f]}")
                

                # Only pass the model features to the pipeline
                input_data = {f: float(request.form[f]) for f in features}
                X = pd.DataFrame([input_data], columns=features)
                prediction = pipeline.predict(X)[0]


        except Exception as e:
            error_msg = str(e)

    return render_template(
        "predict_rul.html",
        features=features,
        feature_labels=feature_labels,
        prediction=prediction,
        error_msg=error_msg
    )

@app.route("/predict/clv", methods=["GET", "POST"])
@login_required
@admin_required
def predict_clv():
    clv = None

    if request.method == "POST":
        X = pd.DataFrame([{
            "Salary": float(request.form["salary"]),
            "Enrollment Month": int(request.form["enroll_month"]),
            "Enrollment Year": int(request.form["enroll_year"]),
            "Education": request.form["education"],
            "Gender": request.form["gender"],
            "Marital Status": request.form["marital"],
            "Loyalty Card": request.form["loyalty_card"]
        }])

        y = clv_pipeline.predict(X)
        clv = float(np.expm1(y[0]))

    return render_template("predict_clv.html", clv=clv)

# ---- CHURN ----
@app.route("/predict/churn", methods=["GET", "POST"])
@login_required
@admin_required
def predict_churn():
    churn = None
    score = None

    if request.method == "POST":
        X = pd.DataFrame([{
            'Total Flights': float(request.form["total_flights"]),
            'Distance': float(request.form["distance"]),
            'Points Accumulated': float(request.form["points_accumulated"]),
            'Points Redeemed': float(request.form["points_redeemed"]),
            'Dollar Cost Points Redeemed': float(request.form["dollar_cost"]),
            'Enrollment Year': int(request.form["enrollment_year"]),
            'Active_Years': int(request.form["active_years"]),
            'CLV': float(request.form["clv"]),
            'Month': int(request.form.get("month", 1)),
            'Education_encoded': int(request.form["education_encoded"]),
            'Gender_Female': int(request.form["gender_female"]),
            'Gender_Male': int(request.form["gender_male"]),
            'Marital Status_Single': int(request.form["marital_single"]),
            'Enrollment Type_Standard': int(request.form["enrollment_standard"]),
            'Enrollment Type_2018 Promotion': int(request.form["enrollment_promo"])
        }])

        raw = churn_pipeline.predict_proba(X)[:, 1][0]
        churn_threshold = churn_model_package.get("threshold", 0.5)
        score = min(raw * 15, 1.0)
        churn = int(score >= churn_threshold)

    return render_template("predict_churn.html", churn=churn, score=score)
@app.route("/predict/plane-cluster", methods=["GET", "POST"])
@login_required
@admin_required
def predict_plane_cluster():
    rul_prediction = None
    cluster = None
    description = None
    error_msg = None

    if request.method == "POST":
        try:
            # 1️⃣ Collect raw user inputs and convert to float
            input_data = {}
            for f in features:
                value = request.form.get(f)
                if value is None or value.strip() == "":
                    raise ValueError(f"Missing value for {feature_labels[f]}")
                input_data[f] = float(value)

            # 2️⃣ Create DataFrame with correct feature order
            X = pd.DataFrame([input_data], columns=features)

            # 3️⃣ Make predictions
            rul_prediction = float(pipeline.predict(X)[0])
            cluster = int(plane_cluster_pipeline.predict(X)[0])
            description = cluster_description.get(cluster, "Unknown engine state")

        except Exception as e:
            error_msg = str(e)
            print("Error in plane cluster prediction:", error_msg)  # 🔹 Print error to console

    # 4️⃣ Render template and pass error_msg to show in HTML
    return render_template(
        "predict_plane_cluster.html",
        features=features,
        feature_labels=feature_labels,
        rul_prediction=rul_prediction,
        cluster=cluster,
        description=description,
        error_msg=error_msg
    )

@app.route("/predict/flight-cluster", methods=["GET", "POST"])
@login_required
@admin_required
def predict_flight_cluster():
    cluster = None

    if request.method == "POST":
        X = pd.DataFrame([{
            "Airline": request.form["airline"],
            "Flight": int(request.form["flight"]),
            "AirportFrom": request.form["airport_from"],
            "AirportTo": request.form["airport_to"],
            "DayOfWeek": int(request.form["day"]),
            "Time": int(request.form["time"]),
            "Length": int(request.form["length"]),
            "Delay": int(request.form["delay"])
        }])

        cluster = int(cluster_pipeline.predict(X)[0])

    return render_template("predict_flight_cluster.html", cluster_flight=cluster)

# ---- CUSTOMER CLUSTER ----
@app.route("/predict/customer_cluster", methods=["GET", "POST"])
@login_required
@admin_required
def predict_customer_cluster():
    cluster = None
    name = None
    description = None

    if request.method == "POST":
        X = pd.DataFrame([{
            'Month': 6,
            'Total Flights': float(request.form["total_flights"]),
            'Distance': float(request.form["distance"]),
            'Points Accumulated': float(request.form["points_accumulated"]),
            'Points Redeemed': float(request.form["points_redeemed"]),
            'Dollar Cost Points Redeemed': float(request.form["dollar_cost"]),
            'Year': int(request.form["year"])
        }])

        cluster = int(cluster_activity_pipeline.predict(X)[0])
        name = cluster_activity_info["cluster_names"][cluster]
        description = {
            0: "Clients très engagés",
            1: "Grands voyageurs",
            2: "Clients peu actifs"
        }[cluster]

    return render_template(
        "predict_customer_cluster.html",
        cluster=cluster,
        name=name,
        description=description
    )


# =========================
# POWER BI (ADMIN ONLY)
# =========================
# ADMIN DASHBOARD
# =========================
@app.route("/admin")
@login_required
@admin_required
def admin():
    # Total counts
    total_comments = CustomerComment.query.count()
    negative_comments = CustomerComment.query.filter(
        CustomerComment.sentiment.ilike("negative")
    ).count()
    positive_comments = CustomerComment.query.filter(
        CustomerComment.sentiment.ilike("positive")
    ).count()

    negative_by_category = db.session.query(
        CustomerComment.category,
        db.func.count(CustomerComment.id)
    ).filter(CustomerComment.sentiment.ilike("negative")) \
     .group_by(CustomerComment.category).all()

    negative_dict = {cat: count for cat, count in negative_by_category}

    # Categories list for charts
    categories = sorted(list(set(list(negative_dict.keys()))))

    # Fill missing categories with 0
    negative_counts = [negative_dict.get(cat, 0) for cat in categories]


    return render_template(
        "bi_dashboard.html",
        total_comments=total_comments,
        positive_comments=positive_comments,
        negative_comments=negative_comments,
        categories=categories or [],
        negative_counts=negative_counts or [],
)

@app.route("/comment", methods=["GET", "POST"])
@login_required
def comment():
    if current_user.role == "admin":
        return redirect(url_for("admin"))

    # Track login status
    is_logged_in = current_user.is_authenticated

    sentiment = None
    category = None
    message = None
    text = ""

    # Only allow POST if user is logged in
    if request.method == "POST" and is_logged_in:
        text = request.form.get("text", "").strip()
        flight_id = request.form.get("flight_id", None)

        if text:
            # === BERT inference ===
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                sent_logits, cat_logits = model(input_ids, attention_mask)

            # Sentiment
            sent_id = torch.argmax(sent_logits, dim=1).item()
            sentiment = inv_sentiment_map[sent_id].strip().lower()

            # Category only if negative
            category = None
            if sentiment == "negative":
                cat_id = torch.argmax(cat_logits, dim=1).item()
                category = category_encoder.inverse_transform([cat_id])[0]

            # Save to DB
            comment_record = CustomerComment(
                user_id=current_user.id,
                flight_id=flight_id,
                text=text,
                sentiment=sentiment,
                category=category,
            )
            db.session.add(comment_record)
            db.session.commit()

            message = "Comment submitted successfully!"
        else:
            message = "Please enter a comment"

    # Load recent positive comments (available to all users)
    positive_comments = CustomerComment.query.filter(
        func.lower(func.trim(CustomerComment.sentiment)) == "positive"
    ).order_by(CustomerComment.created_at.desc()).limit(6).all()

    return render_template(
        "comment.html",
        is_logged_in=is_logged_in,           # <-- NEW
        text=text,
        sentiment=sentiment or "Can't Tell",
        category=category or "Can't Tell",
        message=message,
        positive_comments=positive_comments
    )

from flask import flash, request, redirect, url_for, render_template
from flask_login import login_required
from PIL import Image
import pytesseract
import re
from datetime import datetime
import os
import uuid
import pandas as pd
import joblib

# -----------------------------
# Load full pipeline once
# -----------------------------
delai_pipeline = joblib.load("xgb_pipeline_all.pkl")  # contains preprocessing + XGB

@app.route("/upload_ticket", methods=["POST"])
@login_required
def upload_ticket():
    if "ticket_image" not in request.files:
        flash("No file selected")
        return redirect(url_for("comment"))

    file = request.files["ticket_image"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("comment"))

    # -----------------------------
    # Save uploaded file
    # -----------------------------
    upload_folder = os.path.join(app.root_path, "static/uploads")
    os.makedirs(upload_folder, exist_ok=True)
    
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    # -----------------------------
    # OCR processing
    # -----------------------------
    image = Image.open(file_path).convert("RGB")
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ocr_text_raw = pytesseract.image_to_string(
        image,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:"
    )

    # -----------------------------
    # Normalize OCR
    # -----------------------------
    ocr_text = re.sub(r"\s+", " ", ocr_text_raw).upper()
    ocr_text = ocr_text.replace("AN", "AM").replace("PN", "PM")
    ocr_text = re.sub(r"\bS\b", "SEP", ocr_text)
    ocr_text = re.sub(r"\b22S\b", "22 SEP", ocr_text)
    ocr_text = re.sub(r"H(H)?OU", "HOU", ocr_text)
    ocr_text = re.sub(r"DFWDFW", "DFW", ocr_text)

    # -----------------------------
    # Extract airline, flight, airports, time
    # -----------------------------
    airline_match = re.search(r"([A-Z]{2})\s*(\d{3,4})", ocr_text)
    airline = airline_match.group(1) if airline_match else ""
    flight_number = airline_match.group(2) if airline_match else ""

    flight_pos = ocr_text.find(flight_number)
    from_airport, to_airport = "", ""
    if flight_pos != -1:
        before_flight = ocr_text[:flight_pos][::-1]
        from_match = re.search(r"([A-Z]{3})", before_flight)
        from_airport = from_match.group(1)[::-1] if from_match else ""
        after_flight = ocr_text[flight_pos:]
        to_match = re.search(r"([A-Z]{3})", after_flight)
        to_airport = to_match.group(1) if to_match else ""

    airport_corrections = {"GAT": "HOU", "HHOU": "HOU", "DFWDFW": "DFW", "OU": "HOU", "GRO": "HOU"}
    from_airport = airport_corrections.get(from_airport, from_airport)
    to_airport = airport_corrections.get(to_airport, to_airport)

    # -----------------------------
    # Extract date and time
    # -----------------------------
    time_match = re.search(r"(\d{1,2}:\d{2}[AP]M?)", ocr_text)
    time_str = time_match.group(1) if time_match else "00:00AM"

    day = 1
    month = "JAN"
    if time_match:
        look_back = ocr_text[max(0, time_match.start()-30):time_match.start()]
        date_match = re.search(r"(\d{1,2})\s*([A-Z]{1,3})", look_back)
        if date_match:
            day = int(date_match.group(1))
            month_raw = date_match.group(2)
            month_map = {
                "JAN":"JAN","FEB":"FEB","MAR":"MAR","APR":"APR","MAY":"MAY","JUN":"JUN",
                "JUL":"JUL","AUG":"AUG","SEP":"SEP","S":"SEP",
                "OCT":"OCT","NOV":"NOV","DEC":"DEC"
            }
            month = month_map.get(month_raw, "SEP")
    # Fixed year = 2026
    try:
        date_obj = datetime.strptime(f"{day} {month} 2026", "%d %b %Y")
        dayofweek = date_obj.weekday() + 1
    except:
        date_obj = datetime(2026, 1, 1)
        dayofweek = 1

    # -----------------------------
    # Convert time to numeric for model
    # -----------------------------
    try:
        time_num = int(datetime.strptime(time_str, "%I:%M%p").strftime("%H%M"))
    except:
        time_num = 0

    features_df = pd.DataFrame([{
        "Airline": airline,
        "Flight": int(flight_number) if flight_number.isdigit() else 0,
        "AirportFrom": from_airport,
        "AirportTo": to_airport,
        "DayOfWeek": dayofweek,
        "Time": time_num,
        "Length": 1.0  # placeholder
    }])

    # -----------------------------
    # Predict delay
    # -----------------------------
    delay_pred = delai_pipeline.predict(features_df)[0]
    delay_status = "Delayed" if delay_pred == 1 else "On Time"

    # -----------------------------
    # Return results including delay info
    # -----------------------------
    ticket_result = {
        "Airline": airline,
        "Flight": flight_number,
        "From": from_airport,
        "To": to_airport,
        "Date": date_obj.strftime("%d %b %Y"),  # full date
        "DayOfWeek": dayofweek,
        "Time": time_str,
        "DelayStatus": delay_status,           # <-- added
        "image_url": url_for('static', filename=f"uploads/{unique_filename}")
    }

    return render_template("comment.html", ticket=ticket_result)

@app.route("/my-space")
@login_required
def my_space():
    # last user comments
    my_comments = CustomerComment.query.filter_by(
        user_id=current_user.id
    ).order_by(CustomerComment.created_at.desc()).all()

    return render_template(
        "my_space.html",
        my_comments=my_comments
    )
from flask import send_file
import whisper
import os
import tempfile
import subprocess

# Configuration FFmpeg pour Windows
os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin"

# Vérifier FFmpeg
def check_ffmpeg():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            print("✅ FFmpeg détecté !")
            return True
    except:
        print("❌ FFmpeg NON trouvé")
        return False

# Charger le modèle Whisper
if 'whisper_model' not in globals():
    whisper_model = whisper.load_model("small")
    print("✅ Modèle Whisper chargé !")

ffmpeg_available = check_ffmpeg()


def speech_to_text(file_storage):
    """
    Convertit un fichier audio en texte avec Whisper
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        temp_path = temp_file.name
        file_storage.save(temp_path)
    
    try:
        print(f"🎤 Transcription du fichier : {temp_path}")
        
        # Vérifier que le fichier existe
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Fichier non trouvé : {temp_path}")
        
        file_size = os.path.getsize(temp_path)
        print(f"📦 Taille : {file_size} bytes")
        
        if file_size < 100:
            raise ValueError("Fichier audio trop petit")
        
        # ============================================
        # TRANSCRIPTION AMÉLIORÉE
        # ============================================
        
        # Tentative 1 : Anglais strict
        print("🔊 Tentative 1 : Détection anglais...")
        result = whisper_model.transcribe(
            temp_path,
            language="en",           # Force l'anglais
            fp16=False,
            verbose=False,
            task="transcribe",       # Pas de traduction
            best_of=5,               # Essayer 5 fois et prendre le meilleur
            beam_size=5,             # Meilleure précision
            temperature=0.0,         # Pas de hasard (plus déterministe)
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False  # Pas de contexte précédent
        )
        
        transcribed_text = result["text"].strip()
        
        # Vérifier si la transcription semble valide
        if transcribed_text and len(transcribed_text) > 0:
            # Vérifier qu'il n'y a pas de caractères bizarres (coréen, japonais, etc.)
            import unicodedata
            
            # Compter les caractères non-latins
            non_latin = sum(1 for c in transcribed_text 
                          if unicodedata.category(c).startswith('Lo'))  # Lo = Other Letter
            
            total_letters = sum(1 for c in transcribed_text if c.isalpha())
            
            # Si plus de 30% de caractères non-latins, réessayer
            if total_letters > 0 and (non_latin / total_letters) > 0.3:
                print(f"⚠️ Détection suspecte ({non_latin}/{total_letters} caractères non-latins)")
                print(f"   Texte détecté : {transcribed_text}")
                print("🔄 Tentative 2 : Sans langue forcée...")
                
                # Tentative 2 : Laisser Whisper auto-détecter
                result = whisper_model.transcribe(
                    temp_path,
                    fp16=False,
                    verbose=False,
                    best_of=5,
                    beam_size=5,
                    temperature=0.0
                )
                
                transcribed_text = result["text"].strip()
                detected_lang = result.get("language", "unknown")
                print(f"   Langue détectée : {detected_lang}")
                print(f"   Nouveau texte : {transcribed_text}")
        
        print(f"✅ Transcription finale : {transcribed_text}")
        
        # Vérification finale
        if not transcribed_text or transcribed_text == "":
            raise ValueError("Transcription vide après traitement")
        
        return transcribed_text
    
    except Exception as e:
        print(f"❌ Erreur : {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️ Fichier temporaire supprimé")
            except:
                pass


def text_to_speech(text: str) -> str:
    """
    Convertit du texte en audio avec pyttsx3 (local, pas besoin d'API)
    Retourne le chemin du fichier audio généré
    """
    try:
        import pyttsx3
        
        # Créer fichier temporaire pour l'audio
        audio_file = tempfile.mktemp(suffix=".mp3")
        
        # Initialiser le moteur TTS
        engine = pyttsx3.init()
        
        # Configuration de la voix ANGLAISE
        voices = engine.getProperty('voices')
        english_voice = None
        
        # Chercher une voix anglaise (priorité : US English female)
        for voice in voices:
            name_lower = voice.name.lower()
            # Priorité 1 : Voix féminine US
            if 'zira' in name_lower or ('english' in name_lower and 'female' in name_lower):
                english_voice = voice.id
                print(f"🎙️ Voix sélectionnée : {voice.name}")
                break
            # Priorité 2 : N'importe quelle voix anglaise
            elif 'english' in name_lower or 'david' in name_lower or 'zira' in name_lower:
                english_voice = voice.id
        
        # Si trouvée, appliquer la voix anglaise
        if english_voice:
            engine.setProperty('voice', english_voice)
        else:
            # Fallback : utiliser la première voix disponible
            print("⚠️ Aucune voix anglaise trouvée, utilisation de la voix par défaut")
            if len(voices) > 0:
                engine.setProperty('voice', voices[0].id)
        
        # Paramètres de la voix
        engine.setProperty('rate', 150)    # Vitesse (150 = normal)
        engine.setProperty('volume', 0.9)  # Volume (0.0 à 1.0)
        
        # Sauvegarder l'audio
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        
        print(f"🔊 Audio généré : {audio_file}")
        return audio_file
    
    except ImportError:
        print("⚠️ pyttsx3 non installé. Installez avec : pip install pyttsx3")
        return None
    except Exception as e:
        print(f"❌ Erreur TTS : {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route("/voice/assistant", methods=["GET", "POST"])
@login_required
def voice_assistant():
    if request.method == "POST" and "voice_file" in request.files:
        try:
            voice_file = request.files["voice_file"]
            
            print(f"\n{'='*60}")
            print(f"📨 Requête vocale reçue - {voice_file.filename}")
            print(f"{'='*60}")
            
            # 1️⃣ Convertir l'audio en texte
            user_text = speech_to_text(voice_file)
            
            if not user_text or user_text.strip() == "":
                raise ValueError("Transcription vide")
            
            # 2️⃣ Générer la réponse avec votre LLM local
            state = session.get("dialogue_state", {})
            llm_output = call_llm(user_text, state)
            
            # Sauvegarder l'input utilisateur pour l'historique
            llm_output["user_input"] = user_text
            
            # 3️⃣ Mettre à jour l'état
            state = update_state(state, llm_output)
            session["dialogue_state"] = state
            
            response_text = llm_output.get("response", "")
            
            # 4️⃣ Générer l'audio de la réponse
            audio_file = text_to_speech(response_text)
            
            # 5️⃣ Retourner la réponse
            return jsonify({
                "success": True,
                "transcription": user_text,
                "response": response_text,
                "intent": llm_output.get("intent", ""),
                "slots": llm_output.get("slots", {}),
                "has_audio": audio_file is not None,
                "audio_url": url_for('get_audio', filename=os.path.basename(audio_file)) if audio_file else None
            })
        
        except Exception as e:
            print(f"\n❌ ERREUR : {type(e).__name__}")
            print(f"   Message : {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # GET : affichage initial
    return render_template("comment.html")


@app.route("/voice/audio/<filename>")
@login_required
def get_audio(filename):
    """
    Sert les fichiers audio générés
    """
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/mpeg")
    else:
        return "Audio non trouvé", 404


# =========================
# INIT DB
# =========================
with app.app_context():
    db.create_all()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
