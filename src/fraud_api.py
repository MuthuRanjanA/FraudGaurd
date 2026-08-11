from pathlib import Path
import os

import joblib
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)

from flask_sqlalchemy import SQLAlchemy

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user
)

from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from wtforms import (
    StringField,
    PasswordField,
    SubmitField
)

from wtforms.validators import (
    DataRequired,
    Email,
    Length,
    EqualTo
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR
    / "model"
    / "fraud_model.pkl"
)

SCALER_FILE = (
    BASE_DIR
    / "model"
    / "scaler.pkl"
)


# ============================================================
# FEATURE ORDER
#
# THIS MUST MATCH train_model.py EXACTLY
# ============================================================

FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=str(
        BASE_DIR / "templates"
    ),
    static_folder=str(
        BASE_DIR / "static"
    )
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "dev-secret-change-me"
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'fraudguard.db'}"
)

app.config[
    "SQLALCHEMY_TRACK_MODIFICATIONS"
] = False


# ============================================================
# EXTENSIONS
# ============================================================

db = SQLAlchemy(app)

login_manager = LoginManager(app)

login_manager.login_view = "login"

csrf = CSRFProtect(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[]
)


# ============================================================
# USER MODEL
# ============================================================

class User(
    UserMixin,
    db.Model
):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )


# ============================================================
# LOGIN FORM
# ============================================================

class LoginForm(
    FlaskForm
):

    email = StringField(
        "Email",
        validators=[
            DataRequired(),
            Email()
        ]
    )

    password = PasswordField(
        "Password",
        validators=[
            DataRequired()
        ]
    )

    submit = SubmitField(
        "Login"
    )


# ============================================================
# REGISTER FORM
# ============================================================

class RegisterForm(
    FlaskForm
):

    email = StringField(
        "Email",
        validators=[
            DataRequired(),
            Email()
        ]
    )

    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=6)
        ]
    )

    confirm = PasswordField(
        "Confirm Password",
        validators=[
            DataRequired(),
            EqualTo("password")
        ]
    )

    submit = SubmitField(
        "Create Account"
    )


# ============================================================
# LOGIN USER LOADER
# ============================================================

@login_manager.user_loader
def load_user(user_id):

    return db.session.get(
        User,
        int(user_id)
    )


# ============================================================
# LOAD ML MODEL
# ============================================================

def load_model():

    if (
        not MODEL_FILE.exists()
        or not SCALER_FILE.exists()
    ):

        return None, None

    model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    return model, scaler


# ============================================================
# FRAUD PREDICTION
# ============================================================

def predict(features):

    model, scaler = load_model()

    if model is None:

        raise RuntimeError(
            "Model is not trained yet.\n\n"
            "Run:\n"
            "python src/train_model.py"
        )

    # --------------------------------------------------------
    # Convert input to numbers
    # --------------------------------------------------------

    try:

        cleaned_features = []

        for value in features:

            # Remove * and ** if user accidentally includes them

            value = str(value).replace(
                "*",
                ""
            ).strip()

            cleaned_features.append(
                float(value)
            )

    except ValueError:

        raise ValueError(
            "All transaction values must be numeric."
        )

    # --------------------------------------------------------
    # Convert to numpy array
    # --------------------------------------------------------

    arr = np.asarray(
        cleaned_features,
        dtype=float
    ).reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Check number of features
    # --------------------------------------------------------

    if arr.shape[1] != 30:

        raise ValueError(
            f"Exactly 30 numeric features are required. "
            f"You provided {arr.shape[1]}."
        )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scaled = scaler.transform(
        arr
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(scaled)[0]
    )

    probability = float(
        model.predict_proba(scaled)[0][1]
    )

    return (
        prediction,
        probability
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    model_ready = (
        MODEL_FILE.exists()
        and SCALER_FILE.exists()
    )

    return render_template(
        "home.html",
        model_ready=model_ready
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    form = RegisterForm()

    if form.validate_on_submit():

        email = (
            form.email.data
            .lower()
            .strip()
        )

        existing_user = (
            User.query
            .filter_by(email=email)
            .first()
        )

        if existing_user:

            flash(
                "An account with this email already exists.",
                "warning"
            )

        else:

            user = User(
                email=email,
                password_hash=generate_password_hash(
                    form.password.data
                )
            )

            db.session.add(user)

            db.session.commit()

            flash(
                "Account created. You can now log in.",
                "success"
            )

            return redirect(
                url_for("login")
            )

    return render_template(
        "register.html",
        form=form
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    form = LoginForm()

    if form.validate_on_submit():

        email = (
            form.email.data
            .lower()
            .strip()
        )

        user = (
            User.query
            .filter_by(email=email)
            .first()
        )

        if (
            user
            and check_password_hash(
                user.password_hash,
                form.password.data
            )
        ):

            login_user(user)

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template(
        "login.html",
        form=form
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("home")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
@login_required
@limiter.limit(
    "10 per minute"
)
def predict_page():

    if request.method == "POST":

        try:

            raw = request.form.get(
                "features",
                ""
            )

            # ------------------------------------------------
            # Split comma-separated input
            # ------------------------------------------------

            features = [
                value.strip()
                for value in raw.split(",")
                if value.strip()
            ]

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction, probability = predict(
                features
            )

            # ------------------------------------------------
            # Amount is the LAST feature
            #
            # Feature order:
            #
            # Time
            # V1 ... V28
            # Amount
            # ------------------------------------------------

            amount = float(
                str(features[-1])
                .replace("*", "")
                .strip()
            )

            return render_template(
                "result.html",
                prediction=prediction,
                probability=probability,
                amount=amount
            )

        except ValueError as exc:

            flash(
                str(exc),
                "danger"
            )

        except Exception as exc:

            flash(
                str(exc),
                "danger"
            )

    return render_template(
        "predict.html",
        feature_names=FEATURE_NAMES
    )


# ============================================================
# API PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
@login_required
@limiter.limit(
    "5 per minute"
)
def api_predict():

    data = request.get_json(
        silent=True
    ) or {}

    features = data.get(
        "features"
    )

    if not isinstance(
        features,
        list
    ):

        return jsonify({
            "error":
                "features must be an array of 30 numbers"
        }), 400

    try:

        prediction, probability = predict(
            features
        )

        return jsonify({

            "fraud":
                bool(prediction),

            "prediction":
                prediction,

            "fraud_probability":
                round(
                    probability,
                    6
                ),

            "fraud_percentage":
                round(
                    probability * 100,
                    2
                ),

            "message":
                (
                    "Potential fraud detected"
                    if prediction
                    else
                    "Transaction looks genuine"
                )

        })

    except (
        ValueError,
        TypeError
    ) as exc:

        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# CREATE DATABASE
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FraudGuard Flask Application")
    print("=" * 60)

    print("\nServer:")
    print("http://127.0.0.1:5000")

    print("\nPrediction page:")
    print("http://127.0.0.1:5000/predict")

    print("\nModel:")
    print(
        "READY"
        if MODEL_FILE.exists()
        else
        "NOT TRAINED"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )