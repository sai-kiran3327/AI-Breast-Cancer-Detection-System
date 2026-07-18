import pickle
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from database import db, User

# ==========================
# Flask Configuration
# ==========================

app = Flask(__name__)

app.secret_key = "SaiKiran_MCA_2026"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///breast_cancer.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

# ==========================
# Load ML Model
# ==========================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feature Order
features_name = [
    "clump_thickness",
    "uniform_cell_size",
    "uniform_cell_shape",
    "marginal_adhesion",
    "single_epithelial_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses"
]

# ==========================
# Home
# ==========================

@app.route("/")
def home():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))

# ==========================
# Dashboard
# ==========================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])

    return render_template(
        "dashboard.html",
        user=user
    )

# ==========================
# Register
# ==========================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        confirm = request.form["confirm"]

        if not name or not email or not password or not confirm:

            return render_template(
                "register.html",
                error="All fields are required."
            )

        if password != confirm:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        existing = User.query.filter_by(email=email).first()

        if existing:

            return render_template(
                "register.html",
                error="Email already exists."
            )

        hashed = generate_password_hash(password)

        new_user = User(
            name=name,
            email=email,
            password=hashed
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

# ==========================
# Login
# ==========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form["email"].strip()
        password = request.form["password"]

        if not email or not password:

            return render_template(
                "login.html",
                error="Please enter Email and Password."
            )

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session["user_id"] = user.id
            session["user_name"] = user.name

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid Email or Password."
        )

    return render_template("login.html")


# ==========================
# Prediction Page
# ==========================

@app.route("/prediction")
def prediction():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("index1.html")


# ==========================
# Predict
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    if "user_id" not in session:
        return redirect(url_for("login"))

    try:

        values = []

        for value in request.form.values():
            values.append(float(value))

        df = pd.DataFrame([values], columns=features_name)

        scaled = scaler.transform(df)

        prediction = model.predict(scaled)
        probability = model.predict_proba(scaled)

        print("Prediction :", prediction)
        print("Probability:", probability)

        if prediction[0] == 0:
            result = "Benign (No Breast Cancer Detected)"
        else:
            result = "Malignant (Breast Cancer Detected)"

        return render_template(
            "result.html",
            prediction_text=result
        )

    except Exception as e:

        print("Prediction Error:", e)

        return render_template(
            "result.html",
            prediction_text="Error : " + str(e)
        )


# ==========================
# Logout
# ==========================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

# ==========================
# Run Flask App
# ==========================

if __name__ == "__main__":

    print("\n===================================")
    print(" AI Breast Cancer Detection System ")
    print(" Flask Server Started Successfully ")
    print("===================================\n")

    app.run(debug=True)