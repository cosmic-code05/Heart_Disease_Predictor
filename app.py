import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CardioSense",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #172554);
}

.title {
    text-align: center;
    color: white;
    font-size: 48px;
    font-weight: 800;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 35px;
}

.section {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">❤️ CardioSense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Heart Disease Risk Prediction</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section"><h3>Patient Information</h3></div>',
    unsafe_allow_html=True
)


# =========================================================
# ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    age = st.slider(
        "Age",
        min_value=20,
        max_value=100,
        value=50
    )

with col2:

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

with col3:

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        }[x]
    )


# =========================================================
# ROW 2
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=220,
        value=120
    )

with col2:

    chol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

with col3:

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        format_func=lambda x:
        "<= 120 mg/dL" if x == 0 else "> 120 mg/dL"
    )


# =========================================================
# ROW 3
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x]
    )

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

with col3:

    exang = st.selectbox(
        "Exercise-Induced Angina",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


# =========================================================
# ROW 4
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )

with col2:

    slope = st.selectbox(
        "ST Slope",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )

with col3:

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4]
    )


# =========================================================
# ROW 5
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect",
            3: "Other"
        }[x]
    )


# =========================================================
# BUTTON
# =========================================================

st.write("")

predict = st.button(
    "🔍 Analyse Heart Disease Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # -----------------------------------------------------
    # START WITH ALL TRAINING FEATURES AS ZERO
    # FIX: use 0.0 (float) instead of 0 (int) so the whole
    # DataFrame is float64 from the start. Otherwise the
    # column dtype defaults to int64 and assigning a float
    # value like oldpeak=1.5 either raises an error or gets
    # silently truncated to 1, depending on your pandas
    # version.
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        0.0,
        index=[0],
        columns=columns
    )


    # -----------------------------------------------------
    # NUMERIC FEATURES
    # -----------------------------------------------------

    numeric_values = {

        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal

    }


    for feature, value in numeric_values.items():

        if feature in input_data.columns:

            input_data.at[0, feature] = value


    # -----------------------------------------------------
    # SEX DUMMY
    # -----------------------------------------------------

    if "sex_label_Male" in input_data.columns:

        input_data.at[0, "sex_label_Male"] = int(
            sex == 1
        )


    # -----------------------------------------------------
    # CHEST PAIN DUMMIES
    # -----------------------------------------------------

    if "cp_name_Atypical Angina" in input_data.columns:

        input_data.at[0, "cp_name_Atypical Angina"] = int(
            cp == 1
        )


    if "cp_name_Non-Anginal Pain" in input_data.columns:

        input_data.at[0, "cp_name_Non-Anginal Pain"] = int(
            cp == 2
        )


    if "cp_name_Typical Angina" in input_data.columns:

        input_data.at[0, "cp_name_Typical Angina"] = int(
            cp == 0
        )


    if "cp_name_Asymptomatic" in input_data.columns:

        input_data.at[0, "cp_name_Asymptomatic"] = int(
            cp == 3
        )


    # -----------------------------------------------------
    # FBS DUMMY
    # -----------------------------------------------------

    if "fbs_name_>120 mg/dl" in input_data.columns:

        input_data.at[0, "fbs_name_>120 mg/dl"] = int(
            fbs == 1
        )


    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        input_scaled
    )[0]


    # -----------------------------------------------------
    # RESULT
    # FIX: in this dataset (heart.csv), target=1 means
    # "no disease" and target=0 means "disease present" -
    # the opposite of the usual convention. Verified this
    # directly against the training data's feature
    # correlations (higher age/oldpeak/ca -> target=0,
    # higher max heart rate -> target=1). So the branches
    # below are swapped from the original code.
    # -----------------------------------------------------

    if prediction == 0:

        st.markdown("""
        <div class="result"
        style="background:rgba(239,68,68,0.20);
        color:#fca5a5;">

        ⚠️ Higher Risk Detected

        </div>
        """, unsafe_allow_html=True)

        st.info(
            "The model predicts a higher likelihood "
            "of heart disease based on the provided parameters."
        )

    else:

        st.markdown("""
        <div class="result"
        style="background:rgba(34,197,94,0.20);
        color:#86efac;">

        ✅ Lower Risk Detected

        </div>
        """, unsafe_allow_html=True)

        st.success(
            "The model predicts a lower likelihood "
            "of heart disease based on the provided parameters."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br><br>

<p style="
text-align:center;
color:#94a3b8;
font-size:13px;
">

CardioSense • KNN Machine Learning Project

<br>

For educational purposes only — not a medical diagnosis.

</p>
""", unsafe_allow_html=True)