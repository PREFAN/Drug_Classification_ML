import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Drug Classification App",
    page_icon="💊",
    layout="centered"
)

# ====================== FILE PATHS ======================
MODEL_FILE = "drug_model.pkl"
DATA_FILE = "drug_classification.csv"

# ====================== LOAD DATASET ======================
df = pd.DataFrame()

if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        df.columns = df.columns.str.strip()
    except Exception:
        df = pd.DataFrame()

# ====================== LOAD MODEL ======================
model = None
le_category = None
le_indication = None

try:
    with open(MODEL_FILE, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    le_category = saved["le_category"]
    le_indication = saved["le_indication"]

except Exception:
    st.info("No model found. Please train or create one first.")
    st.stop()

# ====================== FALLBACK ENCODERS ======================
if le_category is None:
    le_category = LabelEncoder()
    le_category.fit(["Analgesic", "Antibiotic", "Antihistamine", "Antacid"])

if le_indication is None:
    le_indication = LabelEncoder()
    le_indication.fit(["Pain", "Infection", "Allergy", "Heartburn"])

# ====================== SESSION STATE ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ====================== OPTIONS ======================
category_options = list(le_category.classes_)
indication_options = list(le_indication.classes_)

# ====================== TRANSLATIONS ======================
texts = {
    "en": {
        "title": "💊 Drug Classification App",
        "input": "Enter Drug Details",
        "category": "Category",
        "indication": "Indication",
        "strength": "Strength (mg)",
        "predict": "🔮 Predict Drug Type",
        "history": "🕘 Prediction History"
    },
    "fr": {
        "title": "💊 Classification des Médicaments",
        "input": "Entrer les détails",
        "category": "Catégorie",
        "indication": "Indication",
        "strength": "Dosage (mg)",
        "predict": "🔮 Prédire",
        "history": "🕘 Historique"
    },
    "es": {
        "title": "💊 Clasificación de Medicamentos",
        "input": "Ingresar detalles",
        "category": "Categoría",
        "indication": "Indicación",
        "strength": "Dosis (mg)",
        "predict": "🔮 Predecir",
        "history": "🕘 Historial"
    }
}

# ====================== PREDICTION FUNCTION ======================
def predict(category, indication, strength):
    cat_enc = le_category.transform([category])[0]
    ind_enc = le_indication.transform([indication])[0]

    input_df = pd.DataFrame({
        "Category_enc": [cat_enc],
        "Indication_enc": [ind_enc],
        "Strength": [strength]
    })

    pred = model.predict(input_df)[0]
    return "OTC" if pred == 0 else "Prescription"

# ====================== TABS ======================
tab_en, tab_fr, tab_es = st.tabs(["English 🇺🇸", "Français 🇫🇷", "Español 🇪🇸"])

# ====================== ENGLISH ======================
with tab_en:
    t = texts["en"]

    st.title(t["title"])
    st.subheader(t["input"])

    category = st.selectbox(t["category"], category_options, key="en_cat")
    indication = st.selectbox(t["indication"], indication_options, key="en_ind")
    strength = st.number_input(t["strength"], 0.0, 5000.0, 100.0, key="en_str")

    if st.button(t["predict"], key="en_btn"):
        result = predict(category, indication, strength)

        st.success(f"Prediction: {result}")

        st.session_state.history.append({
            "Language": "English",
            "Category": category,
            "Indication": indication,
            "Strength": strength,
            "Prediction": result
        })

# ====================== FRENCH ======================
with tab_fr:
    t = texts["fr"]

    st.title(t["title"])
    st.subheader(t["input"])

    category = st.selectbox(t["category"], category_options, key="fr_cat")
    indication = st.selectbox(t["indication"], indication_options, key="fr_ind")
    strength = st.number_input(t["strength"], 0.0, 5000.0, 100.0, key="fr_str")

    if st.button(t["predict"], key="fr_btn"):
        result = predict(category, indication, strength)

        st.success(f"Résultat: {result}")

        st.session_state.history.append({
            "Language": "French",
            "Category": category,
            "Indication": indication,
            "Strength": strength,
            "Prediction": result
        })

# ====================== SPANISH ======================
with tab_es:
    t = texts["es"]

    st.title(t["title"])
    st.subheader(t["input"])

    category = st.selectbox(t["category"], category_options, key="es_cat")
    indication = st.selectbox(t["indication"], indication_options, key="es_ind")
    strength = st.number_input(t["strength"], 0.0, 5000.0, 100.0, key="es_str")

    if st.button(t["predict"], key="es_btn"):
        result = predict(category, indication, strength)

        st.success(f"Resultado: {result}")

        st.session_state.history.append({
            "Language": "Spanish",
            "Category": category,
            "Indication": indication,
            "Strength": strength,
            "Prediction": result
        })

# ====================== HISTORY (FIXED & STRUCTURED) ======================
st.markdown("---")
st.subheader("🕘 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, use_container_width=True)

    st.download_button(
        "📥 Download History",
        history_df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions yet. Start predicting above 👆")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("💊 About this App")

    st.write("""
    Drug Classification App Using Machine Learning algorithm to predict OTC vs Prescription drugs.
    """)

    st.info("""
    **Created by Francis Darko**  
    BSc. Chemistry & MS. Data Science
    """)