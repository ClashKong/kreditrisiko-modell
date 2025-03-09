import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- Pfade definieren ---
MODEL_PATH = os.path.join("data", "notebooks", "src", "webapp", "best_model.pkl")
SCALER_PATH = os.path.join("data", "notebooks", "src", "webapp", "scaler.pkl")

# --- Modell und Scaler laden ---
@st.cache_resource
def load_model():
    """Lädt das Modell und den Scaler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("❌ Modell oder Scaler nicht gefunden! Stelle sicher, dass sie vorhanden sind.")
        return None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Wähle eine Seite:", ["Kreditbewertung", "Datenanalyse", "Modellperformance"])

# --- Seite: Kreditbewertung ---
if page == "Kreditbewertung":
    st.title("🏦 Kreditrisiko-Bewertung")
    st.write("Gib die Kundendaten ein, um die Kreditwürdigkeit zu bewerten.")

    # Eingabefelder für den Nutzer
    kreditbetrag = st.number_input("Kreditbetrag", min_value=1000, max_value=100000, value=10000, step=500)
    laufzeit = st.number_input("Laufzeit (Monate)", min_value=6, max_value=120, value=24, step=1)
    einkommen = st.number_input("Einkommen", min_value=10000, max_value=500000, value=50000, step=5000)
    alter = st.number_input("Alter", min_value=18, max_value=100, value=35, step=1)
    kreditverlauf = st.number_input("Kreditverlauf (Anzahl vergangener Kredite)", min_value=0, max_value=10, value=2, step=1)
    schuldenquote = st.slider("Schuldenquote", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

    # Berechnung der Vorhersage
    if st.button("🔍 Bewerten"):
        if model is None or scaler is None:
            st.error("❌ Das Modell konnte nicht geladen werden. Überprüfe die Dateien.")
        else:
            input_data = np.array([[kreditbetrag, laufzeit, einkommen, alter, kreditverlauf, schuldenquote]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            if prediction[0] == 1:
                st.success("✅ Der Kredit wird voraussichtlich genehmigt!")
            else:
                st.warning("⚠️ Der Kredit wird voraussichtlich abgelehnt!")

# --- Seite: Datenanalyse ---
elif page == "Datenanalyse":
    st.title("📊 Datenanalyse")
    st.write("Hier kannst du eine Beispiel-Kreditrisiko-Datenbank anschauen.")

    data_path = os.path.join("data", "kreditrisiko.csv")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.dataframe(df)
    else:
        st.error("❌ Datei `kreditrisiko.csv` nicht gefunden!")

# --- Seite: Modellperformance ---
elif page == "Modellperformance":
    st.title("📈 Modellperformance")
    st.write("Hier siehst du die Feature-Wichtigkeit deines Modells.")

    feature_importance_path = os.path.join("data", "notebooks", "src", "webapp", "best_model.pkl")

    if os.path.exists(feature_importance_path):
        model = joblib.load(feature_importance_path)
        feature_importance = model.feature_importances_
        feature_cols = ["Kreditbetrag", "Laufzeit", "Einkommen", "Alter", "Kreditverlauf", "Schuldenquote"]

        st.bar_chart(pd.DataFrame({"Feature Importance": feature_importance}, index=feature_cols))
    else:
        st.error("❌ Modell konnte nicht geladen werden!")
