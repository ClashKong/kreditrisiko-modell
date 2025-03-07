import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.preprocessing import StandardScaler

# Lade das trainierte Modell
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Verbindung zur SQLite-Datenbank herstellen
def load_data():
    conn = sqlite3.connect("data/kreditrisiko.db")
    df = pd.read_sql("SELECT * FROM kreditdaten", conn)
    conn.close()
    return df

# Streamlit UI
def main():
    st.title("Kreditrisiko-Bewertung")
    st.write("Gib die Kundendaten ein, um die Kreditwürdigkeit zu bewerten.")
    
    # Eingabefelder
    kreditbetrag = st.number_input("Kreditbetrag", min_value=1000, max_value=50000, value=10000)
    laufzeit = st.number_input("Laufzeit (Monate)", min_value=6, max_value=72, value=24)
    einkommen = st.number_input("Einkommen", min_value=20000, max_value=100000, value=50000)
    alter = st.number_input("Alter", min_value=18, max_value=70, value=35)
    kreditverlauf = st.number_input("Kreditverlauf (Anzahl vergangener Kredite)", min_value=0, max_value=10, value=2)
    schuldenquote = st.slider("Schuldenquote", min_value=0.0, max_value=1.0, value=0.3)
    
    # Vorhersage-Button
    if st.button("Bewerten"):
        # Eingabedaten transformieren
        input_data = np.array([[kreditbetrag, laufzeit, einkommen, alter, kreditverlauf, schuldenquote]])
        input_data_scaled = scaler.transform(input_data)
        
        # Vorhersage
        prediction = model.predict(input_data_scaled)[0]
        
        if prediction == 1:
            st.success("✅ Der Kunde ist kreditwürdig!")
        else:
            st.error("❌ Der Kunde ist nicht kreditwürdig!")

if __name__ == "__main__":
    main()
