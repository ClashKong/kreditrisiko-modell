import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# Lade das trainierte Modell
model = joblib.load("webapp/best_model.pkl")
scaler = joblib.load("webapp/scaler.pkl")

# Verbindung zur SQLite-Datenbank herstellen
def load_data():
    conn = sqlite3.connect("data/kreditrisiko.db")
    df = pd.read_sql("SELECT * FROM kreditdaten", conn)
    conn.close()
    return df

# ROC-Kurve zeichnen
def plot_roc_curve():
    df = load_data()
    X = df[['Kreditbetrag', 'Laufzeit', 'Einkommen', 'Alter', 'Kreditverlauf', 'Schuldenquote']]
    y = df['Kreditwürdigkeit']
    X_scaled = scaler.transform(X)
    
    y_prob = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Kurve des Modells")
    plt.legend()
    st.pyplot(plt)

# Feature-Importance Diagramm
def plot_feature_importance():
    feature_names = ['Kreditbetrag', 'Laufzeit', 'Einkommen', 'Alter', 'Kreditverlauf', 'Schuldenquote']
    importances = model.feature_importances_
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=feature_names, orient='h')
    plt.xlabel("Wichtigkeit")
    plt.ylabel("Feature")
    plt.title("Feature Importance des Modells")
    st.pyplot(plt)

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
    
    # Zusätzliche Analyse
    st.subheader("📊 Modell-Performance: ROC-Kurve")
    plot_roc_curve()
    
    st.subheader("📊 Feature Importance")
    plot_feature_importance()
    
    # Datenanalyse
    st.subheader("📊 Kreditdaten erkunden")
    df = load_data()
    st.dataframe(df.head())
    
    st.subheader("📊 Verteilungen der Merkmale")
    selected_feature = st.selectbox("Wähle ein Feature", ['Kreditbetrag', 'Laufzeit', 'Einkommen', 'Alter', 'Kreditverlauf', 'Schuldenquote'])
    plt.figure(figsize=(6, 4))
    sns.histplot(df[selected_feature], bins=30, kde=True)
    plt.title(f"Verteilung von {selected_feature}")
    st.pyplot(plt)
    
if __name__ == "__main__":
    main()
