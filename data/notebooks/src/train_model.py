import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Logging einrichten
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dateipfade
DATA_PATH = os.path.join("data", "kreditrisiko.csv")
MODEL_PATH = os.path.join("data", "notebooks", "src", "webapp", "best_model.pkl")
SCALER_PATH = os.path.join("data", "notebooks", "src", "webapp", "scaler.pkl")

def load_data():
    """Lädt die Kreditrisiko-Daten als Pandas DataFrame."""
    if not os.path.exists(DATA_PATH):
        logging.error(f"❌ Datei nicht gefunden: {DATA_PATH}")
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {DATA_PATH}")
    
    logging.info("📥 Daten werden geladen...")
    df = pd.read_csv(DATA_PATH)
    logging.info("✅ Daten erfolgreich geladen!")
    return df

def preprocess_data(df):
    """Normalisiert die Features und trennt sie von der Zielvariable."""
    features = ['Kreditbetrag', 'Laufzeit', 'Einkommen', 'Alter', 'Kreditverlauf', 'Schuldenquote']
    target = 'Kreditwürdigkeit'

    X = df[features]
    y = df[target]

    logging.info("📊 Daten werden normalisiert...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X_train, y_train):
    """Trainiert ein RandomForest-Modell und gibt es zurück."""
    logging.info("🧠 Modell wird trainiert...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("✅ Modell erfolgreich trainiert!")
    return model

def save_model(model, scaler):
    """Speichert das Modell und den Scaler als .pkl-Dateien."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"💾 Modell gespeichert unter: {MODEL_PATH}")
    logging.info(f"💾 Scaler gespeichert unter: {SCALER_PATH}")

def main():
    """Hauptfunktion zum Laden, Trainieren und Speichern des Modells."""
    df = load_data()
    X, y, scaler = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    save_model(model, scaler)
    
    logging.info("🎉 Training abgeschlossen!")

if __name__ == "__main__":
    main()
