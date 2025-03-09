import pandas as pd
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def load_model():
    """Lädt das trainierte Modell und den Scaler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logging.error("❌ Modell oder Scaler nicht gefunden!")
        raise FileNotFoundError("❌ Modell oder Scaler nicht gefunden!")
    
    logging.info("📦 Modell & Scaler werden geladen...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("✅ Modell & Scaler erfolgreich geladen!")
    return model, scaler

def preprocess_data(df):
    """Normalisiert die Features und trennt sie von der Zielvariable."""
    features = ['Kreditbetrag', 'Laufzeit', 'Einkommen', 'Alter', 'Kreditverlauf', 'Schuldenquote']
    target = 'Kreditwürdigkeit'

    X = df[features]
    y = df[target]

    logging.info("📊 Daten werden normalisiert...")
    return X, y

def evaluate_model(model, scaler, X, y):
    """Berechnet Metriken zur Modellbewertung und gibt sie aus."""
    logging.info("📊 Modell wird evaluiert...")

    # Skalierung anwenden
    X_scaled = scaler.transform(X)
    
    # Vorhersagen
    y_pred = model.predict(X_scaled)

    # Berechnung der Metriken
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)

    # Ergebnisse ausgeben
    logging.info("📊 Modell-Performance:")
    print("✅ Genauigkeit:", round(accuracy, 4))
    print("✅ Präzision:", round(precision, 4))
    print("✅ Recall:", round(recall, 4))
    print("✅ F1-Score:", round(f1, 4))
    print("✅ ROC-AUC:", round(roc_auc, 4))

    return model.feature_importances_, X.columns

def plot_feature_importance(feature_importance, feature_cols):
    """Erstellt ein Barplot zur Visualisierung der Feature-Importanz."""
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance, y=feature_cols, palette="viridis")
    plt.title("Feature-Importance des Modells")
    plt.xlabel("Feature Importance")
    plt.ylabel("")
    plt.show()

def main():
    """Hauptfunktion zum Laden, Evaluieren und Visualisieren des Modells."""
    df = load_data()
    model, scaler = load_model()
    X, y = preprocess_data(df)
    
    feature_importance, feature_cols = evaluate_model(model, scaler, X, y)
    plot_feature_importance(feature_importance, feature_cols)

    logging.info("🎉 Evaluierung abgeschlossen!")

if __name__ == "__main__":
    main()
