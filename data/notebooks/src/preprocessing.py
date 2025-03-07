import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 1. SQLite-Datenbank erstellen & Daten speichern
def generate_sample_data():
    np.random.seed(42)
    data = {
        'Kreditbetrag': np.random.randint(1000, 50000, 1000),
        'Laufzeit': np.random.randint(6, 72, 1000),
        'Einkommen': np.random.randint(20000, 100000, 1000),
        'Alter': np.random.randint(18, 70, 1000),
        'Kreditverlauf': np.random.randint(0, 10, 1000),
        'Schuldenquote': np.random.uniform(0, 1, 1000),
        'Kreditwürdigkeit': np.random.choice([0, 1], 1000, p=[0.3, 0.7])  # 0 = schlecht, 1 = gut
    }
    df = pd.DataFrame(data)
    
    # Verbindung zur SQLite-Datenbank herstellen
    conn = sqlite3.connect('data/kreditrisiko.db')
    df.to_sql('kreditdaten', conn, if_exists='replace', index=False)
    conn.close()
    return df

# 2. Daten aus der SQL-Datenbank laden & aufbereiten
def load_and_preprocess_data():
    conn = sqlite3.connect('data/kreditrisiko.db')
    df = pd.read_sql('SELECT * FROM kreditdaten', conn)
    conn.close()
    
    # Features auswählen
    X = df[['Kreditbetrag', 'Laufzeit', 'Einkommen', 'Alter', 'Kreditverlauf', 'Schuldenquote']]
    y = df['Kreditwürdigkeit']
    
    # Outlier entfernen (z. B. unrealistische Werte)
    X = X[(X['Schuldenquote'] >= 0) & (X['Schuldenquote'] <= 1)]
    
    # Daten skalieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Modell trainieren (Unverändert)
def train_model(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. Evaluierung (Unverändert)
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    print("Genauigkeit:", accuracy_score(y_test, y_pred))
    print("Bericht:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("Generiere Beispiel-Daten & speichere in SQLite-DB...")
    generate_sample_data()
    
    print("Lade & verarbeite Daten aus SQLite-DB...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Trainiere Modell...")
    model = train_model(X_train, X_test, y_train, y_test)
    
    print("Bewerte Modell...")
    evaluate_model(model, X_test, y_test)
    
    print("Modelltraining abgeschlossen!")