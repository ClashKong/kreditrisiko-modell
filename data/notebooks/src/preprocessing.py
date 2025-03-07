import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
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
    
    # Outlier entfernen
    X = X[(X['Schuldenquote'] >= 0) & (X['Schuldenquote'] <= 1)]
    
    # Daten skalieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), X.columns

# 3. Modell trainieren (verschiedene Modelle testen)
def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"{name} Genauigkeit: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # ROC-Kurve berechnen
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f}')
        
        if acc > best_accuracy:
            best_model = model
            best_accuracy = acc
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Kurve')
    plt.legend()
    plt.savefig("plots/roc_curve.png")
    plt.show()
    
    # Feature Importances für RandomForest oder XGBoost
    if isinstance(best_model, (RandomForestClassifier, XGBClassifier)):
        importances = best_model.feature_importances_
        plt.figure(figsize=(8, 6))
        plt.barh(feature_names, importances)
        plt.xlabel("Wichtigkeit")
        plt.ylabel("Features")
        plt.title("Feature Importance des besten Modells")
        plt.savefig("plots/feature_importance.png")
        plt.show()
    
    return best_model

if __name__ == "__main__":
    print("Generiere Beispiel-Daten & speichere in SQLite-DB...")
    generate_sample_data()
    
    print("Lade & verarbeite Daten aus SQLite-DB...")
    (X_train, X_test, y_train, y_test), feature_names = load_and_preprocess_data()
    
    print("Trainiere & vergleiche Modelle mit Visualisierung...")
    best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)
    
    print("Modelltraining & Analyse abgeschlossen!")
