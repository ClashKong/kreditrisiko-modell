import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 1. Beispiel-CSV-Datei mit neuen Features generieren
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
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/kreditrisiko.csv', index=False)
    return df

# 2. Daten laden & aufbereiten
def load_and_preprocess_data():
    df = pd.read_csv('data/kreditrisiko.csv')
    
    # Features auswählen
    X = df[['Kreditbetrag', 'Laufzeit', 'Einkommen', 'Alter', 'Kreditverlauf', 'Schuldenquote']]
    y = df['Kreditwürdigkeit']
    
    # Outlier entfernen (z. B. unrealistische Werte)
    X = X[(X['Schuldenquote'] >= 0) & (X['Schuldenquote'] <= 1)]
    
    # Daten skalieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
