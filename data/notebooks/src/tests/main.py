from src.preprocessing import generate_sample_data, load_and_preprocess_data
from src.model import train_model
from src.evaluation import evaluate_model

if __name__ == "__main__":
    print("Generiere Beispiel-Daten...")
    generate_sample_data()
    
    print("Lade und verarbeite Daten...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Trainiere Modell...")
    model = train_model(X_train, y_train)
    
    print("Bewerte Modell...")
    evaluate_model(model, X_test, y_test)
    
    print("Modelltraining abgeschlossen!")
