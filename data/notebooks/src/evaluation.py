from sklearn.metrics import accuracy_score, classification_report

# Modell evaluieren
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Genauigkeit:", accuracy_score(y_test, y_pred))
    print("Bericht:")
    print(classification_report(y_test, y_pred))
