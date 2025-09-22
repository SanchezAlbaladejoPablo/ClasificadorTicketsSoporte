
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# Rutas relativas
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "..", "data")
models_dir = os.path.join(script_dir, "..", "models")
sample_tickets_path = os.path.join(data_dir, "sample_tickets.csv")
ticket_classifier_path = os.path.join(models_dir, "ticket_classifier.pkl")

# Cargar el dataset preprocesado
try:
    df = pd.read_csv(sample_tickets_path)
except FileNotFoundError:
    print(f"Error: {sample_tickets_path} no encontrado. Asegúrate de que el archivo esté en la ruta correcta.")
    exit()

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df["cleaned_description"]
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorización de texto con TF-IDF
vectorizer = TfidfVectorizer(max_features=5000) # Limitar a 5000 características para eficiencia
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test_tfidf)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1_score": f1_score(y_test, y_pred, average="weighted")
}

print("Métricas del modelo:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Guardar el modelo entrenado y el vectorizador
with open(ticket_classifier_path, "wb") as f:
    pickle.dump({"model": model, "vectorizer": vectorizer, "metrics": metrics}, f)

print(f"Modelo y vectorizador guardados en {ticket_classifier_path}")


