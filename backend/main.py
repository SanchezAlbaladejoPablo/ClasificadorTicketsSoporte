from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Rutas relativas
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "..", "models")
ticket_classifier_path = os.path.join(models_dir, "ticket_classifier.pkl")

# Cargar el modelo entrenado
try:
    with open(ticket_classifier_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    metrics = model_data["metrics"]
except FileNotFoundError:
    print(f"Error: {ticket_classifier_path} no encontrado. Asegúrate de que el modelo esté entrenado.")
    exit()

app = FastAPI(title="Ticket Classifier API", version="1.0.0")

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TicketRequest(BaseModel):
    text: str

class TicketResponse(BaseModel):
    category: str
    probability: float

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float

@app.get("/")
def read_root():
    return {"message": "Ticket Classifier API is running"}

@app.post("/classify/", response_model=TicketResponse)
def classify_ticket(request: TicketRequest):
    # Vectorizar el texto de entrada
    text_tfidf = vectorizer.transform([request.text])
    
    # Predecir la categoría
    prediction = model.predict(text_tfidf)[0]
    
    # Obtener las probabilidades de cada clase
    probabilities = model.predict_proba(text_tfidf)[0]
    max_probability = np.max(probabilities)
    
    return TicketResponse(category=prediction, probability=float(max_probability))

@app.get("/metrics/", response_model=MetricsResponse)
def get_metrics():
    return MetricsResponse(
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


