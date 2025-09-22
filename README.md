![Portada](docs/portada.png)

## 🚀 Características

- **Clasificación automática** de tickets en tres categorías: Login, Billing y Technical
- **Backend API** con FastAPI para servir el modelo de ML
- **Frontend interactivo** con Streamlit para clasificar tickets y visualizar datos
- **Modelo de ML** entrenado con Regresión Logística y vectorización TF-IDF
- **Dataset real** de tickets de soporte procesado y balanceado

## 📁 Estructura del Proyecto

```
ClasificadorTicketsSoporteUpdated/
├── app/
│   └── streamlit_app.py          # Aplicación frontend con Streamlit
├── backend/
│   └── main.py                   # API backend con FastAPI
├── data/
│   ├── all_tickets.csv           # Dataset completo descargado
│   └── sample_tickets.csv        # Muestra procesada y balanceada
├── models/
│   └── ticket_classifier.pkl     # Modelo entrenado y vectorizador
├── notebooks/
│   ├── preprocess_data.py        # Script de preprocesamiento
│   └── training.py               # Script de entrenamiento del modelo
│   └── training.ipynb            # Notebook de Jupyter para documentar el entrenamiento
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este archivo
```

## 🛠️ Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   # Si ya tienes el proyecto, navega a su directorio
   cd ClasificadorTicketsSoporteUpdated
   ```

2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Descargar recursos de NLTK** (se hace automáticamente al ejecutar los scripts)

## 📊 Preparación de Datos y Entrenamiento

### 1. Preprocesar los datos
```bash
cd notebooks
python preprocess_data.py
```

Este script:
- Carga el dataset de tickets de soporte (`all_tickets.csv`)
- Limpia y procesa el texto (eliminación de stopwords, lemmatización)
- Categoriza automáticamente los tickets en Login, Billing, Technical
- Balancea las categorías y guarda una muestra en `data/sample_tickets.csv`

### 2. Entrenar el modelo
```bash
cd notebooks
python training.py
```

Este script:
- Entrena un modelo de Regresión Logística con vectorización TF-IDF
- Evalúa el modelo y muestra métricas (accuracy, precision, recall, f1-score)
- Guarda el modelo entrenado en `models/ticket_classifier.pkl`

## 🚀 Ejecución

### 1. Iniciar el backend (FastAPI)
En una terminal:
```bash
cd backend
python main.py
```

El backend estará disponible en: `http://localhost:8000`

**Endpoints disponibles:**
- `GET /`: Mensaje de bienvenida
- `POST /classify/`: Clasificar un ticket (envía JSON con campo "text")
- `GET /metrics/`: Obtener métricas del modelo

### 2. Iniciar el frontend (Streamlit)
En **otra** terminal:
```bash
cd app
streamlit run streamlit_app.py
```

La aplicación estará disponible en: `http://localhost:8501`

## 📈 Métricas del Modelo

El modelo actual alcanza las siguientes métricas en el conjunto de prueba:
- **Accuracy**: ~79%
- **Precision**: ~83%
- **Recall**: ~79%
- **F1-Score**: ~79%

## 🎯 Uso de la Aplicación

### Frontend (Streamlit)
1. Ingresa el texto del ticket en el área de texto
2. Haz clic en "Clasificar Ticket"
3. Observa la categoría predicha y la probabilidad
4. Revisa las métricas del modelo en la barra lateral
5. Explora las visualizaciones del dataset

### API (FastAPI)
```bash
# Clasificar un ticket
curl -X POST "http://localhost:8000/classify/" \
     -H "Content-Type: application/json" \
     -d '{"text": "No puedo acceder a mi cuenta"}'

# Obtener métricas
curl -X GET "http://localhost:8000/metrics/"
```

## 📋 Dependencias Principales

- **pandas**: Manipulación de datos
- **scikit-learn**: Machine Learning
- **nltk**: Procesamiento de lenguaje natural
- **fastapi**: Framework web para la API
- **uvicorn**: Servidor ASGI para FastAPI
- **streamlit**: Framework para el frontend

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 🐛 Solución de Problemas

### Error: "Backend no disponible"
- Asegúrate de que FastAPI esté ejecutándose en `http://localhost:8000`
- Verifica que no haya conflictos de puertos

### Error: "Dataset no encontrado"
- Asegúrate de que el archivo `all_tickets.csv` esté en la carpeta `data/`
- Ejecuta `cd notebooks` y luego `python preprocess_data.py` para generar `sample_tickets.csv`

### Error: "Modelo no encontrado"
- Ejecuta `cd notebooks` y luego `python preprocess_data.py` para generar `sample_tickets.csv`
- Luego, ejecuta `cd notebooks` y `python training.py` para entrenar y guardar el modelo
- Verifica que el archivo `ticket_classifier.pkl` esté en la carpeta `models/`
