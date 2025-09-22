
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Descargar recursos de NLTK si no están ya descargados
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

# Inicializar lemmatizer y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower() # Convertir a string y minúsculas
    text = re.sub(r"[^a-z\s]", "", text) # Eliminar caracteres no alfabéticos
    text = re.sub(r"\s+", " ", text).strip() # Eliminar espacios extra
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def categorize_ticket(text):
    text = str(text).lower()
    if re.search(r"login|account|password|username|authentication|access", text):
        return "Login"
    elif re.search(r"bill|payment|invoice|charge|refund|price|cost|transaction", text):
        return "Billing"
    elif re.search(r"technical|error|bug|issue|crash|software|hardware|system|network", text):
        return "Technical"
    else:
        return "Other"

# Rutas relativas
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "..", "data")
all_tickets_path = os.path.join(data_dir, "all_tickets.csv")
sample_tickets_path = os.path.join(data_dir, "sample_tickets.csv")

# Cargar el dataset
try:
    df = pd.read_csv(all_tickets_path)
except FileNotFoundError:
    print(f"Error: {all_tickets_path} no encontrado. Asegúrate de que el archivo esté en la ruta correcta.")
    exit()

# Para este dataset específico, la columna relevante es 'body'
# y crearemos una nueva columna 'category' basada en el contenido del texto.

if "body" not in df.columns:
    print("Error: La columna 'body' no se encuentra en el dataset. Por favor, verifica el nombre de la columna que contiene el texto del ticket.")
    exit()

df["cleaned_description"] = df["body"].apply(clean_text)
df["category"] = df["body"].apply(categorize_ticket)

# Filtrar para quedarse solo con las categorías deseadas y balancear
df_filtered = df[df["category"].isin(["Login", "Billing", "Technical"])]

# Balancear las categorías
min_samples = df_filtered["category"].value_counts().min()
df_balanced = df_filtered.groupby("category").apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Guardar una muestra representativa
df_sample = df_balanced.sample(n=min(500, len(df_balanced)), random_state=42) # Mínimo 500 o el tamaño del df_balanced
df_sample.to_csv(sample_tickets_path, index=False)

print("Preprocesamiento completado y sample_tickets.csv guardado.")
print("Categorías y conteo en sample_tickets.csv:\n" + df_sample["category"].value_counts().to_string())


