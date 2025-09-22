import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Tickets de Soporte",
    page_icon="🎫",
    layout="wide"
)

# URL del backend FastAPI
BACKEND_URL = "http://localhost:8000"

# Título principal
st.title("🎫 Clasificador de Tickets de Soporte")
st.markdown("Esta aplicación clasifica tickets de soporte en las categorías: **Login**, **Billing** y **Technical**")

# Sidebar con métricas del modelo
st.sidebar.header("📊 Métricas del Modelo")

try:
    response = requests.get(f"{BACKEND_URL}/metrics/")
    if response.status_code == 200:
        metrics = response.json()
        st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.sidebar.metric("Precision", f"{metrics['precision']:.3f}")
        st.sidebar.metric("Recall", f"{metrics['recall']:.3f}")
        st.sidebar.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    else:
        st.sidebar.error("Error al obtener métricas del modelo")
except requests.exceptions.RequestException:
    st.sidebar.error("Backend no disponible. Asegúrate de que FastAPI esté ejecutándose.")

# Sección principal para clasificación
st.header("Clasificar Ticket")

# Input de texto
ticket_text = st.text_area(
    "Ingresa el texto del ticket:",
    placeholder="Ejemplo: No puedo acceder a mi cuenta, olvidé mi contraseña...",
    height=150
)

# Botón para clasificar
if st.button("Clasificar Ticket", type="primary"):
    if ticket_text.strip():
        try:
            # Enviar solicitud al backend
            response = requests.post(
                f"{BACKEND_URL}/classify/",
                json={"text": ticket_text}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Mostrar resultado
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Categoría:** {result['category']}")
                
                with col2:
                    st.info(f"**Probabilidad:** {result['probability']:.3f}")
                
                # Mostrar barra de progreso para la probabilidad
                st.progress(result['probability'])
                
            else:
                st.error("Error al clasificar el ticket")
                
        except requests.exceptions.RequestException:
            st.error("Error de conexión. Asegúrate de que el backend esté ejecutándose.")
    else:
        st.warning("Por favor, ingresa el texto del ticket")

# Sección de visualización del dataset
st.header("📈 Análisis del Dataset")

try:
    # Cargar el dataset de muestra
    df = pd.read_csv("../data/sample_tickets.csv")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Categorías")
        category_counts = df['category'].value_counts()
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(8, 6))
        category_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Distribución de Categorías en el Dataset')
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Número de Tickets')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Gráfico de Pastel")
        
        # Crear gráfico de pastel
        fig, ax = plt.subplots(figsize=(8, 6))
        category_counts.plot(
            kind='pie', 
            ax=ax, 
            autopct='%1.1f%%',
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            startangle=90
        )
        ax.set_title('Distribución Porcentual de Categorías')
        ax.set_ylabel('')  # Remover etiqueta del eje y
        st.pyplot(fig)
    
    # Mostrar estadísticas del dataset
    st.subheader("Estadísticas del Dataset")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Tickets", len(df))
    
    with col2:
        st.metric("Número de Categorías", df['category'].nunique())
    
    with col3:
        st.metric("Promedio de Palabras", int(df['cleaned_description'].str.split().str.len().mean()))
    
    # Mostrar muestra del dataset
    st.subheader("Muestra del Dataset")
    st.dataframe(df[['body', 'category']].head(10), use_container_width=True)
    
except FileNotFoundError:
    st.error("Dataset no encontrado. Asegúrate de que sample_tickets.csv esté disponible.")
except Exception as e:
    st.error(f"Error al cargar el dataset: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Desarrollado con ❤️ usando Streamlit y FastAPI")

