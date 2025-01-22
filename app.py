import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Cargar el modelo preentrenado para análisis semántico
modelo = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Título de la aplicación
st.title("Análisis Semántico de Requerimientos")
st.write("Sube un archivo Excel con tus requerimientos y analiza solicitudes similares basándote en el contexto.")

# Subir el archivo Excel
archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

if archivo:
    # Leer el archivo Excel
    df = pd.read_excel(archivo)

    # Verificar si la columna de observaciones existe
    if 'Observaciones' not in df.columns:
        st.error("El archivo debe contener una columna llamada 'Observaciones'.")
    else:
        # Verificar si la columna de territorio está en la columna 'J' (índice 9)
        if df.columns[9] != 'Territorio':
            st.error("El archivo debe tener la columna 'Territorio' en la columna 'J'.")
        else:
            # Preprocesar las observaciones
            df['Observaciones_procesadas'] = df['Observaciones'].fillna('').str.strip()
            observaciones = df['Observaciones_procesadas'].tolist()

            # Generar embeddings para las observaciones
            st.write("Generando embeddings para las observaciones...")
            embeddings = modelo.encode(observaciones, convert_to_tensor=True)

            # Filtrar por territorio
            territorio_seleccionado = st.selectbox("Selecciona un territorio:", df.iloc[:, 9].unique())

            # Filtrar el DataFrame por el territorio seleccionado
            df_filtrado = df[df.iloc[:, 9] == territorio_seleccionado]

            # Mostrar el DataFrame filtrado
            st.write(f"Observaciones del territorio: {territorio_seleccionado}")
            st.dataframe(df_filtrado)

            # Buscar similitudes
            consulta = st.text_input("Escribe una observación para buscar similares:")

            if consulta:
                embedding_consulta = modelo.encode([consulta], convert_to_tensor=True)
                similaridades = util.pytorch_cos_sim(embedding_consulta, embeddings)[0]

                # Mostrar resultados más similares
                st.write("Observaciones más similares:")
                indices_similares = torch.topk(similaridades, k=5).indices

                for idx in indices_similares:
                    st.write(f"- {observaciones[idx]}")

        # Mostrar la tabla original con todas las observaciones si no hay filtro
        if st.checkbox("Mostrar datos originales"):
            st.dataframe(df)
