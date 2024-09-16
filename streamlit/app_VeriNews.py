
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from PIL import Image

# -----------------------------------------------------------------           
# -------------- APP VeriNews para detectar Fake News -------------
# -----------------------------------------------------------------  

# --------------------------------           
# definiciones previas necesarias:
# --------------------------------

# Cargamos el modelo y el vectorizador Tfidf
model = joblib.load('../models/random_forest.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')


# Inicializamos el stemmer y cargamos las stopwords para la transformación
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


# Definimos función de stemming + quitar caracteres especiales, stopwords, a minúsculas...
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content


# Y lo más importante: la función para clasificar la noticia
def classify_news(news):
    
    # Aplicamos transformaciones necesarias
    news_processed = stemming(news)
    
    # Convertimos a vector con Tfidf
    news_vector = vectorizer.transform([news_processed])
    
    # Predecimos con el modelo cargado
    prediction = model.predict(news_vector)
    return prediction[0]



# -----------------------------------------------------------------           
# ----------------- Configuración app Streamlit -------------------
# ----------------------------------------------------------------- 

# ---------------- 
# ---- Inicio ----
# ----------------
st.set_page_config(page_title="VeriNews",
                   page_icon=":mag:",
                   layout="centered")


# ---------------- 
# ---- Título ----
# ----------------
st.markdown("<h1 style='color: #1e355f;'>VeriNews 📰 🔍</h1>", unsafe_allow_html=True)

st.markdown("""
            ¡Bienvenido a VeriNews! La aplicación que le permitirá saber si esa noticia que tanto le interesa es real o es, en realidad, una fake new!
            """)



# ------------------ 
# - Lateral y logo -
# ------------------

logo = Image.open("../images_notebook/fake_news.png")
#st.image(logo, use_column_width=False, width=250, caption="", clamp=True)

# Panel lateral a la izquierda
st.sidebar.header("Acerca VeriNews")
st.sidebar.image(logo, use_column_width=False, width=250, caption="")
st.sidebar.markdown("Somos una App clasificadora de titulares y textos de noticias. Porque la información vale más que nunca, en VeriNews nos dedicamos a clasificar titulares y textos de noticias para que siempre sepas la verdad.")
    
    
# ------------------- 
# -- Input usuario --
# -------------------
st.markdown("""
            <div style='color: #1e355f; font-size: large;'>
                <label for="noticia">Escriba el texto o el titular de la noticia aquí!</label>
            </div>
            """, unsafe_allow_html=True)

noticia = st.text_input(label="", key="noticia")

# ------------------- 
# -- Clasificación --
# -------------------

# Botón para realizar la clasificación

if st.button('Conocer si es real o fake'):
    # Verificamos si se introdujo algún texto
        if not noticia:
            st.warning("Por favor, escriba la noticia antes de realizar la predicción.")
        else:
            # Realizamos la clasificación de la noticia
            prediction = classify_news(noticia)
            
            # Mostramos el resultado de la predicción
            if prediction == 0:
                st.success("Es una noticia real :) ✅")
            
            else:
                st.error("¡Cuidado! Es una fake new ⛔️")
















            
            
