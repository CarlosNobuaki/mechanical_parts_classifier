import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Título da aplicação
st.title("Classificador de Peças")
# --- Links úteis ---
st.markdown("---")
st.subheader("Contatos")
st.markdown("""
- [INOVASKILL](https://mentto.com.br/programa/inova-skill-2025/)
- [FATEC Shunji Nishimura](https://www.fatecpompeia.edu.br/)""")
st.markdown("---")
st.subheader("Desenvolvimento de projetos. Entre em contato!")
st.markdown("""
- [CIAG](https://www.ciag.org.br/#/)
""")

# Upload da imagem
uploaded_file = st.file_uploader("Envie uma imagem da peça - jpg, jpeg ou png", type=["jpg", "jpeg", "png"])

# Carrega o modelo .pt do YOLOv11
@st.cache_resource
def load_model():
    model = YOLO("models/mechanical_parts_best.pt")  # substitua pelo caminho correto do seu modelo
    return model

model = load_model()

# Processamento da imagem e inferência
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_container_width=True)

    # Convertendo para array
    img_array = np.array(image)

    # Rodando inferência
    st.subheader("Resultado da Classificação:")
    results = model(img_array) 
    # Visualizando com as detecções
    annotated_frame = results[0].plot() 
    # Exibindo predição da primeira detecção (posição 0)
    # Acessando resultados de classificação
    first_result = results[0]
    probs = first_result.probs

    if probs is not None:
        confs = probs.data  # Tensor com as probabilidades
        cls_id = int(np.argmax(confs))  # Índice da maior probabilidade
        conf = float(confs[cls_id])     # Valor da confiança
        label = first_result.names[cls_id]  # Nome da classe
        st.markdown(f"### 🔍 Predição principal: `{label}` com confiança `{conf:.2f}`")
    else:
        st.markdown("### ⚠️ Nenhuma predição foi retornada.")


