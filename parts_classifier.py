import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

#st.image("/home/carlos/workspace/synapse/mechanical_parts_classifier/mechanical_parts_classifier/img/Captura de tela de 2025-05-11 20-33-07.png", use_container_width=True)# Título da aplicação
st.title("Agromerica classificador de Peças")
#--- Descrição da aplicação
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
uploaded_file = '/home/carlos/workspace/synapse/mechanical_parts_classifier/mechanical_parts_classifier/img/Captura de tela de 2025-05-11 20-33-07.png'
#uploaded_file = st.file_uploader("Envie uma imagem da peça - jpg, jpeg ou png", type=["jpg", "jpeg", "png"])
print(f"Arquivo enviado: {uploaded_file}")
# Carrega o modelo .pt do YOLOv11
@st.cache_resource
def load_model():
    model = YOLO("/home/carlos/workspace/synapse/mechanical_parts_classifier/mechanical_parts_classifier/models/agromerica_train/weights/best.pt")  # substitua pelo caminho correto do seu modelo
    return model

model = load_model()
print(f"Modelo carregado com sucesso!{model.names}")

uploaded_file = st.file_uploader("Envie uma imagem da peça", type=["jpg", "jpeg", "png"])


# Exemplo de banco de dados fictício de peças
pecas_info = {
    "Junta Cria": {
        "fabricante": "Agromerica Ltda",
        "numeração": "FK104589",
        "instalação": "Junta Basculante",
        "recomendação": "Montagem em eixo de transmissão",
        "preco": "R$ 980,00"
    }
    # Adicione outras peças conforme necessário
}

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_container_width=True)
    img_array = np.array(image)

    st.subheader("Resultado da Classificação:")
    results = model(img_array)
    first_result = results[0]

    boxes = first_result.boxes
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            label = model.names[cls_id]
            st.markdown(f"### 🔍 Detecção: `{label}` com confiança `{conf:.2f}`")

            # Alerta se a confiança estiver abaixo de 50%
            if conf < 0.5:
                st.warning("⚠️ Peça com algum tipo de diferença ou baixa confiança na detecção!")

            # Exibe informações da peça, se existir no "banco"
            if label in pecas_info:
                info = pecas_info[label]
                st.info(f"""
                **Fabricante:** {info['fabricante']}  
                **Numeração:** {info['numeração']}  
                **Instalação:** {info['instalação']}  
                **Recomendação:** {info['recomendação']}  
                **Preço:** {info['preco']}
                """)
            else:
                st.markdown("ℹ️ Informações da peça não encontradas no banco de dados.")

        # Imagem anotada
        annotated_image = first_result.plot()
        st.image(annotated_image, caption="Resultado com detecções", use_container_width=True)
    else:
        st.markdown("### ⚠️ Nenhuma detecção encontrada.")



# # Processamento da imagem e inferência
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     print(f"Imagem carregada: {image}")
#     st.image(image, caption="Imagem enviada", use_container_width=True)

#     # Convertendo para array
#     img_array = np.array(image)

#     # Rodando inferência
#     st.subheader("Resultado da Classificação:")
    
#     results = model(img_array)  
#     print(f"Resultados da inferência: {results}")  
#     # Exibindo predição da primeira detecção (posição 0)
#     # Acessando resultados de classificação
#     first_result = results[0]
#     # probs = first_result.probs
#     print(f"Resultados da primeira detecção: {first_result}")
#     # print(f"Probabilidades: {probs}")
    
    
#     ###################################################################################################################

#     boxes = first_result.boxes
#     print(f"Caixas detectadas: {boxes}")
#     if boxes is not None and len(boxes) > 0:
#         for box in boxes:
#             cls_id = int(box.cls[0].cpu().numpy())
#             conf = float(box.conf[0].cpu().numpy())
#             label = model.names[cls_id]
#             print(f"ID da classe detectada: {cls_id}")
#             st.markdown(f"### 🔍 Detecção: `{label}` com confiança `{conf:.2f}`")
#     else:
#         st.markdown("### ⚠️ Nenhuma detecção encontrada.")
#         print("Nenhuma detecção encontrada.")
###################################################################################################################
    # if probs is not None:
    #     confs = probs.data 
    #     cls_id = int(np.argmax(confs.cpu().numpy())) 
    #     print(f"ID da classe detectada: {cls_id}")
    #     conf = float(confs[cls_id]) 
    #     print(f"ID da classe: {cls_id}, Confiança: {conf}")
    #     label = first_result.names[cls_id] 
    #     print(f"Classe detectada: {label}, Confiança: {conf}")
    #     st.markdown(f"### 🔍 Predição principal: `{label}` com confiança `{conf:.2f}`")
    # else:
    #     st.markdown("### ⚠️ Nenhuma predição foi retornada.")
    #     print("Nenhuma predição retornada.")


