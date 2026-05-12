import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

@st.cache_resource
def load_tm_model():
    return load_model("saved.model.pb", compile=False)

model = load_tm_model()
class_names = open("labels.txt", "r").readlines()

if "pagina" not in st.session_state:
    st.session_state.pagina = "Home"

if "fotos" not in st.session_state:
    st.session_state.fotos = []

with st.sidebar:
    st.title("Menu")

    if st.button("Home"):
        st.session_state.pagina = "Home"

    if st.button("Webcam"):
        st.session_state.pagina = "Webcam"

    if st.button("foto's"):
        st.session_state.pagina = "foto's"



if st.session_state.pagina == "Home":
    st.title("Home")

    image_file = st.file_uploader("Upload een foto", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image)

        # --- Teachable Machine ---
        size = (224, 224)
        image_tm = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        image_array = np.asarray(image_tm)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        merk = class_names[index].strip()
        confidence = prediction[0][index]

        st.success(f"Merk: {merk}")
        st.write(f"Zekerheid: {confidence:.2%}")

    kleur = st.slider("Achtergrondkleur", 0, 255, 255)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: rgb({kleur}, {kleur}, {kleur});
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



elif st.session_state.pagina == "Webcam":
    st.title("Webcam")
    img = st.camera_input("Neem een foto")

    if img:
        st.image(img)
        st.session_state.fotos.append(img)



elif st.session_state.pagina == "foto's":
    st.title("Geschiedenis")

    if st.session_state.fotos:
        for foto in st.session_state.fotos:
            st.image(foto)
    else:
        st.write("Nog geen foto's genomen")
