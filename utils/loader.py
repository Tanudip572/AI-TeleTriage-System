import pickle
import tensorflow as tf
import streamlit as st


@st.cache_resource
def load_models():

    image_model = tf.keras.models.load_model(
        "models/densenet121_best.keras"
    )

    text_model = tf.keras.models.load_model(
        "models/bilstm_best.keras"
    )

    fusion_model = tf.keras.models.load_model(
        "models/fusion_final.keras"
    )

    with open("saved_objects/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("saved_objects/max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    print("LOADER MAX_LEN =", max_len)

    with open("saved_objects/fusion_class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    return (
        image_model,
        text_model,
        fusion_model,
        tokenizer,
        max_len,
        class_names
    )