import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Image Feature Extractor
# --------------------------------------------------

def get_image_features(image_model, image):

    feature_extractor = tf.keras.Model(
        inputs=image_model.input,
        outputs=image_model.get_layer("dense").output
    )

    return feature_extractor.predict(image, verbose=0)


# --------------------------------------------------
# Text Feature Extractor
# --------------------------------------------------

def get_text_features(text_model, text):

    # Build model once
    _ = text_model.predict(text, verbose=0)

    feature_extractor = tf.keras.Model(
        inputs=text_model.inputs,
        outputs=text_model.layers[-2].output
    )

    features = feature_extractor.predict(
        text,
        verbose=0
    )

    return features


# --------------------------------------------------
# Fusion Prediction
# --------------------------------------------------

def predict_disease(
        fusion_model,
        image_features,
        text_features,
        class_names
):

    fusion_input = np.concatenate(
        [image_features, text_features],
        axis=1
    )

    pred = fusion_model.predict(
        fusion_input,
        verbose=0
    )[0]

    idx = np.argmax(pred)

    return (
        class_names[idx],
        float(pred[idx]),
        pred
    )