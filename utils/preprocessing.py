import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(uploaded_file):

    img = Image.open(uploaded_file)

    img = img.convert("RGB")

    img = img.resize((128,128))

    img = np.array(img,dtype=np.float32)/255.0

    img = np.expand_dims(img,axis=0)

    return img

# --------------------------------------------------
# Clinical Note Generator
# --------------------------------------------------

def create_clinical_note(
        age,
        sex,
        occupation,
        location,
        symptoms
):

    note = f"""
Patient age {age} years.
Sex {sex}.
Occupation {occupation}.
Lesion located on {location}.
Symptoms include {symptoms}.
"""

    return note.strip()


# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------

def preprocess_text(
        text,
        tokenizer,
        max_len
):

    seq = tokenizer.texts_to_sequences([text])

    pad = pad_sequences(
        seq,
        maxlen=max_len,
        padding="post"
    )

    return pad