import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from utils.loader import load_models
from utils.preprocessing import (
    preprocess_image,
    preprocess_text,
    create_clinical_note
)
from utils.predictor import (
    get_image_features,
    get_text_features,
    predict_disease
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Tele-Triage System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODELS
# ======================================================

(
    image_model,
    text_model,
    fusion_model,
    tokenizer,
    max_len,
    class_names
) = load_models()

# ======================================================
# CLASS NAME MAPPING
# ======================================================

CLASS_MAPPING = {

    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",

    "bcc": "Basal Cell Carcinoma",

    "bkl": "Benign Keratosis-like Lesion",

    "df": "Dermatofibroma",

    "mel": "Melanoma",

    "nv": "Melanocytic Nevus (Mole)",

    "vasc": "Vascular Lesion"

}

# ======================================================
# DISEASE INFORMATION
# ======================================================

DISEASE_INFO = {

    "Actinic Keratoses / Intraepithelial Carcinoma":
    "A precancerous skin lesion caused by prolonged sun exposure. Early treatment is recommended.",

    "Basal Cell Carcinoma":
    "A slow-growing skin cancer that is highly treatable when detected early.",

    "Benign Keratosis-like Lesion":
    "A benign skin lesion that usually does not require urgent treatment.",

    "Dermatofibroma":
    "A harmless fibrous skin nodule commonly found on the limbs.",

    "Melanoma":
    "A serious and potentially life-threatening skin cancer requiring immediate medical evaluation.",

    "Melanocytic Nevus (Mole)":
    "A common benign mole that should be monitored for changes.",

    "Vascular Lesion":
    "A benign lesion involving blood vessels."
}
# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🩺 AI Tele-Triage")

st.sidebar.success("System Ready")

st.sidebar.markdown("---")

st.sidebar.subheader("Loaded Models")

st.sidebar.write("✅ DenseNet121")
st.sidebar.write("✅ BiLSTM")
st.sidebar.write("✅ Fusion Network")

st.sidebar.markdown("---")

st.sidebar.info(
    """
Upload a dermoscopic skin image and provide
basic clinical information.

The AI combines image features and
clinical text for disease prediction.
"""
)

# ======================================================
# TITLE
# ======================================================

st.title("🩺 AI Tele-Triage System")

st.caption(
    "Multimodal Skin Disease Classification using Deep Learning"
)

st.markdown("---")

# ======================================================
# MAIN LAYOUT
# ======================================================

left, right = st.columns([1, 1])

# ======================================================
# IMAGE UPLOAD
# ======================================================

with left:

    st.subheader("📷 Upload Dermoscopic Image")

    uploaded_file = st.file_uploader(
        "Choose a Skin Lesion Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )
# ======================================================
# PATIENT INFORMATION
# ======================================================

with right:

    st.subheader("📝 Patient Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Student",
            "Software Engineer",
            "Teacher",
            "Doctor",
            "Business",
            "Farmer",
            "Labour",
            "Housewife",
            "Retired",
            "Other"
        ]
    )

    location = st.selectbox(
        "Lesion Location",
        [
            "Face",
            "Scalp",
            "Neck",
            "Chest",
            "Back",
            "Abdomen",
            "Upper Extremity",
            "Lower Extremity",
            "Hand",
            "Foot",
            "Ear",
            "Unknown"
        ]
    )

    symptoms = st.multiselect(
        "Clinical Symptoms",
        [
            "Itching",
            "Pain",
            "Bleeding",
            "Burning",
            "Growing Lesion",
            "Color Change",
            "Irregular Border",
            "Swelling",
            "Dry Skin",
            "No Symptoms"
        ]
    )

    symptoms = ", ".join(symptoms)

# ======================================================
# PREDICT BUTTON
# ======================================================

st.markdown("---")

predict = st.button(
    "🔍 Predict Disease",
    use_container_width=True
)
# ======================================================
# PREDICTION
# ======================================================

if predict:

    # -----------------------------
    # INPUT VALIDATION
    # -----------------------------

    if uploaded_file is None:

        st.error("Please upload a dermoscopic image.")

    elif len(symptoms) == 0:

        st.error("Please select at least one symptom.")

    else:

        # -----------------------------
        # CREATE CLINICAL NOTE
        # -----------------------------

        clinical_note = create_clinical_note(

            age,
            sex,
            occupation,
            location,
            symptoms

        )

        with st.expander("📄 Generated Clinical Note"):

            st.write(clinical_note)

        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------

        with st.spinner("Analyzing image and clinical information..."):

            # Image preprocessing
            image = preprocess_image(uploaded_file)

            # Text preprocessing
            text = preprocess_text(

                clinical_note,

                tokenizer,

                max_len

            )

            # Image feature extraction
            image_features = get_image_features(

                image_model,

                image

            )

            # Text feature extraction
            text_features = get_text_features(

                text_model,

                text

            )

            # Fusion prediction
            disease, confidence, prob = predict_disease(

                fusion_model,

                image_features,

                text_features,

                class_names

            )

        display_disease = CLASS_MAPPING.get(

            disease,

            disease

        )

        st.success("✅ Prediction Completed Successfully")
                # ======================================================
        # RESULT CARD
        # ======================================================

        if confidence >= 0.90:

            st.success(
                f"""
### 🩺 Prediction Result

**Predicted Disease:** {display_disease}

**Confidence:** {confidence*100:.2f}%
"""
            )

        elif confidence >= 0.70:

            st.warning(
                f"""
### 🩺 Prediction Result

**Predicted Disease:** {display_disease}

**Confidence:** {confidence*100:.2f}%
"""
            )

        else:

            st.error(
                f"""
### ⚠ Low Confidence Prediction

**Predicted Disease:** {display_disease}

**Confidence:** {confidence*100:.2f}%
"""
            )

        # ======================================================
        # METRICS
        # ======================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Disease",
                display_disease
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        # ======================================================
        # DISEASE INFORMATION
        # ======================================================

        st.subheader("📖 Disease Information")

        st.info(
            DISEASE_INFO.get(
                display_disease,
                "No information available."
            )
        )

        # ======================================================
        # RISK LEVEL
        # ======================================================

        st.subheader("🚨 Risk Level")

        if display_disease == "Melanoma":

            st.error(
                "🔴 High Risk — Immediate consultation with a dermatologist is strongly recommended."
            )

        elif display_disease == "Basal Cell Carcinoma":

            st.warning(
                "🟠 Moderate Risk — Please consult a dermatologist for further evaluation."
            )

        else:

            st.success(
                "🟢 Low Risk — Continue monitoring the lesion and seek medical advice if any changes occur."
            )

        # ======================================================
        # AI RECOMMENDATION
        # ======================================================

        st.subheader("💡 AI Recommendation")

        if display_disease == "Melanoma":

            st.error("""
• Consult a dermatologist immediately.
• A biopsy may be required.
• Avoid delaying medical evaluation.
""")

        elif display_disease == "Basal Cell Carcinoma":

            st.warning("""
• Schedule a dermatology appointment.
• Protect the area from excessive sun exposure.
• Early treatment usually has excellent outcomes.
""")

        else:

            st.info("""
• Monitor the lesion regularly.
• Watch for changes in color, size or shape.
• Seek medical advice if symptoms worsen.
""")
                    # ======================================================
        # PREDICTION PROBABILITIES
        # ======================================================

        result_df = pd.DataFrame({

            "Disease": [
                CLASS_MAPPING.get(x, x)
                for x in class_names
            ],

            "Probability (%)": (prob * 100).round(2)

        })

        result_df = result_df.sort_values(

            by="Probability (%)",

            ascending=False

        ).reset_index(drop=True)

        st.subheader("📊 Prediction Probabilities")

        st.dataframe(

            result_df,

            use_container_width=True,

            hide_index=True

        )

        # ======================================================
        # TOP 3 PREDICTIONS
        # ======================================================

        st.subheader("🏆 Top 3 Most Likely Diseases")

        top3 = result_df.head(3)

        for i, row in top3.iterrows():

            st.write(

                f"**{i+1}. {row['Disease']}** — {row['Probability (%)']:.2f}%"

            )

        # ======================================================
        # CONFIDENCE CHART
        # ======================================================

        st.subheader("📈 Prediction Confidence")

        fig, ax = plt.subplots(figsize=(10,4))

        colors = ["crimson"] + ["steelblue"] * (len(result_df)-1)

        ax.bar(

            result_df["Disease"],

            result_df["Probability (%)"],

            color=colors

        )

        ax.set_ylabel("Probability (%)")

        ax.set_xlabel("Disease")

        ax.set_ylim(0,100)

        ax.set_title("Prediction Confidence")

        plt.xticks(rotation=20, ha="right")

        st.pyplot(fig)

        # ======================================================
        # DOWNLOAD RESULTS
        # ======================================================

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(

            label="📥 Download Prediction Report",

            data=csv,

            file_name="prediction_results.csv",

            mime="text/csv"

        )

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.caption(
    "⚠️ This AI system is intended for preliminary screening only and should not replace professional medical diagnosis."
)