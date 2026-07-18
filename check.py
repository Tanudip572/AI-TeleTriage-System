import tensorflow as tf

image_model = tf.keras.models.load_model("models/densenet121_best.keras")
text_model = tf.keras.models.load_model("models/bilstm_best.keras")
fusion_model = tf.keras.models.load_model("models/fusion_final.keras")

print("\n========== IMAGE MODEL ==========\n")
image_model.summary()

print("\n========== TEXT MODEL ==========\n")
text_model.summary()

print("\n========== FUSION MODEL ==========\n")
fusion_model.summary()