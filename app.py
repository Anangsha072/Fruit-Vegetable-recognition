import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Load model
model = tf.keras.models.load_model("model/fv_model_tf")


# Load class names (VERY IMPORTANT)
class_names = np.load("class_names.npy", allow_pickle=True).tolist()

# Load calories table
cal_data = pd.read_csv("calories.csv")
cal_dict = dict(zip(cal_data["item"], cal_data["calories_per_100g"]))

st.title("🥗 Fruit & Vegetable Recognition + Calorie Counter")
st.write("Upload an image to detect the food item and calculate calories!")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    index = np.argmax(predictions)

    item_name = class_names[index].lower()

    st.subheader(f"🍎 Detected Item: **{item_name.capitalize()}**")

    calories_100g = cal_dict.get(item_name, "N/A")
    st.write(f"Calories per 100g: **{calories_100g} kcal**")

    quantity = st.number_input("Enter quantity (grams):", min_value=1, value=100)

    if calories_100g != "N/A":
        total_calories = (calories_100g / 100) * quantity
        st.subheader(f"🔥 Total Calories: **{total_calories:.2f} kcal**")
    else:
        st.error("Calorie data not found for this item!")


