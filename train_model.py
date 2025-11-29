import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

train_path = "dataset/train"
val_path = "dataset/validation"
model_save_path = "model/fv_model_tf"

img_size = 224
batch_size = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_path, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode="categorical"
)

# Load MobileNetV2
base_model = MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3), weights="imagenet")
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(len(train_data.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the model
model.save(model_save_path, save_format="tf")

# Save class names
class_names = list(train_data.class_indices.keys())
np.save("class_names.npy", class_names)

print("Model saved at:", model_save_path)
print("Class Names:", class_names)


