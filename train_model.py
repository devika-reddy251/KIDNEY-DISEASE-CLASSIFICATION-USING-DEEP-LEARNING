import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Paths
data_dir = r"C:\Users\SHIVAKUMAR\Desktop\KIDNEY\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
model_dir = 'models'
model_path = os.path.join(model_dir, 'saved_model.h5')

# Ensure 'models' directory exists
os.makedirs(model_dir, exist_ok=True)

# Image preprocessing parameters
img_size = (150, 150)
batch_size = 32

# Data generators with validation split
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class indices for prediction mapping
with open(os.path.join(model_dir, 'class_indices.json'), 'w') as f:
    json.dump(train_gen.class_indices, f)

print("Class indices saved:", train_gen.class_indices)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')  # output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_gen, epochs=10, validation_data=val_gen)

# Save model
model.save(model_path)
print(f"Model saved at: {model_path}")

