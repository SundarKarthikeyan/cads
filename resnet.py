import tensorflow as tf
import sys
import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os

# Change standard output encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Assuming binary classification

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Define image data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load and preprocess the image data
train_generator = datagen.flow_from_directory(
    'arcade/stenosis/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'arcade/stenosis/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Unfreeze some layers and fine-tune the model
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Function to load and preprocess the user input image
def prepare_image(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        img = load_img(filepath, target_size=(224, 224))  # Load the image with the target size
        img_array = img_to_array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model input shape
        img_array = img_array / 255.0  # Rescale the image
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# Predict function
def predict_image(model, img_path):
    img = prepare_image(img_path)
    prediction = model.predict(img)
    return prediction[0][0]

# Example usage
image_path = 'testinput.png'  # Path to your image
try:
    result = predict_image(model, image_path)
    if result > 0.5:
        print("The person has microvascular dysfunction.")
    else:
        print("The person does not have microvascular dysfunction.")
except Exception as e:
    print(f"Error: {e}")
