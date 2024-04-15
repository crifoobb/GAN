import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

## Load and preprocess the dataset
def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    gesture_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for i, gesture_class in enumerate(gesture_classes):
        class_path = os.path.join(data_dir, gesture_class)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if os.path.isfile(img_path):  # Ensure it's a file
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
                img = cv2.resize(img, (64, 64))  # Resize image to ensure consistency
                images.append(img)
                labels.append(i)  # Assign a label based on the index of the gesture class
    
    return np.array(images), np.array(labels)


## CNN model
def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3,), activation='sigmoid', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the dataset
data_dir = '/Users/crifoobb/Downloads/fu/data'
images, labels = load_and_preprocess_data(data_dir)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape images to fit the CNN input shape
X_train = X_train.reshape((-1, 64, 64, 1)).astype('float32') / 255.0
X_test = X_test.reshape((-1, 64, 64, 1)).astype('float32') / 255.0

# Create the model
input_shape = (64, 64, 1)
num_classes = 6#len(np.unique(labels))
model = create_model(input_shape, num_classes)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy}')

# Save the model for future use
model.save('hand_gesture_model.h5')