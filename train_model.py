import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 🔹 Data generators (normalize images)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 🔹 Load training data
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(24, 24),
    batch_size=64,   # increased for speed
    class_mode='binary'
)

# 🔹 Load testing data
test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(24, 24),
    batch_size=64,
    class_mode='binary'
)

# 🔹 Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24,24,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

# 🔹 Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🔹 Train model (reduced epochs for large dataset)
model.fit(
    train_data,
    epochs=3,   # 🔥 optimized (important change)
    validation_data=test_data
)

# 🔹 Save model
model.save("model/drowsiness_model.h5")

print("✅ Model trained and saved successfully!")
