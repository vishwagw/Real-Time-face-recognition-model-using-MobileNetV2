# this file is for developing a training model for training the dataset.
# for full program use 'training.py' script.
from tf_keras.applications import MobileNetV2
from tf_keras.layers import Dense, Flatten, Dropout
from tf_keras.models import Model

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers for classification
x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(len(class_labels), activation="softmax")(x)  # Output layer

# Create model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save the trained model
model.save("facerec_model.h5")

print("Model training complete and saved as facerec_model.h5")
