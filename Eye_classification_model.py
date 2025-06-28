import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam

# Image settings
image_size = 128
batch_size = 32

# Paths
train_dir = r"D:\Proj\Eyes open close\cropped_eyes\big_data\rgb\train"
test_dir = r"D:\Proj\Eyes open close\cropped_eyes\big_data\rgb\test"

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.5, 1.5]
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(image_size, image_size),
                                                    batch_size=batch_size, class_mode='binary', color_mode='rgb')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(image_size, image_size),
                                                  batch_size=batch_size, class_mode='binary', color_mode='rgb',
                                                  shuffle=False)

# Build Model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_generator, epochs=10, validation_data=test_generator)
model.save(r'D:\Proj\Eyes open close\mobilenet_eye_top_layers_only.h5')

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:75]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_generator, epochs=5, validation_data=test_generator)
model.save(r'D:\Proj\Eyes open close\mobilenet_eye_finetuned_model.h5')

# Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nFinal Test Accuracy: {test_acc:.4f} | Final Test Loss: {test_loss:.4f}")

# Predictions
test_generator.reset()
y_probs = model.predict(test_generator)
y_preds = (y_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Report
print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=["Closed", "Open"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Closed", "Open"], yticklabels=["Closed", "Open"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC
roc_auc = roc_auc_score(y_true, y_probs)
fpr, tpr, _ = roc_curve(y_true, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Accuracy & Loss plots
def plot_history(history, phase):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{phase} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{phase} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plot_history(history1, "")
plot_history(history2, "Fine-Tuning")
plt.show()

# Show predictions using OpenCV
label_map = {0: 'Closed', 1: 'Open'}
test_generator.reset()
images, labels = next(test_generator)  # Load one batch
preds = model.predict(images)
preds_binary = (preds > 0.5).astype(int)

grid_images = []
grid_size = 3  # 3x3 grid

for i in range(grid_size * grid_size):
    img = (images[i] * 255).astype(np.uint8)
    img_resized = cv2.resize(img, (150, 150))
    label = label_map[preds_binary[i][0]]
    cv2.putText(img_resized, f'{label}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    grid_images.append(img_resized)

# Stack images into a 3x3 grid
row1 = np.hstack(grid_images[0:3])
row2 = np.hstack(grid_images[3:6])
row3 = np.hstack(grid_images[6:9])
full_grid = np.vstack([row1, row2, row3])

cv2.imshow("Predicted Eye States (3x3 Grid)", full_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
