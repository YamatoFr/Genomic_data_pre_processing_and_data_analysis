import numpy as np
import pandas as pd
import os
import re
import shutil

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

import cv2
from PIL import Image
matplotlib.use('Agg')   # type: ignore
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define dataset directory
images_dir = "images"

# Training parameters
img_size = (32, 32)  # Resize images to 64x64
batch_size = 32
epochs = 15

k_folds = 5  # Number of folds for cross-validation

# Dictionary to store accuracies
cross_val_accuracies = {}

# Function to build a simple CNN model
def cnn_model(input_shape, num_classes):
	model = Sequential([
		Conv2D(32, (5,5), activation='relu', input_shape=input_shape),
		MaxPooling2D(2,2),
		Conv2D(64, (5,5), activation='relu'),
		MaxPooling2D(2,2),
		Flatten(),
		Dense(128, activation='relu'),
		Dense(num_classes, activation='softmax')  # Adjust output neurons based on classes
	])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# Iterate through resolution folders
for resolution_folder in sorted(os.listdir(images_dir), key=lambda x: int(re.search(r'\d+', x).group())):
	resolution_path = os.path.join(images_dir, resolution_folder)

	if os.path.isdir(resolution_path):
		# Iterate through method folders inside resolution
		for method_folder in os.listdir(resolution_path):
			method_path = os.path.join(resolution_path, method_folder)

			if os.path.isdir(method_path):
				print(f"\nCross-validating {resolution_folder}/{method_folder}...")

				# Data Preprocessing
				datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
				dataset = datagen.flow_from_directory(
					method_path, target_size=img_size, batch_size=batch_size,
					class_mode='categorical', shuffle=True
				)

				num_samples = dataset.samples
				num_classes = dataset.num_classes
				input_shape = (img_size[0], img_size[1], 3)

				# K-Fold Cross-Validation
				kfold = KFold(n_splits=k_folds, shuffle=True)
				fold_accuracies = []

				for train_idx, val_idx in kfold.split(np.arange(num_samples)):
					# Re-initialize CNN for each fold
					model = cnn_model(input_shape, num_classes)

					# Create new data generators for train/validation split
					train_data = datagen.flow_from_directory(
						method_path, target_size=img_size, batch_size=batch_size,
						class_mode='categorical', subset='training'
					)
					val_data = datagen.flow_from_directory(
						method_path, target_size=img_size, batch_size=batch_size,
						class_mode='categorical', subset='validation'
					)

					# Train model
					history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)

					# Store best validation accuracy for this fold
					max_fold_acc = max(history.history['val_accuracy'])
					fold_accuracies.append(max_fold_acc)
					print(f"Fold accuracy: {max_fold_acc:.4f}")
					
				# save the model
				model_path = os.path.join("models/kfold", resolution_folder, method_folder)
				os.makedirs(model_path, exist_ok=True)
				model_name = f"{resolution_folder}_{method_folder}_fold_{len(fold_accuracies)}_mask_5.h5"
				model.save(os.path.join(model_path, model_name))
				print(f"Model saved to {model_path}")

				# Store average accuracy for this method-resolution combo
				avg_acc = np.mean(fold_accuracies)
				key = f"{resolution_folder} - {method_folder}"
				cross_val_accuracies[key] = avg_acc
				print(f"Average accuracy for {key}: {avg_acc:.4f}")

# Create positions for bars
num_bars = len(cross_val_accuracies)
x_positions = np.arange(num_bars)

# Add extra space every 10 bars
for i in range(10, num_bars, 10):  
	x_positions[i:] += 1  # Shift everything after every 10th bar

plt.figure(figsize=(15, 10))
plt.bar(x_positions, cross_val_accuracies.values(), color='blue', width=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Custom x-ticks for readability
plt.xticks(x_positions, cross_val_accuracies.keys(), rotation=90)

plt.yticks(np.arange(0, 1.1, 0.2))

plt.ylabel("Validation Accuracy")
plt.title("CNN Validation Results")
plt.tight_layout()
plt.show()

# Extract and sort resolutions numerically
resolutions = sorted(set(k.split(" - ")[0] for k in cross_val_accuracies.keys()), key=lambda x: int(x.replace("px", "")))
methods = sorted(set(k.split(" - ")[1] for k in cross_val_accuracies.keys()))

# Organize data for plotting
data = {method: [cross_val_accuracies.get(f"{res} - {method}", None) for res in resolutions] for method in methods}

# Define a color map for better distinction
colors = plt.cm.get_cmap("tab10", len(methods))  # Use "tab10" color map with enough colors

# Plot the lines
plt.figure(figsize=(16, 10))
for i, (method, accuracies) in enumerate(data.items()):
	plt.plot(resolutions, accuracies, marker="o", linestyle="-", linewidth=2, markersize=8, label=method, color=colors(i))

# Labels and title
plt.xlabel("Resolution", fontsize=12)
plt.ylabel("Max Accuracy", fontsize=12)
plt.title("CNN Validation Accuracy by Resolution and Method", fontsize=14)

# Move legend outside the plot for better clarity
plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

# Add a grid with transparency
plt.grid(True, linestyle="--", alpha=0.6)

# Improve layout to fit legend properly
plt.tight_layout()

# Show plot
plt.show()

plt.savefig("../../imgs/graphs/kfold/cnn_validation_accuracy_mask_5-kfold.png")

# Extract and sort resolutions numerically
resolutions = sorted(set(k.split(" - ")[0] for k in cross_val_accuracies.keys()), key=lambda x: int(x.replace("px", "")))

# Extract unique methods and split into two groups of 5
methods = sorted(set(k.split(" - ")[1] for k in cross_val_accuracies.keys()))
methods_group1 = methods[:5]
methods_group2 = methods[5:]

# Organize data for plotting
data1 = {method: [cross_val_accuracies.get(f"{res} - {method}", None) for res in resolutions] for method in methods_group1}
data2 = {method: [cross_val_accuracies.get(f"{res} - {method}", None) for res in resolutions] for method in methods_group2}

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

colors = plt.cm.get_cmap("tab10", len(methods_group1))  # Use "tab10" color map with enough colors

# Plot first group
for method, accuracies in data1.items():
	axs[0].plot(resolutions, accuracies, marker="o", label=method, color=colors(methods_group1.index(method)))
axs[0].set_title("CNN Validation Accuracy (Group 1)")
axs[0].set_ylabel("Max Accuracy")
axs[0].legend(title="Method")
axs[0].grid(True, linestyle="--", alpha=0.6)

# Plot second group
for method, accuracies in data2.items():
	axs[1].plot(resolutions, accuracies, marker="o", label=method, color=colors(methods_group2.index(method)))
axs[1].set_title("CNN Validation Accuracy (Group 2)")
axs[1].set_xlabel("Resolution")
axs[1].set_ylabel("Max Accuracy")
axs[1].legend(title="Method")
axs[1].grid(True, linestyle="--", alpha=0.6)

# Adjust layout
plt.tight_layout()
plt.show()

plt.savefig("../../imgs/graphs/kfold/cnn_validation_accuracy_groups_mask_5_kfold.png")

# confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras.models import load_model

# Load the model
model_path = "models/kfold/100px/Skew-Diversite-NucleScore/100px_Skew-Diversite-NucleScore_fold_5_mask_5.h5"
model = load_model(model_path)
model.summary()

# Load the test data
test_data_dir = 'images/100px/Chargaff-Composante-Diversite'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_size[0], img_size[1]),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=False
)
# Get the true labels
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Get the predicted labels
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_labels = np.argmax(predictions, axis=1)
# Generate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

plt.savefig("../../imgs/graphs/cnn_confusion_matrix_100px_mask_5-kfold.png")

# Define dataset directory
mosaics_dir = "C:/Users/theof/OneDrive/Documents/Github/genome_color_unpickler/res"

# Training parameters
img_size = (20, 50)  # Resize images to 64x64
batch_size = 32
epochs = 15

k_folds = 5  # Number of folds for cross-validation

# Dictionary to store max accuracies
cross_val_accuracies_mos = {}

# mosaic version
# Iterate through resolution folders (e.g., 4px, 16px)
for resolution_folder in sorted(os.listdir(mosaics_dir), key=lambda x: int(re.search(r'\d+', x).group())):
	resolution_path = os.path.join(mosaics_dir, resolution_folder)

	if os.path.isdir(resolution_path):
		print(f"\nCross-validating {resolution_folder}...")

		# Data Preprocessing
		datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
		dataset = datagen.flow_from_directory(
			resolution_path, target_size=img_size, batch_size=batch_size,
			class_mode='categorical', shuffle=True
		)

		num_samples = dataset.samples
		num_classes = dataset.num_classes
		input_shape = (img_size[0], img_size[1], 3)

		# K-Fold Cross-Validation
		kfold = KFold(n_splits=k_folds, shuffle=True)
		fold_accuracies = []

		for train_idx, val_idx in kfold.split(np.arange(num_samples)):
			# Re-initialize CNN for each fold
			model = cnn_model(input_shape, num_classes)

			# Create new data generators for train/validation split
			train_data = datagen.flow_from_directory(
				resolution_path, target_size=img_size, batch_size=batch_size,
				class_mode='categorical', subset='training'
			)
			val_data = datagen.flow_from_directory(
				resolution_path, target_size=img_size, batch_size=batch_size,
				class_mode='categorical', subset='validation'
			)

			# Train model
			history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)

			# Store best validation accuracy for this fold
			max_fold_acc = max(history.history['val_accuracy'])
			fold_accuracies.append(max_fold_acc)
			print(f"Fold accuracy: {max_fold_acc:.4f}")

		# save the model
		model_path = os.path.join("models/kfold_mosaic", resolution_folder)
		os.makedirs(model_path, exist_ok=True)
		model_name = f"{resolution_folder}_fold_{len(fold_accuracies)}_mosaic_mask_5.h5"
		model.save(os.path.join(model_path, model_name))
		print(f"Model saved to {model_path}")

		# Store average accuracy for this resolution
		avg_acc = np.mean(fold_accuracies)
		cross_val_accuracies_mos[resolution_folder] = avg_acc
		print(f"Average accuracy for {resolution_folder}: {avg_acc:.4f}")

# Generate Bar Graph
plt.figure(figsize=(10,5))
plt.bar(cross_val_accuracies_mos.keys(), cross_val_accuracies_mos.values(), color='blue')
plt.xticks(rotation=45)
plt.ylabel("Cross-Validation Avg Accuracy")
plt.title("CNN Training Results by Resolution")
plt.show()

# Create positions for bars
num_bars = len(cross_val_accuracies_mos)
x_positions = np.arange(num_bars)

plt.figure(figsize=(15, 10))
plt.bar(x_positions, cross_val_accuracies_mos.values(), color='blue', width=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Custom x-ticks for readability
plt.xticks(x_positions, cross_val_accuracies_mos.keys(), rotation=90)

plt.yticks(np.arange(0, 1.1, 0.2))

plt.ylabel("Max Accuracy")
plt.title("CNN Validation Results")
plt.tight_layout()
plt.show()

plt.savefig("../../imgs/graphs/kfold/cnn_validation_accuracy_kfold_mosaics_mask_5.png")

# Extract and sort resolutions numerically
resolutions = sorted(cross_val_accuracies_mos.keys(), key=lambda x: int(x.replace("px", "")))

# Get max accuracy values in the correct order
accuracies = [cross_val_accuracies_mos[res] for res in resolutions]

# Plot the accuracy per resolution
plt.figure(figsize=(12, 6))
plt.plot(resolutions, accuracies, marker="o", linestyle="-", linewidth=2, markersize=8, color="blue", label="Accuracy")

# Labels and title
plt.xlabel("Resolution", fontsize=12)
plt.ylabel("Max Accuracy", fontsize=12)
plt.title("CNN Validation Accuracy by Resolution", fontsize=14)

# Add data points on the plot
for i, acc in enumerate(accuracies):
    plt.text(resolutions[i], acc, f"{acc:.2f}", fontsize=10, ha="right")

# Grid and legend
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right")

# Save and show plot
plt.tight_layout()
plt.show()
plt.savefig("../../imgs/graphs/kfold/cnn_validation_accuracy_kfold_mosaics_line_mask_5.png")

# confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras.models import load_model

# Load the model for the new version
new_model_path = "models/mosaics/100px/100px_mosaic_mask_5.h5"
new_model = load_model(new_model_path)
new_model.summary()

# Load the test data for the new version
new_test_data_dir = 'C:/Users/theof/OneDrive/Documents/Github/genome_color_unpickler/res/100px'

new_test_datagen = ImageDataGenerator(rescale=1./255)
new_test_generator = new_test_datagen.flow_from_directory(
	new_test_data_dir,
	target_size=(img_size[0], img_size[1]),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=False
)

# Get the true labels for the new version
new_true_labels = new_test_generator.classes
new_class_labels = list(new_test_generator.class_indices.keys())

# Get the predicted labels for the new version
new_predictions = new_model.predict(new_test_generator, steps=len(new_test_generator), verbose=1)
new_predicted_labels = np.argmax(new_predictions, axis=1)

# Generate the confusion matrix for the new version
new_cm = confusion_matrix(new_true_labels, new_predicted_labels)

# Plot the confusion matrix for the new version
plt.figure(figsize=(10, 8))
sns.heatmap(new_cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_class_labels, yticklabels=new_class_labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for New Version')
plt.show()

plt.savefig("../../imgs/graphs/kfold/cnn_confusion_matrix_kfold_mosaics_100px_mask_5.png")