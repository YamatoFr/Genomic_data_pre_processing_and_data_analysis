import numpy as np
import pandas as pd
import os
import re
import shutil
import random
from glob import glob
from collections import defaultdict

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import load_model

# Define dataset directory
images_dir = "images"

# Training parameters
img_size = (32, 32)
batch_size = 32
epochs = 15
k_folds = 5
cross_val_accuracies = {}

def cnn_model(input_shape, num_classes):
	model = Sequential([
		Conv2D(32, (5,5), activation='relu', input_shape=input_shape),
		MaxPooling2D(2,2),
		Conv2D(64, (5,5), activation='relu'),
		MaxPooling2D(2,2),
		Flatten(),
		Dense(128, activation='relu'),
		Dense(num_classes, activation='softmax')
	])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# Iterate through resolution folders
for resolution_folder in sorted(os.listdir(images_dir), key=lambda x: int(re.search(r'\d+', x).group())):
	resolution_path = os.path.join(images_dir, resolution_folder)
	if os.path.isdir(resolution_path):
		for method_folder in os.listdir(resolution_path):
			method_path = os.path.join(resolution_path, method_folder)
			if os.path.isdir(method_path):
				print(f"\nCross-validating {resolution_folder}/{method_folder}...")

				# UNDERSAMPLING STEP
				class_dirs = [d for d in os.listdir(method_path) if os.path.isdir(os.path.join(method_path, d))]
				class_to_images = defaultdict(list)
				for class_name in class_dirs:
					class_path = os.path.join(method_path, class_name)
					image_paths = glob(os.path.join(class_path, "*"))
					class_to_images[class_name].extend(image_paths)

				target_size = min(len(v) for k, v in class_to_images.items() if k != 'Euro-American')
				balanced_image_paths = []
				balanced_labels = []
				for cls, imgs in class_to_images.items():
					sampled = random.sample(imgs, target_size) if len(imgs) > target_size else imgs
					balanced_image_paths.extend(sampled)
					balanced_labels.extend([cls] * len(sampled))

				combined = list(zip(balanced_image_paths, balanced_labels))
				random.shuffle(combined)
				balanced_image_paths, balanced_labels = zip(*combined)
				df = pd.DataFrame({'filename': balanced_image_paths, 'class': balanced_labels})

				datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
				train_data = datagen.flow_from_dataframe(
					dataframe=df, x_col='filename', y_col='class',
					target_size=img_size, batch_size=batch_size,
					class_mode='categorical', subset='training', shuffle=True
				)
				val_data = datagen.flow_from_dataframe(
					dataframe=df, x_col='filename', y_col='class',
					target_size=img_size, batch_size=batch_size,
					class_mode='categorical', subset='validation', shuffle=True
				)

				num_samples = len(df)
				num_classes = len(df['class'].unique())
				input_shape = (img_size[0], img_size[1], 3)

				kfold = KFold(n_splits=k_folds, shuffle=True)
				fold_accuracies = []

				for train_idx, val_idx in kfold.split(np.arange(num_samples)):
					model = cnn_model(input_shape, num_classes)
					history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)
					max_fold_acc = max(history.history['val_accuracy'])
					fold_accuracies.append(max_fold_acc)
					print(f"Fold accuracy: {max_fold_acc:.4f}")

				model_path = os.path.join("models/kfold", resolution_folder, method_folder)
				os.makedirs(model_path, exist_ok=True)
				model_name = f"{resolution_folder}_{method_folder}_fold_{len(fold_accuracies)}_mask_5_undersample.h5"
				model.save(os.path.join(model_path, model_name))
				print(f"Model saved to {model_path}")

				avg_acc = np.mean(fold_accuracies)
				key = f"{resolution_folder} - {method_folder}"
				cross_val_accuracies[key] = avg_acc
				print(f"Average accuracy for {key}: {avg_acc:.4f}")

# ---------- MOSAIC VERSION BELOW ----------

mosaics_dir = "C:/Users/theof/OneDrive/Documents/Github/genome_color_unpickler/res"
img_size = (20, 50)
cross_val_accuracies_mos = {}

for resolution_folder in sorted(os.listdir(mosaics_dir), key=lambda x: int(re.search(r'\d+', x).group())):
	resolution_path = os.path.join(mosaics_dir, resolution_folder)
	if os.path.isdir(resolution_path):
		print(f"\nCross-validating {resolution_folder}...")

		# UNDERSAMPLING STEP
		class_dirs = [d for d in os.listdir(resolution_path) if os.path.isdir(os.path.join(resolution_path, d))]
		class_to_images = defaultdict(list)
		for class_name in class_dirs:
			class_path = os.path.join(resolution_path, class_name)
			image_paths = glob(os.path.join(class_path, "*"))
			class_to_images[class_name].extend(image_paths)

		target_size = min(len(v) for k, v in class_to_images.items() if k != 'Euro-American')
		balanced_image_paths = []
		balanced_labels = []
		for cls, imgs in class_to_images.items():
			sampled = random.sample(imgs, target_size) if len(imgs) > target_size else imgs
			balanced_image_paths.extend(sampled)
			balanced_labels.extend([cls] * len(sampled))

		combined = list(zip(balanced_image_paths, balanced_labels))
		random.shuffle(combined)
		balanced_image_paths, balanced_labels = zip(*combined)
		df = pd.DataFrame({'filename': balanced_image_paths, 'class': balanced_labels})

		datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
		train_data = datagen.flow_from_dataframe(
			dataframe=df, x_col='filename', y_col='class',
			target_size=img_size, batch_size=batch_size,
			class_mode='categorical', subset='training', shuffle=True
		)
		val_data = datagen.flow_from_dataframe(
			dataframe=df, x_col='filename', y_col='class',
			target_size=img_size, batch_size=batch_size,
			class_mode='categorical', subset='validation', shuffle=True
		)

		num_samples = len(df)
		num_classes = len(df['class'].unique())
		input_shape = (img_size[0], img_size[1], 3)

		kfold = KFold(n_splits=k_folds, shuffle=True)
		fold_accuracies = []

		for train_idx, val_idx in kfold.split(np.arange(num_samples)):
			model = cnn_model(input_shape, num_classes)
			history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)
			max_fold_acc = max(history.history['val_accuracy'])
			fold_accuracies.append(max_fold_acc)
			print(f"Fold accuracy: {max_fold_acc:.4f}")

		model_path = os.path.join("models/kfold_mosaic", resolution_folder)
		os.makedirs(model_path, exist_ok=True)
		model_name = f"{resolution_folder}_fold_{len(fold_accuracies)}_mosaic_mask_5_undersample.h5"
		model.save(os.path.join(model_path, model_name))
		print(f"Model saved to {model_path}")

		avg_acc = np.mean(fold_accuracies)
		cross_val_accuracies_mos[resolution_folder] = avg_acc
		print(f"Average accuracy for {resolution_folder}: {avg_acc:.4f}")