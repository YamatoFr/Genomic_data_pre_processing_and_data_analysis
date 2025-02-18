import matplotlib.pyplot as plt
import numpy as np

import os, sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.cluster import KMeans

from parser import get_files_in_folder, parse_gene_sequences
from sklearn.svm import SVC

# Load your genomic data
# Assuming you have a folder containing gene sequences in text format
folder_path = "D:\genes\TUBERCULOSE\Prokka7kFFN_FINAL"

print("Loading genomic data...")

# Get the paths to all files in the folder
files = get_files_in_folder(folder_path)

# Parse the gene sequences from each file
gene_sequences = []

for file_path in files[:5]:
	gene_sequences = parse_gene_sequences(file_path)
	
# Convert the gene sequences to a matrix of one-hot encoded vectors
# The matrix will have shape (n_sequences, sequence_length, n_features)
# where n_sequences is the number of gene sequences
sequence_length = max(len(seq) for seq in gene_sequences.values())
n_features = 4  # A, C, G, T

def one_hot_encode(sequence, sequence_length):
	encoding = np.zeros((sequence_length, n_features), dtype=int)
	for i, nucleotide in enumerate(sequence):
		if nucleotide == 'A':
			encoding[i, 0] = 1
		elif nucleotide == 'T':
			encoding[i, 1] = 1
		elif nucleotide == 'C':
			encoding[i, 2] = 1
		elif nucleotide == 'G':
			encoding[i, 3] = 1
	return encoding

encoded_sequences = np.array([one_hot_encode(seq, sequence_length) for seq in gene_sequences.values()])

# Flatten the sequences for clustering
flattened_sequences = encoded_sequences.reshape(len(encoded_sequences), -1)

# Standardize the data
scaler = StandardScaler()
flattened_sequences = scaler.fit_transform(flattened_sequences)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(flattened_sequences)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(2,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

print("Training neural network model...")

model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
model.fit(pca_result, pca_result, epochs=100, batch_size=32)

model.predict(pca_result)

# Apply KMeans clustering to the encoded sequences
kmeans = KMeans(n_clusters=7)
clusters = kmeans.fit_predict(flattened_sequences)


# Plot the PCA-reduced data with cluster assignments
plt.figure(figsize=(10, 8))

for cluster in np.unique(clusters):
	plt.scatter(pca_result[clusters == cluster, 0], pca_result[clusters == cluster, 1], label=f'Cluster {cluster}')

plt.legend()
plt.show()

# Get the encoded sequences from the last hidden layer
hidden_layer_model = Sequential()
hidden_layer_model.add(Dense(128, activation='relu', input_shape=(2,)))
hidden_layer_model.add(Dense(64, activation='relu'))
hidden_layer_model.add(Dense(32, activation='relu'))
hidden_layer_model.add(Dense(16, activation='relu'))
hidden_layer_model.add(Dense(8, activation='relu'))

hidden_layer_model.set_weights(model.get_weights()[:10])

encoded_sequences = hidden_layer_model.predict(pca_result)

# Apply KMeans clustering to the encoded sequences
kmeans = KMeans(n_clusters=7)
clusters = kmeans.fit_predict(encoded_sequences)

# Plot the PCA-reduced data with cluster assignments
plt.figure(figsize=(10, 8))

for cluster in np.unique(clusters):
	plt.scatter(pca_result[clusters == cluster, 0], pca_result[clusters == cluster, 1], label=f'Cluster {cluster}')

plt.legend()
plt.show()