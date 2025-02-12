import numpy as np

import os, sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.cluster import KMeans

from parser import get_files_in_folder, parse_gene_sequences
from damerau_levenshtein import damerau_levenshtein
from gen_algo import crossover, mutate
from keras.utils import to_categorical

# Load your genomic data
# Assuming you have a folder containing gene sequences in text format
folder_path = "D:\genes\TUBERCULOSE\Prokka7kFFN_FINAL"

print("Loading genomic data...")

# Get the paths to all files in the folder
files = get_files_in_folder(folder_path)

# Parse the gene sequences from each file
gene_sequences = []
for file_path in files[:1]:
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
pca = PCA(n_components=100)
reduced_data = pca.fit_transform(flattened_sequences)

# Use Damerau-Levenshtein distance to calculate pairwise distances
distances = np.zeros((len(gene_sequences), len(gene_sequences)))
for i, seq1 in enumerate(gene_sequences.values()):
	for j, seq2 in enumerate(gene_sequences.values()):
		distances[i, j] = damerau_levenshtein(seq1, seq2)

# Apply genetic algorithm operations
population = list(gene_sequences.values())
for generation in range(100):  # Number of generations
	# Selection
	population.sort(key=lambda seq, i=i, pop=population: np.mean([distances[i, j] for j, seq2 in enumerate(pop) if seq != seq2]))
	population = population[:len(population)//2]
	
	# Crossover
	offspring = []
	for i in range(len(population)//2):
		parent1, parent2 = population[2*i], population[2*i+1]
		child1, child2 = crossover(parent1, parent2)
		offspring.extend([child1, child2])
	
	# Mutation
	for i in range(len(offspring)):
		offspring[i] = mutate(offspring[i])
	
	population.extend(offspring)

# Convert the final population to one-hot encoded vectors
final_encoded_sequences = np.array([one_hot_encode(seq, sequence_length) for seq in population])

# Flatten the sequences for clustering
final_flattened_sequences = final_encoded_sequences.reshape(len(final_encoded_sequences), -1)

# Standardize the data
final_flattened_sequences = scaler.transform(final_flattened_sequences)

# Apply PCA for dimensionality reduction
final_reduced_data = pca.transform(final_flattened_sequences)

print("Building neural network model...")

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(100,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

print("Training neural network model...")

model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
model.fit(final_reduced_data, final_reduced_data, batch_size=32, epochs=500, verbose=0)

# Get the cluster assignments
clusters = KMeans(n_clusters=7).fit_predict(final_reduced_data)

import matplotlib.pyplot as plt

# Plot the PCA-reduced data with cluster assignments
plt.figure(figsize=(10, 8))
scatter = plt.scatter(final_reduced_data[:, 0], final_reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('PCA of Genomic Data with KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()