import random as rd
from parser import get_files_in_folder, parse_gene_sequences
from damerau_levenshtein import damerau_levenshtein

def crossover(parent1, parent2):
	"""
	Perform a crossover operation between two parent sequences to produce two offspring sequences.

	Args:
		parent1 (list): The first parent sequence.
		parent2 (list): The second parent sequence.

	Returns:
		tuple: A tuple containing two offspring sequences (child1, child2) resulting from the crossover operation.

	Example:
		>>> parent1 = [1, 2, 3, 4, 5]
		>>> parent2 = [5, 4, 3, 2, 1]
		>>> crossover(parent1, parent2)
		([1, 2, 3, 2, 1], [5, 4, 3, 4, 5])
	"""
	# Select a random crossover point
	crossover_point = rd.randint(0, len(parent1) - 1)
	# Create the first child
	child1 = parent1[:crossover_point] + parent2[crossover_point:]
	# Create the second child
	child2 = parent2[:crossover_point] + parent1[crossover_point:]
	return child1, child2

def mutate(gene):
	"""
	Mutates a given gene sequence by randomly selecting a mutation point and 
	replacing the nucleotide at that point with a randomly chosen nucleotide 
	from 'A', 'C', 'G', or 'T'.

	Args:
		gene (str): The original gene sequence to be mutated.

	Returns:
		str: A new gene sequence with one nucleotide mutated.
	"""
	# Select a random mutation point
	mutation_point = rd.randint(0, len(gene) - 1)
	# Create a new gene with the mutation
	new_gene = gene[:mutation_point] + rd.choice("ACGT") + gene[mutation_point + 1:]
	return new_gene

# folder_path = "D:\genes\TUBERCULOSE\Prokka7kFFN_FINAL"

# files = get_files_in_folder(folder_path)

# gene_sequences = parse_gene_sequences(files[0])

# # Initialize the population with random genes
# population = ["".join(rd.choices("ACGT", k=len(gene_sequences))) for _ in range(5)]

# # Perform crossover and mutation
# for i in range(5):
# 	parent1 = population[rd.randint(0, len(population) - 1)]
# 	parent2 = population[rd.randint(0, len(population) - 1)]
# 	child1, child2 = crossover(parent1, parent2)
# 	child1 = mutate(child1)
# 	child2 = mutate(child2)
# 	population.append(child1)
# 	population.append(child2)

# # Evaluate the fitness of each gene in the population
# fitness = []
# for gene in population:
# 	total_distance = 0
# 	for gene_sequence in gene_sequences.values():
# 		distance = damerau_levenshtein(gene, gene_sequence)
# 		total_distance += distance
# 	fitness.append(total_distance)

# # Select the best genes
# best_genes = [gene for gene, fit in zip(population, fitness) if fit == min(fitness)]

# print(best_genes)