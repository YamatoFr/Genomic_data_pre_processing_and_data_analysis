from parser import get_files_in_folder, parse_gene_sequences

def damerau_levenshtein(s1, s2):
	"""
	Calculate the Damerau-Levenshtein distance between two strings.
	The Damerau-Levenshtein distance is a measure of the similarity between two strings,
	which is the minimum number of operations (insertions, deletions, substitutions, and transpositions)
	required to transform one string into the other.
	Parameters:
	s1 (str): The first string.
	s2 (str): The second string.
	Returns:
	int: The Damerau-Levenshtein distance between the two strings.
	"""
	d = {}
	lenstr1 = len(s1)
	lenstr2 = len(s2)
	
	for i in range(-1, lenstr1 + 1):
		d[(i, -1)] = i + 1
	for j in range(-1, lenstr2 + 1):
		d[(-1, j)] = j + 1

	for i in range(lenstr1):
		for j in range(lenstr2):
			if s1[i] == s2[j]:
				cost = 0
			else:
				cost = 1
			d[(i, j)] = min(
						   d[(i - 1, j)] + 1,  # deletion
						   d[(i, j - 1)] + 1,  # insertion
						   d[(i - 1, j - 1)] + cost,  # substitution
						  )
			if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
				d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

	return d[lenstr1 - 1, lenstr2 - 1]

def compare_gene_sequences(file1, file2):
	"""
	Compare gene sequences from two files using the Damerau-Levenshtein distance.
	Args:
		file1 (str): Path to the first file containing gene sequences.
		file2 (str): Path to the second file containing gene sequences.
	Returns:
		list of tuples: A list of tuples where each tuple contains:
			- gene_id1 (str): Gene ID from the first file.
			- gene_id2 (str): Gene ID from the second file.
			- distance (int): Damerau-Levenshtein distance between the two gene sequences.
	"""
	
	gene_sequences_file1 = parse_gene_sequences(file1)
	gene_sequences_file2 = parse_gene_sequences(file2)

	result = []
	for gene_id1, gene_sequence1, gene_id2, gene_sequence2 in zip(
		gene_sequences_file1.keys(), gene_sequences_file1.values(),
		gene_sequences_file2.keys(), gene_sequences_file2.values()):
			distance = damerau_levenshtein(gene_sequence1, gene_sequence2)
			result.append((gene_id1, gene_id2, distance))
	
	return result


# # Example usage
# folder_path = "D:\genes\TUBERCULOSE\Prokka7kFFN_FINAL"

# files = get_files_in_folder(folder_path)

# for i in range(len(files)):
# 	for j in range(i + 1, len(files)):
# 		result = compare_gene_sequences(files[i], files[j])
# 		print(result)