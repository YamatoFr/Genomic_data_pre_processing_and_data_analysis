import os

def parse_gene_sequences(file_path):
	"""
	Parses a file containing gene sequences in FASTA format.
	Args:
		file_path (str): The path to the file containing gene sequences.
	Returns:
		dict: A dictionary where the keys are the gene IDs and the values are the gene sequences.
	"""
	gene_sequences = {}
	with open(file_path) as file:
		gene_id = None
		gene_sequence = ""
		for line in file:
			if line.startswith(">"):
				if gene_id is not None:
					gene_sequences[gene_id] = gene_sequence
				gene_id = line.strip()[1:]
				gene_sequence = ""
			else:
				gene_sequence += line.strip()
		gene_sequences[gene_id] = gene_sequence

	return gene_sequences

def get_files_in_folder(folder_path):
	"""
	Explores the folder and its subfolders to find all files.

	Args:
		folder_path (str): The path to the folder to explore.

	Returns:
		list: A list of paths to all files found in the folder and its subfolders.
	"""
	files = []
	for root, directories, filenames in os.walk(folder_path):
		for filename in filenames:
			files.append(os.path.join(root, filename))

	return files

def print_to_file(file_path, file_name, content):
	"""
	Writes the content to a file.

	Args:
		file_path (str): The path to the directory where the file should be saved.
		file_name (str): The name of the file.
		content (str): The content to write to the file.
	"""
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	with open(os.path.join(file_path, file_name), 'w') as file:
		file.write(content)

# Example usage

# Define the path to the folder containing gene sequences
# folder_path = "D:\genes\TUBERCULOSE\Prokka7kFFN_FINAL"

# # Get the paths to all files in the folder
# files = get_files_in_folder(folder_path)

# # Parse the gene sequences from each file
# for file_path in files[:1]:
# 	gene_sequences = parse_gene_sequences(file_path)
# 	for gene_id, _ in gene_sequences.items():
# 		print(gene_id)