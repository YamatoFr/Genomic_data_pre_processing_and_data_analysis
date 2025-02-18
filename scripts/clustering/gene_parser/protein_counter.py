from parser import get_files_in_folder, parse_gene_sequences, print_to_file

import pandas as pd

def count_proteins_in_file(file_path):
	"""
	Count the occurrences of each protein in a given gene sequence file.
	Args:
		file_path (str): The path to the file containing gene sequences.
	Returns:
		dict: A dictionary where the keys are protein names and the values are the counts of each protein.
	"""
	protein_counts = {}
	gene_sequences = parse_gene_sequences(file_path)
	
	for gene_id, gene_sequence in gene_sequences.items():
		protein = gene_id.partition(" ")[2]
		if protein not in protein_counts:
			protein_counts[protein] = 1
		else:
			protein_counts[protein] += 1
	
	return protein_counts

def total_proteins_count(folder_path):
	"""
	Count the total appearances of each proteins in all files.
	Args:
		folder_path (str): The path to the folder containing the count files, each file is a CSV file with the protein counts.
	Returns:
		dict: A dictionary where the keys are protein names and the values are the counts of each protein.
	"""
	
	total_counts = {}

	for files in get_files_in_folder(folder_path):
		df = pd.read_csv(files, sep=";")
		for index, row in df.iterrows():
			if row["Protein"] not in total_counts:
				total_counts[row["Protein"]] = row["Count"]
			else:
				total_counts[row["Protein"]] += row["Count"]
    
	return total_counts

def to_csv(output_folder, protein_counts, file_name):
	"""
	Save protein counts to a CSV file.
	Args:
		output_folder (str): The path to the folder where the CSV file should be saved.
		protein_counts (dict): A dictionary where the keys are protein names and the values are the counts of each protein.
		file_name (str): The name of the file to save the protein counts.
	"""
	
	filename = file_name.split(".")[0]
	
	print_to_file(
		output_folder, filename + ".csv",
		pd.DataFrame(
			protein_counts.items(),
			columns=["Protein", "Count"]).to_csv(index=False, sep=";"))
 
# Define the path to the folder containing gene sequences
# folder_path = "D:\genes\TUBERCULOSE\Prokka7kFFN_FINAL"
output_folder = "documents\\tables"

# # Get the paths to all files in the folder
# files = get_files_in_folder(folder_path)

# # Count the proteins in each file and save the results to a CSV file
# for file_path in files:
# 	file_name = file_path.split("\\")[-1].split(".")[0] + '_' + file_path.split("\\")[-1].split(".")[1]
# 	protein_counts = count_proteins_in_file(file_path)
# 	# to_csv("documents\\tables", protein_counts, file_name)

# Count the total appearances of each proteins in all files and save the results to a CSV file
total_proteins = total_proteins_count(output_folder)

to_csv(output_folder, total_proteins, "total_proteins_count.csv")

def remove_blank_lines_from_csv(file_path):
	"""
	Remove blank lines from a CSV file.
	Args:
		file_path (str): The path to the CSV file.
	"""
	df = pd.read_csv(file_path, sep=";")
	df = df.dropna(how='all')
	df.to_csv(file_path, index=False, sep=";")

for files in get_files_in_folder(output_folder):
	remove_blank_lines_from_csv(files)