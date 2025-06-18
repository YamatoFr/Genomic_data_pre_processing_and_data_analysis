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
	
	for gene_id, _ in gene_sequences.items():
		protein = gene_id
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

def all_to_table(folder_path):
	"""
	Count the total appearances of each proteins in all files and save the results to a table as a CSV file.
	The table has the following structure:
	| Protein name | file 1 | file 2 | ... | file n |
	|--------------|--------|--------|-----|--------|
	| protein 1    | count  | count  | ... | count  |
	| protein 2    | count  | count  | ... | count  |
	| ...          | ...    | ...    | ... | ...    |
	| protein n    | count  | count  | ... | count  |
	If a protein does not appear in a file, the count should be 0.

	Args:
		folder_path (str): The path to the folder containing the count files, each file is a CSV file with the protein counts.
	"""
	
	total_proteins = total_proteins_count(folder_path)
	
	# Create a table with the protein names as the first column
	table = pd.DataFrame(list(total_proteins.keys()), columns=["Protein"])
	
	# Add a column for each file in the folder
	for file_path in get_files_in_folder(folder_path):
		file_name = file_path.split("\\")[-1].split(".")[0]
		df = pd.read_csv(file_path, sep=";")
		table = table.merge(df, on="Protein", how="left").fillna(0)
		table = table.rename(columns={"Count": file_name})
	
	# Save the table to a CSV file
	print_to_file(folder_path, "total_proteins_table.csv", table.to_csv(index=False, sep=";"))

def all_to_table_v2(folder_path):
	"""
	Count the total appearances of each proteins in all files and save the results to a table as a CSV file.
	The table has the following structure:
	| Bacteria name | protein 1 | protein 2 | ... | protein n |
	|---------------|-----------|-----------|-----|-----------|
	| file 1        | count     | count     | ... | count     |
	| file 2        | count     | count     | ... | count     |
	| ...           | ...       | ...       | ... | ...       |
	| file n        | count     | count     | ... | count     |
 
	If a protein does not appear in a file, the count should be 0.

	Args:
		folder_path (str): The path to the folder containing the count files, each file is a CSV file with the protein counts.
	"""

	total_proteins = total_proteins_count(folder_path)
	
	# Create a table with the bacteria names as the first column, and the protein names as the rest of the columns
	# the bacteria names are the names of the files in the folder
	table = pd.DataFrame(columns=["Bacteria"] + list(total_proteins.keys()))

	# with the protein counts for each bacteria
	for file_path in get_files_in_folder(folder_path):
		file_name = file_path.split("\\")[-1].split(".")[0]
		df = pd.read_csv(file_path, sep=";")
		file_proteins = {row["Protein"]: row["Count"] for index, row in df.iterrows()}
		table.loc[len(table)] = [file_name] + [file_proteins.get(protein, 0) for protein in total_proteins.keys()]

	
	# Save the table to a CSV file
	print_to_file(folder_path, "total_proteins_table_v2.csv", table.to_csv(index=False, sep=";"))


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
 
def remove_blank_lines_from_csv(file_path):
	"""
	Remove blank lines from a CSV file.
	Args:
		file_path (str): The path to the CSV file.
	"""
	df = pd.read_csv(file_path, sep=";")
	df = df.dropna(how='all')
	df.to_csv(file_path, index=False, sep=";")

 
# Define the path to the folder containing gene sequences
# folder_path = "D:\genes\TUBERCULOSE\Prokka7kFFN_FINAL"
# output_folder = "documents\\tables"

# # Get the paths to all files in the folder
# files = get_files_in_folder(folder_path)

# all_to_table_v2(output_folder)

# Count the proteins in each file and save the results to a CSV file
# for file_path in files[:1]:
# 	file_name = file_path.split("\\")[-1].split(".")[0] + '_' + file_path.split("\\")[-1].split(".")[1]
# 	protein_counts = count_proteins_in_file(file_path)
# 	# to_csv("documents\\tables", protein_counts, file_name)

# Count the total appearances of each proteins in all files and save the results to a CSV file
# total_proteins = total_proteins_count(output_folder)

# to_csv(output_folder, total_proteins, "total_proteins_count.csv")

# remove_blank_lines_from_csv("documents\\tables\\total_proteins_table_v2.csv")

# # convert all float to int, ignore the first column and header
# df = pd.read_csv("documents\\tables\\total_proteins_table.csv", sep=";")
# df = df.astype(int, errors='ignore')
# df.to_csv("documents\\tables\\total_proteins_table_clean.csv", index=False, sep=";")

# print(pd.read_csv("scripts\clustering\gene_parser\dataset.csv", sep=";"))

# remove hypothetical proteins column from the dataset
df = pd.read_csv("scripts\clustering\gene_parser\dataset_V2 copy.csv", sep=";")
df = df.drop(columns=["putative protein"])
df.to_csv("scripts\clustering\gene_parser\dataset_V2 copy.csv", index=False, sep=";")
