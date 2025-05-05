import re
import pandas as pd

def extract_protein_clusters(file_path):
	with open(file_path, "r") as file:
		lines = file.readlines()
	
	# Find the start of the @data section
	data_start = lines.index("@data\n") + 1
	data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
	
	# Extract protein names and cluster assignments
	results = []
	for line in data_lines:
		parts = line.split(",")
		protein_match = re.search(r"'([^']+)'", parts[1])  # Extract protein name
		if protein_match:
			protein_name = protein_match.group(1)
			cluster_assignment = parts[-1].strip()
			results.append((protein_name, cluster_assignment))
	
	return results

def extract_protein_clusters_v2(file_path):
	with open(file_path, "r") as file:
		lines = file.readlines()
	
	# Find the start of the @data section
	data_start = lines.index("@data\n") + 1
	data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
	
	# Extract bacteria names and cluster assignments
	results = []
	for line in data_lines:
		parts = line.split(",")
		bacteria_name_match = re.search(r"(GCF[_\d\.]+[^,]*)", parts[1])  # Extract bacteria name
		if bacteria_name_match:
			protein_name = bacteria_name_match.group(1)
			cluster_assignment = parts[-1].strip()
			results.append((protein_name, cluster_assignment))
	
	return results

# Example usage
file_path = "scripts\weka\\result-cluster_V2.arff"
protein_clusters = extract_protein_clusters_v2(file_path)

# Print sample results
for protein, cluster in protein_clusters[:10]:
	print(f"Protein: {protein}, Cluster: {cluster}")
	
# save to a file
df = pd.DataFrame(protein_clusters, columns=["Protein", "Cluster"])
df.to_csv("scripts\weka\\clusters_bacteria.csv", index=False, sep=",")

