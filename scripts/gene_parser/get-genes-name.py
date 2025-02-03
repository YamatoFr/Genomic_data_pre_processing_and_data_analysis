import re
import os



def extract_between_symbols(text):
	"""
	Extracts substrings from the input text that are located between the '>' symbol and a newline character.

	Args:
		text (str): The input text from which to extract substrings.

	Returns:
		list: A list of substrings found between the '>' symbol and a newline character.
	"""
	pattern = r'>(.*?)\n'
	matches = "\n".join(re.findall(pattern, text, re.DOTALL))
	return matches

def get_file(file_path):
	"""
	Reads the content of a file and returns it as a string.

	Args:
		file_path (str): The path to the file to read.

	Returns:
		str: The content of the file as a string.
	"""
	with open(file_path, 'r') as file:
		return file.read()

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

# Define the path to the folder containing the files
folder_path = "D:\genes\TUBERCULOSE"

# Get the paths to all files in the folder
files = get_files_in_folder(folder_path)

# Iterate over the files
for file in files:
	# Read the content of the file
	content = get_file(file)

	# Extract the gene names from the content
	gene_names = extract_between_symbols(content)

	# Print the gene names to a file
	print_to_file("D:\genes\TUBERCULOSE\genes", f"{os.path.basename(file)}.txt", gene_names)
