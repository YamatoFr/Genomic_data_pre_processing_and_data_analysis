# Genomic Data Pre-processing and Data Analysis

This repository contains tools and scripts for pre-processing and analyzing genomic data.

Please note that this project contains tentative code that is either, incomplete, or not fully functional.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (TensorFlow, NumPy, Pandas, Matplotlib, etc.)
- Provide your own genomic datasets

### Folder explanation

- `documents`: for storing documents generated or other relevant documents.
- `imgs`: for storing images generated or other relevant images.
- `report`: contains .tex files for writing the report (includes a .bib file for references).
- `scripts`: comes with scripts for checking references and citations in the report (inclused a PowerShell for easy use). You may also add your own scripts here.

### Includes scripts from template:

Requires bibtexparser Python package.

- `scripts/citationcheck.py`: check a given .tex file for citations and references and compare them with the present .bib file. <br>
User can specify the .tex file and chose if they want the results to be printed to the console, saved to a file, or both.
- `scripts/citationmapper.py`: check all the outputted .csv and compile them into a single .csv file (does not work).
- `scripts/launch.ps1`: PowerShell script to run the citation check script easily. User can specify the .tex file and chose if  <br>
they want the results to be printed to the console, saved to a file, or both.

### Project structure

- `scripts/clustering`: contains scripts for parsing and clustering genomic data.
	- `damereau_levenshtein.py`: script for calculating the Damerau-Levenshtein distance between sequences. [Functional, not actively used]
	- `parser.py`: parse the FASTA files and extract the sequences with their ID. Is used by other scripts.
	- `protein_counter.py`: count the number of time a protein appears in a file.
	- `gen_algo.py`: script for running a genetic algorithm on the genomic data. [Tentative]
	- `network.py/ipynb`: most of the clustering work. [Need a better name]

- `scripts/img_analysis`: contains scripts image analysis and CNNs.
	- `move-imgs.py`: moves unsorted images to the folder corresponding to their class.
	- `cnn.py`: first attempt at a CNN for image classification. Needed to manually point to the dataset.
	- `auto-cnn.ipynb`: Jupyter notebook for an automated CNN for image classification. Can automatically train a CNN on all datasets with a <br>
	parent folder containing the datasets.
		- `*k-fold*`: adds k-fold cross-validation to the CNN training.
		- `*undersample*`: adds undersampling to the CNN training.
	- `dataset-merger.ipynb`: Jupyter notebook for merging datasets into a single dataset for training. Used to merge real images with synthetic images.
	- `matrix-calc.ipynb`: Jupyter notebook for calculating metrics from the confusion matrix of a trained CNN.
- `scripts/model`: the saved models from the CNN training.