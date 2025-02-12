import unittest
from unittest.mock import patch
from damerau_levenshtein import damerau_levenshtein, compare_gene_sequences

class TestDamerauLevenshtein(unittest.TestCase):

	def test_damerau_levenshtein(self):
		self.assertEqual(damerau_levenshtein("kitten", "sitting"), 3)
		self.assertEqual(damerau_levenshtein("flaw", "lawn"), 2)
		self.assertEqual(damerau_levenshtein("gumbo", "gambol"), 2)
		self.assertEqual(damerau_levenshtein("book", "back"), 2)
		self.assertEqual(damerau_levenshtein("", ""), 0)
		self.assertEqual(damerau_levenshtein("a", ""), 1)
		self.assertEqual(damerau_levenshtein("", "a"), 1)
		self.assertEqual(damerau_levenshtein("a", "a"), 0)
		self.assertEqual(damerau_levenshtein("a", "b"), 1)
		self.assertEqual(damerau_levenshtein("ab", "ba"), 1)

	@patch('damerau_levenshtein.parse_gene_sequences')
	def test_compare_gene_sequences(self, mock_parse_gene_sequences):
		mock_parse_gene_sequences.side_effect = [
			{'gene1': 'AGCT', 'gene2': 'CGTA'},
			{'gene1': 'AGTT', 'gene2': 'CGTT'}
		]
		
		result = compare_gene_sequences('file1', 'file2')
		expected_result = [('gene1', 'gene1', 1), ('gene2', 'gene2', 1)]
		
		self.assertEqual(result, expected_result)

if __name__ == '__main__':
	unittest.main()