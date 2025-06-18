import unittest
from gen_algo import crossover, mutate

class TestGenAlgo(unittest.TestCase):

	def test_crossover(self):
		parent1 = [1, 2, 3, 4, 5]
		parent2 = [5, 4, 3, 2, 1]
		child1, child2 = crossover(parent1, parent2)
		
		# Check if the children are of the same length as parents
		self.assertEqual(len(child1), len(parent1))
		self.assertEqual(len(child2), len(parent2))
		
		# Check if the children contain elements from both parents
		self.assertTrue(any(elem in parent1 for elem in child1))
		self.assertTrue(any(elem in parent2 for elem in child1))
		self.assertTrue(any(elem in parent1 for elem in child2))
		self.assertTrue(any(elem in parent2 for elem in child2))

	def test_mutate(self):
		gene = "ACGTACGT"
		mutated_gene = mutate(gene)
		
		# Check if the mutated gene is of the same length as the original gene
		self.assertEqual(len(mutated_gene), len(gene))
		
		# Check if the mutated gene differs from the original gene by exactly one nucleotide
		differences = sum(1 for a, b in zip(gene, mutated_gene) if a != b)
		self.assertEqual(differences, 1)

if __name__ == '__main__':
	unittest.main()	