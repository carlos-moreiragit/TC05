import unittest
from pathlib import Path
from src.entities.pede import Pede

class TestAddFunction(unittest.TestCase):
    
    def test_destination_path(self):
        """Testa se o detination_path está correto"""
        result = Pede.get_destination_folder
        self.assertEqual(result, r'..\passos_repo\data') #


if __name__ == '__main__':
    unittest.main() #