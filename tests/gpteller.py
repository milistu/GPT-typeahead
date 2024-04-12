import unittest
from qteller.GPTeller import GPTeller


class GPTellerTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.gpteller = GPTeller()

    def test_complet(self):
        self.assertEqual(
            len(self.gpteller.complete(query="Hello Alien", num_return_sequences=5)), 5
        )


if __name__ == "__main__":
    unittest.main()
