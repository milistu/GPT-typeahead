import unittest
from qteller.GPTeller import GPTeller


class GPTellerTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.teller = GPTeller()
        self.num_return_sequences = 3
        self.compleations = self.teller.complete(
            query="a red",
            context="car dealership",
            max_length=5,
            num_return_sequences=self.num_return_sequences,
        )

    def test_num_return_sequences(self):
        self.assertEqual(len(self.compleations), self.num_return_sequences)


if __name__ == "__main__":
    unittest.main()
