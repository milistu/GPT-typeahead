from abc import ABC, abstractmethod


class TellerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self):
        """Method to perform query completion."""
        pass
