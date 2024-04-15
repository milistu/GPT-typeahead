from abc import ABC, abstractmethod
from typing import List


class TellerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def encode(self):
        """Method to convert query and context to tokens."""
        pass

    @abstractmethod
    def decode(self):
        """Method to convert model output tokens to text."""
        pass

    @abstractmethod
    def complete(self) -> List[str]:
        """Method to perform query completion."""
        pass
