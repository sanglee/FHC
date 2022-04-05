
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from .ad_dataset import FedDataset
from .base_net import BaseNet


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 weight_decay: float, device: str):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.weight_decay = weight_decay
        self.device = device

    @abstractmethod
    def train(self, dataset: DataLoader, validset: DataLoader, net: BaseNet) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset: DataLoader, net: BaseNet):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass