
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np

class FedDataset(BaseDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0,
                pin_memory: bool = False) -> (DataLoader, DataLoader, DataLoader):

        train_loaders = []
        valid_loaders = []
        test_loaders = []

        for traindt, validdt, testdt in zip(self.train_set, self.valid_set, self.test_set):

            train_loader = DataLoader(dataset=traindt, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
            valid_loader = DataLoader(dataset=validdt, batch_size=batch_size, shuffle=shuffle_test,
                                      num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            test_loader = DataLoader(dataset=testdt, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            train_loaders.append(train_loader)
            valid_loaders.append(valid_loader)
            test_loaders.append(test_loader)

        return train_loaders, valid_loaders, test_loaders


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)