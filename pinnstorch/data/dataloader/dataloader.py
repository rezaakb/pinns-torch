import torch


class PINNDataLoader:
    """Custom DataLoader for the PINN datasets.

    This allows you to have a fast dataloader.
    """

    def __init__(self, dataset, batch_size=None, ignore=False, shuffle=False):
        """Initialize a PINNDataLoader.

        :param dataset: The dataset to load data from.
        :param batch_size: The batch size for the dataloader.
        :param ignore: Whether to ignore incomplete batches (default is False).
        :param shuffle: Whether to shuffle the dataset (default is False).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = len(self.dataset)
        self.shuffle = shuffle
        self.current_index = 0
        self.ignore = ignore

        if self.shuffle:
            self.indices = torch.randperm(self.dataset_size)
        else:
            self.indices = torch.arange(self.dataset_size)

    def __len__(self):
        """Get the number of batches in the dataloader.

        :return: The number of batches.
        """
        if self.batch_size is None:
            return 1
        if self.ignore:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size // self.batch_size) + 1

    def __iter__(self):
        """Initialize the data loader iterator.

        :return: The data loader iterator.
        """
        self.current_index = 0
        return self

    def __next__(self):
        """Generate the next batch of data.

        :return: The next batch of data.
        """
        if self.current_index >= len(self.indices):
            raise StopIteration

        # If batch_size is None, return the entire dataset as a single batch
        if self.batch_size is None:
            batch = self.dataset[:]
            self.current_index += self.dataset_size
            return batch

        batch_indices = self.indices[self.current_index : self.current_index + self.batch_size]
        batch = self.dataset[batch_indices]
        self.current_index += self.batch_size

        return batch
