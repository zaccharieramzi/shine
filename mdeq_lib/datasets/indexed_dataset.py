from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.internal_dataset = dataset

    def __getitem__(self, index):
        data = self.internal_dataset[index]
        return (*data, index)

    def __len__(self):
        return len(self.internal_dataset)
