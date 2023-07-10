from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from hyperparameters import BATCHSIZE


class STDataset(Dataset):
    def __init__(self, t_input, target):
        self.input = t_input
        self.target = target

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]


inputs = torch.load('/home/igardner/hpoSiDBTransformer/denseinputs.pth')
labels = torch.load('/home/igardner/hpoSiDBTransformer/denselabels.pth')

train_inputs, tv_inputs, train_labels, tv_labels = train_test_split(inputs, labels, test_size=0.4, random_state=42)
train_dataset = STDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, pin_memory=True, num_workers=4)
test_inputs, valid_inputs, test_labels, valid_labels = train_test_split(tv_inputs, tv_labels, test_size=0.5, random_state=42)
test_dataset = STDataset(test_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=True,  num_workers=4)
valid_dataset = STDataset(valid_inputs, valid_labels)
valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=True,  num_workers=4)
