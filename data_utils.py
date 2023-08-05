import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# create the train dataset
class NucleusTypeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.nuclei_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.nuclei_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.nuclei_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.nuclei_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
def load_data(config, data_dir, csv_file):
    transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = NucleusTypeDataset(csv_file=csv_file, root_dir= data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    return train_loader, val_loader, loader