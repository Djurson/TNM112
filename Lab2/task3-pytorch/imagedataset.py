from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        img = self.x[idx]
        if self.transform:
            img = self.transform(img)
        if self.y is not None:
            return img, self.y[idx]
        return img