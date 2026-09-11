class TGSDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load grayscale image (101x101)
        # Convert 1-channel to 3-channel for standard ResNet50 backbone compatibility
        img_path = self.image_paths[idx]
        image = torch.load(img_path) if img_path.endswith('.pt') else TF.to_tensor(Image.open(img_path).convert('RGB'))
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # Expand 1-channel grayscale to 3 channels

        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = torch.load(mask_path) if mask_path.endswith('.pt') else TF.to_tensor(Image.open(mask_path).convert('L'))
            mask = (mask > 0.5).float()  # Binary mask (0: Background/Sediment, 1: Salt)
        else:
            mask = torch.zeros((1, image.shape[1], image.shape[2]))

        # Apply research transform pipeline
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


class TGSDataModule(LightningDataModule):
    def __init__(self, train_imgs, train_masks, val_imgs, val_masks, batch_size=16, transform=None):
        super().__init__()
        self.train_imgs = train_imgs
        self.train_masks = train_masks
        self.val_imgs = val_imgs
        self.val_masks = val_masks
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.train_dataset = TGSDataset(self.train_imgs, self.train_masks, transform=self.transform)
        self.val_dataset = TGSDataset(self.val_imgs, self.val_masks, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)