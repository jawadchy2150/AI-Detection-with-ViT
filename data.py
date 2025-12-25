from torchvision import transforms
from torch.utils.data import DataLoader
from text_image_dataset import TextImageDataset

def get_dataloaders(
    csv_file,
    image_dir,
    batch_size=32,
    num_workers=4
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = TextImageDataset(
        csv_file, image_dir, split="train", transform=transform
    )
    val_ds = TextImageDataset(
        csv_file, image_dir, split="val", transform=transform
    )
    test_ds = TextImageDataset(
        csv_file, image_dir, split="test", transform=transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader