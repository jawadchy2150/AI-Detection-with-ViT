from torchvision import transforms
from torch.utils.data import DataLoader
from text_image_dataset import TextImageDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds = TextImageDataset(
    "metadata_exp1_split.csv",
    "triplet_output",
    split="train",
    transform=transform
)

val_ds = TextImageDataset(
    "metadata_exp1_split.csv",
    "triplet_output",
    split="val",
    transform=transform
)

test_ds = TextImageDataset(
    "metadata_exp1_split.csv",
    "triplet_output",
    split="test",
    transform=transform
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)
