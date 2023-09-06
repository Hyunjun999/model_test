import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from customset import Fashionset
from tqdm import tqdm


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setting
    model = mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, 22)
    model.load_state_dict(torch.load("./MobileNetV2_77.pt"))

    # Test transforms with normalization
    test_transforms = A.Compose(
        [
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    test_dataset = Fashionset("./test(WH)/", transform=test_transforms)

    # Increase batch size if GPU memory allows
    batch_size = 1
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test acc : {100 * correct / total:.2f}%")


if __name__ == "__main__":
    test()
