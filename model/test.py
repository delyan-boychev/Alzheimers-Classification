from dataset import AlzheimerDataset
import torch
import torch.nn as nn
from model import ConvNetwork
from torchsummary import summary
from torch.utils.data import DataLoader
import os
import torchvision.transforms.transforms as transforms
from tools import print_final_accuracy

if os.path.exists("../results/trained_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    batch_size = 10

    transform = transforms.Compose(
        [transforms.ToTensor()])

    test = AlzheimerDataset(
        "../data/test.csv", "../data/Images", transform=transform)
    test_loader = DataLoader(
        dataset=test, batch_size=batch_size, shuffle=False)

    model = ConvNetwork().to(device)
    model.load_state_dict(torch.load("../results/trained_model.pt"))
    classes = ("Non demented", "Very mild demented",
               "Mild demented", "Moderate demented")
    print(summary(model, input_size=(1, 128, 128)))
    print_final_accuracy(test_loader, device, model, batch_size, classes)
else:
    print("Model not trained")
