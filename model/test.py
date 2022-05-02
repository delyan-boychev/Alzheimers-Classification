from dataset import AlzheimerDataset
import torch
import torch.nn as nn
from model import ConvNetwork
from torchsummary import summary
from torch.utils.data import DataLoader
import os
import torchvision.transforms.transforms as transforms

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
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
else:
    print("Model not trained")
