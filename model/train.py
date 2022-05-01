from dataset import AlzhimerDataset
import torch
import torch.nn as nn
from model import ConvNetwork
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")


batch_size = 10
lr = 0.0001
num_epochs = 72


transform = transforms.Compose(
    [transforms.ToTensor()])


train = AlzhimerDataset(
    "../data/train.csv", "../data/Images", transform=transform)
test = AlzhimerDataset(
    "../data/test.csv", "../data/Images", transform=transform)
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

classes = ("Non demented", "Very mild demented",
           "Mild demented", "Moderate demented")
print("Data loaded")

model = ConvNetwork()
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_his = []
n_total_steps = len(train_loader)
print("Start training")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 512 == 0:
            loss_his.append(loss.cpu().detach().numpy())
            print(
                f"Epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}")
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    n_samples += labels.shape[0]
                    n_correct += (predictions == labels).sum().item()
                acc = 100 * n_correct/n_samples
                print(f"accuracy={acc:.4f}")
print("Training finished")
torch.save(model.state_dict(), "./trained_model.pt")
iters = range(1, len(loss_his)+1)
plt.plot(iters, loss_his, 'r--')
plt.plot(iters, loss_his, 'b-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print("Trained model saved")
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
