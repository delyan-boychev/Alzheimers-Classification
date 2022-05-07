from dataset import AlzheimerDataset
import torch
import torch.nn as nn
from model import ConvNetwork
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")
if not os.path.exists("../results"):
    os.mkdir("../results")

batch_size = 10
lr = 0.0001
num_epochs = 100


transform = transforms.Compose(
    [transforms.ToTensor()])


train = AlzheimerDataset(
    "../data/train.csv", "../data/Images", transform=transform)
test = AlzheimerDataset(
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
train_loss_his = []
test_loss_his = []
acc_train_his = []
acc_test_his = []
n_total_steps = len(train_loader)
print("Start training")
for epoch in range(num_epochs):
    loss_val = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_val.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 512 == 0:
            l = sum(loss_val)/len(loss_val)
            train_loss_his.append(l)
            print(
                f"Epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_total_steps}, loss: {l:.10f}")
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                loss_test_val = []
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss_test_val.append(criterion(
                        outputs, labels).cpu().detach().numpy())
                    _, predictions = torch.max(outputs, 1)
                    n_samples += labels.shape[0]
                    n_correct += (predictions == labels).sum().item()
                acc = 100 * n_correct/n_samples
                acc_test_his.append(acc)
                test_loss_his.append(sum(loss_test_val)/len(loss_test_val))
                print(f"accuracy_test={acc:.4f}")
                n_correct = 0
                n_samples = 0
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    n_samples += labels.shape[0]
                    n_correct += (predictions == labels).sum().item()
                acc = 100 * n_correct/n_samples
                acc_train_his.append(acc)
                print(f"accuracy_train={acc:.4f}")
print("Training finished")
torch.save(model.state_dict(), "../results/trained_model.pt")
iters = range(1, len(train_loss_his)+1)
plt.plot(iters, train_loss_his, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("../results/loss_train.png")
plt.clf()
plt.plot(iters, test_loss_his, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("../results/loss_test.png")
plt.clf()
plt.plot(iters, test_loss_his, 'r-', label="Test loss")
plt.plot(iters, train_loss_his, 'b-', label="Train loss")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("../results/loss_cmp.png")
plt.clf()
plt.plot(iters, acc_test_his, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("../results/acc.png")
plt.clf()
plt.plot(iters, acc_test_his, 'r-', label="Test accuracy")
plt.plot(iters, acc_train_his, 'b-', label="Train accuracy")
plt.legend()
plt.xlabel('Epoch')
plt.savefig("../results/acc_cmp.png")
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
