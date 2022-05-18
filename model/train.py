from dataset import AlzheimerDataset
import torch
import torch.nn as nn
from model import ConvNetwork
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as transforms
import matplotlib.pyplot as plt
import os
from tools import accuracy_loss, accuracy, print_final_accuracy, save_graphs

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
    acc_train = accuracy(
        train_loader, model, device)
    loss_train = sum(loss_val)/len(loss_val)
    acc_test, loss_test = accuracy_loss(
        test_loader, model, device, criterion)
    acc_train_his.append(acc_train)
    acc_test_his.append(acc_test)
    train_loss_his.append(loss_train)
    test_loss_his.append(loss_test)
    print(
        f"Epoch: {epoch+1}/{num_epochs}, loss_train: {loss_train:.10f}, loss_test: {loss_test:.10f}, accuracy_train: {acc_train:.4f}, accuracy_test: {acc_test:.4f}")
print("Training finished")
torch.save(model.state_dict(), "../results/trained_model.pt")
save_graphs(train_loss_his, test_loss_his, acc_test_his, acc_train_his)
print("Trained model saved")
print_final_accuracy(test_loader, device, model, batch_size, classes)
