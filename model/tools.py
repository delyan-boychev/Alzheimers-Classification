import torch
import matplotlib.pyplot as plt


def accuracy_loss(dataset_loader, model, device, criterion):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        loss_dataset_val = []
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_dataset_val.append(criterion(
                outputs, labels).cpu().detach().numpy())
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
        acc = 100 * n_correct/n_samples
        loss = sum(loss_dataset_val)/len(loss_dataset_val)
        return acc, loss


def accuracy(dataset_loader, model, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
        acc = 100 * n_correct/n_samples
        return acc


def print_final_accuracy(test_loader, device, model, batch_size, classes):
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


def save_graphs(train_loss_his, test_loss_his, acc_test_his, acc_train_his):
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
