from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import time
from model import LeNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Preparing for Data
print('==> Preparing data..')

# Training Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Testing Data preparation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
writer = SummaryWriter("runs/regularized")

def main():
    time0 = time.time()
    # Training settings
    batch_size = 8
    epochs = 20
    lr = 1.2e-3 #6e-4
    # dropout_rate = 0.3
    save_model = False
    torch.manual_seed(100)
    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_size = int(0.8 * len(trainset))  # 80% of the dataset for training
    valid_size = len(trainset) - train_size

    train_subset, valid_subset = torch.utils.data.random_split(trainset, [train_size, valid_size])


    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

    model = LeNet(regularized=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


    train(model, device, train_loader, valid_loader, optimizer, epochs)
    test( model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"cifar_lenet.pt")
    time1 = time.time() 
    print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))   

def train(model, device, train_loader, val_loader, optimizer, num_epochs):
    
    count = 0
    criterion = nn.NLLLoss(reduction='mean')
    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        writer.add_scalar('val epoch acc', val_accs[epoch], epoch)

    print("Training finished.")
    plt.plot(range(1, num_epochs + 1), val_accs, label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('dropout tuned result.png')

def test(model, device, test_loader):
    print("Testing start.")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()
