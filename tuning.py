from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LeNet
import numpy as np

torch.manual_seed(100)

def main():
    config = {
        #The lr (learning rate) should be uniformly sampled between 0.0001 and 0.1. Lastly, the batch size is a choice between 2, 4, 8, and 16.
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([8]),
        "prob": tune.quniform(0.2, 0.5, 0.05)
    }

    scheduler = ASHAScheduler( #use the ASHAScheduler which will terminate bad performing trials early
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_cifar),
            resources={"cpu": 2, "gpu": 2}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy", #Now Ray Tune search algorithm tries to minimize loss that is used to session.report
            mode="max", #mode: Must be one of [min, max]. Determines whether objective is minimizing or maximizing
            scheduler=scheduler,
            num_samples=8,
        ),
        param_space=config,
        run_config=air.RunConfig(local_dir="./results", name="test_experiment")
    )
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["accuracy"]))


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ])

    trainset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    
    train_size = int(0.8 * len(trainset))  # 80% of the dataset for training
    valid_size = len(trainset) - train_size
    train_subset, valid_subset = torch.utils.data.random_split(trainset, [train_size, valid_size])

    testset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return train_subset, valid_subset

def train_cifar(config):
    trainset, testset = load_data()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        testset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet(config["prob"])
    
    model.to(device)

    criterion = nn.NLLLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=5e-4)

    train_losses = []
    val_losses = []

    for epoch in range(10):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)

        # Compute loss for validation set
        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(valloader)
        val_acc = correct / total

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Val Acc: {val_acc:.2f}" )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        session.report({"loss": (val_loss), "accuracy": val_acc})

if __name__ == "__main__":
    main()