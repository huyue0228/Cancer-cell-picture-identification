import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from data_utils import load_data
from model import ModelCNN


def train_model(config, data_dir, csv_file):
    train_loader, val_loader, _ = load_data(config, data_dir, csv_file)
    
    model = ModelCNN(config["l1"], config["l2"])    
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
   
    train_lossSet = []
    val_lossSet = []
    val_accuSet = []
    for epoch in range(10):
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        train_loss = round(train_loss / len(train_loader), 3)
        train_lossSet.append(train_loss)
        

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = round(train_loss / len(val_loader), 3)
        val_accuracy = round(val_correct / val_total, 3)
        val_lossSet.append(val_loss)
        val_accuSet.append(val_accuracy)

        print("train_loss:",train_loss, " val_loss:",val_loss," val_acc:",val_accuracy)

    plt.figure()
    plt.plot(train_lossSet, color='b', label="Training loss")
    plt.plot(val_lossSet, color='r', label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.xticks(range(len(train_lossSet)))  # show every epoch
    plt.grid()
    legend = plt.legend(shadow=True)


    plt.figure()
    plt.plot(val_accuSet, color='r',label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.xticks(range(len(val_accuSet)))  # show every epoch
    plt.grid()
    legend = plt.legend(shadow=True)

    return model