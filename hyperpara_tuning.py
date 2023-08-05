import os
import numpy as np
import functools
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from data_utils import load_data
from model import ModelCNN


def train(config, data_dir, csv_file, checkpoint_dir=None):
    train_loader, val_loader, _ = load_data(config, data_dir, csv_file)

    model = ModelCNN(config['l1'], config['l2'])
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)


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

        train_loss = round(train_loss / (len(train_loader)), 3)

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

        val_loss = round(val_loss / len(val_loader), 3)
        val_accuracy = round(val_correct / val_total, 3)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(train_loss=train_loss, val_accuracy=val_accuracy, val_loss=val_loss)


def hyperparameter_tuning(data_dir, csv_file):
    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(5, 9)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(5, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    reporter = CLIReporter(
        metric_columns=["train_loss", "val_accuracy", "val_loss"],
        max_progress_rows=10
    )

    scheduler = ASHAScheduler(max_t=100, grace_period=1)

    train_func = functools.partial(train, data_dir=data_dir, csv_file=csv_file)

    analysis = tune.run(
        train_func,
        config=config,
        metric="val_accuracy",
        mode="max",
        resources_per_trial={'cpu': 20, 'gpu': 1},
        num_samples=3,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="hyperparameter_tuning"
    )
    
    best_trial = analysis.get_best_trial("val_loss", "min", "last")

    return best_trial
