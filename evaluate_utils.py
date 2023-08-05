import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import torch.cuda as cuda
from data_utils import load_data


def evaluate(model, config, data_dir, csv_file):
    _, _, loader = load_data(config, data_dir, csv_file)
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    # Get actual and predicted values
    y_actual=[]
    y_pred=[]
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            size = labels.size(0)
            for i in range(0,size):
                y_actual.append(int(labels[i]))
                y_pred.append(int(predicted[i]))

    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_actual, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3])
    cm_display.plot()

    plt.show()