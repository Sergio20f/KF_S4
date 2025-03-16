import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(model, data_loader, num_classes, test=False):
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            if test:
                outputs, y_KF = model(inputs)
                outputs = outputs.to(device)
            else:
                outputs, _ = model(inputs)
                outputs = outputs.to(device)

            predicted_labels = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            # Calculate error distribution
            incorrect_labels = labels[predicted_labels != labels]
            unique_labels = torch.unique(incorrect_labels)
            error_counts = torch.bincount(incorrect_labels, minlength=num_classes)
            error_distribution = {label.item(): count.item() for label, count in zip(unique_labels, error_counts) if count > 0}
    
    accuracy = correct / total

    if test:
        return accuracy, y_KF, error_distribution
    else:
        return accuracy, error_distribution

def calculate_accuracy_KF(args, model, data_loader, num_classes, y_KF, device):
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    alpha = 1e-2
    Sigma_pred = alpha * torch.eye(args.hidden_dim)

    y_KF = y_KF[0] # TODO: This only works when we have one reservoir block. Need to change this for multiple blocks.
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            R = torch.var(y_KF).to(device) * torch.eye(args.hidden_dim).to(device) # TODO: Change for dense matrix eventually
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            outputs, _ = model(inputs.to(device), y_KF=y_KF.to(device), R=R.to(device), Sigma_pred=Sigma_pred.to(device))

            predicted_labels = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            # Calculate error distribution
            incorrect_labels = labels[predicted_labels != labels]
            unique_labels = torch.unique(incorrect_labels)
            error_counts = torch.bincount(incorrect_labels, minlength=num_classes)
            error_distribution = {label.item(): count.item() for label, count in zip(unique_labels, error_counts) if count > 0}

    accuracy = correct / total
    return accuracy, error_distribution

def init_reservoir_matrix(hidden_size):
    W = torch.randn(hidden_size, hidden_size)
    Q, R = torch.linalg.qr(W)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q
