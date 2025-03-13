import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(args, model, data_loader, num_classes):
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            outputs = model(inputs).to(device)
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
