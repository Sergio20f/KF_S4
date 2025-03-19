import os
import torch
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(model, data_loader, num_classes, test=False, verbose=False):
    """
    TODO: The way we collect y_KF does not work for multiple batches (it overwrites the previous batch).
    """
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            if test:
                outputs, y_KF, _ = model(inputs, y_KF=None)
                outputs = outputs.to(device)
            else:
                outputs, _, _ = model(inputs, y_KF=None)
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

def calculate_accuracy_KF(args, model, data_loader, num_classes, y_KF, R_est, device):
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    alpha = 1e-6
    Sigma_pred = alpha * torch.eye(args.hidden_dim)
    R = R_est * torch.eye(args.hidden_dim).to(device) # TODO: Change for dense matrix eventually

    y_KF = y_KF[0] # TODO: This only works when we have one reservoir block. Need to change this for multiple blocks and batches
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            outputs, y_KF_test, wt_plot = model(inputs.to(device), y_KF=y_KF.to(device), R=R.to(device), Sigma_pred=Sigma_pred.to(device))
            plot_full_time_series(
                inputs=inputs,
                w_t=wt_plot[0][0],
                w_t_update=wt_plot[0][1],
                filename=f"full_plot_{idx}.png"
            )
            # plot_and_store_time_series(inputs, filename=f"input_{idx}.png")
            # plot_w(wt_plot[0][0], filename=f"wt_{idx}.png")
            # plot_w(wt_plot[0][1], ylabel="w_t_update", title="w_t_update values", filename=f"wt_update_{idx}.png")

            predicted_labels = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            # Calculate error distribution
            if predicted_labels == labels:
                print("this is correct")
            else:
                print("this is incorrect")
            print("*" * 50)
            incorrect_labels = labels[predicted_labels != labels]
            unique_labels = torch.unique(incorrect_labels)
            error_counts = torch.bincount(incorrect_labels, minlength=num_classes)
            error_distribution = {label.item(): count.item() for label, count in zip(unique_labels, error_counts) if count > 0}

    accuracy = correct / total
    return accuracy, error_distribution, y_KF_test

def init_reservoir_matrix(hidden_size):
    print("Initializing reservoir matrix")
    W = torch.randn(hidden_size, hidden_size)
    Q, R = torch.linalg.qr(W)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def plot_full_time_series(inputs, w_t, w_t_update, filename, save_path="time_series_plots"):
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Convert `inputs` to NumPy array if it's a PyTorch tensor
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()

    # Extract time series from batch dimension
    series = inputs[0]  # Shape: (N, D)
    n_steps, n_dims = series.shape  # Get time steps and dimensions

    # Create figure with (n_dims + 2) subplots
    fig, axes = plt.subplots(n_dims + 2, 1, figsize=(8, 4 * (n_dims + 2)), sharex=True)

    # Ensure `axes` is always a list (even if there's only one subplot)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Plot the time series (first n_dims subplots)
    for j in range(n_dims):
        axes[j].plot(series[:, j], label=f"Dimension {j+1}")
        axes[j].set_ylabel(f"Dim {j+1}")
        axes[j].legend()
        axes[j].grid()

    # Plot w_t (next subplot)
    axes[n_dims].plot(w_t, marker="o", linestyle="-", label="w_t values", color="tab:blue")
    axes[n_dims].set_ylabel("w_t")
    axes[n_dims].legend()
    axes[n_dims].grid()

    # Plot w_t_update (last subplot)
    axes[n_dims + 1].plot(w_t_update, marker="o", linestyle="-", label="w_t_update values", color="tab:red")
    axes[n_dims + 1].set_ylabel("w_t_update")
    axes[n_dims + 1].legend()
    axes[n_dims + 1].grid()

    # Add shared x-axis label
    plt.xlabel("Time Steps")
    plt.suptitle("Time Series and w_t Plots")

    # Save the figure
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    plt.close(fig)  # Free memory

    print(f"Plot saved at '{save_file}'.")
