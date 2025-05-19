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
    y_KF_list = []

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
            if test:
                y_KF_list.append(y_KF)
    
    accuracy = correct / total

    if test:
        return accuracy, y_KF_list, error_distribution
    else:
        return accuracy, error_distribution

def calculate_accuracy_KF(args, model, data_loader, num_classes, y_KF_list, R_est, device):
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    alpha = 1e-6
    Sigma_pred = alpha * torch.eye(args.hidden_dim)
    R = R_est * torch.eye(args.hidden_dim).to(device) # TODO: Change for dense matrix eventually

    # y_KF = y_KF[0] # TODO: This only works when we have one reservoir block. Need to change this for multiple blocks and batches
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            y_KF = y_KF_list[idx][0]
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            outputs, y_KF_test, wt_plot = model(inputs.to(device), y_KF=y_KF.to(device), R=R.to(device), Sigma_pred=Sigma_pred.to(device))
            plot_full_time_series(
                inputs=inputs,
                w_t=wt_plot[0][0],
                w_t_update=wt_plot[0][1],
                wt_cont_list=wt_plot[0][2],
                err_list=wt_plot[0][3],
                filename=f"full_plot_{idx}.png"
            )

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

def plot_full_time_series(inputs, w_t, w_t_update, wt_cont_list, err_list, filename, save_path="time_series_plots"):
    zero_indices = np.where(np.array(w_t) == 0)[0]
    print(f"Indices where w_t is 0: {zero_indices}" if zero_indices.size else "No indices where w_t is 0")

    os.makedirs(save_path, exist_ok=True)
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()

    series = inputs[0]
    n_steps, n_dims = series.shape
    total_plots = n_dims + 4

    fig, axes = plt.subplots(total_plots, 1, figsize=(8, 4 * total_plots), sharex=True)
    axes = np.atleast_1d(axes)

    # Get default Matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot each original dimension
    for j in range(n_dims):
        axes[j].plot(series[:, j], label=f"Dim {j+1}", color=colors[j % len(colors)])
        axes[j].set_ylabel(f"Dim {j+1}")
        axes[j].legend()
        axes[j].grid()

    # w_t
    axes[n_dims].plot(w_t, marker="o", linestyle="-", label="w_t", color=colors[(n_dims + 0) % len(colors)])
    axes[n_dims].set_ylabel("w_t")
    axes[n_dims].legend()
    axes[n_dims].grid()

    # w_t_update
    axes[n_dims + 1].plot(w_t_update, marker="o", linestyle="-", label="w_t_update", color=colors[(n_dims + 1) % len(colors)])
    axes[n_dims + 1].set_ylabel("w_t_update")
    axes[n_dims + 1].legend()
    axes[n_dims + 1].grid()

    # w_t_cont
    axes[n_dims + 2].plot(wt_cont_list, marker="o", linestyle="-", label="w_t_cont", color=colors[(n_dims + 2) % len(colors)])
    axes[n_dims + 2].set_ylabel("w_t_cont")
    axes[n_dims + 2].legend()
    axes[n_dims + 2].grid()

    # err
    axes[n_dims + 3].plot(err_list, marker="o", linestyle="-", label="err", color=colors[(n_dims + 3) % len(colors)])
    axes[n_dims + 3].set_ylabel("err")
    axes[n_dims + 3].legend()
    axes[n_dims + 3].grid()

    plt.xlabel("Time Steps")
    plt.suptitle("Time Series + w_t / Correction / Error")

    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    plt.close(fig)

    print(f"Plot saved at '{save_file}'.")
