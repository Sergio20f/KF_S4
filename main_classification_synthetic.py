import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import *
import time
from helpers import calculate_accuracy, calculate_accuracy_KF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch Transformer on Time series forecasting')

parser.add_argument('--batch_size', default=1, type=int, # Changed from 10
                    help='mini-batch size (default: 64)')
parser.add_argument('--eval_batch_size', default=-1, type=int,
                    help='eval_batch_size default is equal to training batch_size')
parser.add_argument('--nlayers', default=4, type=int,
                    help='number of layers')
parser.add_argument('--epoch', default=1, type=int, # Changed
                    help='epoch (default: 1)')
parser.add_argument('--lr',default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay',default=0, type=float,
                    help='weight_decay')
parser.add_argument("--dataset", default='sinusoidal',type=str,
                    help="Dataset you want to train")
parser.add_argument('--optimizer',default="Adam", type=str,
                    help='Choice BERTAdam or Adam')
parser.add_argument("--pretrained_model_path", default='',type=str,
                    help="location of the dataset to keep trainning")
parser.add_argument('--hyperp_tuning', default=False, action='store_true',
                    help='whether to perform hyperparameter tuning')
parser.add_argument("--model", default='lstm',type=str,
                    help="Model you want to train. [linear-RNN, LSTM]")
parser.add_argument('--hidden_dim', default=128, type=int, 
                    help='number of layers')
# Added
parser.add_argument('--encode_dim', default=128, type=int, # 2 to match input_size # 128
                    help='encode_dim')
parser.add_argument('--mlp_hidden_dim', default=64, type=int,
                    help='mlp_hidden_dim')
parser.add_argument('--mlp_num_layers', default=1, type=int,
                    help='mlp_num_layers')

global args
args = parser.parse_args()

def main(hyperp_tuning=False):

    print(args)

    # Create dataset and data loader
    if(args.eval_batch_size ==-1):
        eval_batch_size = args.batch_size
    else:
        eval_batch_size = args.eval_batch_size
    
    if (args.dataset == 'sinusoidal'):
        # Hyperparameters for the dataset and dataloader
        num_samples = 1000
        seq_length = 100
        seq_length_orig = seq_length

        num_features = 2 # TODO: Does this make the seq_length = seq_length/num_features?

        freq_min=10 
        freq_max=500 
        num_classes=10
    
        dataset = SinusoidalDataset(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = SinusoidalDataset(int(num_samples/4), seq_length, num_features, freq_min, freq_max, num_classes)
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        # test_dataset = SinusoidalDataset(int(num_samples/2), seq_length, num_features, freq_min, freq_max, num_classes, add_outlier=5, outlier_factor=5)
        test_dataset = SinusoidalDataset(1, seq_length, num_features, freq_min, freq_max, num_classes, add_outlier=10, outlier_factor=5)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    
    elif (args.dataset == 'sinusoidal_long'):
        # Hyperparameters for the dataset and dataloader
        num_samples = 1000
        seq_length = 100

        num_features = 2

        freq_min=10 
        freq_max=500 
        num_classes=10

        dataset = SinusoidalDatasetLong(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = SinusoidalDatasetLong(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        # Create test dataset and loader
        test_dataset = SinusoidalDatasetLong(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    else:
        raise ValueError('Dataset not supported')
    
    print('Num features: ', num_features)
  
    # Initialize the model, loss function, and optimizer
    if(args.model == 'lstm'):
        model = LSTM(input_size=num_features, hidden_size=args.hidden_dim, num_layers=args.nlayers, batch_first=True, num_classes=num_classes).to(device)
    elif(args.model == 'linear-RNN'):
        model = LinearRNN(input_size=num_features, hidden_size=args.hidden_dim, num_layers=args.nlayers, batch_first=True, num_classes=num_classes).to(device)
    
    elif(args.model == 'reservoir-linear-RNN'):
        model = ReservoirLinearRNN(input_size=num_features, encode_dim=args.encode_dim, hidden_size=args.hidden_dim,
                                   mlp_hidden_dim=args.mlp_hidden_dim, mlp_num_layers=args.mlp_num_layers, num_layers=args.nlayers,
                                   num_classes=num_classes).to(device)
    else:
        raise ValueError('Model not supported')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_list = []
    val_accuracy_list = []
    test_accuracy_list = []

    start_time = time.time()

    # Training loop
    for epoch in range(args.epoch):            
        epoch_loss = 0.0  # Variable to store the total loss for the epoch
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            
            outputs = model(inputs) # 2 outputs if reservoir-linear-RNN
            if args.model == 'reservoir-linear-RNN':
                outputs = outputs[0]
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()  # Add the loss of the current batch to the epoch loss

            train_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_accuracy, _ = calculate_accuracy(model, val_loader, num_classes)
        test_accuracy, _ = calculate_accuracy(model, test_loader, num_classes)

        model.train()

        val_accuracy_list.append(val_accuracy)
        test_accuracy_list.append(test_accuracy)
        
        print(f'Epoch {epoch + 1}/{args.epoch}, Loss: {loss.item():.4f}, Valid Accuracy: {val_accuracy * 100:.2f}%')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training time: {execution_time:.2f} seconds")

    if (args.model == 'reservoir-linear-RNN'):
        # TODO: After training, we will need to carry out a forward pass with the test data to get the y_KFs for the KF-based model
        model.eval()
        # Let us estimate the diag values of R from a validation dataset
        _, R_est, _ = calculate_accuracy(model, val_loader, num_classes, test=True)
        R_est = torch.var(R_est[0])
        print(f'R_est: {R_est.cpu().numpy():.2f}')

        print("Forward pass on test data to collect y_KFs multiple times")
        for i in range(1):
            test_accuracy, y_KF, _ = calculate_accuracy(model, test_loader, num_classes, test=True)
            print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

        print("Calculating accuracy using KF-based model multiple times")
        for i in range(3):
            test_accuracy_KF, _ = calculate_accuracy_KF(args, model, test_loader, num_classes, y_KF, R_est, device)
            print(f'Test Accuracy KF: {test_accuracy_KF * 100:.2f}%')
            test_accuracy_list.append(test_accuracy_KF*100)

if __name__ == '__main__':
    main()