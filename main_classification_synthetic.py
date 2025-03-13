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
from torchsummary import summary
from helpers import calculate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='PyTorch Transformer on Time series forecasting')

parser.add_argument('--batch_size', default=10, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--eval_batch_size', default=-1, type=int,
                    help='eval_batch_size default is equal to training batch_size')
parser.add_argument('--nlayers', default=4, type=int,
                    help='number of layers')
parser.add_argument('--epoch', default=1000, type=int,
                    help='epoch (default: 20)')
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
parser.add_argument('--encode_dim', default=2, type=int, # 2 to match input_size
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

    
        num_features = 2

        freq_min=10 
        freq_max=500 
        num_classes=10
    
        dataset = SinusoidalDataset(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = SinusoidalDataset(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        # Create test dataset and loader
        test_dataset = SinusoidalDataset(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
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
    global_step = 0

    start_time = time.time()

    # Training loop
    for epoch in range(args.epoch):            
        epoch_loss = 0.0  # Variable to store the total loss for the epoch
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()  # Add the loss of the current batch to the epoch loss

            global_step += 1
            train_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_accuracy, _ = calculate_accuracy(args, model, val_loader, num_classes)
        test_accuracy, _ = calculate_accuracy(args, model, test_loader, num_classes)

        model.train()

        val_accuracy_list.append(val_accuracy)
        test_accuracy_list.append(test_accuracy)
        
        print(f'Epoch {epoch + 1}/{args.epoch}, Loss: {loss.item():.4f}, Valid Accuracy: {val_accuracy * 100:.2f}%')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main(hyperp_tuning=args.hyperp_tuning)