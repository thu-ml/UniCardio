#%%
import time
import os
# os.chdir('/root/autodl-tmp/BioDiffuser')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.nn.functional as F
from ts2vg import NaturalVG
import networkx as nx
import timm
import torchvision.models as models


device = torch.device("cuda:1")


all_results = []
all_labels = []
all_truth = []

# Load and process data
for i in range(0, 20):
    results_name = f'results_batch_mimic{i}.pt'
    truth_name = f'input_batch_mimic{i}.pt'
    
    if i == 0:
        print(torch.load(results_name).shape)

    mean_results = torch.load(results_name).median(dim=1).values[:,0,:].detach().cpu().numpy()
    truth = torch.load(truth_name)[:,0,:].detach().cpu().numpy()

    all_results.append(mean_results)
    all_truth.append(truth)

all_results = np.concatenate(all_results, axis=0)
all_truth = np.concatenate(all_truth, axis=0)

all_results = torch.tensor(all_results, dtype=torch.float32)
all_truth = torch.tensor(all_truth, dtype=torch.float32)

BP_truth = all_truth[:,500:1000]*50+100

PPG = all_truth[:,0:500]
ECG = all_truth[:,1000:1500]
ECG_generated = all_results


#%%

PPG_train, PPG_temp = train_test_split(
    PPG, test_size=16000, random_state=42
)
PPG_val, PPG_test = train_test_split(
    PPG_temp, test_size=8000, random_state=42
)

ECG_train, ECG_temp = train_test_split(
    ECG, test_size=16000, random_state=42
)
ECG_val, ECG_test = train_test_split(
    ECG_temp, test_size=8000, random_state=42
)

BP_train, BP_temp = train_test_split(
    BP_truth, test_size=16000, random_state=42
)
BP_val, BP_test = train_test_split(
    BP_temp, test_size=8000, random_state=42
)


#%%
from torch.utils.data import Dataset
class Image_dataset(Dataset):
    def __init__(self, ppg, ecg, BP):
        self.ppg = ppg
        self.ecg = ecg
        self.BP = BP

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        
        return self.ppg[None, idx], self.ecg[None, idx], torch.concat([torch.max(self.BP[idx,None], dim = 1)[0], torch.min(self.BP[idx,None], dim = 1)[0]], dim = 0)

train_set = Image_dataset(PPG_train, ECG_train, BP_train)
train_loader = DataLoader(train_set, batch_size=128, pin_memory=True)


val_set = Image_dataset(PPG_val, ECG_val, BP_val)
val_loader = DataLoader(val_set, batch_size=64, pin_memory=True, shuffle=True)

test_set = Image_dataset(PPG_test, ECG_test, BP_test)
test_loader = DataLoader(test_set, batch_size=64, pin_memory=True)

#%%
class ConvLSTMModel(nn.Module):
    def __init__(self):
        super(ConvLSTMModel, self).__init__()
        
        # Conv1D layers for ECG signals
        self.conv1d_ecg = nn.Sequential(
            nn.BatchNorm1d(1),  # BatchNorm for input normalization
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=4, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Conv1D layers for PPG signals
        self.conv1d_ppg = nn.Sequential(
            nn.BatchNorm1d(1),  # BatchNorm for input normalization
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=4, stride=2),
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=28, hidden_size=1, num_layers=4, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(16),  # For LSTM output + 2 R-to-R intervals
            nn.Linear(16, 8),  # 32 from LSTM output + 2 for R-to-R interval features
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)  # Single output for SBP or DBP
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(16),  # For LSTM output + 2 R-to-R intervals
            nn.Linear(16, 8),  # 32 from LSTM output + 2 for R-to-R interval features
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),  
        )

    def forward(self, x_ppg, x_ecg):
        # Process ECG and PPG signals through their respective Conv1D layers
        x_ecg = self.conv1d_ecg(x_ecg)
        x_ppg = self.conv1d_ppg(x_ppg)
        # Interleave ECG and PPG features
        x = torch.concat([x_ecg, x_ppg], dim = 2)
        # Prepare the input for the LSTM layer
        x, _ = self.lstm(x)
        # Concatenate the LSTM output with R-to-R interval features
        #x = torch.cat((x, rr_intervals), dim=1)

        # Pass through the fully connected layers
        x1 = self.fc1(x[:,:,0])
        x2 = self.fc2(x[:,:,0])
        return torch.concat([x1,x2], dim = 1)
    
#%%
import torch.optim as optim
from tqdm import tqdm

# Instantiate model, criterion, and optimizer
model = ConvLSTMModel()
criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train(model, train_loader, criterion1, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    
    for i, (inputs_ppg, inputs_ecg, labels) in loop:
        inputs_ppg, inputs_ecg, labels =inputs_ppg.to(device), inputs_ecg.to(device), labels.to(device)  
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs_ppg, inputs_ecg)
        loss = criterion1(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update tqdm loop with the loss value
        loop.set_description(f"Training Batch [{i+1}/{len(train_loader)}]")
        loop.set_postfix(loss=loss.item())
    
    return running_loss / len(train_loader)

# Validation loop
def validate(model, val_loader, criterion2, device):
    model.eval()
    val_loss = 0.0
    loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    
    with torch.no_grad():
        for i, (inputs_ppg, inputs_ecg, labels) in loop:
            inputs_ppg, inputs_ecg, labels =inputs_ppg.to(device), inputs_ecg.to(device), labels.to(device)  
            
            # Forward pass
            outputs = model(inputs_ppg, inputs_ecg)
            loss = criterion2(outputs, labels)
            
            val_loss += loss.item()
            
            # Update tqdm loop with the loss value
            loop.set_description(f"Validation Batch [{i+1}/{len(val_loader)}]")
            loop.set_postfix(loss=loss.item())
    
    return val_loss / len(val_loader)

# Test loop

# Test loop
def test(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    sys = 0
    dys = 0
    numbers = 0
    with torch.no_grad():
        for i, (inputs_ppg, inputs_ecg, labels) in loop:
            inputs_ppg, inputs_ecg, labels =inputs_ppg.to(device), inputs_ecg.to(device), labels.to(device)  
            
            # Forward pass
            outputs = model(inputs_ppg, inputs_ecg)
            print(outputs.shape)
            print(labels.shape)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())
            
            sys += criterion2(outputs[0,0], labels[0,0])
            dys += criterion2(outputs[0,1], labels[0,1])
            numbers += 1
            # Update tqdm loop with a basic message
            loop.set_description(f"Testing Batch [{i+1}/{len(test_loader)}]")
    
    return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0), sys/numbers, dys/numbers

# Example training process
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

import csv 
csv_file_path = 'training_log_gen.csv'

# Check if the CSV file exists; if not, create it and write the header
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train(model, train_loader, criterion2, optimizer, device)
    val_loss = validate(model, val_loader, criterion2, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_loss])

SYS = []
DYS = []
for iteration in range(10):
    model = ConvLSTMModel().to(device)
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, criterion2, optimizer, device)
        val_loss = validate(model, val_loader, criterion2, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, val_loss])
        
    # Test the model after training
    print("Testing the model...")
    predictions, actuals, sys, dys = test(model, test_loader, device)
    print(sys)
    print(dys)
    SYS.append(sys)
    DYS.append(dys)

SYS = torch.stack(SYS)    
DYS = torch.stack(DYS)    
print("sys mean:", torch.mean(SYS))
print("sys std:", torch.std(SYS))
print("dys mean:", torch.mean(DYS))
print("dys std:", torch.std(DYS))