#%%
import sys
import scipy
import matplotlib.pyplot as plt
import numpy as np
import torch
import neurokit2 as nk
import scipy.io
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml
from diffusion_model_no_compress import CSDI_base
from utils_together2 import train
from self_process import imputation_pattern, AddNoise
#%%
device = torch.device("cuda")
file_path = 'base_no_compress.yaml'
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

label = np.load('label_seg.npy')
ecg = np.load("x_II_seg.npy")
ecg_missing, mask = imputation_pattern(ecg[:,None,:].copy(), extended = True)
# ecg_missing = AddNoise(ecg[:,None,:].copy(), SNR = 15)
ecg = torch.tensor(ecg[:,None,:].copy(), dtype=torch.float32)
ecg_noisy = torch.tensor(ecg_missing, dtype=torch.float32)
cleaned_signal = []
for index in range(4):
    sig = np.load(f'IMP{index}.npy')
    sig = np.mean(sig, axis = 1)
    for i in range(sig.shape[0]):
        sig[i] = -1 + 2*(sig[i] - sig[i].min())/(sig[i].max() - sig[i].min())
    
    cleaned_signal.append(sig[:,0,:])


cleaned_signal = np.concatenate(cleaned_signal, axis = 0)


class custom_dataset(Dataset):
    def __init__(self, signal, label):
        self.signal = signal
        self.label = label
    def __len__(self):
        return len(self.signal)
    def __getitem__(self, idx):
        return self.signal[idx], self.label[idx]

train_sig, test_sig, train_label, test_label = train_test_split(ecg, label, test_size=0.2, random_state=42)
train_sig, val_sig, train_label, val_label = train_test_split(train_sig, train_label, test_size=0.2, random_state=42)

train_mask = np.logical_or(train_label == 0, train_label == 4)

train_sig = torch.tensor(train_sig[train_mask], dtype=torch.float32)
train_label = torch.tensor(train_label[train_mask], dtype=torch.float32)
train_label[train_label!=0] = 1
train_dataset = custom_dataset(train_sig, train_label)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#64

val_mask = np.logical_or(val_label == 0, val_label == 4)

val_sig = torch.tensor(val_sig[val_mask], dtype=torch.float32)
val_label = torch.tensor(val_label[val_mask], dtype=torch.float32)
val_label[val_label!=0] = 1
val_dataset = custom_dataset(val_sig, val_label)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_mask = np.logical_or(test_label == 0, test_label == 4)

test_sig = torch.tensor(test_sig[test_mask], dtype=torch.float32)
test_label = torch.tensor(test_label[test_mask], dtype=torch.float32)
test_label[test_label!=0] = 1
test_dataset = custom_dataset(test_sig, test_label)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


train_sig_noisy, test_sig_noisy, train_label_noisy, test_label_noisy = train_test_split(ecg_noisy, label, test_size=0.2, random_state=42)
test_sig_noisy = torch.tensor(test_sig_noisy[test_mask], dtype=torch.float32)
test_dataset_noisy = custom_dataset(test_sig_noisy, test_label)
test_noisy_loader = DataLoader(test_dataset_noisy, batch_size=64, shuffle=False)

diff = test_sig - test_sig_noisy 
cleaned_signal = torch.tensor(cleaned_signal, dtype=torch.float32)
CL = cleaned_signal[:,None,:][[test_mask]]
for i in range(diff.shape[0]):
    mask = diff[i,0,:]
    mask[mask> 0] = 1
    CL[i,0,:][mask != 1] = test_sig[i,0,:][mask != 1]
test_dataset_cleaned = custom_dataset(CL, test_label)
test_cleaned_loader = DataLoader(test_dataset_cleaned, batch_size=64, shuffle=False)


#%%

import torch.nn as nn

class VGG16_1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        """
        1D variant of VGG16 network adapted for sequence classification
        
        Args:
            input_channels (int): Number of input channels in the sequence data
            num_classes (int): Number of output classes for classification
        """
        super(VGG16_1D, self).__init__()
        
        # Block 1: Input → 64 channels
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 2: 64 → 128 channels
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 3: 128 → 256 channels
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 4: 256 → 512 channels
        self.block4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 5: 512 → 512 channels
        self.block5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to handle variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(7)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights
        # self._initialize_weights()
        
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Apply convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class ImprovedVGG16_1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        """
        Enhanced 1D variant of VGG16 with modern architectural improvements
        
        Improvements include:
        1. Residual connections for better gradient flow
        2. Squeeze-and-Excitation blocks for adaptive channel weighting
        3. Global context block for capturing long-range dependencies
        4. Efficient channel attention
        5. Layer scaling for stable training
        """
        super(ImprovedVGG16_1D, self).__init__()
        self.scale_factor = 1e-5  # Layer scaling factor
        
        # Squeeze-and-Excitation block
        class SEBlock(nn.Module):
            def __init__(self, channel, reduction=16):
                super(SEBlock, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1)
                return x * y.expand_as(x)
        
        # Efficient Channel Attention
        class ECABlock(nn.Module):
            def __init__(self, channel, gamma=2, b=1):
                super(ECABlock, self).__init__()
                t = int(abs((torch.log2(torch.tensor(channel)).item() + b) / gamma))
                k = t if t % 2 else t + 1
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
                self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                y = self.avg_pool(x)
                y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
                y = self.sigmoid(y)
                return x * y.expand_as(x)
        
        # Global Context Block
        class GlobalContextBlock(nn.Module):
            def __init__(self, channel):
                super(GlobalContextBlock, self).__init__()
                self.context_modeling = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channel, 1, kernel_size=1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                context = self.context_modeling(x)
                return x * context
        
        # Convolutional block with residual connection and attention
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                SEBlock(out_channels),
                ECABlock(out_channels)
            )
        
        # Block 1: Input → 64 channels with residual connection
        self.block1 = nn.ModuleList([
            conv_block(input_channels, 64),
            conv_block(64, 64)
        ])
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gc1 = GlobalContextBlock(64)
        
        # Block 2: 64 → 128 channels
        self.block2 = nn.ModuleList([
            conv_block(64, 128),
            conv_block(128, 128)
        ])
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gc2 = GlobalContextBlock(128)
        
        # Block 3: 128 → 256 channels
        self.block3 = nn.ModuleList([
            conv_block(128, 256),
            conv_block(256, 256),
            conv_block(256, 256)
        ])
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gc3 = GlobalContextBlock(256)
        
        # Block 4: 256 → 512 channels
        self.block4 = nn.ModuleList([
            conv_block(256, 512),
            conv_block(512, 512),
            conv_block(512, 512)
        ])
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gc4 = GlobalContextBlock(512)
        
        # Block 5: 512 → 512 channels
        self.block5 = nn.ModuleList([
            conv_block(512, 512),
            conv_block(512, 512),
            conv_block(512, 512)
        ])
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gc5 = GlobalContextBlock(512)
        
        # Adaptive pooling and feature aggregation
        self.adaptive_pool = nn.AdaptiveAvgPool1d(7)
        
        # Improved classifier with reduced parameters and dropout
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.3),  # Reduced dropout for better feature preservation
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, num_classes)
        )
        
        # self._initialize_weights()
        
    def forward(self, x):
        # Block 1 with residual
        identity = x
        for layer in self.block1:
            x = layer(x)
        if x.size(1) == identity.size(1):  # Channel matching check
            x = x + identity
        x = self.pool1(x)
        x = self.gc1(x)
        
        # Block 2
        for layer in self.block2:
            x = layer(x)
        x = self.pool2(x)
        x = self.gc2(x)
        
        # Block 3
        for layer in self.block3:
            x = layer(x)
        x = self.pool3(x)
        x = self.gc3(x)
        
        # Block 4
        for layer in self.block4:
            x = layer(x)
        x = self.pool4(x)
        x = self.gc4(x)
        
        # Block 5
        for layer in self.block5:
            x = layer(x)
        x = self.pool5(x)
        x = self.gc5(x)
        
        # Final processing
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x   # Layer scaling for stable training
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization for better convergence
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
precision_imputed = []
specificity_imputed = []
sensitivity_imputed = []

precision_noisy = []
specificity_noisy = []
sensitivity_noisy = []

precision_clean = []
specificity_clean = []
sensitivity_clean = []

from tqdm import tqdm
for iterations in range(10):
    epochs = 100
    lr = 1e-5
    model = VGG16_1D(input_channels=1, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 2], dtype=torch.float32))
    model = model.to(device)
    criterion = criterion.to(device)
    patience = 10
    best_loss = np.inf
    counter = 0
    for epoch in range(epochs):
        bar = tqdm(enumerate(train_loader), desc="Training")
        for batch_index, batch in bar:
            optimizer.zero_grad()
            input_sig, label = batch
            input_sig = input_sig.to(device)
            label = label.to(device)
            output = model(input_sig)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=loss.item())
        bar = tqdm(enumerate(val_loader), desc="Validation")
        val_loss = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in bar:
                input_sig, label = batch
                input_sig = input_sig.to(device)
                label = label.to(device)
                output = model(input_sig)
                loss = criterion(output, label.long())
                val_loss += loss.item()
                bar.set_postfix(loss=loss.item())
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'vgg16_1d.pth')
            counter = 0
        else:
            counter += 1
        if counter == patience:
            break
            print(1)
        bar = tqdm(enumerate(test_loader), desc="Test")
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in bar:
                input_sig, label = batch
                input_sig = input_sig.to(device)
                label = label.to(device)
                output = model(input_sig)
                loss = criterion(output, label.long())
                test_loss += loss.item()

                # Calculate accuracy
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                
                test_acc = 100. * correct / total
                bar.set_postfix({
                    'test loss': f'{test_loss/(batch_index + 1):.4f}',
                    'test acc': f'{test_acc:.2f}%'
                })

        bar = tqdm(enumerate(test_noisy_loader), desc="Test")
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in bar:
                input_sig, label = batch
                input_sig = input_sig.to(device)
                label = label.to(device)
                output = model(input_sig)
                loss = criterion(output, label.long())
                test_loss += loss.item()

                # Calculate accuracy
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                
                test_acc = 100. * correct / total
                bar.set_postfix({
                    'test loss': f'{test_loss/(batch_index + 1):.4f}',
                    'test acc': f'{test_acc:.2f}%'
                })
        
        bar = tqdm(enumerate(test_cleaned_loader), desc="Test")
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in bar:
                input_sig, label = batch
                input_sig = input_sig.to(device)
                label = label.to(device)
                output = model(input_sig)
                loss = criterion(output, label.long())
                test_loss += loss.item()

                # Calculate accuracy
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                
                test_acc = 100. * correct / total
                bar.set_postfix({
                    'test loss': f'{test_loss/(batch_index + 1):.4f}',
                    'test acc': f'{test_acc:.2f}%'
                })

    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

    def evaluate(model, data_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for ppg_batch, label_batch in data_loader:
                ppg_batch = ppg_batch.to(device)
                label_batch = label_batch.to(device)
                
                outputs = model(ppg_batch)
                _, predicted = outputs.max(1)
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label_batch.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Calculate per-class metrics
        class_report = classification_report(all_labels, all_preds, 
                                        target_names=['Normal', 'ST'],
                                        output_dict=True)
        
        # Print results
        print(f"Overall Accuracy: {acc*100:.2f}%")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nPer-class Metrics:")
        counter = 0
        for cls in ['Normal', 'ST']:
            print(f"\n{cls}:")
            # print(f"Precision: {class_report[cls]['precision']:.3f}")
            if counter == 0:
                print(f"specificity: {class_report[cls]['recall']*100:.2f}%")
            else:
                print(f"sensitivity: {class_report[cls]['recall']*100:.2f}%")
            # print(f"F1-score: {class_report[cls]['f1-score']:.3f}")
            counter+= 1
        
        return acc, conf_matrix, class_report, 
    # model.load_state_dict(torch.load("/mnt/vepfs/audio/lichang/HeartDiffuser/Heart_diffuser/vgg16_1d.pth"))
    model = model.to(device)
    test_metrics = evaluate(model, test_loader, device)
    test_metrics2 = evaluate(model, test_noisy_loader, device)
    test_metrics3 = evaluate(model, test_cleaned_loader, device)

    precision_clean.append(test_metrics[0])
    specificity_clean.append(test_metrics[2]['Normal']['recall'])
    sensitivity_clean.append(test_metrics[2]['ST']['recall'])

    precision_noisy.append(test_metrics2[0])
    specificity_noisy.append(test_metrics2[2]['Normal']['recall'])
    sensitivity_noisy.append(test_metrics2[2]['ST']['recall'])

    precision_imputed.append(test_metrics3[0])
    specificity_imputed.append(test_metrics3[2]['Normal']['recall'])
    sensitivity_imputed.append(test_metrics3[2]['ST']['recall'])
#%%
precision_clean = np.array(precision_clean)
specificity_clean = np.array(specificity_clean)
sensitivity_clean = np.array(sensitivity_clean)

precision_noisy = np.array(precision_noisy)
specificity_noisy = np.array(specificity_noisy)
sensitivity_noisy = np.array(sensitivity_noisy)

precision_imputed = np.array(precision_imputed)
specificity_imputed = np.array(specificity_imputed)
sensitivity_imputed = np.array(sensitivity_imputed)
#%%
print("Clean ACC", np.mean(precision_clean))
print("Clean Specificity",np.mean(specificity_clean))
print("Clean Sensitivity", np.mean(sensitivity_clean))

print("Noisy ACC", np.mean(precision_noisy))
print("Noisy Specificity",np.mean(specificity_noisy))
print("Noisy Sensitivity", np.mean(sensitivity_noisy))

print("Imputed ACC", np.mean(precision_imputed))
print("Imputed Specificity",np.mean(specificity_imputed))
print("Imputed Sensitivity", np.mean(sensitivity_imputed))


#%%

print("Clean Mean", np.mean(precision_clean), np.mean(specificity_clean), np.mean(sensitivity_clean))
print("Noisy Mean", np.mean(precision_noisy), np.mean(specificity_noisy), np.mean(sensitivity_noisy))
print("Imputed Mean", np.mean(precision_imputed), np.mean(specificity_imputed), np.mean(sensitivity_imputed))

print("Clean STD", np.std(precision_clean), np.std(specificity_clean), np.std(sensitivity_clean))
print("Noisy STD", np.std(precision_noisy), np.std(specificity_noisy), np.std(sensitivity_noisy))
print("Imputed STD", np.std(precision_imputed), np.std(specificity_imputed), np.std(sensitivity_imputed))

