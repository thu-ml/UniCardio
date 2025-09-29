#%%
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.nn.functional as F
from ts2vg import NaturalVG
import yaml
from sklearn.metrics import confusion_matrix

#%%
all_results = []
all_labels = []
all_truth = []
ppg = []
# Load and process data
for i in range(0, 18):
    results_name = f'results_batch_large_{i}.pt'
    truth_name = f'input_batch_large{i}.pt'
    labels_name = f'labels_batch_large{i}.pt'
    if i == 0:
        print(torch.load(results_name).shape)

    mean_results = torch.load(results_name)[:,0:100,0,None,:].mean(dim=1)[:,0,:].detach().cpu().numpy()
    truth = torch.load(truth_name)[:,0,1000:1500].detach().cpu().numpy()
    labels = torch.load(labels_name).detach().cpu().numpy()

    all_results.append(mean_results)
    all_truth.append(truth)
    all_labels.append(labels)
    ppg1 = torch.load(truth_name)[:,0,0:500].detach().cpu().numpy()
    ppg.append(ppg1)

all_results = np.concatenate(all_results, axis=0)
all_truth = np.concatenate(all_truth, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
ppg = np.concatenate(ppg, axis=0)

all_labels = torch.nn.functional.one_hot(torch.tensor(all_labels, dtype=int), num_classes=2).squeeze(dim=1)
all_results = torch.tensor(all_results, dtype=torch.float32)
all_truth = torch.tensor(all_truth, dtype=torch.float32)
ppg = torch.tensor(ppg, dtype=torch.float32)

#%%
import scipy

def detect_errors_advanced(ppg_signals, slope_threshold=0.55, var_threshold=0.1, low_freq_threshold=0.5, low_freq_ratio_threshold=0.5, dominant_freq_threshold = 0.8):
    """
    Detect errors in PPG signals using advanced statistical measures and frequency analysis.

    Parameters:
    ppg_signals (list of np.array): List of PPG signals, each signal should be a 1D numpy array.
    slope_threshold (float): Threshold for detecting abnormal slopes.
    var_threshold (float): Threshold for detecting outliers based on the variance of the signal.
    low_freq_threshold (float): Threshold for detecting low-frequency content.
    low_freq_ratio_threshold (float): Ratio of low-frequency content to detect abnormal signals.

    Returns:
    list of bool: A list indicating whether each signal is erroneous (True) or not (False).
    """
    error_flags = []
    
    for signal in ppg_signals:
        # Slope-based detection
        '''
        slopes = np.diff(signal)
        if np.any(np.abs(slopes) > slope_threshold):
            error_flags.append(True)
            continue
        error_flags.append(False)
        '''
        # Variance-based detection
        signal2 = signal - signal.mean()
        var_value = np.var(signal2)
        if var_value < var_threshold:
            error_flags.append(True)
            continue
        '''
        # Low-frequency content detection
        freqs, power_spectrum = scipy.signal.periodogram(signal)
        low_freq_power = np.sum(power_spectrum[freqs < low_freq_threshold])
        total_power = np.sum(power_spectrum)
        
        if (low_freq_power / total_power) > low_freq_ratio_threshold:
            error_flags.append(True)
            continue
        '''
        freqs, power_spectrum = scipy.signal.periodogram(signal2, fs = 125)
        dominant_freq = freqs[np.argmax(power_spectrum)]
        if dominant_freq < dominant_freq_threshold:
            error_flags.append(True)
            continue

        # If no errors detected
        error_flags.append(False)
        
    
    return error_flags

error_flags = detect_errors_advanced(ppg.numpy())
print(error_flags)
index = np.zeros((len(error_flags), 1))
index[error_flags] = 1
print(index.sum())
print(np.where(index==1))

loc, v = np.where(index==1)
all_results = torch.tensor(np.delete(all_results.numpy(), loc, axis = 0), dtype = torch.float32) 
all_truth = torch.tensor(np.delete(all_truth.numpy(), loc, axis = 0), dtype = torch.float32) 
all_labels = torch.tensor(np.delete(all_labels.numpy(), loc, axis = 0), dtype = torch.float32) 
#%%
class VGGEcg(nn.Module):
    def __init__(self):
        super(VGGEcg, self).__init__()
        # Branch for the first modality (truth)
        self.conv1_1_truth = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv1_2_truth = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool1_truth = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv2_1_truth = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv2_2_truth = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2_truth = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv3_1_truth = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3_2_truth = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_truth = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool3_truth = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv4_1_truth = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv4_2_truth = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_truth = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.pool4_truth = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv5_1_truth = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv5_2_truth = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv5_3_truth = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool5_truth = nn.AvgPool1d(kernel_size=2, stride=2)

        # Branch for the second modality (mean_results)
        self.conv1_1_results = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv1_2_results = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool1_results = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv2_1_results = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv2_2_results = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2_results = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv3_1_results = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3_2_results = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_results = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool3_results = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv4_1_results = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv4_2_results = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_results = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.pool4_results = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv5_1_results = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv5_2_results = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv5_3_results = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool5_results = nn.AvgPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 15, 1024)  # Adjust the size here based on input size after conv and pooling
        self.fc1_2 = nn.Linear(256 * 15*2, 2048)  # Adjust the size here based on input size after conv and pooling
        self.fc1_3 = nn.Linear(2048, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)
        self.layer_norm_truth = nn.LayerNorm([3840])
        self.layer_norm_results = nn.LayerNorm([3840])
        
        
    def forward(self, truth, results):
        # Process truth modality
        truth = F.relu(self.conv1_1_truth(truth))
        truth = F.relu(self.conv1_2_truth(truth))
        truth = self.pool1_truth(truth)
        
        truth = F.relu(self.conv2_1_truth(truth))
        truth = F.relu(self.conv2_2_truth(truth))
        truth = self.pool2_truth(truth)
        
        truth = F.relu(self.conv3_1_truth(truth))
        truth = F.relu(self.conv3_2_truth(truth))
        truth = F.relu(self.conv3_3_truth(truth))
        truth = self.pool3_truth(truth)
        
        truth = F.relu(self.conv4_1_truth(truth))
        truth = F.relu(self.conv4_2_truth(truth))
        truth = F.relu(self.conv4_3_truth(truth))
        truth = self.pool4_truth(truth)
        
        truth = F.relu(self.conv5_1_truth(truth))
        truth = F.relu(self.conv5_2_truth(truth))
        truth = F.relu(self.conv5_3_truth(truth))
        truth = self.pool5_truth(truth)
        
        # Process results modality
        results = F.relu(self.conv1_1_results(results))
        results = F.relu(self.conv1_2_results(results))
        results = self.pool1_results(results)
        
        results = F.relu(self.conv2_1_results(results))
        results = F.relu(self.conv2_2_results(results))
        results = self.pool2_results(results)
        
        results = F.relu(self.conv3_1_results(results))
        results = F.relu(self.conv3_2_results(results))
        results = F.relu(self.conv3_3_results(results))
        results = self.pool3_results(results)
        
        results = F.relu(self.conv4_1_results(results))
        results = F.relu(self.conv4_2_results(results))
        results = F.relu(self.conv4_3_results(results))
        results = self.pool4_results(results)
        
        results = F.relu(self.conv5_1_results(results))
        results = F.relu(self.conv5_2_results(results))
        results = F.relu(self.conv5_3_results(results))
        results = self.pool5_results(results)
        truth = truth.view(truth.size(0), -1)
        results = results.view(results.size(0), -1)
        
        results = self.layer_norm_results(results)
        truth = self.layer_norm_truth(truth)
        x = truth
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Example usage
device = torch.device('cuda:1')
model = VGGEcg().to(device)
#%%

from sklearn.model_selection import KFold
import torch.optim as optim
from sklearn.metrics import f1_score
ACC = []
SPC = []
SENS = []
for iteration in range(10):
    kf = KFold(n_splits=10, shuffle=True, random_state = 42)
    fold = 0

    all_fold_accuracies = []
    all_fold_f1_scores = []
    all_fold_spec = []
    all_fold_sens = []
    # 10-Fold Cross-Validation
    for train_index, test_index in kf.split(all_results):
        fold += 1
        print(f'Fold {fold}')

        X_train_results, X_test_results = all_results[train_index], all_results[test_index]
        X_train_truth, X_test_truth = all_truth[train_index], all_truth[test_index]
        y_train, y_test = all_labels.type(torch.float32)[train_index], all_labels.type(torch.float32)[test_index]

        # Further split train into train and validation sets
        X_train_results, X_val_results, y_train, y_val = train_test_split(X_train_results, y_train, test_size=0.1, random_state=42)
        X_train_truth, X_val_truth = train_test_split(X_train_truth, test_size=0.1, random_state=42)

        # Create data loaders
        batch_size = 128
        train_dataset = TensorDataset(X_train_truth, X_train_results, y_train)
        val_dataset = TensorDataset(X_val_truth, X_val_results, y_val)
        test_dataset = TensorDataset(X_test_truth, X_test_results, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model, loss, and optimizer
        model = VGGEcg().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        best_val_loss = float('inf')
        best_test_accuracy = 0.0
        best_test_f1 = 0.0
        best_test_spec = 0.0
        best_test_sens = 0.0

        num_epochs = 25
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs_truth, inputs_results, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs_truth.unsqueeze(dim=1).to(device), inputs_results.unsqueeze(dim=1).to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs_truth, inputs_results, labels in val_loader:
                    outputs = model(inputs_truth.unsqueeze(dim=1).to(device), inputs_results.unsqueeze(dim=1).to(device))
                    loss = criterion(outputs, labels.to(device))
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted.detach().cpu() == labels.max(1)[1]).sum().item()
            val_accuracy = 100 * correct / total
            val_loss = val_loss / len(val_loader)
            print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')

            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.eval()
                correct = 0
                total = 0
                all_labels_test = []
                all_predictions = []
                with torch.no_grad():
                    for inputs_truth, inputs_results, labels in test_loader:
                        outputs = model(inputs_truth.unsqueeze(dim=1).to(device), inputs_results.unsqueeze(dim=1).to(device))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted.detach().cpu() == labels.max(1)[1]).sum().item()
                        all_labels_test.extend(labels.max(1)[1].cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                test_accuracy = 100 * correct / total
                test_f1 = f1_score(all_labels_test, all_predictions, average='macro')
                print(f'Best model updated. Test Accuracy: {test_accuracy}%, Test F1 Score: {test_f1}')


                # Inside the test evaluation loop where all_labels_test and all_predictions are defined
                conf_matrix = confusion_matrix(all_labels_test, all_predictions)

                if conf_matrix.shape == (1, 1):  # Only one class is present in the data
                    if np.unique(all_labels_test) == 0:  # All labels are 0
                        TN = conf_matrix[0, 0]
                        specificity = 1.0  # Specificity is 100%
                    elif np.unique(all_labels_test) == 1:  # All labels are 1
                        TP = conf_matrix[0, 0]
                        sensitivity = 1.0  # Sensitivity is 100%
                elif conf_matrix.shape == (2, 2):  # Both classes are present
                    # Extract True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
                    TN = conf_matrix[0, 0]
                    FP = conf_matrix[0, 1]
                    FN = conf_matrix[1, 0]
                    TP = conf_matrix[1, 1]
                    # Calculate Sensitivity (Recall) and Specificity
                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

                print(f'Sensitivity (Recall): {sensitivity}')
                print(f'Specificity: {specificity}')

                best_test_accuracy = test_accuracy
                best_test_f1 = test_f1
                best_test_spec = specificity
                best_test_sens = sensitivity

                all_fold_accuracies.append(best_test_accuracy)
                all_fold_f1_scores.append(best_test_f1)
                all_fold_spec.append(specificity)
                all_fold_sens.append(sensitivity)


    # Average accuracy and F1 score across folds
    average_accuracy = np.mean(all_fold_accuracies)
    average_f1_score = np.mean(all_fold_f1_scores)
    average_spec = np.mean(all_fold_spec)
    average_sens = np.mean(all_fold_sens)

    print(f'Average Test Accuracy: {average_accuracy}%')
    print(f'Test Accuracy STD: { np.std(all_fold_accuracies)}%')
    print(f'Average Test F1 Score: {average_f1_score}')
    print(f'Test F1 Score STD: {np.std(all_fold_f1_scores)}')
    print(f'Average Test spec Score: {average_spec}')
    print(f'Test spec Score STD: {np.std(all_fold_spec)}')
    print(f'Average Test Sens Score: {average_sens}')
    print(f'Test Sens Score STD: {np.std(all_fold_sens)}')
    ACC.append(average_accuracy)
    SPC.append(average_spec)
    SENS.append(average_sens)

ACC = np.array(ACC)
SPC = np.array(SPC)
SENS = np.array(SENS)
print(f'Average Test Accuracy: {ACC.mean()}%')
print(f'STD Test Accuracy: {ACC.std()}%')
print(f'Average Test spec Score: {SPC.mean()}')
print(f'STD Test spec Score: {SPC.std()}')
print(f'Average Test Sens Score: {SENS.mean()}')
print(f'STD Test Sens Score: {SENS.std()}')
np.save('ACC_e.npy', ACC)
np.save('SPC_e.npy', SPC)
np.save('SENS_e.npy', SENS)
