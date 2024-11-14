import os
import re
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

# Read data
data_folder = 'Data'
data = {}

csv_files = [f for f in os.listdir(data_folder) if re.match(r'State_Control_60sec_\d+_\d+_\d+\.csv', f)]

all_states = []
all_controls = []

for file_name in csv_files:
    # Extract initial conditions from the filename
    tokens = re.findall(r'_(\d+)_(\d+)_(\d+)\.csv$', file_name)
    if tokens:
        initial_condition = tokens[0]
        initial_str = f'init_{initial_condition[0]}_{initial_condition[1]}_{initial_condition[2]}'

        # Read the CSV file
        full_path = os.path.join(data_folder, file_name)
        csv_data = pd.read_csv(full_path)

        time = csv_data.iloc[:, 0].values
        state = csv_data.iloc[:, 1:13].values
        control = csv_data.iloc[:, 13:17].values

        # Save to dictionary with initial conditions as key
        data[initial_str] = {'time': time, 'state': state, 'control': control}

        all_states.append(state)
        all_controls.append(control)

# Partition
total_files = len(data)

train_percent = 0.8
val_percent = 0.1

train_count = int(train_percent * total_files)
val_count = int(val_percent * total_files)
test_count = total_files - train_count - val_count

# Randomize order of scenarios
keys = list(data.keys())
np.random.shuffle(keys)

train_keys = keys[:train_count]
val_keys = keys[train_count:train_count + val_count]
test_keys = keys[train_count + val_count:]

# Initialize new dictionaries to store the partitioned data
train_data = {key: data[key] for key in train_keys}
val_data = {key: data[key] for key in val_keys}
test_data = {key: data[key] for key in test_keys}

print(f'Total data: {total_files} sets.\nTraining data: {train_count} sets.\nValidation data: {val_count} sets.\nTesting data: {test_count} sets.')

# Window class
class Window(Dataset):
    def __init__(self, time_data, state_data, control_data, input_width, prediction_width, predict_columns=None):
        # Store the data
        self.time_data = time_data
        self.state_data = state_data
        self.control_data = control_data

        # Store the sequence length
        self.input_width = input_width
        self.prediction_width = prediction_width

        # Handle label columns
        self.predict_columns = predict_columns
        if predict_columns is not None:
            # If the data is a DataFrame, convert column names to indices
            if isinstance(control_data, pd.DataFrame):
                self.label_columns_indices = [control_data.columns.get_loc(col) for col in predict_columns]
            else:
                # If it's a NumPy array, assume columns are already represented by integer indices
                self.label_columns_indices = predict_columns

        # Calculate the number of valid input-output pairs
        self.num_samples = len(state_data) - input_width - prediction_width + 1

        self.split_window()

    def split_window(self):
        # Precompute the inputs and targets
        self.inputs_time = []
        self.targets_time = []
        self.inputs = []
        self.targets = []

        for i in range(self.num_samples):
            # Time 
            input_time_sequence = self.time_data[i:i + self.input_width]
            target_time_control = self.time_data[i + self.input_width:i + self.input_width + self.prediction_width]

            # Input: the sequence of state data
            input_sequence = self.state_data[i:i + self.input_width]
            
            # Target: the next predict time steps of control data
            target_control = self.control_data[i + self.input_width:i + self.input_width + self.prediction_width]

            self.inputs_time.append(input_time_sequence)
            self.targets_time.append(target_time_control)
            self.inputs.append(input_sequence)
            self.targets.append(target_control)

    def __repr__(self):
        self.total_window_size = self.input_width + self.prediction_width

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.predict_slice = slice(0, self.prediction_width)
        self.predict_indices = np.arange(self.total_window_size)[self.predict_slice]
        
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Predict indices: {self.predict_indices}',
            f'Label column name(s): {self.predict_columns}'])
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_time': torch.tensor(self.inputs_time[idx], dtype=torch.float32),
            'input': torch.tensor(self.inputs[idx], dtype=torch.float32),
            'target_time': torch.tensor(self.targets_time[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

# Use a specific key from the test data, for example, the first one
key = test_keys[0]
w1 = Window(
    time_data=test_data[key]['time'],
    state_data=test_data[key]['state'],
    control_data=test_data[key]['control'],
    input_width=10,
    prediction_width=10,
    predict_columns=['thrust', 'roll angle', 'pitch angle', 'yaw angle']
)

# Create DataLoader for the windowed dataset
scenario_loader = DataLoader(w1, batch_size=20, shuffle=False)

# Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out) 
        return out

input_size = 12
hidden_size = 64
output_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleRNN(input_size, hidden_size, output_size).to(device)
PATH = 'Training_output_draft/trained_model.pth'
model.load_state_dict(torch.load(PATH, weights_only=True))
print(f'Using device: {device}')

model.eval()

example_batch = next(iter(scenario_loader))
input_sequence = example_batch['input']  
true_output = example_batch['target']

input_sequence = input_sequence[0] 
true_output = true_output[0]  

input_sequence = input_sequence.to(device)
true_output = true_output.to(device)

all_true_ctrl = []
all_pred_ctrl = []

current_state = input_sequence.unsqueeze(0)
with torch.no_grad():
    for t in range(true_output.shape[0]):
        # Predict using the model
        predicted_output = model(current_state)  # Shape: (1, input_width, num_controls)
        
        # Get the prediction for the current time step (take the last time step output)
        predicted_control = predicted_output[:, -1, :]  # Shape: (1, num_controls)

        # Store the true and predicted control values for the first control variable (index 0)
        all_true_ctrl.append(true_output[t, 0].item())
        all_pred_ctrl.append(predicted_control[0, 0].item())

        # Update the current state with the predicted control to predict the next step
        # Here you can extend the logic to incorporate the predicted values back into the state

# Convert the lists to numpy arrays for plotting
all_true_ctrl = np.array(all_true_ctrl)
all_pred_ctrl = np.array(all_pred_ctrl)

# Plot the true vs predicted control values over the entire time sequence
plt.figure(figsize=(15, 6))
time_steps = range(len(all_true_ctrl))

plt.plot(time_steps, all_true_ctrl, linestyle='--', color='blue', label='True Control 1', alpha=0.7)
plt.plot(time_steps, all_pred_ctrl, linestyle='-', color='red', label='Predicted Control 1', alpha=0.7)

plt.xlabel('Time Steps')
plt.ylabel('Control 1 Value')
plt.title('True vs Predicted Control 1 Values Over Time Sequence')
plt.legend()
plt.show()