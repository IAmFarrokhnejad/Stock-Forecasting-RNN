import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = pd.read_csv('amazon.csv')
data = data[['Date', 'Close']]

data['Date'] = pd.to_datetime(data['Date'])
plt.plot(data['Date'], data['Close'])
plt.title('Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Prepare data for the LSTM model
def prepare_dataframe_for_model(df, lookback):
    """Create lagged features for time series forecasting."""
    df = dc(df)
    df.set_index('Date', inplace=True)

    for i in range(1, lookback + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    return df

lookback = 10
shifted_df = prepare_dataframe_for_model(data, lookback)
shifted_df_as_np = shifted_df.to_numpy()

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

x = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

# Flip x for sequential processing
x = dc(np.flip(x, axis=1))

# Split data into training and testing sets
split_index = int(len(x) * 0.95)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape for LSTM input
x_train = x_train.reshape((-1, lookback, 1))
x_test = x_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Convert to PyTorch tensors
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

# DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# GRU model definition
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize models
input_size = 1
hidden_size = 7
num_layers = 1
lstm_model = LSTM(input_size, hidden_size, num_layers).to(device)
gru_model = GRU(input_size, hidden_size, num_layers).to(device)

# Training and validation functions
def train_one_epoch(model, loader, optimizer, loss_function):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def validate_one_epoch(model, loader, loss_function):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    return running_loss / len(loader)

# Hyperparameters
learning_rate = 0.01
num_epochs = 77
loss_function = nn.MSELoss()

# Training loop for LSTM
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
print("Training LSTM model...")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(lstm_model, train_loader, lstm_optimizer, loss_function)
    val_loss = validate_one_epoch(lstm_model, test_loader, loss_function)
    print(f"LSTM Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Training loop for GRU
gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=learning_rate)
print("\nTraining GRU model...")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(gru_model, train_loader, gru_optimizer, loss_function)
    val_loss = validate_one_epoch(gru_model, test_loader, loss_function)
    print(f"GRU Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plot training predictions for LSTM
with torch.no_grad():
    lstm_train_predictions = lstm_model(x_train.to(device)).cpu().numpy().flatten()
    y_train_actual = y_train.numpy().flatten()

plt.plot(y_train_actual, label='Actual Close')
plt.plot(lstm_train_predictions, label='LSTM Predicted Close')
plt.title('Training Data: Actual vs LSTM Predicted')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Plot training predictions for GRU
with torch.no_grad():
    gru_train_predictions = gru_model(x_train.to(device)).cpu().numpy().flatten()

plt.plot(y_train_actual, label='Actual Close')
plt.plot(gru_train_predictions, label='GRU Predicted Close')
plt.title('Training Data: Actual vs GRU Predicted')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Plot testing predictions for LSTM
with torch.no_grad():
    lstm_test_predictions = lstm_model(x_test.to(device)).cpu().numpy().flatten()
    y_test_actual = y_test.numpy().flatten()

plt.plot(y_test_actual, label='Actual Close')
plt.plot(lstm_test_predictions, label='LSTM Predicted Close')
plt.title('Testing Data: Actual vs LSTM Predicted')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Plot testing predictions for GRU
with torch.no_grad():
    gru_test_predictions = gru_model(x_test.to(device)).cpu().numpy().flatten()

plt.plot(y_test_actual, label='Actual Close')
plt.plot(gru_test_predictions, label='GRU Predicted Close')
plt.title('Testing Data: Actual vs GRU Predicted')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.show()