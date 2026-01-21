import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def parse_trajectory(row):
    values = row['trajectory'][1:-1].split()
    values = np.array(list(map(lambda x: float(x), values)))
    return values


# Define the autoencoder
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.latent_size = 48
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 2, 3), stride=1, padding=0),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.Linear(128, self.latent_size)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.latent_size, out_channels=1, kernel_size=(4, 3, 4), stride=1,
                               padding=0),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape((x.shape[0], self.latent_size, 1, 1, 1))
        x = self.decoder(x)
        return x


v3_prolific_binary = pd.read_csv("../outputs/v3_prolific/v3_prolific_binary_intra.csv")
v3_prolific_binary['trajectory'] = v3_prolific_binary.apply(parse_trajectory, axis=1)

input_data = np.vstack(v3_prolific_binary['trajectory'].values)
scaler = MinMaxScaler()
scaler.fit(input_data)
input_data = scaler.transform(input_data)
# Assuming your numpy array is named 'input_data'
# Convert numpy array to PyTorch tensor and reshape
input_data_tensor = torch.tensor(input_data).float()
N, C = input_data_tensor.shape[0], 1
D, H, W = 4, 3, 4
input_data_tensor = input_data_tensor.view(N, D, H, W, C)
input_data_tensor = input_data_tensor.permute(0, 4, 1, 2, 3)  # Reshape to (N, C, D, H, W)
# Split into train / test :
input_data_train, input_data_test = train_test_split(input_data_tensor, test_size=0.01, random_state=42)
batch_size = 8  # Adjust as needed
train_dataset = TensorDataset(input_data_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(input_data_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the autoencoder
autoencoder = Autoencoder()
# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)


# Train the model:
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        # Iterate over data
        for inputs in train_loader:
            inputs = inputs[0]  # DataLoader returns a list
            optimizer.zero_grad()  # Zero the parameter gradients
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        # Evaluate the model on the test set
        test_loss = evaluate_model(model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.4f}')
        if epoch % 1000 == 0:
            save_reconstructed_images(model, test_loader, epoch)


def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # Turn off gradients for validation
        for inputs in test_loader:
            inputs = inputs[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item() * inputs.size(0)
    total_loss = running_loss / len(test_loader.dataset)
    return total_loss


def save_reconstructed_images(model, test_loader, epoch, save_path="reconstructions"):
    Path(f"{save_path}/").mkdir(parents=True, exist_ok=True)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for validation
        # Get a batch of test data
        inputs = next(iter(test_loader))[0]
        inputs = next(iter(train_loader))[0]
        # Forward pass through the autoencoder
        reconstructed = model(inputs)

        # Convert tensors to numpy arrays for visualization
        inputs = inputs.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()

        # Plot and save the first image in the batch for comparison
        fig, axs = plt.subplots(2, 1, figsize=(5, 10))
        axs[0].imshow(inputs[0, 0, :, :, :].reshape(D, H, W)[0, :, :], cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(reconstructed[0, 0, :, :, :].reshape(D, H, W)[0, :, :], cmap='gray')
        axs[1].set_title('Reconstructed Image')

        # Save the figure
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/reconstruction_epoch_{epoch}.png")
        plt.close()


# Train the model
train_model(autoencoder, train_loader, test_loader, criterion, optimizer, num_epochs=100000)
