import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import h5py
import matplotlib.pyplot as plt
import numpy as np

hr_path = # Path to high resolution image
lr_path = # Path to low resolution image

class ctdataset:
    def __init__(self, hrpath, lrpath):
        maxi = #Highest pixel intensity is stored in "maxi"
        with h5py.File(hrpath, 'r') as hdf:
            self.y_data = torch.tensor(hdf['HR'][:].astype('float32')/maxi)
        with h5py.File(lrpath, 'r') as hdf:
            self.x_data = torch.tensor(hdf['LR'][:].astype('float32')/maxi)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx].unsqueeze(0)
        y = self.y_data[idx].unsqueeze(0)
        return x, y

dataset = ctdataset(hr_path,lr_path)

train_split = # Test-Val-Train Split
train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = # Set Batch size
num_workers = # Set Number of workers for data loading
training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder
        # input: 512x512x1
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 512x512x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 512x512x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 256x256x64

        # input: 256x256x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 256x256x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 256x256x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 128x128x128

        # input: 128x128x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 128x128x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 128x128x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 64x64x256

        # input: 64x64x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 32x32x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 32x32x1024


        # Decoder
        #input: 32x32x1024
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        #output: Before concatenating with encoder: 64x64x512;
        #output: After concatenating with encoder: 64x64x1024;
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512

        #input: 64x64x512
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        #output: Before concatenating with encoder: 128x128x256;
        #output: After concatenating with encoder: 128x128x512;
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # output: 128x128x256
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 128x128x256

        #input: 128x128x256
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        #output: Before concatenating with encoder: 256x256x128;
        #output: After concatenating with encoder: 256x256x256;
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # output: 256x256x128
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 256x256x128

        #input: 256x256x128
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        #output: Before concatenating with encoder: 512x512x64;
        #output: After concatenating with encoder: 512x512x128;
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # output: 512x512x64
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 512x512x64

        #input: 512x512x64
        self.outconv = nn.Conv2d(64,1, kernel_size=1) #output: 512x512x1

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out

# Defining the Mean Squared Logarithmic Error
def msle(y_true, y_pred):
    y_true = torch.flatten(y_true).to(y_pred.dtype)  # Flatten and cast
    y_pred = torch.flatten(y_pred)
    first_log = torch.log(torch.clamp(torch.abs(y_pred), min=torch.finfo(y_pred.dtype).eps))
    second_log = torch.log(torch.clamp(torch.abs(y_true), min=torch.finfo(y_true.dtype).eps))
    return torch.mean(torch.square(first_log - second_log))

# Defining the Mean Squared Error
def mse(y_true, y_pred):
    y_true = y_true.to(y_pred.dtype)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return torch.mean(torch.square(y_true - y_pred))


# Hyperparameters
learning_rate = # Set learning Rate
epochs = # Set number of epochs
print(' initial learning rate = ', learning_rate, ' batch size = ', batch_size, ' num epochs =', epochs)

# Model, loss function, optimizer
model = UNet()
criterion = mse
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Decay LR by a factor of 0.1 every 20 epochs

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

avg_training_loss = []
avg_val_loss = []

for epoch in range(epochs):
    #Training
    model.train()
    running_loss = 0.0
    for inputs, targets in training_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average training loss
    avg_training_loss.append(running_loss / len(training_loader))

    # Validation
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in validation_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            # Forward pass
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_running_loss += val_loss.item()

    # Calculate average validation loss
    avg_val_loss.append(val_running_loss / len(validation_loader))
    
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_training_loss[-1]}, Validation Loss: {avg_val_loss[-1]}")

    
    if (epoch + 1) % 5 == 0:
        # Save model checkpoint and plot results every 5 epochs

        # Plot the results
        img_inp = val_inputs.cpu()[0][0]
        img_tar = val_targets.cpu()[0][0]
        img_out = val_outputs.detach().cpu()[0][0]

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(img_inp, cmap='gray')
        plt.title("Input image")
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(img_tar, cmap='gray')
        plt.title("Target image")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(img_out, cmap='gray')
        plt.title("Model result")
        plt.colorbar()

        plt.show()

print("Training finished.")

# Load the model and do the testing
