import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

import mlp


# setup dataset
class FeatureData():
    def __init__(self, df: pd.DataFrame):
        # self.df = copy.deepcopy(df)
        self.df = df.copy(deep=True)
        # Input data
        self.data = torch.stack(
            [
                torch.tensor(self.df["pca0"],
                             dtype=torch.float32, requires_grad=True),
                torch.tensor(self.df["pca1"],
                             dtype=torch.float32, requires_grad=True),
                torch.tensor(self.df["pca2"],
                             dtype=torch.float32, requires_grad=True),
                torch.tensor(self.df["pca3"],
                             dtype=torch.float32, requires_grad=True),
                torch.tensor(self.df["aggregated_consumption"],
                             dtype=torch.float32, requires_grad=True)
            ],
            dim=1,
        )

    def __getitem__(self, index: int) -> tuple:
        return self.data[index]

    def __len__(self) -> int:
        return self.df.shape[0]

# train model
# evaluate model
# test model


if __name__ == '__main__':

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare dataset
    data = pd.read_csv("./basic_models/preprocessed_data.csv")
    data = FeatureData(data.data)
    trainloader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=True)

    # Initialize the MLP
    input_dim = 4
    hidden_dim = [32, 32, 1]
    model = mlp.Mlp(input_dim, hidden_dim)

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum

        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            inputs, targets = data

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')
