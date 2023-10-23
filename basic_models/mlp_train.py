from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple, List
import json

import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from mlp import Mlp
from args import parse_args


def dump_json(obj, fname: str):
    with open(fname, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=2)


# Dataset class
class FeatureData():
    def __init__(self, arr: np.ndarray):
        '''
        `arr` is a 2D array where each row is 6-dimensional, and the first 5
        dimensions are features and the last dimension is the label.
        '''
        self.features = torch.tensor(arr[:, :-1], dtype=torch.float32)
        self.labels = torch.tensor(arr[:, -1], dtype=torch.float32) / 100000 # Scaled down

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "input": self.features[index],
            "label": self.labels[index],
        }

    def __len__(self) -> int:
        return self.labels.shape[0]


def evaluate(model: Mlp, data: FeatureData, batch_size: int, device: str) -> dict:
    '''
    Evaluate `model` on `data`, and return the loss and RMAE.
    '''
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    model.eval()

    print("====== Evaluation ======")
    print(f"Number of samples: {len(data)}")

    all_losses = []
    all_rmae = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs, labels=labels)
            all_losses.append(outputs['loss'].item())
            # the shape of `preds` is (batch_size, 1, 1)
            # and the shape `labels` is (batch_size)
            # So we use squeeze to turn `preds` into (batch_size)
            preds = outputs['preds'].squeeze()
            all_rmae += (abs(preds - labels) / labels).tolist()
    loss = sum(all_losses) / len(all_losses)
    rmae = sum(all_rmae) / len(all_rmae)
    return {
        "loss": loss,
        "rmae": rmae,
    }


def load_data(fname: str, test_size: float = 0.1) -> Tuple[FeatureData, FeatureData, FeatureData]:
    # Prepare dataset
    raw_data = pd.read_csv(fname)
    array = raw_data.values
    num_examples = len(array)
    test_cnt = int(num_examples * test_size)
    dev_cnt = test_cnt
    train_cnt = num_examples - test_cnt - dev_cnt

    # Shuffle array and split
    np.random.shuffle(array)
    train_data = array[:train_cnt]
    dev_data = array[train_cnt:train_cnt + dev_cnt]
    test_data = array[train_cnt + dev_cnt:]
    return FeatureData(train_data), FeatureData(dev_data), FeatureData(test_data)


def train(
    model: Mlp,
    train_data: FeatureData,
    dev_data: FeatureData,
    device: str,
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-2,
    lr_gamma: float = 0.9,
    log_interval: int = 100,
) -> Dict[str, List[float]]:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_gamma)

    dev_losses = []
    dev_rmae = []
    train_losses = []
    train_rmae = []

    # Run the training loop
    for epoch in range(0, num_epochs):  # 5 epochs at maximum
        print(f'Starting epoch {epoch+1}')
        # Set current loss value
        ep_losses = []
        ep_rmae = []

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs = data["input"].to(device)
            targets = data["label"].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs, labels=targets)
            loss = outputs['loss']

            # Backward
            loss.backward()
            optimizer.step()

            # Log statistics
            ep_losses.append(loss.item())
            batch_rmae = (abs(outputs['preds'].squeeze() - targets) / targets).tolist()
            ep_rmae += batch_rmae
            if (i + i) % log_interval == 0:
                avg_loss = sum(ep_losses) / len(ep_losses)
                avg_rmae = sum(batch_rmae) / len(batch_rmae)
                print(dict(
                    step=i,
                    mse=round(avg_loss, 5),
                    rmae=round(avg_rmae, 5),
                ))

        # Step the scheduler
        lr_scheduler.step()

        # Evaluate
        result = evaluate(model, dev_data, batch_size, device=device)
        print(f'Average test loss: {result["loss"]:.5f}')
        dev_losses.append(result['loss'])
        dev_rmae.append(result['rmae'])
        train_losses.append(sum(ep_losses) / len(ep_losses))
        train_rmae.append(sum(ep_rmae) / len(ep_rmae))

    # Process is complete.
    print('Training process has finished.')
    return {
        'train_losses': train_losses,
        'dev_losses': dev_losses,
    }


def main():
    args: Namespace = parse_args()
    torch.manual_seed(args.seed)

    # Data
    train_data, dev_data, test_data = load_data("./preprocessed_data.csv")

    # Model
    input_dim = 5
    hidden_dim = [args.hidden_dim] * args.num_layers
    hidden_dim += [1]
    model = Mlp(input_dim, hidden_dim, args.act_fn).to(args.device)

    # Train and test
    train_result = train(
        model,
        train_data,
        dev_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=200,
        device=args.device,
    )
    output_dir = Path(
        'result',
        f'd{args.hidden_dim}_L{args.num_layers}_actfn{args.act_fn}',
        f"bs{args.batch_size}_lr{args.lr}",
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    dump_json(train_result, output_dir / 'train_result.json')

    test_result = evaluate(
        model, test_data, batch_size=args.batch_size, device=args.device)
    print("Test loss:", test_result['loss'])
    print("Test RMAE:", test_result['rmae'])

    dump_json(test_result, output_dir / 'test_result.json')


if __name__ == "__main__":
    main()
