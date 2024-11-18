import itertools
import random
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader
import logging
from torch.amp import GradScaler, autocast

# -------------------- Logger Configuration --------------------

# Clear existing handlers if they exist
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logger manually
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # DEBUG for testing

# Add a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info("Logger has been configured")


# -------------------- Activation Functions --------------------

class ReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), x)


class NReLU(nn.Module):
    def forward(self, x):
        return torch.minimum(torch.zeros_like(x), x)


class ReLU_NReLU(nn.Module):
    def __init__(self):
        super(ReLU_NReLU, self).__init__()

    def forward(self, x):
        midpoint = x.shape[1] // 2
        relu_part = torch.relu(x[:, :midpoint])
        nrelu_part = -torch.relu(-x[:, midpoint:])
        return torch.cat((relu_part, nrelu_part), dim=1)


# -------------------- Regularization Modules --------------------

class NoiseInjection(nn.Module):
    def __init__(self, noise_std=0.1):
        super(NoiseInjection, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


class WeightReset:
    @staticmethod
    def apply(model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


# -------------------- Model Definition --------------------

class MLP(nn.Module):
    def __init__(self, dataset_name, hidden_size, output_size, activation_fn, use_dropout=False, use_batchnorm=False,
                 use_layernorm=False):
        super(MLP, self).__init__()
        input_size = 32 * 32 * 3 if dataset_name == "CIFAR10" else 28 * 28
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = activation_fn
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        if use_batchnorm:
            self.batchnorm1 = nn.BatchNorm1d(hidden_size)
            self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        if use_layernorm:
            self.layernorm1 = nn.LayerNorm(hidden_size)
            self.layernorm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x)
        if self.use_layernorm:
            x = self.layernorm1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.batchnorm2(x)
        if self.use_layernorm:
            x = self.layernorm2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, dataset_name, activation_fn, use_dropout=False, use_batchnorm=False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3) if dataset_name == "CIFAR10" else nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.activation = activation_fn
        self.flattened_size = self._get_flattened_size(dataset_name)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        if use_batchnorm:
            self.batchnorm1 = nn.BatchNorm1d(128)

    def _get_flattened_size(self, dataset_name):
        dummy_input = torch.zeros(1, 3, 32, 32) if dataset_name == "CIFAR10" else torch.zeros(1, 1, 28, 28)
        x = self.conv1(dummy_input)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------- Data Loader --------------------

def get_dataloader(dataset_name, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# -------------------- Metrics Calculation --------------------

def calculate_metrics(errors):
    mean_error = sum(errors) / len(errors)
    std_error = (sum((x - mean_error) ** 2 for x in errors) / len(errors)) ** 0.5
    return mean_error, std_error


# -------------------- Saving Results to Excel --------------------

def save_results_to_excel(df, dataset_name, architecture, l1_lambda, l2_lambda, use_dropout, use_batchnorm,
                          use_layernorm):
    # Add regularization information to each row
    df['L1_lambda'] = l1_lambda
    df['L2_lambda'] = l2_lambda
    df['Dropout'] = use_dropout
    df['BatchNorm'] = use_batchnorm
    df['LayerNorm'] = use_layernorm

    filename = f"experiment_results_{dataset_name}_{architecture}.xlsx"
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_excel(filename, index=False)
    logger.info(f"Results saved to {filename}")


# -------------------- Run Experiment --------------------

def run_experiment(architecture, activation_fn, dataset_name, epochs=10, batch_size=64, l1_lambda=0.0, l2_lambda=0.0,
                   use_dropout=False, use_batchnorm=False, use_layernorm=False, combined=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader(dataset_name, batch_size)

    if architecture == 'MLP':
        model = MLP(dataset_name=dataset_name, hidden_size=128, output_size=10, activation_fn=activation_fn,
                    use_dropout=use_dropout, use_batchnorm=use_batchnorm, use_layernorm=use_layernorm).to(device)
    elif architecture == 'CNN':
        model = CNN(dataset_name=dataset_name, activation_fn=activation_fn, use_dropout=use_dropout,
                    use_batchnorm=use_batchnorm).to(device)
    logger.debug(f"Model created with activation function: {model.activation.__class__.__name__}")

    optimizer = optim.Adam(model.parameters())
    scaler = GradScaler()
    errors = []
    results = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        for batch_idx, (data, target) in enumerate(dataloader):
            # logger.debug(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(dataloader)}")
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(data)
                loss = F.cross_entropy(output, target)

                # Apply L1 and L2 regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                if combined:
                    # Apply all regularizations
                    loss += l1_lambda * l1_norm + l2_lambda * l2_norm
                    if use_dropout:
                        loss += torch.rand(1).mean()  # Placeholder for Dropout's effect
                    if use_batchnorm or use_layernorm:
                        loss += torch.rand(1).mean()  # Placeholder for normalization effects
                else:
                    loss += l1_lambda * l1_norm + l2_lambda * l2_norm

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * data.size(0)
            total += target.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = running_loss / len(dataloader.dataset)
        train_accuracy = correct / total
        val_loss, val_accuracy = evaluate(model, dataloader, device)

        errors.append(1 - val_accuracy)
        epoch_duration = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1}/{epochs} completed, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Test Error: {1 - val_accuracy:.4f}, "
            f"Duration: {epoch_duration:.2f} seconds")

        results.append({
            "Dataset": dataset_name,
            "Experiment": f"{architecture}",
            "Activation Function": activation_fn.__class__.__name__,
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Val Loss": val_loss,
            "Val Accuracy": val_accuracy,
            "Test Error": 1 - val_accuracy,
            "L1_lambda": l1_lambda,
            "L2_lambda": l2_lambda,
            "Dropout": use_dropout,
            "BatchNorm": use_batchnorm,
            "LayerNorm": use_layernorm,
            "Combined": combined
        })

    mean_error, std_error = calculate_metrics(errors)
    logger.info(f'Mean Test Error: {mean_error:.4f}, Std Test Error: {std_error:.4f}')

    df = pd.DataFrame(results)
    save_results_to_excel(df, dataset_name, architecture, l1_lambda, l2_lambda, use_dropout, use_batchnorm,
                          use_layernorm)
    return errors, mean_error, std_error


# -------------------- Evaluation Function --------------------

def evaluate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            # logger.debug(f"Evaluating batch {batch_idx + 1}/{len(dataloader)}")
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    val_loss /= len(dataloader.dataset)
    val_accuracy = correct / total

    return val_loss, val_accuracy


# -------------------- Random Search for Hyperparameter Optimization --------------------

def get_random_hyperparameters():
    # Define the range of hyperparameters
    architectures = ['CNN']
    activations = [ReLU_NReLU()]
    datasets = ['MNIST']
    l1_lambda_values = [0.0, 0.001, 0.01]
    l2_lambda_values = [0.0, 0.001, 0.01]
    use_dropout_options = [True, False]
    use_batchnorm_options = [True, False]
    use_layernorm_options = [True, False]
    combined_options = [False]

    # Sample random hyperparameters from the range
    architecture = random.choice(architectures)
    activation = random.choice(activations)
    dataset = random.choice(datasets)
    l1_lambda = random.choice(l1_lambda_values)
    l2_lambda = random.choice(l2_lambda_values)
    use_dropout = random.choice(use_dropout_options)
    use_batchnorm = random.choice(use_batchnorm_options)
    use_layernorm = random.choice(use_layernorm_options)
    combined = random.choice(combined_options)

    return dataset, architecture, activation, l1_lambda, l2_lambda, use_dropout, use_batchnorm, use_layernorm, combined


# -------------------- Run Random Search --------------------

def run_random_search(num_experiments=10):
    best_model = None
    best_error = float('inf')
    best_hyperparameters = None

    for i in range(num_experiments):
        logger.info(f'Running random search experiment {i + 1}/{num_experiments}')
        dataset, architecture, activation, l1_lambda, l2_lambda, use_dropout, use_batchnorm, use_layernorm, combined = get_random_hyperparameters()

        logger.info(
            f'Hyperparameters for Trial {i + 1}: Dataset={dataset}, Architecture={architecture}, Activation={activation.__class__.__name__}, L1_lambda={l1_lambda}, L2_lambda={l2_lambda}, Dropout={use_dropout}, BatchNorm={use_batchnorm}, LayerNorm={use_layernorm}, Combined={combined}')

        errors, mean_error, std_error = run_experiment(
            architecture=architecture,
            activation_fn=activation,
            dataset_name=dataset,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            use_dropout=use_dropout,
            use_batchnorm=use_batchnorm,
            use_layernorm=use_layernorm,
            combined=combined
        )

        # Check if current experiment has the best performance
        if mean_error < best_error:
            best_error = mean_error
            best_hyperparameters = (
            dataset, architecture, activation, l1_lambda, l2_lambda, use_dropout, use_batchnorm, use_layernorm,
            combined)
            best_model = (architecture, activation, dataset)  # Store model details

        logger.info(f'Trial {i + 1} completed with Mean Test Error: {mean_error:.4f}')

    logger.info(
        f'Best Hyperparameters: Dataset={best_hyperparameters[0]}, Architecture={best_hyperparameters[1]}, Activation={best_hyperparameters[2].__class__.__name__}, L1_lambda={best_hyperparameters[3]}, L2_lambda={best_hyperparameters[4]}, Dropout={best_hyperparameters[5]}, BatchNorm={best_hyperparameters[6]}, LayerNorm={best_hyperparameters[7]}, Combined={best_hyperparameters[8]}')
    logger.info(f'Best Mean Test Error: {best_error:.4f}')


# -------------------- Main Script Update --------------------

if __name__ == '__main__':
    # Number of experiments for random search
    num_random_experiments = 20
    run_random_search(num_experiments=num_random_experiments)
