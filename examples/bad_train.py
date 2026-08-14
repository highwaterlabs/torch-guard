"""A training script containing every bug torch-preflight currently detects.

Run ``torch-preflight check examples/bad_train.py`` to see the report.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),  # TG005: CrossEntropyLoss applies log_softmax itself
        )

    def forward(self, x):
        return self.backbone(x)


def train(model, dataset, val_dataset, device):
    loader = DataLoader(dataset, batch_size=64)  # TG004: no num_workers, no pin_memory
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    total_loss = 0.0

    class_weights = torch.ones(10)

    for epoch in range(10):
        # TG011: no model.train() here, so only epoch 0 trains with dropout on
        for images, targets in loader:
            images = images.to(device)
            weights = class_weights.to(device)  # TG013: same copy every iteration
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()          # TG003: nothing calls optimizer.zero_grad()
            optimizer.step()

            losses.append(loss)      # TG001: keeps the whole graph alive
            total_loss += loss       # TG001: chains every step's graph together

        # TG007: a sync per element instead of one reduction on the device
        for i in range(len(losses)):
            total_loss += losses[i].item()

        # TG002: validation forward pass with autograd still enabled
        model.eval()
        for images, targets in val_loader:
            logits = model(images.to(device))
            probs = F.softmax(logits, dim=1)
            val_loss = criterion(probs, targets)  # TG005: softmax before CrossEntropy
            print(val_loss.item())

    return losses, total_loss


def train_binary(model, loader, device):
    """A second head trained with binary cross-entropy."""
    criterion = nn.BCEWithLogitsLoss()  # applies sigmoid itself
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for images, targets in loader:
        logits = model(images.to(device))
        probs = torch.sigmoid(logits)
        loss = criterion(probs, targets)  # TG006: sigmoid applied twice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_unseeded(model, loader, device):
    """Nothing seeds the generators this run draws from."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    noise = torch.randn(784)  # TG008: no seed, so the run cannot be reproduced

    for images, targets in loader:
        loss = criterion(model(images.to(device) + noise), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_with_accumulation(model, loader, device):
    """Gradient accumulation, but the loss is never scaled."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    accumulation_steps = 4

    for step, (images, targets) in enumerate(loader):
        loss = criterion(model(images.to(device)), targets)
        loss.backward()  # TG014: summed over 4 micro-batches, never divided by 4
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
def train_distributed(model, dataset, device):
    """Multi-GPU training where every rank sees the same data."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("nccl")
    model = DistributedDataParallel(model)
    # TG012: no DistributedSampler, so all ranks iterate the whole dataset
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
                        pin_memory=True)
    return loader


def evaluate(model, loader, device):
    """TG002: an eval routine with no @torch.no_grad()."""
    correct = 0
    for images, targets in loader:
        logits = model(images.to(device))
        correct += (logits.argmax(dim=1) == targets).sum().item()
    return correct
