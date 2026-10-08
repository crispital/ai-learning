"""Train a small CNN using PyTorch, then evaluate on MNIST test data."""
import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import ARTIFACTS, load_mnist, preprocess
from model import DigitCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=12000,
                        help="0 uses all 60,000 training images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.train_limit < 0 or args.batch_size < 1:
        parser.error("epochs/batch-size must be positive; train-limit must be >= 0")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check your PyTorch installation or use --device cpu.")
    print(f"Training device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    images, labels = load_mnist("train")
    indices = np.random.default_rng(42).permutation(len(labels))
    if args.train_limit:
        indices = indices[:args.train_limit]
    dataset = TensorDataset(torch.from_numpy(preprocess(images[indices])),
                            torch.from_numpy(labels[indices]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_images, test_labels = load_mnist("test")
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(preprocess(test_images)), torch.from_numpy(test_labels)),
        batch_size=256, num_workers=0)
    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    started = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        mean_loss = total_loss / len(dataset)
        history.append(mean_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={mean_loss:.4f}", flush=True)
    model.eval()
    correct = 0
    with torch.inference_mode():
        for inputs, targets in test_loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == targets).sum())
    accuracy = correct / len(test_labels)
    ARTIFACTS.mkdir(exist_ok=True)
    weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(weights, ARTIFACTS / "digits.pth")
    report = {"training_device": device, "training_samples": len(dataset),
              "epochs": args.epochs, "loss": history, "test_samples": len(test_labels),
              "test_accuracy": accuracy, "elapsed_seconds": time.perf_counter() - started}
    (ARTIFACTS / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PyTorch test accuracy: {accuracy:.2%}")
    print(f"Weights saved: {ARTIFACTS / 'digits.pth'}")


if __name__ == "__main__":
    main()
