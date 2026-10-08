"""MNIST download and preprocessing. Inference does not need PyTorch."""
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download(name):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / name
    if not destination.exists():
        print(f"Downloading {name} ...", flush=True)
        temporary = destination.with_suffix(".part")
        try:
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
                with temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def load_mnist(split):
    prefix = {"train": "train", "test": "t10k"}[split]
    with gzip.open(download(prefix + "-images-idx3-ubyte.gz"), "rb") as source:
        magic, count, height, width = struct.unpack(">IIII", source.read(16))
        if magic != 2051 or (height, width) != (28, 28):
            raise ValueError("Invalid MNIST image header")
        images = np.frombuffer(source.read(), dtype=np.uint8).reshape(count, 28, 28).copy()
    with gzip.open(download(prefix + "-labels-idx1-ubyte.gz"), "rb") as source:
        magic, label_count = struct.unpack(">II", source.read(8))
        if magic != 2049 or label_count != count:
            raise ValueError("Invalid MNIST label header")
        labels = np.frombuffer(source.read(), dtype=np.uint8).astype(np.int64)
    return images, labels


def preprocess(images):
    """[N,28,28] uint8 -> [N,1,28,28] float32, pixel range [0,1]."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Expected images shaped [N, 28, 28]")
    return np.ascontiguousarray(images[:, None].astype(np.float32) / 255.0)
