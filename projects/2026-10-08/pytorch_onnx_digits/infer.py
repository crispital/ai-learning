"""Independent ONNX Runtime inference; does not import the PyTorch model."""
import argparse
import json

import numpy as np
import onnxruntime as ort

from data import ARTIFACTS, load_mnist, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--index", type=int, default=0, help="MNIST test image index")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all 10,000 test images")
    parser.add_argument("--input-npy", help="Optional uint8 NumPy images shaped [N,28,28]")
    parser.add_argument("--model", default=str(ARTIFACTS / "digits.onnx"))
    args = parser.parse_args()
    available = ort.get_available_providers()
    use_cuda = args.device == "cuda" or (args.device == "auto" and "CUDAExecutionProvider" in available)
    if use_cuda:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA provider unavailable. Install a compatible GPU package or use --device cpu.")
        # Finds compatible CUDA/cuDNN DLLs in installed PyTorch/NVIDIA packages.
        ort.preload_dlls()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    active = session.get_providers()
    if use_cuda and "CUDAExecutionProvider" not in active:
        raise RuntimeError("CUDA initialization failed; session fell back to CPU. Check DLL/driver errors above.")
    print(f"Session providers: {active}")
    print(f"Model input: {session.get_inputs()[0].name}, {session.get_inputs()[0].shape}")
    if args.input_npy:
        images = np.load(args.input_npy, allow_pickle=False)
        labels = None
        if images.dtype != np.uint8:
            raise ValueError("Custom images must be uint8 pixels in range 0..255.")
    else:
        images, labels = load_mnist("test")
        if not 0 <= args.index < len(images):
            parser.error(f"index must be between 0 and {len(images) - 1}")
        if not args.evaluate:
            images = images[args.index:args.index + 1]
            labels = labels[args.index:args.index + 1]
    all_predictions = []
    for start in range(0, len(images), 128):
        inputs = preprocess(images[start:start + 128])
        logits = session.run(["logits"], {"images": inputs})[0]
        if not np.isfinite(logits).all():
            raise RuntimeError("Non-finite inference output")
        all_predictions.append(logits.argmax(axis=1))
    if not all_predictions:
        raise ValueError("Input contains no images")
    predictions = np.concatenate(all_predictions)
    if args.evaluate:
        if labels is None:
            parser.error("--evaluate requires the labelled MNIST test set; omit --input-npy")
        accuracy = float(np.mean(predictions == labels))
        print(f"ONNX Runtime test accuracy: {accuracy:.2%} ({len(labels)} images)")
        report = {"requested_device": args.device, "session_providers": active,
                  "test_samples": len(labels), "test_accuracy": accuracy}
        ARTIFACTS.mkdir(exist_ok=True)
        filename = "inference_cuda.json" if use_cuda else "inference_cpu.json"
        (ARTIFACTS / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(f"Predictions: {predictions.tolist()}")
        if labels is not None:
            print(f"True labels: {labels.tolist()}")
        # An ASCII preview of the first image: bright strokes on a dark background.
        characters = np.array(list(" .:-=+*#%@"))
        for row in images[0]:
            print("".join(characters[(row.astype(np.int32) * 9 // 255)]))


if __name__ == "__main__":
    main()
