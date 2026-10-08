"""Export trained weights and verify ONNX against PyTorch on real test images."""
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

from data import ARTIFACTS, load_mnist, preprocess
from model import DigitCNN


def main():
    # Some Windows terminals use GBK, which cannot encode exporter status icons.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    torch.set_num_threads(4)
    weights_path = ARTIFACTS / "digits.pth"
    if not weights_path.exists():
        raise FileNotFoundError("Run train.py first.")
    model = DigitCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    output_path = ARTIFACTS / "digits.onnx"
    torch.onnx.export(
        model, (torch.zeros(2, 1, 28, 28),), str(output_path),
        input_names=["images"], output_names=["logits"],
        dynamo=True, opset_version=18, external_data=False,
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
    )
    onnx.checker.check_model(str(output_path))
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    images, _ = load_mnist("test")
    max_error = 0.0
    for size in (1, 8, 64):
        inputs = preprocess(images[:size])
        with torch.inference_mode():
            reference = model(torch.from_numpy(inputs)).numpy()
        actual = session.run(["logits"], {"images": inputs})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(actual.argmax(1), reference.argmax(1))
        max_error = max(max_error, float(np.max(np.abs(actual - reference))))
    report = {"onnx_check": "passed", "comparison": "passed",
              "tested_batch_sizes": [1, 8, 64], "max_absolute_error": max_error,
              "input": "float32 [batch,1,28,28], pixels / 255",
              "output": "float32 [batch,10] logits", "opset": 18}
    (ARTIFACTS / "export_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"ONNX saved: {output_path}")
    print(f"Verification PASSED. Maximum absolute error: {max_error:.8f}")


if __name__ == "__main__":
    main()
