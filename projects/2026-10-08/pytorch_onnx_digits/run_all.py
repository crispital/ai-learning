"""Run training -> ONNX export/verification -> CPU and optionally GPU inference."""
import argparse
import subprocess
import sys

from data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=12000)
    args = parser.parse_args()
    commands = [
        ["train.py", "--device", args.device, "--epochs", str(args.epochs),
         "--train-limit", str(args.train_limit)],
        ["export_onnx.py"],
        ["infer.py", "--device", "cpu", "--evaluate"],
    ]
    if args.device != "cpu":
        commands.append(["infer.py", "--device", args.device, "--evaluate"])
    commands.append(["infer.py", "--device", args.device, "--index", "0"])
    for command in commands:
        print("\n>>> " + " ".join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
