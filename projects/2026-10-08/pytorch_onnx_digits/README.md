# PyTorch → ONNX → ONNX Runtime：手写数字识别

这是一个教学项目：输入一张 28×28 的灰度手写数字图片，输出 0～9 中的一个数字。
训练数据使用 MNIST，首次运行从公开 MNIST 镜像下载约 12MB 数据，之后复用本地缓存。
默认随机选取 12,000 张训练图，训练 3 轮，用独立的 10,000 张测试图检查准确率。
这是演示流程的小模型，并非生产级识别系统。没有将测试集用于梯度更新。

## 在你的 Windows 电脑上运行

你现有环境在 `D:\onnx_runtime\.venv`，已安装 CUDA 12.8 版 PyTorch、ONNX、ONNX Script 和 GPU 版 ONNX Runtime。
无需重新创建环境，也无需执行 requirements.txt 安装命令。

1. 打开本项目文件夹，在资源管理器地址栏输入 `powershell` 并回车。
2. 在打开的 PowerShell 中执行：

```powershell
& 'D:\onnx_runtime\.venv\Scripts\python.exe' .\run_all.py --device cuda
```

一条命令会完成：GPU 训练 → 保存权重 → 导出 ONNX → 数值一致性检查 → CPU/GPU 推理 → 展示第一个测试数字。
只想使用 CPU 时，将 `--device cuda` 改为 `--device cpu`。
所有数据和产物都保存到本项目内部，脚本使用自身目录定位文件。

## 分步学习（按顺序运行）

```powershell
# 第一步：PyTorch 训练，用 GPU 调整模型参数
& 'D:\onnx_runtime\.venv\Scripts\python.exe' .\train.py --device cuda

# 第二步：加载已训练的权重，导出 ONNX，并与 PyTorch 输出对比
& 'D:\onnx_runtime\.venv\Scripts\python.exe' .\export_onnx.py

# 第三步：ONNX Runtime 在 CPU 上识别一个测试数字
& 'D:\onnx_runtime\.venv\Scripts\python.exe' .\infer.py --device cpu --index 0

# 第四步：同一个 ONNX 文件改用 GPU 推理
& 'D:\onnx_runtime\.venv\Scripts\python.exe' .\infer.py --device cuda --index 0

# 第五步：检查全部测试图的预测准确率
& 'D:\onnx_runtime\.venv\Scripts\python.exe' .\infer.py --device cuda --evaluate
```

`--index` 表示第几张测试图片，从 0 开始；例如 `--index 25`。
`--train-limit 0 --epochs 5` 可用于训练全部 60,000 张训练图。
显存不足时，在 train.py 命令后增加 `--batch-size 32`。

## 代码应该按什么顺序阅读

- `data.py`：下载数据；把 uint8 像素除以 255，变成 float32；补上通道维度。
- `model.py`：两层卷积和一个小分类器，输出十个类别的分数（logits）。
- `train.py`：前向计算、计算损失、反向传播、更新参数、独立测试集评估。
- `export_onnx.py`：加载权重，调用 eval()，用 dynamo 导出并验证 ONNX。
- `infer.py`：只用 NumPy 和 ONNX Runtime 进行推理，不导入 model.py 或 torch。
- `run_all.py`：按顺序运行上述步骤；使用同一个 Python 解释器。

## 生成的文件

- `data/*.gz`：下载的 MNIST 数据缓存。
- `artifacts/digits.pth`：PyTorch 权重；重新训练可覆盖。
- `artifacts/digits.onnx`：包含权重的单个 ONNX 文件。
- `artifacts/training.json`：训练设置、损失、测试准确率。
- `artifacts/export_verification.json`：导出校验结果和最大输出误差。
- `artifacts/inference_cpu.json`、`inference_cuda.json`：两种推理环境的测试准确率。

模型输入名 `images`，形状 `[batch, 1, 28, 28]`，类型 float32，范围 0～1。
模型输出名 `logits`，形状 `[batch, 10]`，取 argmax 得到类别；分数本身不是概率。
batch 是动态维度，允许一次输入一张或多张图，图片高宽固定为 28。
导出时使用 opset 18；这不保证任意 NPU 转换器都兼容，需另外检查。

## 自己的数据

infer.py 可接受 `.npy` 文件，内容必须是 `[N,28,28]` 的 uint8 灰度图片，黑底白字。

```powershell
& 'D:\onnx_runtime\.venv\Scripts\python.exe' .\infer.py --device cpu --input-npy .\my_digits.npy
```

手机照片不能直接代入：要处理背景、裁剪、居中、缩放和颜色反转。
训练时是 MNIST 风格的黑底白字，自己的数据分布不同时识别效果会下降。

## 以后部署到 ARM64 CPU

复制 `infer.py`、`data.py`、`artifacts/digits.onnx` 到板上，保持相对目录结构。
安装适配板端 Python/ARM64 的 numpy 和 CPU 版 onnxruntime，然后执行 `python3 infer.py --device cpu`。
首次运行会下载测试数据；离线使用时复制 `data/`，或通过 `--input-npy` 提供自己的输入。
CPU 推理不需要安装 PyTorch。GPU 推理在本 PC 上可借用已安装 PyTorch 的 CUDA/cuDNN DLL；
只有复制 ONNX 文件，不会同时复制这些运行库。
RK3588 NPU 仍需 RKNN 转换和专门的板端代码，此项目不包含 NPU 部署。

## 常见问题

- 找不到 torch/onnxruntime：检查是否用了 `D:\onnx_runtime\.venv\Scripts\python.exe`。
- 找不到 digits.pth：先运行 train.py。
- 找不到 digits.onnx：先运行 export_onnx.py。
- GPU 无法初始化：先用 `--device cpu` 验证流程，再检查 CUDA/cuDNN 和驱动匹配。
- 下载失败：检查网络后重新运行；完成的文件会复用，未完成下载不会当成有效缓存。
- GPU provider 出现在列表中，只表示会话启用了它，不表示每个算子均在 GPU 上执行。
- 小模型的 GPU 推理不一定比 CPU 快；本项目先验证功能和精度，没有声称性能提升。
- 导出时提示未安装 torchvision/triton：这个模型未使用相关功能，可以忽略这些可选组件提示。

## 已完成的本机验证

在 RTX 3050 Laptop GPU、PyTorch 2.11.0+cu128、ONNX Runtime GPU 1.26.0 上运行：

- 默认 12,000 张训练图，3 轮训练，PyTorch 测试准确率 93.84%。
- ONNX 结构校验通过；batch 为 1、8、64 时，类别与 PyTorch 一致。
- ONNX CPU 与 PyTorch logits 最大绝对误差约 0.00000286。
- ONNX Runtime CPU、CUDA 在全部 10,000 张测试图上的准确率均为 93.84%。

不同运行环境和重新训练可能产生轻微差异；这是功能与精度检查，不是性能基准。

## 新电脑的 CPU 安装方式

已有 GPU 环境请跳过本节。新环境可以执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install onnx==1.23.2 onnxscript==0.7.2 onnxruntime==1.26.0 numpy
.\.venv\Scripts\python.exe .\run_all.py --device cpu
```

也可使用 requirements.txt，但 torch 的具体 CPU/CUDA 构建由所选安装源决定。
不要在同一环境同时安装 onnxruntime 和 onnxruntime-gpu。

## 参考

- MNIST 数据与格式说明：https://yann.lecun.com/exdb/mnist/
- PyTorch ONNX 导出：https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
- ONNX Runtime Python：https://onnxruntime.ai/docs/api/python/api_summary.html
- ONNX Runtime CUDA：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
