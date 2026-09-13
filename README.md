# 语音信号处理与去噪

语音信号处理实验项目，包含 **深度学习语音去噪模型** 与 **Tkinter 桌面应用**（支持录音、播放、波形可视化、去噪推理）。

## 项目简介

围绕语音信号处理开展的去噪研究与应用：

- 采用帧长 25ms / 帧移 10ms 的分帧加窗、FFT（512 点）等经典语音信号处理流程；
- 训练轻量化神经网络（257 维输入 → 128 维隐藏层）实现语音去噪，模型参数保存于 `denoise_model_light.pth`；
- 基于 Tkinter 构建桌面界面：录音、文件选择、波形/频谱可视化（Matplotlib 嵌入）、实时去噪演示。

## 技术栈

- Python + Tkinter（桌面 GUI）
- NumPy / SciPy / librosa / soundfile / sounddevice（音频处理与采集）
- PyTorch（去噪模型）
- Matplotlib（可视化）

## 目录结构

```
speech-signal/
├── mainpy.py                  # 主程序（GUI + 去噪 + 可视化）
└── denoise_model_light.pth    # 轻量化去噪模型权重
```

## 运行方式

```bash
pip install numpy scipy matplotlib torch librosa soundfile sounddevice
python mainpy.py
```

> 注意：模型权重 `denoise_model_light.pth` 体积较大，已被 .gitignore 忽略、不会提交到 GitHub。首次获取可运行代码内置的训练逻辑生成，或使用你本地的模型文件。

## 功能说明

| 功能 | 说明 |
| --- | --- |
| 录音 | 通过 sounddevice 采集语音（16kHz） |
| 去噪 | 加载 `denoise_model_light.pth` 模型对含噪语音推理去噪 |
| 可视化 | 显示原始/去噪语音的波形与频谱（汉明窗、512 点 FFT） |
| 模型训练 | 代码内置训练逻辑（15 epochs，Adam 优化），可重新训练生成模型权重 |
