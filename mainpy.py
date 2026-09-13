
import os
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa
import soundfile as sf
import sounddevice as sd
from collections import deque
import matplotlib.pyplot as plt

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 或者使用系统自带的黑体：plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题

# ==================== 全局参数 ====================
SAMPLE_RATE = 16000  # 采样率 (助听器常用16kHz)
FRAME_LEN = 0.025  # 帧长 25ms
FRAME_SHIFT = 0.010  # 帧移 10ms
N_FFT = 512  # FFT点数
HOP_LEN = int(FRAME_SHIFT * SAMPLE_RATE)  # 160
WIN_LEN = int(FRAME_LEN * SAMPLE_RATE)  # 400
WIN_TYPE = 'hamming'

# 深度学习参数（轻量化模型）
INPUT_DIM = N_FFT // 2 + 1  # 257
HIDDEN_DIM = 128  # 减小隐藏层，轻量化
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_PATH = "denoise_model_light.pth"  # 新模型文件名

# 实时处理参数
CHUNK_DURATION = 0.025  # 每次处理25ms（一帧）
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)  # 400
OVERLAP_FACTOR = 0.5  # 50%重叠（OLA）
REAL_TIME_GAIN = 1.0  # 降噪强度，可调


# ==================== 信号处理基础函数 ====================
# 分帧加窗
def enframe(signal, frame_len, frame_shift, win_func=np.hamming):
    """分帧加窗"""
    signal_len = len(signal)
    num_frames = 1 + int((signal_len - frame_len) / frame_shift)
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_len
        frames[i, :] = signal[start:end] * win_func(frame_len)
    return frames

# STFT幅度谱提取
def stft_magnitude(signal, n_fft, hop_len, win_len, win_type='hamming'):
    """STFT幅度谱和相位"""
    window = np.hanning(win_len) if win_type == 'hann' else np.hamming(win_len)
    stft_mat = librosa.stft(signal, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window)
    mag = np.abs(stft_mat).T
    phase = np.angle(stft_mat).T
    return mag, phase


def istft_from_mag_phase(mag, phase, hop_len, win_len):
    """幅度+相位重建信号"""
    complex_spec = mag * np.exp(1j * phase)
    complex_spec = complex_spec.T
    window = np.hanning(win_len)
    signal = librosa.istft(complex_spec, hop_length=hop_len, win_length=win_len, window=window)
    return signal


def endpoint_detection(signal, frame_len, frame_shift, sample_rate):
    """短时能量+过零率端点检测"""
    frame_len_samples = frame_len
    frame_shift_samples = frame_shift
    frames = enframe(signal, frame_len_samples, frame_shift_samples, np.hamming)
    energy = np.sum(frames ** 2, axis=1)
    energy_db = 10 * np.log10(energy + 1e-6)
    thr_energy = np.max(energy_db) - 20
    zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / (2 * frame_len_samples)
    thr_zcr = 0.1 * np.max(zcr)
    vad = (energy_db > thr_energy) | (zcr > thr_zcr)
    vad_smooth = np.convolve(vad, np.ones(5) / 5, mode='same') > 0.3
    if not np.any(vad_smooth):
        return 0, len(signal)
    start_frame = np.argmax(vad_smooth)
    end_frame = len(vad_smooth) - np.argmax(vad_smooth[::-1]) - 1
    start_sample = max(0, start_frame * frame_shift_samples)
    end_sample = min(len(signal), (end_frame + 1) * frame_shift_samples + frame_len_samples)
    return start_sample, end_sample


# ==================== 轻量化深度学习模型（适合实时） ====================
#DNN网络定义
class LightDenoiseDNN(nn.Module):
    """轻量化DNN：单隐藏层128单元，参数量约33k，适合低延迟设备"""

    def __init__(self, input_dim=257, hidden_dim=128):
        super(LightDenoiseDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def generate_synthetic_speech(duration=2.0, sr=16000):
    """合成模拟语音"""
    t = np.linspace(0, duration, int(sr * duration))
    f0 = 130 + 20 * np.sin(2 * np.pi * 2 * t)
    signal = np.zeros_like(t)
    for harm in range(1, 8):
        amp = 1.0 / harm
        phase = np.cumsum(2 * np.pi * f0 * harm / sr)
        signal += amp * np.sin(phase)
    envelope = np.exp(-2 * np.abs(t - duration / 2))
    signal = signal * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal


def generate_noise(duration=2.0, sr=16000, noise_type='white'):
    """生成噪声"""
    n_samples = int(sr * duration)
    if noise_type == 'white':
        noise = np.random.randn(n_samples)
    elif noise_type == 'pink':
        white = np.random.randn(n_samples)
        noise = np.cumsum(white)
        noise = noise / (np.max(np.abs(noise)) + 1e-6)
    else:
        noise = np.random.randn(n_samples)
    return noise

#训练数据生成
def prepare_training_data(num_samples=200, duration=1.5, sr=16000, snr_range=(0, 15)):
    """生成训练数据（幅度谱帧对）"""
    X_list, y_list = [], []
    for _ in range(num_samples):
        clean = generate_synthetic_speech(duration, sr)
        noise_type = np.random.choice(['white', 'pink'])
        noise = generate_noise(duration, sr, noise_type)
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        clean_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        noise = noise * np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
        noisy = clean + noise

        mag_clean, _ = stft_magnitude(clean, N_FFT, HOP_LEN, WIN_LEN, WIN_TYPE)
        mag_noisy, _ = stft_magnitude(noisy, N_FFT, HOP_LEN, WIN_LEN, WIN_TYPE)
        min_frames = min(mag_clean.shape[0], mag_noisy.shape[0])
        mag_clean = mag_clean[:min_frames, :]
        mag_noisy = mag_noisy[:min_frames, :]
        max_val = max(np.max(mag_clean), np.max(mag_noisy))
        if max_val > 0:
            mag_clean = mag_clean / max_val
            mag_noisy = mag_noisy / max_val
        X_list.append(mag_noisy)
        y_list.append(mag_clean)
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X.astype(np.float32), y.astype(np.float32)


def train_model(model, X_train, y_train, epochs, batch_size, lr, device):
    """训练模型（后台线程）"""
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    return losses


# ==================== 实时降噪处理器（助听器前端） ====================
class RealTimeDenoiser:
    def __init__(self, model, device, sample_rate=SAMPLE_RATE, hop_len=HOP_LEN, win_len=WIN_LEN, n_fft=N_FFT):
        self.model = model
        self.device = device
        self.sr = sample_rate
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_fft = n_fft
        self.model.eval()

        # 环形缓冲区（保存上一帧的尾部，用于重叠相加）
        self.ring_buffer = np.zeros(win_len)  # 用于拼接新数据
        self.output_buffer = np.zeros(win_len)  # 用于OLA累积

        # 相位缓存（简单处理：每一帧独立估计相位，无需跨帧）
        self.last_phase = None

    def process_frame(self, frame):
        """处理单帧（加窗后）信号，返回增强后的帧（时域）"""
        # 计算幅度谱和相位
        mag, phase = stft_magnitude(frame, self.n_fft, self.hop_len, self.win_len, WIN_TYPE)
        if mag.shape[0] == 0:
            return np.zeros_like(frame)
        # 取第一帧（实际输入只有一帧，但stft可能会输出多个帧，这里简单处理）
        mag = mag[0:1, :]  # (1, F)
        phase = phase[0:1, :]

        # 归一化
        max_mag = np.max(mag)
        if max_mag < 1e-6:
            return frame
        mag_norm = mag / max_mag

        # 模型预测增益
        with torch.no_grad():
            gain = self.model(torch.tensor(mag_norm, dtype=torch.float32).to(self.device))
            gain = gain.cpu().numpy()[0]  # (F,)

        # 应用增益，反归一化
        enhanced_mag = gain * max_mag
        enhanced_mag = enhanced_mag.reshape(1, -1)

        # 合成时域（使用当前帧的相位）
        enhanced_frame = istft_from_mag_phase(enhanced_mag, phase, self.hop_len, self.win_len)
        # 裁剪到与原始帧相同长度
        if len(enhanced_frame) < len(frame):
            enhanced_frame = np.pad(enhanced_frame, (0, len(frame) - len(enhanced_frame)))
        else:
            enhanced_frame = enhanced_frame[:len(frame)]
        return enhanced_frame

    def denoise_chunk(self, input_chunk):
        """
        处理一段连续的音频块（长度可以为任意，内部会自动分帧重叠相加）
        输入: input_chunk (numpy array, 1D)
        输出: 增强后的音频块 (numpy array)
        """
        # 将新数据拼接到环形缓冲区
        self.ring_buffer = np.concatenate([self.ring_buffer, input_chunk])
        # 当缓冲区数据不少于一帧时进行处理
        output_chunks = []
        while len(self.ring_buffer) >= self.win_len:
            # 取一帧
            frame = self.ring_buffer[:self.win_len]
            # 加窗
            window = np.hamming(self.win_len)
            framed = frame * window
            # 增强
            enhanced_frame = self.process_frame(framed)
            # 重叠相加（OLA）
            if len(self.output_buffer) < self.win_len:
                self.output_buffer = np.pad(self.output_buffer, (0, self.win_len - len(self.output_buffer)))
            self.output_buffer += enhanced_frame * window
            # 输出重叠部分（帧移长度）
            out_seg = self.output_buffer[:self.hop_len]
            output_chunks.append(out_seg)
            # 移位缓冲区
            self.output_buffer = self.output_buffer[self.hop_len:]
            self.ring_buffer = self.ring_buffer[self.hop_len:]
        return np.concatenate(output_chunks) if output_chunks else np.array([])


# ==================== GUI 界面（含实时降噪） ====================
class SpeechEnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能助听器前端降噪系统 - 语音信号处理课程设计")
        self.root.geometry("1300x850")

        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightDenoiseDNN(INPUT_DIM, HIDDEN_DIM).to(self.device)
        self.load_or_train_model()

        # 实时处理控制
        self.realtime_running = False
        self.realtime_denoiser = None
        self.input_stream = None
        self.output_stream = None
        self.realtime_gain = tk.DoubleVar(value=1.0)

        # 数据存储
        self.original_signal = None
        self.enhanced_signal = None
        self.original_mag = None
        self.original_phase = None
        self.frames = None
        self.vad_start = 0
        self.vad_end = 0

        # 创建界面
        self.create_widgets()

    def load_or_train_model(self):
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            messagebox.showinfo("模型加载", f"已加载轻量化模型 {MODEL_PATH}")
        else:
            answer = messagebox.askyesno("训练模型", "未找到轻量化模型。是否立即训练？（约1-2分钟）")
            if answer:
                self.train_model_background()
            else:
                messagebox.showwarning("提示", "未训练模型，降噪效果较差。")

    def train_model_background(self):
        def train_task():
            progress_win = tk.Toplevel(self.root)
            progress_win.title("训练中")
            progress_label = tk.Label(progress_win, text="生成训练数据...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_win, mode='indeterminate')
            progress_bar.pack(pady=10)
            progress_bar.start()
            self.root.update()

            X_train, y_train = prepare_training_data(num_samples=300, duration=1.2, sr=SAMPLE_RATE)
            progress_label.config(text="训练神经网络（轻量化）...")
            losses = train_model(self.model, X_train, y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE, self.device)
            torch.save(self.model.state_dict(), MODEL_PATH)
            progress_bar.stop()
            progress_win.destroy()
            messagebox.showinfo("完成", f"模型保存至 {MODEL_PATH}\n最终损失: {losses[-1]:.6f}")

        threading.Thread(target=train_task, daemon=True).start()

    def create_widgets(self):
        # 顶部控制栏（第一行：文件处理）
        control_frame1 = tk.Frame(self.root)
        control_frame1.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(control_frame1, text="打开音频文件", command=self.load_audio, width=14).pack(side=tk.LEFT, padx=3)
        tk.Button(control_frame1, text="端点检测", command=self.show_endpoint_detection, width=12).pack(side=tk.LEFT,
                                                                                                    padx=3)
        tk.Button(control_frame1, text="分帧加窗分析", command=self.show_framing, width=12).pack(side=tk.LEFT, padx=3)
        tk.Button(control_frame1, text="语谱图分析", command=self.show_spectrogram, width=12).pack(side=tk.LEFT, padx=3)
        tk.Button(control_frame1, text="文件降噪", command=self.enhance_speech, width=12).pack(side=tk.LEFT, padx=3)
        tk.Button(control_frame1, text="保存降噪音频", command=self.save_enhanced_audio, width=14).pack(side=tk.LEFT, padx=3)
        tk.Button(control_frame1, text="播放原始", command=lambda: self.play_audio(self.original_signal), width=10).pack(
            side=tk.LEFT, padx=3)
        tk.Button(control_frame1, text="播放降噪", command=lambda: self.play_audio(self.enhanced_signal), width=10).pack(
            side=tk.LEFT, padx=3)

        # 第二行：实时降噪控制
        control_frame2 = tk.Frame(self.root)
        control_frame2.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame2, text="助听器实时降噪:").pack(side=tk.LEFT, padx=5)
        self.btn_realtime = tk.Button(control_frame2, text="启动实时降噪", command=self.toggle_realtime, bg="lightgreen",
                                      width=15)
        self.btn_realtime.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame2, text="降噪强度:").pack(side=tk.LEFT, padx=5)
        gain_scale = tk.Scale(control_frame2, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                              variable=self.realtime_gain, length=150)
        gain_scale.pack(side=tk.LEFT, padx=5)
        self.gain_label = tk.Label(control_frame2, text="1.0")
        self.gain_label.pack(side=tk.LEFT)
        self.realtime_gain.trace("w", lambda *_: self.gain_label.config(text=f"{self.realtime_gain.get():.1f}"))

        tk.Label(control_frame2, text=" | 麦克风设备: ").pack(side=tk.LEFT, padx=(20, 0))
        self.device_list = tk.StringVar(value="默认")
        device_menu = tk.OptionMenu(control_frame2, self.device_list, "默认",
                                    *[str(d) for d in range(len(sd.query_devices()))])
        device_menu.pack(side=tk.LEFT)

        # 第三行：状态和说明
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=2)
        self.status_var = tk.StringVar(value="就绪 | 可打开文件或启动实时降噪")
        status_label = tk.Label(status_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)

        # 绘图区域
        self.figure = plt.Figure(figsize=(12, 7), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ========== 文件处理相关 ==========
    def load_audio(self):
        """优化的音频加载函数，解决格式、路径、损坏等主要问题"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("所有支持的音频", "*.wav;*.mp3;*.flac;*.m4a"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3")
            ]
        )
        if not file_path:
            return

        # --- 1. 检查文件是否存在并处理路径空格 ---
        if not os.path.exists(file_path):
            messagebox.showerror("错误", f"文件不存在：\n{file_path}")
            return

        # 检查路径中是否有特殊字符
        if any(c in file_path for c in ['#', '%', '&']):
            if not messagebox.askyesno("警告", "文件路径包含特殊字符，可能导致加载失败。是否仍尝试加载？"):
                return

        # --- 2. 开始加载，使用更多的异常捕获 ---
        self.status_var.set(f"正在加载: {os.path.basename(file_path)} ...")
        self.root.update()

        try:
            # 使用 librosa 加载，它内部会自动尝试 soundfile -> audioread -> ffmpeg
            # 强制重采样到 SAMPLE_RATE，并转为单声道
            data, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE,  # 统一采样率
                mono=True,  # 转为单声道
                res_type='kaiser_fast'  # 快速重采样算法
            )

            # --- 3. 检查加载结果，处理可能的静音信号 ---
            if data is None or len(data) == 0:
                raise ValueError("文件加载后未获得任何数据")

            max_val = np.max(np.abs(data))
            if max_val < 1e-6:
                raise ValueError("音频文件音量过低，可能为静音文件")

            # --- 4. 数据归一化 ---
            data = data / max_val

            # --- 5. 保存到对象属性 ---
            self.original_signal = data
            self.current_file = file_path
            self.enhanced_signal = None
            self.original_mag, self.original_phase = stft_magnitude(
                self.original_signal, N_FFT, HOP_LEN, WIN_LEN, WIN_TYPE
            )

            # --- 6. 更新状态和界面 ---
            duration = len(data) / SAMPLE_RATE
            self.status_var.set(f"成功加载: {os.path.basename(file_path)} | 时长: {duration:.2f}秒")
            self.plot_waveform(self.original_signal, "原始语音波形")
            messagebox.showinfo("加载成功", f"文件 '{os.path.basename(file_path)}' 加载成功！\n时长: {duration:.2f}秒")

        # --- 7. 更详细的错误捕获与提示 ---
        except FileNotFoundError:
            messagebox.showerror("错误", f"文件未找到：\n{file_path}")
        except Exception as e:
            error_msg = str(e)
            # 识别常见的具体错误并给出针对性建议
            if "NoBackendError" in error_msg:
                detail = "\n\n建议：请安装 ffmpeg 并添加到环境变量，或尝试转换为 WAV 格式。"
            elif "PySoundFile failed" in error_msg:
                detail = "\n\n建议：请安装 ffmpeg（详见文档），或使用 WAV 格式文件。"
            elif "not found" in error_msg or "No such file" in error_msg:
                detail = "\n\n建议：请检查文件路径是否包含中文或空格，尝试将文件移动至英文目录下。"
            else:
                detail = "\n\n建议：请检查文件格式是否为 WAV/MP3，或尝试重新安装 librosa 和 soundfile。"

            messagebox.showerror(
                "加载失败",
                f"无法读取文件：{os.path.basename(file_path)}\n\n"
                f"错误信息：{error_msg[:200]}"
                f"{detail}\n\n"
                f"如需更多帮助，可尝试：\n1. 将文件转换为标准 WAV 格式\n2. 确认文件未损坏且能被其他播放器打开"
            )
            self.status_var.set("加载失败，请重试或检查文件")

    def plot_waveform(self, signal, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        t = np.arange(len(signal)) / SAMPLE_RATE
        ax.plot(t, signal)
        ax.set_title(title)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("幅值")
        ax.grid(True)
        self.canvas.draw()

    def show_endpoint_detection(self):
        if self.original_signal is None:
            messagebox.showwarning("警告", "请先加载音频文件")
            return
        start, end = endpoint_detection(self.original_signal, WIN_LEN, HOP_LEN, SAMPLE_RATE)
        self.vad_start, self.vad_end = start, end
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        t = np.arange(len(self.original_signal)) / SAMPLE_RATE
        ax.plot(t, self.original_signal, label='语音信号')
        # 修复：直接使用采样点计算时间，避免索引越界
        ax.axvspan(start / SAMPLE_RATE, end / SAMPLE_RATE, alpha=0.3, color='green', label='语音段')
        ax.set_title("端点检测结果 (基于短时能量+过零率)")
        ax.set_xlabel("时间 (s)")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
        self.status_var.set(f"端点检测完成: 语音起始 {start / SAMPLE_RATE:.2f}s, 结束 {end / SAMPLE_RATE:.2f}s")

    def show_framing(self):
        if self.original_signal is None:
            messagebox.showwarning("警告", "请先加载音频文件")
            return
        frames = enframe(self.original_signal, WIN_LEN, HOP_LEN, np.hamming)
        self.frames = frames
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        num_show = min(5, frames.shape[0])
        for i in range(num_show):
            t = np.arange(WIN_LEN) / SAMPLE_RATE
            ax.plot(t + i * HOP_LEN / SAMPLE_RATE, frames[i] + i * 0.5, label=f'帧{i + 1}')
        ax.set_title("分帧加窗 (前5帧垂直偏移)")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("幅值")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
        self.status_var.set(f"总帧数: {frames.shape[0]}")

    def show_spectrogram(self):
        if self.original_signal is None:
            messagebox.showwarning("警告", "请先加载音频信号")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        D = librosa.amplitude_to_db(self.original_mag.T, ref=np.max)
        img = librosa.display.specshow(D, sr=SAMPLE_RATE, hop_length=HOP_LEN, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title("语谱图")
        self.figure.colorbar(img, ax=ax, format="%+2.0f dB")
        self.canvas.draw()

    def enhance_speech(self):
        if self.original_signal is None:
            messagebox.showwarning("警告", "请先加载音频文件")
            return
        if self.original_mag is None:
            self.original_mag, self.original_phase = stft_magnitude(self.original_signal, N_FFT, HOP_LEN, WIN_LEN,
                                                                    WIN_TYPE)
        self.status_var.set("文件降噪中...")
        self.root.update()
        mag_noisy = self.original_mag
        max_mag = np.max(mag_noisy)
        if max_mag == 0:
            max_mag = 1
        mag_norm = mag_noisy / max_mag
        self.model.eval()
        with torch.no_grad():
            mag_tensor = torch.tensor(mag_norm, dtype=torch.float32).to(self.device)
            gain = self.model(mag_tensor).cpu().numpy()
        mag_enhanced = gain * max_mag
        enhanced = istft_from_mag_phase(mag_enhanced, self.original_phase, HOP_LEN, WIN_LEN)
        if len(enhanced) > len(self.original_signal):
            enhanced = enhanced[:len(self.original_signal)]
        else:
            enhanced = np.pad(enhanced, (0, len(self.original_signal) - len(enhanced)))
        self.enhanced_signal = enhanced
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        t = np.arange(len(self.original_signal)) / SAMPLE_RATE
        ax1.plot(t, self.original_signal)
        ax1.set_title("原始语音")
        ax1.grid(True)
        ax2 = self.figure.add_subplot(212)
        ax2.plot(t, self.enhanced_signal)
        ax2.set_title("降噪后语音")
        ax2.set_xlabel("时间 (s)")
        ax2.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()
        self.status_var.set("文件降噪完成")

    def save_enhanced_audio(self):
        if self.enhanced_signal is None:
            messagebox.showwarning("警告", "没有降噪后的语音，请先执行“文件降噪”")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if path:
            sf.write(path, self.enhanced_signal, SAMPLE_RATE)
            self.status_var.set(f"已保存: {path}")

    def play_audio(self, signal):
        if signal is None:
            messagebox.showwarning("警告", "没有可播放的音频")
            return
        try:
            sd.play(signal, SAMPLE_RATE)
            sd.wait()
        except:
            tmp = "_temp.wav"
            sf.write(tmp, signal, SAMPLE_RATE)
            os.system(f"start {tmp}" if os.name == 'nt' else f"aplay {tmp}")
            threading.Timer(5, lambda: os.remove(tmp) if os.path.exists(tmp) else None).start()

    # ========== 实时降噪核心 ==========
    def toggle_realtime(self):
        if self.realtime_running:
            self.stop_realtime()
        else:
            self.start_realtime()

    def start_realtime(self):
        if self.model is None:
            messagebox.showerror("错误", "模型未加载")
            return
        # 初始化实时处理器
        self.realtime_denoiser = RealTimeDenoiser(self.model, self.device, SAMPLE_RATE, HOP_LEN, WIN_LEN, N_FFT)
        self.realtime_running = True
        self.btn_realtime.config(text="停止实时降噪", bg="lightcoral")
        self.status_var.set("实时降噪运行中... 请对着麦克风说话")

        # 获取设备索引
        dev = self.device_list.get()
        device_idx = None if dev == "默认" else int(dev)

        # 定义回调函数
        def audio_callback(indata, outdata, frames, time, status):
            if not self.realtime_running:
                outdata.fill(0)
                return
            # indata: (frames, channels) 我们取第一通道
            mic_signal = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
            # 应用用户强度缩放（模拟助听器增益调节）
            processed = self.realtime_denoiser.denoise_chunk(mic_signal)
            # 如果处理输出不足一帧，补零
            if len(processed) < len(mic_signal):
                processed = np.pad(processed, (0, len(mic_signal) - len(processed)))
            else:
                processed = processed[:len(mic_signal)]
            # 乘以降噪强度系数
            processed *= self.realtime_gain.get()
            outdata[:, 0] = processed

        try:
            self.input_stream = sd.InputStream(device=device_idx, samplerate=SAMPLE_RATE, channels=1,
                                               blocksize=CHUNK_SAMPLES)
            self.output_stream = sd.OutputStream(device=device_idx, samplerate=SAMPLE_RATE, channels=1,
                                                 blocksize=CHUNK_SAMPLES)
            self.input_stream.start()
            self.output_stream.start()
            # 注意：简单起见，可以使用一个单独的线程读取/处理/写入，但为了低延迟，这里采用更直接的方法：
            # 重新定义使用 sd.Stream 同时输入输出
            self.stream = sd.Stream(device=device_idx, samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SAMPLES,
                                    callback=audio_callback)
            self.stream.start()
            self.status_var.set("实时降噪已启动 | 调整降噪强度滑块体验效果")
        except Exception as e:
            self.realtime_running = False
            self.btn_realtime.config(text="启动实时降噪", bg="lightgreen")
            messagebox.showerror("错误", f"无法启动音频流: {e}")

    def stop_realtime(self):
        self.realtime_running = False
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        self.btn_realtime.config(text="启动实时降噪", bg="lightgreen")
        self.status_var.set("实时降噪已停止")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechEnhancementApp(root)
    root.mainloop()