# Qwen3-ASR Windows Client (WSL Backend)

一个基于 Windows 的本地音频转录工具，通过 WSL 后端调用 Qwen3-ASR 模型。支持本地文件及 URL 下载，内置 VAD 检测、降噪及智能分段处理，让 Windows 用户也能轻松使用 Qwen3-ASR 进行高准确度语音转录。*

<img width="1654" height="1319" alt="图片" src="https://github.com/user-attachments/assets/a962d388-7609-433f-a6de-f7b35e8e2e3f" />
<img width="3040" height="1249" alt="图片" src="https://github.com/user-attachments/assets/d6ba2729-d817-4015-b948-b3e3570749ce" />



## 功能特点

- 支持多种音频/视频格式（wav, mp3, m4a, flac, ogg, aac, mp4, avi, mkv, mov 等）
- VAD 智能语音检测与分段，支持超长音频文件
- 可选降噪处理，提升转录质量
- 两种工作模式：本地文件 / 网址下载（B站、YouTube 等）
- 自动拼接转录结果，根据静音时长智能插入换行

## 环境要求

- **Windows**：Python 3.12+， ffmpeg
- **WSL**：Ubuntu 22.04 或其他支持 WSL 的发行版

## 快速开始

### 1. 环境配置

**Windows 端依赖**：
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**WSL 端安装 Qwen3-ASR**：
参考 [Qwen3-ASR 官方文档](https://github.com/QwenLM/Qwen3-ASR)：
```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U qwen-asr[vllm]
```

**模型下载**：
```bash
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./model/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./model/Qwen3-ASR-0.6B
```

### 2. 修改启动脚本配置

编辑 `start_asr_service.bat`，修改【必须配置】区域的路径：
- `PYTHON_PATH`：Windows 上的 Python 解释器路径
- `WSL_DISTRO`：你的 WSL 发行版名称
- `WSL_USER`：WSL 用户名
- `PROJECT_PATH`：WSL 中 Qwen3-ASR 项目路径
- `MODEL_0_6B` / `MODEL_1_7B`：模型文件路径

### 3. 启动服务

```bash
start_asr_service.bat
```

1. 选择模型（0.6B 快速 / 1.7B 精度更高）
2. 等待服务启动
3. 选择工作模式（本地文件 / 网址下载）
4. 拖入文件或输入网址开始转录

输入 `help` 查看详细帮助，输入 `quit` 退出程序。

## 配置说明

编辑 `start_asr_service.bat` 中的配置区域：

### 必须配置（【必须配置】区域）

| 配置项 | 说明 |
|--------|------|
| `PYTHON_PATH` | Windows Python 解释器路径 |
| `WSL_DISTRO` | WSL 发行版名称（如 Ubuntu-22.04） |
| `WSL_USER` | WSL 用户名 |
| `PROJECT_PATH` | WSL 中 Qwen3-ASR 项目路径 |
| `MODEL_0_6B` | Qwen3-ASR-0.6B 模型路径 |
| `MODEL_1_7B` | Qwen3-ASR-1.7B 模型路径 |

### 可选配置（【可选配置】区域）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `SERVICE_PORT` | 服务端口 | 8001 |
| `OUTPUT_MODE` | 输出模式：FIXED/SAME | FIXED |
| `OUTPUT_DIR` | 固定输出目录 | D:\Download\ |
| `enable-denoise` | 是否启用降噪 | true |
| `denoise-strength` | 降噪强度：light/medium/heavy | medium |
| `vad-threshold` | VAD检测灵敏度 (0-1) | 0.5 |
| `vad-min-silence` | VAD静音阈值 (秒) | 2 |
| `silence-newline` | 沉默超过此值插入换行 (秒) | 5 |
| `MAX_CHUNK_DURATION` | 单片段最长时长 (秒) | 600 |
| `RETRY_VAD_MIN_SILENCE` | 二次VAD静音阈值 (秒) | 0.6 |
| `GPU_MEMORY` | GPU 显存占用比例 | 0.7 |
| `MAX_MODEL_LEN` | 最大模型上下文长度 | 1024 |
| `WAIT_TIMEOUT` | 等待服务启动超时 (秒) | 60 |

## 命令行参数

客户端支持直接运行（不通过 bat）：

```bash
python asr_client.py [选项]

主要选项：
  --port              服务端口 (默认 8001)
  --output-mode      FIXED|SAME (默认 FIXED)
  --output-dir       输出目录 (默认 D:\Download\)
  --enable-denoise   true|false (默认 false)
  --vad-threshold    VAD灵敏度 0-1 (默认 0.5)
  --vad-min-silence  静音阈值秒数 (默认 0.3)
  --silence-newline  换行阈值秒数 (默认 5.0)
  --input-mode       FILE|URL (默认 FILE)
  --max-chunk-duration   片段最长秒数 (默认 60.0)
  --retry-vad-min-silence 二次VAD阈值 (默认 0.6)
```

## WSL 服务手动启动

```bash
cd ~/software/Qwen3-ASR
source .venv/bin/activate
qwen-asr-serve ~/software/Qwen3-ASR/model/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.7 --host 0.0.0.0 --port 8001 --max-model-len 1024
```

## 注意事项

- **端口**：bat 和 argparse 默认均为 8001，确保一致
- **WAV转换**：启用降噪时，所有文件都会重新编码为 16kHz mono PCM
- **VAD模型**：首次运行会下载 silero-vad 模型，需要网络
- **临时文件**：存储在 `%TEMP%\qwen_asr_temp\YYYY-MM\`，退出时自动清理超过30天的文件
- **日志文件**：存储在 `logs/asr_client_YYYYMM.log`
- **网址下载**：需要安装 yt-dlp（已包含在 requirements.txt 中）

## 项目结构

```
qwen3-asr-win-client/
├── asr_client.py          # Windows 客户端
├── start_asr_service.bat  # 启动脚本
├── requirements.txt       # Python 依赖
├── README.md              # 本文档
└── logs/                  # 日志目录
```

## 常见问题

**Q: 转录速度慢？**
A: 增大 `vad-min-silence`（减少片段数）或增大 `max-chunk-duration`。

**Q: ASR 报错？**
A: 减小 `max-chunk-duration`。

**Q: 识别不到语音？**
A: 降低 `vad-threshold`（如设为 0.3）。

**Q: 片段太长硬切后转录失败？**
A: 减小 `max-chunk-duration`，建议从 60 开始尝试。

## 相关链接

- [Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR 官方文档](https://github.com/QwenLM/Qwen3-ASR#readme)
