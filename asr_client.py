#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-ASR 音频转录客户端
功能：接收音频文件，VAD检测，智能分段，降噪，发送到WSL服务进行转录

第三方库：
pip install silero-vad
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
...
"""

import os
import re
import sys
import time
import argparse
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil
import torch
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps


class Colors:
    """终端颜色代码"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class ASRClient:
    """ASR客户端（增强版）"""

    def __init__(
        self,
        port=8000,
        output_mode="FIXED",
        output_dir="D:\\Download\\",
        timeout=60,
        enable_denoise=True,
        denoise_strength="medium",
        vad_threshold=0.5,
        vad_min_silence_duration=0.3,
        silence_for_newline=5.0,
        input_mode="FILE",
        max_chunk_duration=60.0,
        retry_vad_min_silence=0.6,
    ):
        self.service_url = f"http://localhost:{port}/v1/audio/transcriptions"
        self.output_mode = output_mode
        self.output_dir = output_dir
        self.timeout = timeout
        self.request_timeout = 300
        self.max_retries = 1

        # 降噪配置
        self.enable_denoise = enable_denoise
        self.denoise_params = {
            "light": "afftdn=nf=-20",
            "medium": "afftdn=nf=-25",
            "heavy": "afftdn=nf=-35",
        }
        self.denoise_filter = self.denoise_params.get(
            denoise_strength, self.denoise_params["medium"]
        )

        # VAD配置
        self.vad_threshold = vad_threshold
        self.vad_min_silence_duration = (
            vad_min_silence_duration  # VAD检测：多长的静音算片段边界（秒）
        )

        # 拼接标点配置
        self.silence_for_newline = silence_for_newline  # >2秒 → 换行

        # 输入模式与片段时长控制
        self.input_mode = input_mode  # FILE 或 URL
        self.max_chunk_duration = max_chunk_duration  # 单片段最长时长（秒）
        self.retry_vad_min_silence = retry_vad_min_silence  # 二次VAD静音阈值（秒）

        # 临时文件目录（按月管理）
        self.temp_base_dir = Path(tempfile.gettempdir()) / "qwen_asr_temp"
        self.temp_base_dir.mkdir(exist_ok=True)
        self.current_month_dir = self.temp_base_dir / datetime.now().strftime("%Y-%m")
        self.current_month_dir.mkdir(exist_ok=True)

        # 日志配置
        self.log_dir = Path(__file__).parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = (
            self.log_dir / f"asr_client_{datetime.now().strftime('%Y%m')}.log"
        )

        # 支持的格式
        self.audio_formats = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
        self.video_formats = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}

        # 初始化VAD模型
        self.vad_model = load_silero_vad()

    def log(self, message, level="INFO"):
        """写入日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
        if level == "ERROR":
            print(f"{Colors.FAIL}{log_message}{Colors.ENDC}")
        elif level == "WARNING":
            print(f"{Colors.WARNING}{log_message}{Colors.ENDC}")
        elif level == "SUCCESS":
            print(f"{Colors.OKGREEN}{log_message}{Colors.ENDC}")
        else:
            print(log_message)

    def wait_for_service(self):
        """等待服务就绪"""
        print(f"正在检测服务状态: {self.service_url}")
        print(f"最长等待时间: {self.timeout} 秒\n")
        for i in range(self.timeout):
            try:
                response = requests.get(
                    f"http://localhost:{self.service_url.split(':')[2].split('/')[0]}/health",
                    timeout=1,
                )
                if response.status_code == 200:
                    self.log("服务已就绪！", "SUCCESS")
                    return True
                time.sleep(1)
            except Exception:
                pass

            remaining = self.timeout - i - 1
            print(
                f"\r等待服务启动... ({i + 1}/{self.timeout}秒) 剩余: {remaining}秒",
                end="",
                flush=True,
            )
        print()
        self.log("服务启动超时，请检查WSL中的服务是否正常运行", "ERROR")
        return False

    def check_ffmpeg(self):
        """检查ffmpeg是否可用"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except FileNotFoundError:
            self.log("ffmpeg未安装或不在PATH中", "ERROR")
            return False
        except subprocess.TimeoutExpired:
            self.log("ffmpeg检查超时", "ERROR")
            return False
        except Exception as e:
            self.log(f"ffmpeg检查失败: {e}", "ERROR")
            return False

    def check_ffprobe(self):
        """检查ffprobe是否可用"""
        try:
            result = subprocess.run(
                ["ffprobe", "-version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            self.log(f"ffprobe检查失败: {e}", "ERROR")
            return False

    def check_ytdlp(self):
        """检查yt-dlp是否可用（Windows侧全局命令）"""
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except FileNotFoundError:
            self.log("yt-dlp未安装或不在PATH中，请先安装: pip install yt-dlp", "ERROR")
            return False
        except subprocess.TimeoutExpired:
            self.log("yt-dlp检查超时", "ERROR")
            return False
        except Exception as e:
            self.log(f"yt-dlp检查失败: {e}", "ERROR")
            return False

    def download_url(self, url):
        """使用yt-dlp下载网址对应的音频，返回本地文件路径；失败返回None"""
        if not self.check_ytdlp():
            return None

        self.log(f"开始下载: {url}", "INFO")
        download_dir = self.current_month_dir / f"download_{int(time.time())}"
        download_dir.mkdir(exist_ok=True)
        output_template = str(download_dir / "%(title).200B [%(id)s].%(ext)s")
        cmd = [
            "yt-dlp",
            "-f",
            "bestaudio[abr<=128k]/bestaudio",
            "-S",
            "abr:asc",
            "-x",
            "--audio-format",
            "m4a",
            "-o",
            output_template,
            url,
        ]
        self.log(f"yt-dlp指令：{' '.join(cmd)}", "INFO")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            if result.returncode != 0:
                self.log(f"yt-dlp下载失败: {result.stderr}", "ERROR")
                shutil.rmtree(download_dir, ignore_errors=True)
                return None

            files = list(download_dir.glob("*.m4a"))
            if not files:
                self.log(f"下载目录未找到m4a文件: {download_dir}", "ERROR")
                shutil.rmtree(download_dir, ignore_errors=True)
                return None

            downloaded_file = files[0]
            target_file = self.current_month_dir / downloaded_file.name
            shutil.move(str(downloaded_file), str(target_file))
            shutil.rmtree(download_dir, ignore_errors=True)

            self.log(f"下载完成: {target_file}", "SUCCESS")
            return str(target_file)
        except subprocess.TimeoutExpired:
            self.log("yt-dlp下载超时（超过10分钟）", "ERROR")
            shutil.rmtree(download_dir, ignore_errors=True)
            return None
        except Exception as e:
            self.log(f"yt-dlp下载异常: {e}", "ERROR")
            shutil.rmtree(download_dir, ignore_errors=True)
            return None

    def merge_short_segments_into_chunks(self, segments):
        """贪心合并过短片段，使每组语音内容时长不超过 max_chunk_duration。
        返回嵌套列表 [[seg1, seg2, ...], [seg3], ...]，每个内层列表是一组要拼接的段（无中间静音）。
        """
        if not segments:
            return []

        result = []
        current_group = [(segments[0][0], segments[0][1])]
        speech_in_group = segments[0][1] - segments[0][0]

        for seg_start, seg_end in segments[1:]:
            seg_duration = seg_end - seg_start
            if speech_in_group + seg_duration <= self.max_chunk_duration:
                current_group.append((seg_start, seg_end))
                speech_in_group += seg_duration
            else:
                result.append(current_group)
                current_group = [(seg_start, seg_end)]
                speech_in_group = seg_duration

        result.append(current_group)

        if len(result) < len(segments):
            self.log(f"合并短片段: {len(segments)} -> {len(result)} 组", "INFO")

        return result

    def split_long_segment(self, wav_file, start, end, chunk_index):
        """对超过max_chunk_duration的单个片段做二次处理：
        先用更小的VAD静音阈值重新切分，再贪心合并，仍超长则硬切。
        返回嵌套列表 [[(abs_start, abs_end), ...], ...]，文件名带sub_index防冲突。"""
        duration = end - start
        self.log(
            f"片段 {chunk_index} 时长 {duration:.1f}s 超过上限 {self.max_chunk_duration}s，启动二次切分",
            "WARNING",
        )

        tmp_file = (
            self.current_month_dir / f"chunk_{chunk_index}_long_{int(time.time())}.wav"
        )
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(wav_file),
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-y",
                str(tmp_file),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", timeout=60
            )
            if result.returncode != 0:
                self.log(f"提取超长片段失败，将直接硬切: {result.stderr}", "WARNING")
                tmp_file = None
        except Exception as e:
            self.log(f"提取超长片段异常，将直接硬切: {e}", "WARNING")
            tmp_file = None

        sub_groups = []

        if tmp_file and Path(tmp_file).exists():
            try:
                wav_data, sr = sf.read(str(tmp_file))
                wav_tensor = torch.from_numpy(wav_data).float()
                timestamps = get_speech_timestamps(
                    wav_tensor,
                    self.vad_model,
                    threshold=self.vad_threshold,
                    return_seconds=True,
                )
                if timestamps:
                    # 注意：tmp_file 是从 start 处裁剪的，VAD 返回的是相对于 tmp_file 的时间戳，
                    # 必须加上 start 偏移量才能还原为原始 wav_file 中的绝对时间戳。
                    rel_segs = [
                        (float(t["start"]) + start, float(t["end"]) + start) for t in timestamps
                    ]
                    merged = []
                    cs, ce = rel_segs[0]
                    for ss, se in rel_segs[1:]:
                        if ss - ce <= self.retry_vad_min_silence:
                            ce = se
                        else:
                            merged.append((cs, ce))
                            cs, ce = ss, se
                    merged.append((cs, ce))

                    current_group = [merged[0]]
                    sp_in_g = merged[0][1] - merged[0][0]
                    for ms, me in merged[1:]:
                        seg_dur = me - ms
                        if sp_in_g + seg_dur <= self.max_chunk_duration:
                            current_group.append((ms, me))
                            sp_in_g += seg_dur
                        else:
                            sub_groups.append(current_group)
                            current_group = [(ms, me)]
                            sp_in_g = seg_dur
                    sub_groups.append(current_group)

                    self.log(
                        f"二次VAD切分片段 {chunk_index}: 得到 {len(sub_groups)} 个子组",
                        "INFO",
                    )
            except Exception as e:
                self.log(f"二次VAD切分失败: {e}，将直接硬切", "WARNING")
                sub_groups = []

        final_groups = []
        if sub_groups:
            for group in sub_groups:
                group_speech = sum(seg[1] - seg[0] for seg in group)
                if group_speech > self.max_chunk_duration:
                    self.log(
                        f"子组语音时长 {group_speech:.1f}s 仍超长，强制硬切",
                        "WARNING",
                    )
                    for seg in group:
                        abs_s, abs_e = seg
                        seg_dur = abs_e - abs_s
                        if seg_dur > self.max_chunk_duration:
                            cur = abs_s
                            while cur < abs_e:
                                hard_end = min(cur + self.max_chunk_duration, abs_e)
                                if final_groups and len(final_groups[-1]) == 0:
                                    final_groups[-1].append((cur, hard_end))
                                else:
                                    final_groups.append([(cur, hard_end)])
                                cur = hard_end
                        else:
                            final_groups.append([(abs_s, abs_e)])
                else:
                    final_groups.append(group)
        else:
            cur = start
            while cur < end:
                hard_end = min(cur + self.max_chunk_duration, end)
                if final_groups and len(final_groups[-1]) == 0:
                    final_groups[-1].append((cur, hard_end))
                else:
                    final_groups.append([(cur, hard_end)])
                cur = hard_end
            self.log(f"硬切片段 {chunk_index}: 切为 {len(final_groups)} 段", "INFO")

        return final_groups

    def get_audio_duration(self, audio_file):
        """获取音频时长（秒）"""
        get_timeout = 35  # 获取音频时长最大等待时间
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_file),
            ]
            self.log(f"获取音频时长（秒）的指令为：{' '.join(cmd)}", "INFO")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=get_timeout
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                self.log(f"无法获取音频时长: {result.stderr}", "WARNING")
                return None
        except subprocess.TimeoutExpired:
            self.log(f"获取音频时长超时（>{get_timeout}秒）", "WARNING")
            return None
        except Exception as e:
            self.log(f"获取音频时长异常: {e}", "WARNING")
            return None

    def convert_to_wav(self, input_file):
        """转换音频/视频文件为WAV格式，可选降噪"""
        input_path = Path(input_file)
        output_file = (
            self.current_month_dir / f"{input_path.stem}_{int(time.time())}.wav"
        )

        if self.enable_denoise:
            self.log(f"正在转换并降噪: {input_path.name} -> WAV格式", "INFO")
        else:
            self.log(f"正在转换文件: {input_path.name} -> WAV格式", "INFO")

        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
            ]
            if self.enable_denoise:
                cmd.extend(["-af", self.denoise_filter])
            cmd.extend(["-y", str(output_file)])

            self.log(f"转换音频/视频文件为WAV格式的指令为：{' '.join(cmd)}", "INFO")
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", timeout=300
            )
            if result.returncode != 0:
                self.log(f"转换失败: {result.stderr}", "ERROR")
                return None
            self.log(f"转换成功: {output_file}", "SUCCESS")
            return str(output_file)
        except subprocess.TimeoutExpired:
            self.log("转换超时（超过5分钟）", "ERROR")
            return None
        except Exception as e:
            self.log(f"转换异常: {e}", "ERROR")
            return None

    def detect_speech_segments(self, wav_file):
        """使用 VAD 检测语音片段，返回时间段列表 [(start, end), ...]"""

        try:
            wav, sr = sf.read(wav_file)
            wav = torch.from_numpy(wav).float()
            speech_timestamps = get_speech_timestamps(
                wav, self.vad_model, threshold=self.vad_threshold, return_seconds=True
            )

            if not speech_timestamps:
                self.log("未检测到语音片段", "WARNING")
                return []

            self.log(f"检测到 {len(speech_timestamps)} 个语音片段", "SUCCESS")

            # 确保返回的是数值类型的元组列表
            segments = []
            for ts in speech_timestamps:
                # 强制转换为浮点数
                start_time = float(ts.get("start", 0))
                end_time = float(ts.get("end", 0))
                segments.append((start_time, end_time))

            return segments

        except Exception as e:
            self.log(f"VAD 检测失败: {e}", "ERROR")
            return None

    def merge_nearby_segments(self, segments):
        """合并间隔很短的语音片段（避免过度碎片化）"""
        if not segments:
            return []

        merged = []
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            if start - current_end <= self.vad_min_silence_duration:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))

        if len(merged) < len(segments):
            self.log(f"合并相近片段: {len(segments)} -> {len(merged)}", "INFO")

        return merged

    def extract_audio_chunk(
        self, wav_file, start_time, end_time, chunk_index, sub_index=None
    ):
        """使用 ffmpeg 提取音频片段；sub_index 用于二次切分时防止文件名冲突"""
        if sub_index is not None:
            chunk_file = (
                self.current_month_dir
                / f"chunk_{chunk_index}_{sub_index}_{int(time.time())}.wav"
            )
        else:
            chunk_file = (
                self.current_month_dir / f"chunk_{chunk_index}_{int(time.time())}.wav"
            )
        try:
            duration = end_time - start_time
            cmd = [
                "ffmpeg",
                "-i",
                str(wav_file),
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-y",
                str(chunk_file),
            ]
            self.log(f"提取音频片段的指令为：{' '.join(cmd)}", "INFO")
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", timeout=30
            )
            if result.returncode != 0:
                self.log(f"提取音频片段失败: {result.stderr}", "ERROR")
                return None
            return str(chunk_file)
        except Exception as e:
            self.log(f"提取音频片段异常: {e}", "ERROR")
            return None

    def extract_and_concat_segments(self, wav_file, segments, chunk_index):
        """提取多个音频片段并拼接成一个无静音的连续音频文件。

        Args:
            wav_file: 源音频文件路径
            segments: 段列表，如 [(10,15), (18,22)]
            chunk_index: 当前片段索引（用于文件名防冲突）

        Returns:
            拼接后的文件路径，或 None（失败）
        """
        if not segments:
            return None

        if len(segments) == 1:
            return self.extract_audio_chunk(
                wav_file, segments[0][0], segments[0][1], chunk_index
            )

        temp_files = []
        try:
            for i, (start, end) in enumerate(segments):
                tmp_file = (
                    self.current_month_dir
                    / f"chunk_{chunk_index}_sub_{i}_{time.time_ns()}.wav"
                )
                duration = end - start
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(wav_file),
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),
                    "-c",
                    "copy",
                    "-y",
                    str(tmp_file),
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8", timeout=30
                )
                if result.returncode != 0:
                    self.log(f"提取片段 {i} 失败: {result.stderr}", "ERROR")
                    return None
                temp_files.append(str(tmp_file))

            concat_list_file = (
                self.current_month_dir / f"concat_{chunk_index}_{int(time.time())}.txt"
            )
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")

            output_file = (
                self.current_month_dir / f"chunk_{chunk_index}_{int(time.time())}.wav"
            )
            cmd = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_file),
                "-c",
                "copy",
                "-y",
                str(output_file),
            ]
            self.log(f"拼接音频片段指令：{' '.join(cmd)}", "INFO")
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", timeout=60
            )
            if result.returncode != 0:
                self.log(f"拼接音频片段失败: {result.stderr}", "ERROR")
                return None

            for tf in temp_files:
                try:
                    Path(tf).unlink()
                except Exception:
                    pass
            try:
                concat_list_file.unlink()
            except Exception:
                pass

            self.log(f"拼接完成: {output_file} ({len(segments)} 段拼接)", "SUCCESS")
            return str(output_file)

        except Exception as e:
            self.log(f"拼接音频片段异常: {e}", "ERROR")
            for tf in temp_files:
                try:
                    Path(tf).unlink()
                except Exception:
                    pass
            return None

    def transcribe_audio(self, audio_file):
        """发送音频文件进行转录"""
        try:
            file_name = Path(audio_file).name
            with open(audio_file, "rb") as f:
                files = {"file": (file_name, f, "audio/wav")}
                response = requests.post(
                    self.service_url, files=files, timeout=self.request_timeout
                )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                error_msg = f"服务器返回错误: {response.status_code} - {response.text}"
                self.log(error_msg, "ERROR")
                return None
        except requests.exceptions.Timeout:
            self.log(f"请求超时（超过{self.request_timeout}秒）", "ERROR")
            return None
        except requests.exceptions.ConnectionError:
            self.log("无法连接到服务，请检查服务是否正常运行", "ERROR")
            return None
        except Exception as e:
            self.log(f"转录异常: {e}", "ERROR")
            return None

    def merge_transcripts_with_punctuation(self, transcript_segments):
        """智能拼接转录文本，根据片段间的沉默时长插入标点"""
        if not transcript_segments:
            return ""

        result_parts = []
        for i, (text, segment_info) in enumerate(transcript_segments):
            if not text:
                continue

            result_parts.append(text.strip())

            # 如果不是最后一段，根据沉默时长决定标点
            if i < len(transcript_segments) - 1:
                current_end = segment_info[1]
                next_start = transcript_segments[i + 1][1][0]
                silence_duration = next_start - current_end

                if silence_duration >= self.silence_for_newline:
                    result_parts.append("\n")

        return "".join(result_parts)

    def save_transcript(self, original_file, text):
        """保存转录文本"""
        original_path = Path(original_file)
        if self.output_mode == "SAME":
            output_dir = original_path.parent
        else:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{original_path.stem}.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            self.log(f"转录结果已保存: {output_file}", "SUCCESS")
            return str(output_file)
        except Exception as e:
            self.log(f"保存文件失败: {e}", "ERROR")
            return None

    def cleanup_old_temp_folders(self):
        """清理超过1个月的临时文件夹"""
        try:
            if not self.temp_base_dir.exists():
                return

            current_date = datetime.now()
            one_month_ago = current_date - timedelta(days=30)

            for folder in self.temp_base_dir.iterdir():
                if not folder.is_dir():
                    continue

                try:
                    # 解析文件夹名称（格式：YYYY-MM）
                    folder_date = datetime.strptime(folder.name, "%Y-%m")
                    if folder_date < one_month_ago:
                        self.log(f"清理过期临时文件夹: {folder.name}", "INFO")
                        shutil.rmtree(folder)
                except ValueError:
                    # 文件夹名称格式不对，跳过
                    continue
                except Exception as e:
                    self.log(f"清理文件夹失败 {folder.name}: {e}", "WARNING")
        except Exception as e:
            self.log(f"清理临时文件夹异常: {e}", "WARNING")

    def process_file(self, file_path):
        """处理单个文件 - 主流程"""
        print("\n" + "=" * 60)

        file_path_obj = Path(file_path.strip('"').strip("'"))
        # 验证文件存在
        if not file_path_obj.exists():
            self.log(f"文件不存在: {file_path_obj}", "ERROR")
            return False
        file_ext = file_path_obj.suffix.lower()

        # 检查文件格式
        if file_ext not in self.audio_formats and file_ext not in self.video_formats:
            self.log(f"不支持的文件格式: {file_ext}", "ERROR")
            self.log(f"支持的音频格式: {', '.join(self.audio_formats)}", "INFO")
            self.log(f"支持的视频格式: {', '.join(self.video_formats)}", "INFO")
            return False

        # 检查工具
        if not self.check_ffmpeg():
            self.log("ffmpeg不可用，无法处理文件", "ERROR")
            return False
        if not self.check_ffprobe():
            self.log("ffprobe不可用，无法获取音频信息", "ERROR")
            return False

        try:
            # ============ 步骤1：转换为 WAV（带降噪） ============
            if file_ext != ".wav" or self.enable_denoise:
                wav_file = self.convert_to_wav(file_path_obj)
                if not wav_file:
                    return False
            else:
                wav_file = file_path_obj

            # ============ 步骤2：获取音频时长 ============
            total_duration = self.get_audio_duration(wav_file)
            if total_duration:
                self.log(f"音频总时长: {total_duration:.1f} 秒", "INFO")

            # ============ 步骤3：VAD 检测语音片段 ============
            # 返回数据(speech_segments数据)：[{'start': 0.5, 'end': 4.3}, {'start': 4.5, 'end': 7.0}, {'start': 7.2, 'end': 10.5}, ...]
            speech_segments = self.detect_speech_segments(wav_file)
            if speech_segments is None:
                self.log("VAD 检测失败，无法继续处理", "ERROR")
                return False
            if not speech_segments:
                self.log("未检测到语音，文件可能为空或全是静音", "WARNING")
                return False

            # ============ 步骤4：合并相近片段（避免过度碎片化） ============
            speech_segments = self.merge_nearby_segments(speech_segments)
            if not speech_segments:
                self.log("合并后没有有效片段", "WARNING")
                return False

            self.log(f"VAD片段数量: {len(speech_segments)}", "INFO")

            # ============ 步骤4.5：合并短片段 + 处理超长片段 ============
            speech_groups = self.merge_short_segments_into_chunks(speech_segments)
            final_groups = []
            for idx, seg_list in enumerate(speech_groups, 1):
                group_speech = sum(seg[1] - seg[0] for seg in seg_list)
                if group_speech > self.max_chunk_duration:
                    sub = self.split_long_segment(
                        wav_file, seg_list[0][0], seg_list[-1][1], idx
                    )
                    final_groups.extend(sub)
                else:
                    final_groups.append(seg_list)
            self.log(f"最终片段组数量: {len(final_groups)}", "INFO")

            # ============ 步骤5：提取音频片段（去静音拼接） ============
            self.log(f"开始提取 {len(final_groups)} 个音频片段", "INFO")
            audio_chunks = []
            for idx, seg_list in enumerate(final_groups, 1):
                total_speech = sum(seg[1] - seg[0] for seg in seg_list)
                seg_repr = ", ".join(f"({s:.1f},{e:.1f})" for s, e in seg_list)
                self.log(
                    f"提取片段 [{idx}/{len(final_groups)}]: {seg_repr} (语音总时长: {total_speech:.1f}s)",
                    "INFO",
                )
                chunk_file = self.extract_and_concat_segments(wav_file, seg_list, idx)
                if chunk_file:
                    audio_chunks.append((chunk_file, (seg_list[0][0], seg_list[-1][1])))
                else:
                    self.log(f"片段 {idx} 提取失败，跳过", "WARNING")

            if not audio_chunks:
                self.log("所有片段提取失败", "ERROR")
                return False

            # ============ 步骤6：逐段转录 ============
            self.log(f"开始转录 {len(audio_chunks)} 个音频片段", "INFO")
            transcript_segments = []
            for idx, (chunk_file, segment_info) in enumerate(audio_chunks, 1):
                # 尝试转录（带重试）
                text = None
                for retry in range(self.max_retries + 1):
                    text = self.transcribe_audio(chunk_file)
                    if text is not None:
                        break
                    if retry < self.max_retries:
                        self.log(f"片段 {idx} 转录失败，3秒后重试...", "WARNING")
                        time.sleep(3)

                if text is None:
                    self.log(
                        f"片段 {idx} 转录失败（已重试{self.max_retries}次），使用占位符",
                        "ERROR",
                    )
                    text = f"[{chunk_file} 转录失败]"

                # 替换语言标签`language Chinese<asr_text>`为换行符
                text = re.sub(r"language\s+\S+?\s*<asr_text>", "\n", text)

                transcript_segments.append((text, segment_info))
                self.log(
                    f"片段 [{idx}/{len(audio_chunks)}] 转录完成: {len(text)} 字符",
                    "SUCCESS",
                )

            if not transcript_segments:
                self.log("所有片段转录失败", "ERROR")
                return False

            # ============ 步骤7：智能拼接 ============
            self.log("正在智能拼接转录结果...", "INFO")
            final_text = self.merge_transcripts_with_punctuation(transcript_segments)

            # ============ 步骤8：保存结果 ============
            output_file = self.save_transcript(file_path_obj, final_text)
            if not output_file:
                return False

            # 显示结果预览
            print(f"\n{Colors.OKCYAN}{'=' * 60}")
            print("转录结果预览（前200字符）：")
            print(f"{'=' * 60}{Colors.ENDC}")
            preview = final_text[:200] + ("..." if len(final_text) > 200 else "")
            print(preview)
            print(f"{Colors.OKCYAN}{'=' * 60}{Colors.ENDC}\n")

            self.log(
                f"转录完成！共处理 {len(transcript_segments)} 个片段，总计 {len(final_text)} 字符",
                "SUCCESS",
            )
            return True

        except Exception as e:
            self.log(f"处理文件时发生异常: {e}", "ERROR")
            import traceback

            self.log(traceback.format_exc(), "ERROR")
            return False

    def show_mode_selection(self):
        """显示模式选择菜单，返回用户选择的模式"""
        while True:
            print(f"\n{Colors.OKCYAN}{'=' * 60}")
            print("请选择工作模式：")
            print("  [1] 拖入本地文件")
            print("  [2] 输入网址下载")
            print(f"{'=' * 60}{Colors.ENDC}")

            try:
                choice = (
                    input(f"{Colors.BOLD}请输入数字 [1/2]: {Colors.ENDC}")
                    .strip()
                    .lower()
                )

                if choice == "1":
                    self.input_mode = "FILE"
                    print(f"\n{Colors.OKGREEN}已选择: 本地文件模式{Colors.ENDC}")
                    return "FILE"
                elif choice == "2":
                    self.input_mode = "URL"
                    print(f"\n{Colors.OKGREEN}已选择: 网址下载模式{Colors.ENDC}")
                    return "URL"
                else:
                    print(f"{Colors.WARNING}无效选择，请输入 1 或 2{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}操作已取消{Colors.ENDC}")
                return self.input_mode

    def show_help(self):
        """显示帮助信息"""
        help_text = f"""
{Colors.OKCYAN}{"=" * 60}
Qwen3-ASR 音频转录客户端 - 使用帮助
{"=" * 60}{Colors.ENDC}

{Colors.BOLD}基本命令：{Colors.ENDC}
  help / ?        - 显示此帮助信息
  quit / exit / q - 退出程序
  clear           - 清屏
  b                - 返回模式选择

{Colors.BOLD}使用方法：{Colors.ENDC}
  1. FILE 模式：将音频或视频文件拖入窗口
     URL 模式：输入视频网址
  2. 按回车键开始转录
  3. 等待转录完成（转录结果保存为.txt文件）
  4. 输入 'b' 可返回模式选择切换工作模式

{Colors.BOLD}支持的格式：{Colors.ENDC}
  • 音频: {" ".join(sorted(self.audio_formats))}
  • 视频: {" ".join(sorted(self.video_formats))}
  • 若遇到需要解析的其它格式，可尝试自行在函数convert_to_wav中添加

{Colors.BOLD}当前配置：{Colors.ENDC}
  工作模式：{"网址下载模式 (URL)" if self.input_mode == "URL" else "本地文件模式 (FILE)"}
  降噪处理：{"已启用 (" + self.denoise_filter + ")" if self.enable_denoise else "已禁用"}
  VAD检测灵敏度：{self.vad_threshold} (决定人声识别效果，0-1)
  VAD静音阈值：{self.vad_min_silence_duration}秒 (语音片段切分边界)
  换行静音阈值：{self.silence_for_newline}秒 (文本换行插入)
  片段最长时长：{self.max_chunk_duration}秒 (超出触发二次切分)
  二次VAD静音阈值：{self.retry_vad_min_silence}秒 (超长片段再切分时使用)

{Colors.BOLD}输出设置：{Colors.ENDC}
  输出模式：{self.output_mode}
  {"输出目录：" + self.output_dir if self.output_mode == "FIXED" else "输出位置：与源文件相同目录"}

{Colors.BOLD}功能说明：{Colors.ENDC}
  • VAD语音检测：自动识别有语音的片段，跳过静音部分
  • 智能分段：长音频自动切分为多个片段，每段不超过最长时长上限
  • 超长片段处理：先用更小的静音阈值重新VAD切分，仍超长则硬切
  • 降噪处理：可选的音频降噪，提升转录质量
  • 智能换行：根据沉默时长自动在文本中插入换行
  • 网址下载：支持通过yt-dlp下载B站/YouTube等网站的音视频后转录

{Colors.BOLD}性能优化提示：{Colors.ENDC}
  • 如果转录速度慢：可以增大VAD静音阈值(减少片段数)或增大片段最长时长
  • 如果ASR报错：可以减小片段最长时长上限
  • 如果识别不到语音：可以降低VAD检测灵敏度

{Colors.BOLD}临时文件：{Colors.ENDC}
  位置：{self.current_month_dir}
  清理策略：退出时自动删除超过1个月的临时文件

{Colors.BOLD}日志文件：{Colors.ENDC}
  {self.log_file}

{Colors.OKCYAN}{"=" * 60}{Colors.ENDC}
"""
        print(help_text)

    def run(self):
        """主运行循环"""
        if not self.wait_for_service():
            return

        print()
        print(f"{Colors.OKGREEN}{'=' * 60}")
        print(
            f"{Colors.BOLD}Qwen3-ASR 服务已就绪，可以开始转录！{Colors.ENDC}{Colors.OKGREEN}"
        )
        print(f"{'=' * 60}{Colors.ENDC}")
        print()

        print(f"{Colors.OKCYAN}【提示】输入 help 查看配置详情与帮助；输入 q 退出。{Colors.ENDC}")
        print()
        print(f"{'=' * 60}")
        print()

        self.show_mode_selection()

        while True:
            self.run_processing_loop()

    def run_processing_loop(self):
        """处理任务内层循环"""
        while True:
            if self.input_mode == "URL":
                prompt = f"{Colors.BOLD}请输入网址: {Colors.ENDC}"
            else:
                prompt = f"{Colors.BOLD}请拖入文件: {Colors.ENDC}"

            try:
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                command = user_input.lower()

                if command in ["quit", "exit", "q"]:
                    print(f"\n{Colors.WARNING}正在退出并清理临时文件...{Colors.ENDC}")
                    self.cleanup_old_temp_folders()
                    self.log("用户退出程序", "INFO")
                    sys.exit(0)

                if command == "help":
                    self.show_help()
                    continue

                if command == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue

                if command == "b":
                    print(f"\n{Colors.OKCYAN}返回模式选择...{Colors.ENDC}")
                    self.show_mode_selection()
                    continue

                if self.input_mode == "URL":
                    url = user_input.strip()
                    local_file = self.download_url(url)
                    if local_file:
                        self.process_file(local_file)
                else:
                    file_path = user_input.strip('"').strip("'")
                    self.process_file(file_path)

            except KeyboardInterrupt:
                print(
                    f"\n\n{Colors.WARNING}检测到 Ctrl+C，正在退出并清理临时文件...{Colors.ENDC}"
                )
                self.cleanup_old_temp_folders()
                self.log("用户中断程序", "INFO")
                sys.exit(0)
            except Exception as e:
                self.log(f"未预期的错误: {e}", "ERROR")
                continue


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 音频转录客户端（增强版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置说明：
  降噪配置：
    --enable-denoise         启用降噪（true/false，默认false）
    --denoise-strength       降噪强度：light/medium/heavy（默认medium）
    
  VAD配置：
    --vad-threshold          VAD检测灵敏度，决定人声识别效果（0-1，推荐0.5）
    --vad-min-silence        VAD检测静音阈值（秒，默认2）
                            含义：检测到超过此时长的静音时，会将音频切分为独立片段
                            此值太小会分太多段，导致识别速度下降明显
                            如果音频片段太长导致ASR报错，可以适当减小此值
    
  文本格式配置：
    --silence-newline        沉默时长大于此值插入换行（秒，默认5）
                            含义：两个语音片段之间的静音时长超过此值时，在转录文本中插入换行
    
  输出配置：
    --output-mode            输出模式：FIXED(固定目录) / SAME(与音频文件同目录)
    --output-dir             当output-mode=FIXED时的输出目录
    """,
    )

    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    parser.add_argument(
        "--output-mode",
        type=str,
        default="FIXED",
        choices=["FIXED", "SAME"],
        help="输出模式",
    )
    parser.add_argument(
        "--output-dir", type=str, default="D:\\Download\\", help="固定输出目录"
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="等待服务启动的最大超时时间（秒）"
    )
    parser.add_argument(
        "--enable-denoise",
        type=lambda x: x.lower() == "true",
        default=False,
        help="启用降噪（默认False）",
    )
    parser.add_argument(
        "--denoise-strength",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="降噪强度",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=0.5, help="VAD检测灵敏度（0-1）"
    )
    parser.add_argument(
        "--vad-min-silence", type=float, default=0.3, help="VAD检测静音阈值（秒）"
    )
    parser.add_argument(
        "--silence-newline", type=float, default=5.0, help="沉默>此值插入换行（秒）"
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="FILE",
        choices=["FILE", "URL"],
        help="输入模式：FILE=本地文件，URL=网址下载",
    )
    parser.add_argument(
        "--max-chunk-duration",
        type=float,
        default=60.0,
        help="单片段最长时长（秒），超出触发二次切分，默认60",
    )
    parser.add_argument(
        "--retry-vad-min-silence",
        type=float,
        default=0.6,
        help="超长片段二次VAD切分时的静音阈值（秒），默认0.6",
    )

    args = parser.parse_args()

    # 创建并运行客户端
    client = ASRClient(
        port=args.port,
        output_mode=args.output_mode,
        output_dir=args.output_dir,
        timeout=args.timeout,
        enable_denoise=args.enable_denoise,
        denoise_strength=args.denoise_strength,
        vad_threshold=args.vad_threshold,
        vad_min_silence_duration=args.vad_min_silence,
        silence_for_newline=args.silence_newline,
        input_mode=args.input_mode,
        max_chunk_duration=args.max_chunk_duration,
        retry_vad_min_silence=args.retry_vad_min_silence,
    )
    client.run()


if __name__ == "__main__":
    main()
