from typing import Literal, BinaryIO
from faster_whisper import WhisperModel
import ctranslate2
import subprocess
import shutil


def get_device():
    # 如果安装的是 GPU 版 CTranslate2，名字一般会带 "-cuda"
    has_cuda_lib = "cuda" in ctranslate2.__version__ or shutil.which("nvidia-smi")

    if has_cuda_lib:
        try:
            # 进一步检测 nvidia-smi 是否返回正常
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return "cuda"
        except Exception:
            pass

    # Apple M 系列芯片
    try:
        if ctranslate2.Device.supports_device("mps"):
            return "mps"
    except Exception:
        pass

    return "cpu"


def generate_subtitles(
    audio: BinaryIO, type: Literal["text", "timestamped"], model_size: str = "small"
) -> str:
    device = get_device()
    print("Using device:", device)
    print(f"Loading whisper model: {model_size}")

    # 加载模型
    model = WhisperModel(model_size, device=device)

    # 转录
    print("Transcribing...")
    segments, info = model.transcribe(audio, beam_size=5)

    if type == "text":
        return "\n".join([segment.text.strip() for segment in segments])
    else:
        return "\n".join(
            [
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n{segment.text.strip()}\n"
                for segment in segments
            ]
        )


def format_timestamp(seconds: float) -> str:
    """将秒转换为 SRT 时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
