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
    audio: BinaryIO, type: Literal["text", "timestamped"], model_size: str = "base"
) -> str:
    device = get_device()

    # 针对低配置/低内存 VPS 的优化：
    # 1. CPU 环境下默认使用 int8 量化，显著降低内存占用并提升速度
    # 2. auto 会在 GPU 上尝试 float16，不兼容则回退
    compute_type = "int8" if device == "cpu" else "default"

    print(f"Using device: {device}, compute_type: {compute_type}")
    print(f"Loading whisper model: {model_size}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    print("Transcribing...")
    segments, info = model.transcribe(audio)

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
