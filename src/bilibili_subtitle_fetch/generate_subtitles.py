import shutil
import subprocess
from typing import BinaryIO, Literal

import ctranslate2
from faster_whisper import WhisperModel


def get_device() -> str:
    has_cuda_runtime = "cuda" in ctranslate2.__version__ or shutil.which("nvidia-smi")
    if has_cuda_runtime:
        try:
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return "cuda"
        except Exception:
            pass

    try:
        if ctranslate2.Device.supports_device("mps"):
            return "mps"
    except Exception:
        pass

    return "cpu"


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def generate_subtitles(
    audio: BinaryIO,
    type: Literal["text", "timestamped"],
    model_size: str = "base",
) -> str:
    device = get_device()
    compute_type = "int8" if device == "cpu" else "default"

    print(f"Using device: {device}, compute_type: {compute_type}")
    print(f"Loading whisper model: {model_size}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    print("Transcribing...")
    segments, _ = model.transcribe(audio)

    if type == "text":
        return "\n".join(segment.text.strip() for segment in segments)

    return "\n".join(
        (
            f"{format_timestamp(segment.start)} --> "
            f"{format_timestamp(segment.end)}\n"
            f"{segment.text.strip()}\n"
        )
        for segment in segments
    )
