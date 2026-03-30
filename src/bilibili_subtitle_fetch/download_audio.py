from io import BytesIO
from typing import BinaryIO, Optional

from bilibili_api import HEADERS, get_client, video


async def download(url: str, file: BinaryIO) -> None:
    download_id = await get_client().download_create(url, HEADERS)
    downloaded = 0
    total = get_client().download_content_length(download_id)

    while downloaded < total:
        chunk = await get_client().download_chunk(download_id)
        downloaded += file.write(chunk)


def pick_download_url(streams: list[object], is_muxed_stream: bool) -> Optional[str]:
    if not streams:
        return None

    if is_muxed_stream:
        return getattr(streams[0], "url", None)

    for stream in streams:
        if stream is None:
            continue
        if hasattr(stream, "audio_quality"):
            return getattr(stream, "url", None)

    return None


async def download_audio(v: video.Video) -> BinaryIO:
    file = BytesIO()
    download_url_data = await v.get_download_url(0)
    detecter = video.VideoDownloadURLDataDetecter(data=download_url_data)
    streams = detecter.detect_best_streams()

    media_url = pick_download_url(streams, detecter.check_flv_mp4_stream())
    if not media_url:
        raise RuntimeError("Could not find a downloadable audio/media stream.")

    await download(media_url, file)
    file.seek(0)
    return file
