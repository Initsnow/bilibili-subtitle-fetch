from bilibili_api import HEADERS, get_client, video
from typing import BinaryIO
from io import BytesIO


async def download(url: str,  intro: str, file: BinaryIO):
    dwn_id = await get_client().download_create(url, HEADERS)
    bts = 0
    tot = get_client().download_content_length(dwn_id)
    while True:
        bts += file.write(await get_client().download_chunk(dwn_id))
        # print(f"{intro} - [{bts} / {tot}]", end="\r")
        if bts == tot:
            break
    print()


async def download_audio(v: video.Video) -> BinaryIO:
    # 创建内存中的二进制文件对象
    file = BytesIO()

    # 获取视频下载链接
    download_url_data = await v.get_download_url(0)
    # 解析视频下载信息
    detecter = video.VideoDownloadURLDataDetecter(data=download_url_data)
    streams = detecter.detect_best_streams()
    # 有 MP4 流 / FLV 流两种可能
    if not detecter.check_flv_mp4_stream():
        await download(streams[1].url, "Downloading audio stream", file)
    else:
        print("MP4 stream not found")

    # 重置文件指针到开头
    file.seek(0)
    return file
