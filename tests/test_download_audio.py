import unittest
from types import SimpleNamespace

from bilibili_subtitle_fetch.download_audio import pick_download_url


class DownloadAudioHelpersTest(unittest.TestCase):
    def test_pick_download_url_prefers_audio_stream_for_dash(self) -> None:
        streams = [
            SimpleNamespace(url="https://video.example", video_quality=80),
            SimpleNamespace(url="https://audio.example", audio_quality=30280),
        ]
        self.assertEqual(
            pick_download_url(streams, is_muxed_stream=False),
            "https://audio.example",
        )

    def test_pick_download_url_uses_first_stream_for_muxed_media(self) -> None:
        streams = [SimpleNamespace(url="https://media.example")]
        self.assertEqual(
            pick_download_url(streams, is_muxed_stream=True),
            "https://media.example",
        )

    def test_pick_download_url_returns_none_when_no_audio_found(self) -> None:
        streams = [SimpleNamespace(url="https://video.example", video_quality=80)]
        self.assertIsNone(pick_download_url(streams, is_muxed_stream=False))


if __name__ == "__main__":
    unittest.main()
