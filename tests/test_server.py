import unittest

from bilibili_subtitle_fetch.server import (
    choose_subtitle,
    format_subtitle_body,
    normalize_subtitle_url,
    parse_bilibili_url,
    select_cid,
)


class ServerHelpersTest(unittest.TestCase):
    def test_parse_bilibili_url_extracts_bvid_and_page(self) -> None:
        bvid, page = parse_bilibili_url(
            "https://www.bilibili.com/video/BV1fz4y1j7Mf?p=2"
        )
        self.assertEqual(bvid, "BV1fz4y1j7Mf")
        self.assertEqual(page, 2)

    def test_parse_bilibili_url_ignores_invalid_page(self) -> None:
        bvid, page = parse_bilibili_url(
            "https://www.bilibili.com/video/BV1fz4y1j7Mf?p=oops"
        )
        self.assertEqual(bvid, "BV1fz4y1j7Mf")
        self.assertIsNone(page)

    def test_select_cid_prefers_requested_page(self) -> None:
        info = {
            "cid": 100,
            "pages": [
                {"cid": 101},
                {"cid": 102},
            ],
        }
        self.assertEqual(select_cid(info, 2), 102)

    def test_select_cid_falls_back_to_default(self) -> None:
        info = {"cid": 100, "pages": [{"cid": 101}]}
        self.assertEqual(select_cid(info, 5), 100)

    def test_choose_subtitle_prefers_exact_match(self) -> None:
        subtitles = [
            {"lan": "zh-CN", "subtitle_url": "//manual.example", "ai_type": 0},
            {"lan": "en", "subtitle_url": "//english.example", "ai_type": 0},
        ]
        self.assertEqual(
            choose_subtitle(subtitles, "en"),
            ("//english.example", "en"),
        )

    def test_choose_subtitle_falls_back_to_manual(self) -> None:
        subtitles = [
            {"lan": "ai-zh", "subtitle_url": "//ai.example", "ai_type": 1},
            {"lan": "zh-CN", "subtitle_url": "//manual.example", "ai_type": 0},
        ]
        self.assertEqual(
            choose_subtitle(subtitles, "en"),
            ("//manual.example", "zh-CN"),
        )

    def test_normalize_subtitle_url(self) -> None:
        self.assertEqual(
            normalize_subtitle_url("//i0.hdslb.com/subtitle.json"),
            "https://i0.hdslb.com/subtitle.json",
        )
        self.assertIsNone(normalize_subtitle_url("ftp://invalid"))

    def test_format_subtitle_body_text(self) -> None:
        body = [
            {"content": "line 1"},
            {"content": "line 2"},
        ]
        self.assertEqual(format_subtitle_body(body, "text"), "line 1\nline 2")

    def test_format_subtitle_body_timestamped(self) -> None:
        body = [{"from": 1.25, "to": 2.5, "content": "hello"}]
        self.assertEqual(
            format_subtitle_body(body, "timestamped"),
            "00:00:01.250 --> 00:00:02.500\nhello",
        )


if __name__ == "__main__":
    unittest.main()
