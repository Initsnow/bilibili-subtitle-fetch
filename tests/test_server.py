import io
import unittest
from unittest.mock import AsyncMock, patch

from bilibili_subtitle_fetch.credentials import CredentialStoreError
from bilibili_subtitle_fetch.server import (
    choose_subtitle,
    copy_to_clipboard,
    format_subtitle_body,
    main,
    normalize_subtitle_url,
    parse_cli_video_input,
    parse_bilibili_url,
    run_fetch_command,
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

    def test_parse_cli_video_input_distinguishes_bvid(self) -> None:
        self.assertEqual(
            parse_cli_video_input("BV1fz4y1j7Mf"),
            (None, "BV1fz4y1j7Mf"),
        )
        self.assertEqual(
            parse_cli_video_input("https://www.bilibili.com/video/BV1fz4y1j7Mf"),
            ("https://www.bilibili.com/video/BV1fz4y1j7Mf", None),
        )

    def test_copy_to_clipboard_uses_pyperclip(self) -> None:
        with patch("bilibili_subtitle_fetch.server.pyperclip.copy") as copy_mock:
            copy_to_clipboard("字幕内容")

        copy_mock.assert_called_once_with("字幕内容")

    def test_run_fetch_command_prints_and_copies_subtitle(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch(
            "bilibili_subtitle_fetch.server.fetch_bilibili_subtitle_text",
            new=AsyncMock(return_value="字幕内容"),
        ) as fetch_mock:
            with patch(
                "bilibili_subtitle_fetch.server.copy_to_clipboard"
            ) as clipboard_mock:
                run_fetch_command(
                    "BV1fz4y1j7Mf",
                    preferred_lang="zh-CN",
                    output_format="text",
                    stdout=stdout,
                    stderr=stderr,
                )

        fetch_mock.assert_awaited_once_with(
            url=None,
            bvid="BV1fz4y1j7Mf",
            preferred_lang="zh-CN",
            output_format="text",
        )
        clipboard_mock.assert_called_once_with("字幕内容")
        self.assertEqual(stdout.getvalue(), "字幕内容\n")
        self.assertEqual(stderr.getvalue(), "Copied subtitles to clipboard.\n")

    def test_run_fetch_command_warns_when_clipboard_copy_fails(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch(
            "bilibili_subtitle_fetch.server.fetch_bilibili_subtitle_text",
            new=AsyncMock(return_value="字幕内容"),
        ):
            with patch(
                "bilibili_subtitle_fetch.server.copy_to_clipboard",
                side_effect=RuntimeError("clipboard missing"),
            ):
                run_fetch_command(
                    "BV1fz4y1j7Mf",
                    preferred_lang="zh-CN",
                    output_format="text",
                    stdout=stdout,
                    stderr=stderr,
                )

        self.assertEqual(stdout.getvalue(), "字幕内容\n")
        self.assertIn("Warning: Failed to copy subtitles to clipboard", stderr.getvalue())

    def test_main_errors_when_config_missing(self) -> None:
        with patch(
            "bilibili_subtitle_fetch.server.CREDENTIAL_MANAGER.validate_runtime_config",
            side_effect=CredentialStoreError("missing config"),
        ):
            with patch("sys.argv", ["bilibili-subtitle-fetch", "serve"]):
                with self.assertRaises(SystemExit) as ctx:
                    main()

        self.assertEqual(str(ctx.exception), "Error: missing config")

    def test_main_fetch_requires_video_input(self) -> None:
        with patch("sys.argv", ["bilibili-subtitle-fetch", "fetch"]):
            with self.assertRaises(SystemExit) as ctx:
                main()

        self.assertEqual(str(ctx.exception), "Error: fetch requires a BVID or video URL.")

    def test_main_fetch_runs_cli_command(self) -> None:
        with patch(
            "bilibili_subtitle_fetch.server.run_fetch_command"
        ) as run_fetch_mock:
            with patch(
                "sys.argv",
                [
                    "bilibili-subtitle-fetch",
                    "fetch",
                    "BV1fz4y1j7Mf",
                    "--output-format",
                    "timestamped",
                    "--no-clipboard",
                ],
            ):
                main()

        run_fetch_mock.assert_called_once_with(
            "BV1fz4y1j7Mf",
            preferred_lang="zh-CN",
            output_format="timestamped",
            copy_result=False,
        )


if __name__ == "__main__":
    unittest.main()
