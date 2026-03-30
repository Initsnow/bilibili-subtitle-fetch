# Bilibili Subtitle Fetch

MCP server for fetching Bilibili video subtitles with language and format options.

`uv tool install --python 3.13 bilibili-subtitle-fetch`

支持以下输入：

- `BV` 号，例如 `BV1fz4y1j7Mf`
- 标准视频链接，例如 `https://www.bilibili.com/video/BV1fz4y1j7Mf?p=2`
- 短链，例如 `https://b23.tv/FAm7Xn4`

## Configuration

### Environment Variables

- `BILIBILI_SESSDATA`, `BILIBILI_BILI_JCT`, `BILIBILI_BUVID3` - Required Bilibili credentials
- `BILIBILI_PREFERRED_LANG` - Default subtitle language (default: zh-CN)
- `BILIBILI_OUTPUT_FORMAT` - Subtitle format (text/timestamped, default: text)

### CLI Arguments

- `--preferred-lang` - Override default subtitle language
- `--output-format` - Override output format

[Get Bilibili credentials](https://nemo2011.github.io/bilibili-api/#/get-credential.md)
