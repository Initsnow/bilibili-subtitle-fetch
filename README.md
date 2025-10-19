# Bilibili Subtitle Fetch

MCP server for fetching Bilibili video subtitles with language and format options.

## Configuration

### Environment Variables

- `BILIBILI_SESSDATA`, `BILIBILI_BILI_JCT`, `BILIBILI_BUVID3` - Required Bilibili credentials
- `BILIBILI_PREFERRED_LANG` - Default subtitle language (default: zh-CN)
- `BILIBILI_OUTPUT_FORMAT` - Subtitle format (text/timestamped, default: text)

### CLI Arguments

- `--preferred-lang` - Override default subtitle language
- `--output-format` - Override output format

[Get Bilibili credentials](https://nemo2011.github.io/bilibili-api/#/get-credential.md)
