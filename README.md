# Bilibili Subtitle Fetch

MCP server for fetching Bilibili video subtitles with language and format options.

## Quick Start

1. Set required environment variables:

```bash
export BILIBILI_SESSDATA=your_sessdata
export BILIBILI_BILI_JCT=your_jct 
export BILIBILI_BUVID3=your_buvid3
```

2. Run the server with optional parameters:

```bash
scoop install uv
uvx bilibili-subtitle-fetch
```

## Configuration

### Environment Variables

- `BILIBILI_SESSDATA`, `BILIBILI_BILI_JCT`, `BILIBILI_BUVID3` - Required Bilibili credentials
- `BILIBILI_PREFERRED_LANG` - Default subtitle language (default: zh-CN)
- `BILIBILI_OUTPUT_FORMAT` - Subtitle format (text/timestamped, default: text)

### CLI Arguments

- `--preferred-lang` - Override default subtitle language
- `--output-format` - Override output format

## MCP Tool Usage

```json
{
  "tool_name": "get_bilibili_subtitle",
  "arguments": {
    "url": "bilibili_video_url",
    "preferred_lang": "optional_lang_code", 
    "output_format": "text|timestamped"
  }
}
```

[Get Bilibili credentials](https://nemo2011.github.io/bilibili-api/#/get-credential.md)
