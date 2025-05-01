# Bilibili Subtitle Fetch MCP Server

This project provides an MCP (Model Context Protocol) server that allows you to fetch subtitles for Bilibili videos.

## Features

- Fetches subtitles for Bilibili videos using a given URL.
- Supports specifying a preferred subtitle language.
- Supports outputting subtitles in plain text or with timestamps.
- Uses environment variables for Bilibili credentials.

## Installation

Optionally: Install node.js, this will cause the fetch server to use a different HTML simplifier that is more robust.

### Using uv (recommended)

When using [`uv`](https://docs.astral.sh/uv/) no specific installation is needed. We will
use [`uvx`](https://docs.astral.sh/uv/guides/tools/) to directly run *bilibili-subtitle-fetch*.

### Using PIP

Alternatively you can install `bilibili-subtitle-fetch` via pip:

```
pip install bilibili-subtitle-fetch
```

After installation, you can run it as a script using:

```
python -m bilibili_subtitle_fetch
```

To run the MCP server, execute the `server.py` file:

```bash
python server.py
```

The server will start and listen for incoming requests.

## Using the `get_bilibili_subtitle` Tool

Once the server is running and connected to your MCP client, you can use the `get_bilibili_subtitle` tool.

**Tool Name:** `get_bilibili_subtitle`

**Description:** Fetches subtitles for a given Bilibili video URL.

**Parameters:**

- `url` (required, string): The URL of the Bilibili video (e.g., "<https://www.bilibili.com/video/BV1fz4y1j7Mf/?p=2>").
- `preferred_lang` (optional, string): The preferred subtitle language code (e.g., 'zh-CN', 'ai-zh', 'en'). Defaults to 'zh-CN'. Check the video page for available languages. 'ai-zh' is often AI-generated Chinese.
- `output_format` (optional, string): The desired format for the subtitles ('text' for plain text, 'timestamped' for text with timestamps). Defaults to 'text'.

**Example Usage (via MCP Client):**

```json
{
  "server_name": "bilibili-subtitle-getter",
  "tool_name": "get_bilibili_subtitle",
  "arguments": {
    "url": "https://www.bilibili.com/video/BV1fz4y1j7Mf/?p=2",
    "preferred_lang": "en",
    "output_format": "timestamped"
  }
}
```

Replace `"your_server_name"` with the actual name or identifier your MCP client uses to connect to this server.
