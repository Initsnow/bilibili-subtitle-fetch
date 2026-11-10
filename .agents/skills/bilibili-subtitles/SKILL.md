---
name: bilibili-subtitles
description: >-
  Fetch and process Bilibili video subtitles using the bilibili-subtitle-fetch CLI.
  Use whenever the user provides a Bilibili video link (bilibili.com, b23.tv) or BVID (BV...)
  to query video content, generate summaries, extract notes/code/steps, or write articles.
---

# Bilibili Subtitles

## Overview

Use `bilibili-subtitle-fetch` to retrieve video subtitles, then analyze, summarize, or extract information from the transcript to answer the user's request.

## Fetching Subtitles

Run the CLI command:

```bash
bilibili-subtitle-fetch fetch "<bvid_or_url>"
```

### Supported Inputs
- BVID: `BV1yZ4n6MEHR`
- URL: `https://www.bilibili.com/video/BV1fz4y1j7Mf?p=2`
- Short link: `https://b23.tv/FAm7Xn4`

### Options
- **Plain text (Default)**:
  ```bash
  bilibili-subtitle-fetch fetch "<bvid_or_url>"
  ```
- **Timestamped text**:
  ```bash
  bilibili-subtitle-fetch fetch "<bvid_or_url>" --output-format timestamped
  ```
- **Force local ASR fallback**:
  ```bash
  bilibili-subtitle-fetch fetch "<bvid_or_url>" --asr
  ```
- **Preferred language**:
  ```bash
  bilibili-subtitle-fetch fetch "<bvid_or_url>" --preferred-lang zh-CN
  ```

### Handling Missing Subtitles or Login
1. **Unauthenticated / Hidden subtitles**:
   If the command indicates credentials are missing or expired, instruct the user to run:
   ```bash
   bilibili-subtitle-fetch init
   ```
2. **Video has no subtitles**:
   Use `--asr` to transcribe audio locally.

## Output Processing

1. **ASR Correction**:
   Silently correct obvious speech recognition typos and homophones based on context and domain knowledge. See [asr-writing.md](./references/asr-writing.md) for detailed guidelines.
2. **User Request Alignment**:
   - **Direct Q&A**: Answer questions directly using the relevant parts of the transcript.
   - **Summarization**: Extract key points, structure, and takeaways into concise notes.
   - **Procedures / Tutorials**: Retain exact order of operations, commands, parameters, and caveats.
   - **File Output**: Only write to a file if explicitly requested by the user. Otherwise, output directly.
