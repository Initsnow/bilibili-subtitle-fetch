---
name: bilibili-subtitle-writer
description: Fetch Bilibili subtitles with this repository's bilibili-subtitle-fetch CLI and turn them into faithful summaries, tutorials, or polished Chinese articles. Use when Codex is asked to process a Bilibili BVID or video URL, retrieve official or ASR subtitles, clean ASR errors, summarize transcript content, or rewrite subtitles into a concise article without adding outside information.
---

# Bilibili Subtitle Writer

## Overview

Use this repository's `bilibili-subtitle-fetch` command through `uv run` to obtain Bilibili subtitles, then transform the transcript into either a compact summary or a polished Chinese article/tutorial. Preserve the video's meaning, correct obvious ASR mistakes from context, and avoid adding external facts unless the user explicitly asks for a separate fact-check.

## Workflow

1. Identify the input: accept a BVID such as `BV...` or a Bilibili video URL.
2. Resolve `<skill-dir>` as the directory containing this `SKILL.md`.
3. Choose output targets before fetching. Save raw subtitle cache under `subtitles/<safe-video-id>.txt` for stable lookup and collision avoidance. Save the final deliverable as Markdown by default under `result/<safe-title-or-topic>.md` when the user does not specify a path. Do not use the BVID as the Markdown filename; derive a concise safe stem from the video title, requested topic, or transcript content.
4. Fetch subtitles directly with the project CLI through `uv run`; capture stdout into the raw subtitle cache. In PowerShell, write UTF-8 explicitly:

```powershell
New-Item -ItemType Directory -Force subtitles | Out-Null
uv run bilibili-subtitle-fetch fetch "BV..." --output-format text --no-clipboard | Set-Content -Encoding utf8 subtitles/BV....txt
```

Use options when needed:

```powershell
uv run bilibili-subtitle-fetch fetch "https://www.bilibili.com/video/BV..." --output-format timestamped --asr --no-clipboard | Set-Content -Encoding utf8 subtitles/BV....txt
uv run bilibili-subtitle-fetch fetch "BV..." --preferred-lang zh-CN --config path/to/config.toml --no-clipboard | Set-Content -Encoding utf8 subtitles/BV....txt
```

5. If the fetch command reports missing credentials or setup, run:

```bash
uv run bilibili-subtitle-fetch init
```

Then retry the fetch command.

6. Inspect the fetched subtitle file before writing. If it is empty, mostly error text, or clearly not the requested video, stop and report the problem.
7. Write the requested deliverable:
   - For summaries, produce concise sectioned notes with key claims, steps, definitions, examples, and caveats from the transcript.
   - For articles or tutorials, first read `references/asr-writing.md`, then rewrite the transcript into coherent written Chinese.
8. Save a Markdown file by default. Append the source BVID at the end of the Markdown body, for example `BVID: BV...`, instead of encoding it in the filename. Only respond directly without writing a file when the user explicitly asks for inline output.

## Fetching Details

The CLI prints subtitle text to stdout. Always pass `--no-clipboard` for agent workflows, then save stdout to the raw cache file and inspect it before writing the polished output. On PowerShell, prefer `Set-Content -Encoding utf8` instead of bare `>` redirection.

```bash
uv run bilibili-subtitle-fetch fetch <video> --output-format text --no-clipboard
```

Use `--output-format timestamped` when timestamps are useful for auditing, chaptering, or quoting. Use `--no-asr` when only official subtitles are acceptable; use `--asr` when the user wants fallback ASR even if the config disables it.

Do not print secrets from config files or environment variables. If the CLI requires login cookies, ask the user to initialize or update `uv run bilibili-subtitle-fetch` credentials rather than exposing credential values.

## Writing Rules

Base the output strictly on subtitle content. Fix obvious recognition, punctuation, and wording issues silently. Remove filler words, repeated starts, greetings, and low-information transitions.

For technical content, preserve operation order, parameter names, formulas, code, configuration values, numerical claims, conditional limits, and causal relationships. If a transcript section is ambiguous, choose the most reasonable reading from local context and avoid speculative expansion.

Use third-person or impersonal Chinese prose for articles and tutorials. Avoid `我`, `你`, `大家`, empty introductions, and generic closing paragraphs. Start directly with the topic.

## Long Transcripts

For long subtitle files, work in passes:

1. Split by timestamps, visible sections, or natural topic changes.
2. Extract compact notes for each segment without adding information.
3. Merge notes into a single outline.
4. Draft the final summary or article from the outline.
5. Check the final text against the subtitle notes for unsupported claims.

Keep raw subtitles separate from polished output so future revisions can trace back to the source.
