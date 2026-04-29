import asyncio
import os
import re
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Optional, TextIO
from urllib.parse import parse_qs, urlparse

import httpx
import pyperclip
from bilibili_api import search, video
from mcp.server.fastmcp import Context, FastMCP

from bilibili_subtitle_fetch.credentials import (
    CredentialManager,
    CredentialStoreError,
    initialize_credential_file,
    load_asr_config,
)
from bilibili_subtitle_fetch.download_audio import download_audio
from bilibili_subtitle_fetch.generate_subtitles import generate_subtitles

CREDENTIAL_MANAGER = CredentialManager()

BVID_PATTERN = re.compile(r"^BV[1-9A-HJ-NP-Za-km-z]{10}$")
SHORT_LINK_HOSTS = {"b23.tv", "www.b23.tv", "bili2233.cn", "www.bili2233.cn"}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
)
DEFAULT_REQUEST_TIMEOUT = 10.0
DEFAULT_PREFERRED_LANG = os.environ.get("BILIBILI_PREFERRED_LANG", "zh-CN")
DEFAULT_OUTPUT_FORMAT: Literal["text", "timestamped"] = os.environ.get(
    "BILIBILI_OUTPUT_FORMAT", "text"
)
if DEFAULT_OUTPUT_FORMAT not in {"text", "timestamped"}:
    DEFAULT_OUTPUT_FORMAT = "text"

# ASR defaults — can be overridden by [asr] section in config.toml or env vars.
# enable_asr=True means: fall back to audio transcription when no subtitle track exists.
_env_enable_asr = os.environ.get("BILIBILI_ENABLE_ASR", "").strip().lower()
DEFAULT_ENABLE_ASR: bool = _env_enable_asr not in {"0", "false", "no"}
DEFAULT_ASR_MODEL_SIZE: str = os.environ.get("BILIBILI_ASR_MODEL_SIZE", "tiny")

mcp = FastMCP(name="bilibili-subtitle-fetch")
LogHandler = Callable[[str, str], Awaitable[None]]


def parse_bilibili_url(url: str) -> tuple[Optional[str], Optional[int]]:
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")
    bvid = next((part for part in path_parts if BVID_PATTERN.match(part)), None)

    page = None
    query_params = parse_qs(parsed_url.query)
    if "p" in query_params:
        try:
            candidate = int(query_params["p"][0])
            if candidate > 0:
                page = candidate
        except (ValueError, IndexError):
            page = None

    return bvid, page


def is_bilibili_short_url(url: str) -> bool:
    return urlparse(url).netloc.lower() in SHORT_LINK_HOSTS


def create_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
    defaults: dict[str, Any] = {
        "follow_redirects": True,
        "headers": {"User-Agent": DEFAULT_USER_AGENT},
        "timeout": DEFAULT_REQUEST_TIMEOUT,
    }
    defaults.update(kwargs)
    return httpx.AsyncClient(**defaults)


async def resolve_bilibili_short_url(url: str) -> str:
    async with create_httpx_client() as client:
        response = await client.get(url)
        response.raise_for_status()
        return str(response.url)


async def log_message(
    logger: Optional[LogHandler],
    level: str,
    message: str,
) -> None:
    if logger is not None:
        await logger(level, message)


async def resolve_video_input(
    url: Optional[str],
    bvid: Optional[str],
    logger: Optional[LogHandler] = None,
) -> tuple[str, Optional[int]]:
    if url and bvid:
        await log_message(
            logger, "error", "Both URL and BVID provided. Please provide only one."
        )
        raise ValueError("Both URL and BVID provided. Please provide only one.")

    if not url and not bvid:
        await log_message(
            logger, "error", "Neither URL nor BVID provided. Please provide one."
        )
        raise ValueError("Neither URL nor BVID provided. Please provide one.")

    if bvid:
        if not BVID_PATTERN.match(bvid):
            await log_message(logger, "error", f"Invalid BVID format: {bvid}")
            raise ValueError(f"Invalid BVID format: {bvid}")
        return bvid, None

    assert url is not None
    parsed_bvid, page = parse_bilibili_url(url)
    if parsed_bvid:
        return parsed_bvid, page

    if is_bilibili_short_url(url):
        await log_message(logger, "info", f"Resolving Bilibili short URL: {url}")
        try:
            resolved_url = await resolve_bilibili_short_url(url)
        except httpx.HTTPError as exc:
            await log_message(
                logger, "error", f"Failed to resolve short URL {url}: {exc}"
            )
            raise ValueError(f"Failed to resolve Bilibili short URL: {url}") from exc

        await log_message(logger, "info", f"Resolved short URL to: {resolved_url}")
        parsed_bvid, page = parse_bilibili_url(resolved_url)
        if parsed_bvid:
            return parsed_bvid, page

    await log_message(logger, "error", f"Could not extract bvid from URL: {url}")
    raise ValueError(f"Could not extract a valid bvid from the URL: {url}")


def select_cid(info: dict[str, Any], page: Optional[int]) -> Optional[int]:
    pages = info.get("pages")
    if not page:
        return info.get("cid")

    if isinstance(pages, list) and 0 < page <= len(pages):
        page_info = pages[page - 1]
        if isinstance(page_info, dict) and page_info.get("cid"):
            return page_info["cid"]

    return info.get("cid")


def choose_subtitle(
    available_subtitles: list[dict[str, Any]],
    preferred_lang: str,
) -> tuple[Optional[str], Optional[str]]:
    for subtitle in available_subtitles:
        if subtitle.get("lan") == preferred_lang:
            return subtitle.get("subtitle_url"), subtitle.get("lan")

    for subtitle in available_subtitles:
        if subtitle.get("ai_type", 0) == 0:
            return subtitle.get("subtitle_url"), subtitle.get("lan")

    if available_subtitles:
        first = available_subtitles[0]
        return first.get("subtitle_url"), first.get("lan")

    return None, None


def normalize_subtitle_url(subtitle_url: Optional[str]) -> Optional[str]:
    if not subtitle_url:
        return None
    if subtitle_url.startswith("//"):
        return "https:" + subtitle_url
    if subtitle_url.startswith(("http://", "https://")):
        return subtitle_url
    return None


def extract_subtitles_from_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    subtitle = payload.get("subtitle")
    if isinstance(subtitle, dict):
        subtitles = subtitle.get("subtitles")
        if isinstance(subtitles, list):
            return subtitles

    data = payload.get("data")
    if isinstance(data, dict):
        subtitle = data.get("subtitle")
        if isinstance(subtitle, dict):
            subtitles = subtitle.get("subtitles")
            if isinstance(subtitles, list):
                return subtitles

    return []


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    whole_seconds = int(seconds_remainder)
    milliseconds = int((seconds_remainder - whole_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def format_subtitle_body(
    body: list[dict[str, Any]],
    output_format: Literal["text", "timestamped"],
) -> str:
    if output_format == "timestamped":
        chunks = []
        for item in body:
            start = float(item.get("from", 0.0))
            end = float(item.get("to", 0.0))
            content = str(item.get("content", ""))
            chunks.append(
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n{content}"
            )
        return "\n\n".join(chunks).strip()

    return "\n".join(str(item.get("content", "")) for item in body).strip()


async def fetch_subtitle_data(subtitle_url: str, bvid: str) -> dict[str, Any]:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
    }
    async with create_httpx_client(headers=headers) as client:
        response = await client.get(subtitle_url)
        response.raise_for_status()
        return response.json()


async def fetch_available_subtitles(
    aid: int,
    cid: int,
    bvid: str,
    credential: Any,
) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
    }
    params = {"aid": aid, "oid": cid, "type": 1}
    async with create_httpx_client(
        headers=headers,
        cookies=credential.get_cookies(),
    ) as client:
        response = await client.get("https://api.bilibili.com/x/v2/dm/view", params=params)
        response.raise_for_status()
        payload = response.json()

    code = payload.get("code")
    if code not in (None, 0):
        message = payload.get("message") or payload.get("msg") or "unknown error"
        raise ValueError(f"Failed to fetch subtitle metadata: {message} (code: {code})")

    return extract_subtitles_from_response(payload)


def get_effective_preferred_lang(preferred_lang: Optional[str]) -> str:
    return preferred_lang or DEFAULT_PREFERRED_LANG


def get_effective_output_format(
    output_format: Optional[Literal["text", "timestamped"]],
) -> Literal["text", "timestamped"]:
    return output_format or DEFAULT_OUTPUT_FORMAT


async def get_runtime_credential(logger: Optional[LogHandler] = None) -> Any:
    try:
        credential, refresh_note = await CREDENTIAL_MANAGER.get_credential()
    except CredentialStoreError as exc:
        await log_message(logger, "error", str(exc))
        raise ValueError(str(exc)) from exc

    if refresh_note:
        level = "warning" if "failed" in refresh_note.lower() else "info"
        await log_message(logger, level, refresh_note)

    return credential


def parse_cli_video_input(video_input: str) -> tuple[Optional[str], Optional[str]]:
    normalized = video_input.strip()
    if normalized.startswith("BV"):
        return None, normalized
    return normalized, None


def format_subtitle_fetch_error(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return f"Error: {exc}"

    if isinstance(exc, httpx.HTTPStatusError):
        detail = f"HTTP Status {exc.response.status_code}"
        try:
            detail += f" - Response: {exc.response.text[:200]}"
        except Exception:
            pass
        return f"Error fetching subtitle content: {detail}"

    if isinstance(exc, httpx.RequestError):
        return f"Error fetching subtitle content (network issue): {exc}"

    return f"An unexpected error occurred: {type(exc).__name__} - {exc}"


async def log_subtitle_fetch_error(
    exc: Exception,
    logger: Optional[LogHandler] = None,
) -> None:
    if isinstance(exc, ValueError):
        return

    if isinstance(exc, httpx.HTTPStatusError):
        await log_message(
            logger,
            "error",
            "HTTP error fetching subtitle content: "
            f"{exc.response.status_code} for URL {exc.request.url}",
        )
        return

    if isinstance(exc, httpx.RequestError):
        request_url = exc.request.url if exc.request else "unknown"
        await log_message(
            logger,
            "error",
            f"Network error fetching subtitle content for URL {request_url}: {exc}",
        )
        return

    await log_message(logger, "error", f"An unexpected error occurred: {exc}")


async def fetch_bilibili_subtitle_text(
    url: Optional[str] = None,
    bvid: Optional[str] = None,
    preferred_lang: Optional[str] = None,
    output_format: Optional[Literal["text", "timestamped"]] = None,
    logger: Optional[LogHandler] = None,
) -> str:
    preferred_lang = get_effective_preferred_lang(preferred_lang)
    output_format = get_effective_output_format(output_format)

    await log_message(
        logger,
        "info",
        "Received request for URL: "
        f"{url}, BVID: {bvid}, lang: {preferred_lang}, format: {output_format}",
    )

    resolved_bvid, page = await resolve_video_input(url, bvid, logger=logger)
    await log_message(logger, "info", f"Parsed bvid: {resolved_bvid}, page: {page}")

    credential = await get_runtime_credential(logger)
    bilibili_video = video.Video(bvid=resolved_bvid, credential=credential)
    info = await bilibili_video.get_info()
    cid = select_cid(info, page)
    if not cid:
        await log_message(logger, "error", "Could not determine CID for the video.")
        raise ValueError("Could not determine the video part (CID).")

    aid = info.get("aid")
    if not aid:
        await log_message(logger, "error", "Could not determine AID for the video.")
        raise ValueError("Could not determine the video aid.")

    await log_message(
        logger,
        "info",
        f"Fetching subtitle track list via x/v2/dm/view (aid={aid}, cid={cid}).",
    )
    available_subtitles = await fetch_available_subtitles(
        aid=aid,
        cid=cid,
        bvid=resolved_bvid,
        credential=credential,
    )
    if not available_subtitles:
        await log_message(logger, "warning", "No subtitles found for this video part.")
        return (
            "Info: No subtitles available for this video part. "
            "This could happen if the video actually lacks subtitles, "
            "or if the configured Bilibili cookie is invalid/expired "
            "(Bilibili hides AI subtitles from unauthenticated API requests)."
        )

    subtitle_url, found_lang = choose_subtitle(available_subtitles, preferred_lang)
    subtitle_url = normalize_subtitle_url(subtitle_url)
    if not subtitle_url:
        await log_message(logger, "error", "Could not find any valid subtitle URL.")
        raise ValueError("Could not find any subtitle URL in the metadata.")

    await log_message(
        logger,
        "info",
        f"Fetching subtitle content from: {subtitle_url} (Language: {found_lang})",
    )
    subtitle_data = await fetch_subtitle_data(subtitle_url, resolved_bvid)
    body = subtitle_data.get("body", [])
    if not body:
        await log_message(
            logger, "warning", "Subtitle file fetched but contains no content."
        )
        return "Info: Subtitle file is empty."

    formatted_subtitle = format_subtitle_body(body, output_format)
    await log_message(logger, "info", f"Formatted subtitles as {output_format}.")
    return formatted_subtitle


def copy_to_clipboard(text: str) -> None:
    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException as exc:
        raise RuntimeError(str(exc)) from exc





@mcp.tool(
    name="get_bilibili_subtitle",
    description="Fetch video subtitles. You should automatically try to correct any ASR errors in the returned text.",
)
async def get_bilibili_subtitle(
    ctx: Context,
    url: Optional[str] = None,
    bvid: Optional[str] = None,
    preferred_lang: Optional[str] = None,
    output_format: Optional[Literal["text", "timestamped"]] = None,
) -> str:
    try:
        result = await fetch_bilibili_subtitle_text(
            url=url,
            bvid=bvid,
            preferred_lang=preferred_lang,
            output_format=output_format,
            logger=ctx.log,
        )
        # If no subtitle track was found and ASR is enabled, fall back to audio.
        if result.startswith("Info: No subtitles") and DEFAULT_ENABLE_ASR:
            await ctx.log(
                "info",
                f"No subtitle track found. Falling back to ASR (model: {DEFAULT_ASR_MODEL_SIZE}).",
            )
            resolved_bvid, _page = await resolve_video_input(url, bvid, logger=ctx.log)
            credential = await get_runtime_credential(ctx.log)
            bilibili_video = video.Video(bvid=resolved_bvid, credential=credential)
            audio_file = await download_audio(bilibili_video)
            effective_format = get_effective_output_format(output_format)
            return await asyncio.to_thread(
                generate_subtitles, audio_file, effective_format, DEFAULT_ASR_MODEL_SIZE
            )
        return result
    except Exception as exc:
        await log_subtitle_fetch_error(exc, logger=ctx.log)
        return format_subtitle_fetch_error(exc)


class TimeRange(Enum):
    Under10Minutes = 10
    From10to30Minutes = 30
    From30to60Minutes = 60
    Over60Minutes = 61


@mcp.tool(
    name="search_bilibili_videos",
)
async def search_bilibili_videos(
    keyword: str,
    order_type: search.OrderVideo = search.OrderVideo.TOTALRANK,
    time_range: Optional[TimeRange] = None,
    page: int = 1,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
) -> str:
    result = await search.search_by_type(
        keyword,
        search_type=search.SearchObjectType.VIDEO,
        order_type=order_type,
        time_range=time_range.value if time_range is not None else -1,
        page=page,
        time_start=time_start,
        time_end=time_end,
    )

    lines = []
    for item in result.get("result", []):
        senddate = datetime.fromtimestamp(item["senddate"]).strftime("%y%m%d")
        title = re.sub(r'<em class="keyword">(.*?)</em>', r"\1", item["title"])
        lines.append(
            f"{title} by {item['author']} (play {item['play']}, "
            f"fav {item['favorites']}, {senddate}, id {item['bvid']})"
        )

    return "\n".join(lines)


@mcp.tool(
    name="get_bilibili_video_desc",
)
async def get_bilibili_video_desc(bvid: str) -> str:
    result = await video.Video(bvid=bvid).get_info()
    return result["desc"].strip()


def _apply_asr_config_from_file() -> None:
    """Load [asr] section from config.toml and update module-level ASR defaults."""
    global DEFAULT_ENABLE_ASR, DEFAULT_ASR_MODEL_SIZE
    asr_cfg = load_asr_config(CREDENTIAL_MANAGER.config_path)
    if "enable_asr" in asr_cfg:
        val = asr_cfg["enable_asr"]
        if isinstance(val, bool):
            DEFAULT_ENABLE_ASR = val
        elif isinstance(val, str):
            DEFAULT_ENABLE_ASR = val.strip().lower() not in {"0", "false", "no"}
    if "model_size" in asr_cfg:
        val = asr_cfg["model_size"]
        if isinstance(val, str) and val.strip():
            DEFAULT_ASR_MODEL_SIZE = val.strip()


def run_fetch_command(
    video_input: str,
    preferred_lang: str,
    output_format: Literal["text", "timestamped"],
    copy_result: bool = True,
    use_asr: Optional[bool] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> None:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    url, bvid = parse_cli_video_input(video_input)

    # Resolve effective ASR setting for this invocation.
    effective_enable_asr = DEFAULT_ENABLE_ASR if use_asr is None else use_asr

    async def _fetch() -> str:
        result = await fetch_bilibili_subtitle_text(
            url=url,
            bvid=bvid,
            preferred_lang=preferred_lang,
            output_format=output_format,
        )
        if result.startswith("Info: No subtitles") and effective_enable_asr:
            print(
                f"No subtitle track found. Falling back to ASR (model: {DEFAULT_ASR_MODEL_SIZE})…",
                file=stderr,
            )
            resolved_bvid, _page = await resolve_video_input(url, bvid)
            credential = await get_runtime_credential()
            bilibili_video = video.Video(bvid=resolved_bvid, credential=credential)
            audio_file = await download_audio(bilibili_video)
            return await asyncio.to_thread(
                generate_subtitles, audio_file, output_format, DEFAULT_ASR_MODEL_SIZE
            )
        return result

    try:
        subtitle = asyncio.run(_fetch())
    except Exception as exc:
        raise SystemExit(format_subtitle_fetch_error(exc)) from exc

    print(subtitle, file=stdout)

    if not copy_result:
        return

    try:
        copy_to_clipboard(subtitle)
    except Exception as exc:
        print(
            f"Warning: Failed to copy subtitles to clipboard: {exc}",
            file=stderr,
        )
    else:
        print("Copied subtitles to clipboard.", file=stderr)


def main() -> None:
    import argparse

    global DEFAULT_OUTPUT_FORMAT
    global DEFAULT_PREFERRED_LANG
    global CREDENTIAL_MANAGER

    parser = argparse.ArgumentParser(description="Bilibili Subtitle Fetch MCP Server")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["serve", "init", "fetch"],
        default="serve",
        help="Run the MCP server, initialize credentials, or fetch subtitles locally.",
    )
    parser.add_argument(
        "video_input",
        nargs="?",
        help="BVID or video URL for the fetch command.",
    )
    parser.add_argument(
        "--config",
        help="Path to config.toml. Defaults to the user config directory.",
    )
    parser.add_argument(
        "--preferred-lang",
        default=DEFAULT_PREFERRED_LANG,
        help="Preferred subtitle language (default: zh-CN)",
    )
    parser.add_argument(
        "--output-format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["text", "timestamped"],
        help="Subtitle output format (text or timestamped)",
    )
    parser.add_argument(
        "--no-clipboard",
        action="store_true",
        help="Do not copy fetched subtitles to the clipboard.",
    )

    # ASR flags (only relevant for `fetch` and `serve` commands)
    asr_group = parser.add_mutually_exclusive_group()
    asr_group.add_argument(
        "--no-asr",
        action="store_true",
        help="Disable ASR fallback even if it is enabled in config.",
    )
    asr_group.add_argument(
        "--asr",
        action="store_true",
        help="Force-enable ASR fallback for this invocation.",
    )

    args = parser.parse_args()

    if args.config:
        CREDENTIAL_MANAGER.set_config_path(args.config)

    if args.command == "init":
        try:
            initialize_credential_file(CREDENTIAL_MANAGER.config_path)
        except CredentialStoreError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        return

    # Resolve CLI ASR override (None means "use config/env default").
    cli_asr_override: Optional[bool] = None
    if args.no_asr:
        cli_asr_override = False
    elif args.asr:
        cli_asr_override = True

    if args.command == "fetch":
        if not args.video_input:
            raise SystemExit("Error: fetch requires a BVID or video URL.")
        # Load ASR settings from config file before running.
        _apply_asr_config_from_file()
        run_fetch_command(
            args.video_input,
            preferred_lang=args.preferred_lang,
            output_format=args.output_format,
            copy_result=not args.no_clipboard,
            use_asr=cli_asr_override,
        )
        return

    DEFAULT_PREFERRED_LANG = args.preferred_lang
    DEFAULT_OUTPUT_FORMAT = args.output_format
    try:
        CREDENTIAL_MANAGER.validate_runtime_config()
    except CredentialStoreError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    # Load ASR settings from config file for the MCP server.
    _apply_asr_config_from_file()
    mcp.run()


if __name__ == "__main__":
    main()
