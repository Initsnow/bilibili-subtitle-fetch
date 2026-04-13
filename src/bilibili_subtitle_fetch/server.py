import asyncio
import os
import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from bilibili_api import search, video
from mcp.server.fastmcp import Context, FastMCP

from bilibili_subtitle_fetch.credentials import (
    CredentialManager,
    CredentialStoreError,
    initialize_credential_file,
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

mcp = FastMCP(name="bilibili-subtitle-fetch")


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


async def resolve_video_input(
    ctx: Context,
    url: Optional[str],
    bvid: Optional[str],
) -> tuple[str, Optional[int]]:
    if url and bvid:
        await ctx.log("error", "Both URL and BVID provided. Please provide only one.")
        raise ValueError("Both URL and BVID provided. Please provide only one.")

    if not url and not bvid:
        await ctx.log("error", "Neither URL nor BVID provided. Please provide one.")
        raise ValueError("Neither URL nor BVID provided. Please provide one.")

    if bvid:
        if not BVID_PATTERN.match(bvid):
            await ctx.log("error", f"Invalid BVID format: {bvid}")
            raise ValueError(f"Invalid BVID format: {bvid}")
        return bvid, None

    assert url is not None
    parsed_bvid, page = parse_bilibili_url(url)
    if parsed_bvid:
        return parsed_bvid, page

    if is_bilibili_short_url(url):
        await ctx.log("info", f"Resolving Bilibili short URL: {url}")
        try:
            resolved_url = await resolve_bilibili_short_url(url)
        except httpx.HTTPError as exc:
            await ctx.log("error", f"Failed to resolve short URL {url}: {exc}")
            raise ValueError(f"Failed to resolve Bilibili short URL: {url}") from exc

        await ctx.log("info", f"Resolved short URL to: {resolved_url}")
        parsed_bvid, page = parse_bilibili_url(resolved_url)
        if parsed_bvid:
            return parsed_bvid, page

    await ctx.log("error", f"Could not extract bvid from URL: {url}")
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
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
    }
    async with create_httpx_client(headers=headers) as client:
        response = await client.get(subtitle_url)
        response.raise_for_status()
        return response.json()


def get_effective_preferred_lang(preferred_lang: Optional[str]) -> str:
    return preferred_lang or DEFAULT_PREFERRED_LANG


def get_effective_output_format(
    output_format: Optional[Literal["text", "timestamped"]],
) -> Literal["text", "timestamped"]:
    return output_format or DEFAULT_OUTPUT_FORMAT


async def get_runtime_credential(ctx: Context) -> Any:
    try:
        credential, refresh_note = await CREDENTIAL_MANAGER.get_credential()
    except CredentialStoreError as exc:
        await ctx.log("error", str(exc))
        raise ValueError(str(exc)) from exc

    if refresh_note:
        level = "warning" if "failed" in refresh_note.lower() else "info"
        await ctx.log(level, refresh_note)

    return credential


@mcp.tool(
    name="get_bilibili_subtitle",
    description="Fetches subtitles for a given Bilibili video URL or BVID",
)
async def get_bilibili_subtitle(
    ctx: Context,
    url: Optional[str] = None,
    bvid: Optional[str] = None,
    preferred_lang: Optional[str] = None,
    output_format: Optional[Literal["text", "timestamped"]] = None,
) -> str:
    preferred_lang = get_effective_preferred_lang(preferred_lang)
    output_format = get_effective_output_format(output_format)

    await ctx.log(
        "info",
        f"Received request for URL: {url}, BVID: {bvid}, lang: {preferred_lang}, format: {output_format}",
    )

    try:
        resolved_bvid, page = await resolve_video_input(ctx, url, bvid)
        await ctx.log("info", f"Parsed bvid: {resolved_bvid}, page: {page}")

        credential = await get_runtime_credential(ctx)
        bilibili_video = video.Video(bvid=resolved_bvid, credential=credential)
        info = await bilibili_video.get_info()
        cid = select_cid(info, page)
        if not cid:
            await ctx.log("error", "Could not determine CID for the video.")
            return "Error: Could not determine the video part (CID)."

        available_subtitles = (await bilibili_video.get_subtitle(cid=cid)).get(
            "subtitles", []
        )
        if not available_subtitles:
            await ctx.log("warning", "No subtitles found for this video part.")
            return (
                "Info: No subtitles available for this video part. "
                "This could happen if the video actually lacks subtitles, "
                "or if the configured Bilibili cookie is invalid/expired "
                "(Bilibili hides AI subtitles from unauthenticated API requests)."
            )

        subtitle_url, found_lang = choose_subtitle(available_subtitles, preferred_lang)
        subtitle_url = normalize_subtitle_url(subtitle_url)
        if not subtitle_url:
            await ctx.log("error", "Could not find any valid subtitle URL.")
            return "Error: Could not find any subtitle URL in the metadata."

        await ctx.log(
            "info",
            f"Fetching subtitle content from: {subtitle_url} (Language: {found_lang})",
        )
        subtitle_data = await fetch_subtitle_data(subtitle_url, resolved_bvid)
        body = subtitle_data.get("body", [])
        if not body:
            await ctx.log("warning", "Subtitle file fetched but contains no content.")
            return "Info: Subtitle file is empty."

        formatted_subtitle = format_subtitle_body(body, output_format)
        await ctx.log("info", f"Formatted subtitles as {output_format}.")
        return formatted_subtitle

    except ValueError as exc:
        return f"Error: {exc}"
    except httpx.HTTPStatusError as exc:
        await ctx.log(
            "error",
            f"HTTP error fetching subtitle content: {exc.response.status_code} for URL {exc.request.url}",
        )
        detail = f"HTTP Status {exc.response.status_code}"
        try:
            detail += f" - Response: {exc.response.text[:200]}"
        except Exception:
            pass
        return f"Error fetching subtitle content: {detail}"
    except httpx.RequestError as exc:
        request_url = exc.request.url if exc.request else "unknown"
        await ctx.log(
            "error",
            f"Network error fetching subtitle content for URL {request_url}: {exc}",
        )
        return f"Error fetching subtitle content (network issue): {exc}"
    except Exception as exc:
        await ctx.log("error", f"An unexpected error occurred: {exc}")
        return f"An unexpected error occurred: {type(exc).__name__} - {exc}"


class TimeRange(Enum):
    Under10Minutes = 10
    From10to30Minutes = 30
    From30to60Minutes = 60
    Over60Minutes = 61


@mcp.tool(
    name="search_bilibili_videos",
    description="Searches for Bilibili videos.",
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
    description="Fetches the description of a Bilibili video by its BVID.",
)
async def get_bilibili_video_desc(bvid: str) -> str:
    result = await video.Video(bvid=bvid).get_info()
    return result["desc"].strip()


@mcp.tool(
    name="get_subtitle_from_audio",
    description="Generates subtitles from a Bilibili video by its BVID. Default model size is 'base'.",
)
async def get_subtitle_from_audio(
    ctx: Context,
    bvid: str,
    type: Literal["text", "timestamped"] = "text",
    model_size: Literal["tiny", "base", "small", "medium", "large"] = "base",
) -> str:
    try:
        await ctx.log(
            "info",
            f"Generating subtitles for bvid: {bvid} with model size: {model_size}",
        )
        credential = await get_runtime_credential(ctx)
        bilibili_video = video.Video(bvid=bvid, credential=credential)
        audio_file = await download_audio(bilibili_video)
        return await asyncio.to_thread(generate_subtitles, audio_file, type, model_size)
    except Exception as exc:
        await ctx.log("error", f"Error: {exc}")
        return f"Error: {exc}"


def main() -> None:
    import argparse

    global DEFAULT_OUTPUT_FORMAT
    global DEFAULT_PREFERRED_LANG
    global CREDENTIAL_MANAGER

    parser = argparse.ArgumentParser(description="Bilibili Subtitle Fetch MCP Server")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["serve", "init"],
        default="serve",
        help="Run the MCP server or initialize the local credential file.",
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
    args = parser.parse_args()

    if args.config:
        CREDENTIAL_MANAGER.set_config_path(args.config)

    if args.command == "init":
        try:
            initialize_credential_file(CREDENTIAL_MANAGER.config_path)
        except CredentialStoreError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        return

    DEFAULT_PREFERRED_LANG = args.preferred_lang
    DEFAULT_OUTPUT_FORMAT = args.output_format
    try:
        CREDENTIAL_MANAGER.validate_runtime_config()
    except CredentialStoreError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    mcp.run()


if __name__ == "__main__":
    main()
