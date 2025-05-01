import os
import re
import httpx
from typing import Optional, Literal
from urllib.parse import urlparse, parse_qs

from bilibili_api import video, Credential
from mcp.server.fastmcp import FastMCP, Context

# Your Bilibili Credentials
# Get credentials from environment variables
BILIBILI_CREDENTIAL = Credential(
    sessdata=os.environ.get("BILIBILI_SESSDATA"),
    bili_jct=os.environ.get("BILIBILI_BILI_JCT"),
    buvid3=os.environ.get("BILIBILI_BUVID3"),
)


# Helper function to parse Bilibili URL
def parse_bilibili_url(url: str) -> tuple[Optional[str], Optional[int]]:
    """
    Parses a Bilibili video URL to extract bvid and page number.
    Handles URLs like:
    - https://www.bilibili.com/video/BVxxxxxxxxxx/
    - https://www.bilibili.com/video/BVxxxxxxxxxx?p=2
    - https://m.bilibili.com/video/BVxxxxxxxxxx
    - https://m.bilibili.com/video/BVxxxxxxxxxx?p=3
    """
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")
    bvid = None
    page = None

    # Find BV ID in path
    for part in path_parts:
        if re.match(r"^BV[1-9A-HJ-NP-Za-km-z]{10}$", part):
            bvid = part
            break

    # Find page number in query parameters
    query_params = parse_qs(parsed_url.query)
    if "p" in query_params:
        try:
            page = int(query_params["p"][0])
        except (ValueError, IndexError):
            pass  # Ignore invalid page numbers

    return bvid, page


# Define the MCP Server
mcp = FastMCP(
    name="bilibili-subtitle-getter",
    description="An MCP server to fetch subtitles from Bilibili videos.",
    version="0.1.0",
)


# Define the tool
@mcp.tool(
    name="get_bilibili_subtitle",
    description="Fetches subtitles for a given Bilibili video URL",
)
async def get_bilibili_subtitle(
    ctx: Context,
    url: str,
    preferred_lang: str = "zh-CN",  # Default to simplified Chinese
    output_format: Literal["text", "timestamped"] = "text",  # Default to plain text
) -> str:
    """
    Fetches subtitles for a given Bilibili video URL.

    :param url: The URL of the Bilibili video (e.g., "https://www.bilibili.com/video/BV1fz4y1j7Mf/?p=2").
    :param preferred_lang: The preferred subtitle language code (e.g., 'zh-CN', 'ai-zh', 'en'). Defaults to 'zh-CN'.
                           Check the video page for available languages. 'ai-zh' is often AI-generated Chinese.
    :param output_format: The desired format for the subtitles ('text' for plain text, 'timestamped' for text with timestamps). Defaults to 'text'.
    :return: The formatted subtitle string, or an error message.
    """
    ctx.log(
        "info",
        f"Received request for URL: {url}, lang: {preferred_lang}, format: {output_format}",
    )

    bvid, page = parse_bilibili_url(url)

    if not bvid:
        ctx.log("error", f"Could not extract bvid from URL: {url}")
        return f"Error: Could not extract a valid bvid from the URL: {url}"

    ctx.log("info", f"Parsed bvid: {bvid}, page: {page}")

    try:
        v = video.Video(bvid=bvid, credential=BILIBILI_CREDENTIAL)

        # Get video info to find the correct cid
        info = await v.get_info()
        ctx.log("debug", f"Video info fetched for {bvid}")

        cid: Optional[int] = None
        # Check if 'pages' key exists and is a list before accessing it
        pages_info = info.get("pages")
        if page and isinstance(pages_info, list) and len(pages_info) >= page:
            # Check if page number is valid (page is 1-based index)
            if 0 < page <= len(pages_info):
                cid = pages_info[page - 1].get("cid")  # Use .get for safety
                if cid:
                    ctx.log("info", f"Found cid {cid} for page {page}")
                else:
                    ctx.log(
                        "warning",
                        f"Page {page} found in 'pages' list, but 'cid' key is missing for that page.",
                    )
                    # Fallback to default cid if specific page cid is missing
                    cid = info.get("cid")
            else:
                ctx.log(
                    "warning",
                    f"Invalid page number {page} for video with {len(pages_info)} pages. Falling back to default page.",
                )
                cid = info.get(
                    "cid"
                )  # Fallback to the default cid if page is out of range
        else:
            if page:
                ctx.log(
                    "warning",
                    f"Page {page} requested but video seems to be single-part or page info missing/invalid. Using default cid.",
                )
            cid = info.get(
                "cid"
            )  # Default cid for single-part videos or if page not specified/found
            if cid:
                ctx.log("info", f"Using default cid {cid}")

        if not cid:
            ctx.log("error", "Could not determine CID for the video.")
            return "Error: Could not determine the video part (CID)."

        # Get available subtitles metadata
        subtitle_info = await v.get_subtitle(cid=cid)
        ctx.log("debug", f"Subtitle metadata fetched: {subtitle_info}")

        available_subtitles = subtitle_info.get("subtitles", [])
        if not available_subtitles:
            ctx.log("warning", "No subtitles found for this video part.")
            return "Info: No subtitles available for this video part."

        # Find the preferred subtitle URL
        subtitle_url: Optional[str] = None
        found_lang: Optional[str] = None

        # Prioritize exact match for preferred language
        for sub in available_subtitles:
            if sub.get("lan") == preferred_lang:
                subtitle_url = sub.get("subtitle_url")
                found_lang = sub.get("lan")
                ctx.log(
                    "info", f"Found exact match for preferred language: {found_lang}"
                )
                break

        # If exact match not found, try finding *any* subtitle (prioritizing non-AI)
        if not subtitle_url:
            ctx.log(
                "warning",
                f"Preferred language '{preferred_lang}' not found. Searching for alternatives.",
            )
            # Try non-AI first
            for sub in available_subtitles:
                # Check if 'ai_type' exists and is 0 (manual/official) or if 'ai_type' doesn't exist
                is_manual = sub.get("ai_type", 0) == 0
                if is_manual:
                    subtitle_url = sub.get("subtitle_url")
                    found_lang = sub.get("lan")
                    ctx.log("info", f"Found alternative non-AI subtitle: {found_lang}")
                    break
            # If still no subtitle found, take the first available AI one
            if not subtitle_url and available_subtitles:
                subtitle_url = available_subtitles[0].get("subtitle_url")
                found_lang = available_subtitles[0].get("lan")
                ctx.log("info", f"Found first available AI subtitle: {found_lang}")

        if not subtitle_url:
            ctx.log("error", "Could not find any subtitle URL.")
            return "Error: Could not find any subtitle URL in the metadata."

        # Ensure URL starts with http: or https:
        if subtitle_url.startswith("//"):
            subtitle_url = "https:" + subtitle_url
        elif not subtitle_url.startswith(("http:", "https:")):
            ctx.log("error", f"Invalid subtitle URL scheme: {subtitle_url}")
            return f"Error: Invalid subtitle URL format: {subtitle_url}"

        ctx.log(
            "info",
            f"Fetching subtitle content from: {subtitle_url} (Language: {found_lang})",
        )

        # Fetch the actual subtitle JSON content
        async with httpx.AsyncClient() as client:
            # Add headers to mimic browser request, might help avoid blocks
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": f"https://www.bilibili.com/video/{bvid}/",  # Add referer
            }
            response = await client.get(
                subtitle_url, headers=headers, follow_redirects=True
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            subtitle_data = response.json()
            ctx.log("debug", "Subtitle JSON data fetched successfully.")

        # Format the subtitle content
        body = subtitle_data.get("body", [])
        if not body:
            ctx.log("warning", "Subtitle file fetched but contains no content.")
            return "Info: Subtitle file is empty."

        formatted_subtitle = ""
        if output_format == "timestamped":
            for item in body:
                start = item.get("from", 0.0)
                end = item.get("to", 0.0)
                content = item.get("content", "")
                # Simple timestamp format HH:MM:SS.ms
                start_h, start_rem = divmod(start, 3600)
                start_m, start_s = divmod(start_rem, 60)
                start_ms = int((start_s - int(start_s)) * 1000)

                end_h, end_rem = divmod(end, 3600)
                end_m, end_s = divmod(end_rem, 60)
                end_ms = int((end_s - int(end_s)) * 1000)

                formatted_subtitle += f"{int(start_h):02}:{int(start_m):02}:{int(start_s):02}.{start_ms:03} --> "
                formatted_subtitle += (
                    f"{int(end_h):02}:{int(end_m):02}:{int(end_s):02}.{end_ms:03}\n"
                )
                formatted_subtitle += f"{content}\n\n"
            ctx.log("info", "Formatted subtitles with timestamps.")
        else:  # Default to plain text
            lines = [item.get("content", "") for item in body]
            formatted_subtitle = "\n".join(lines)
            ctx.log("info", "Formatted subtitles as plain text.")

        return formatted_subtitle.strip()

    except httpx.HTTPStatusError as e:
        ctx.log(
            "error",
            f"HTTP error fetching subtitle content: {e.response.status_code} for URL {e.request.url}",
        )
        # Provide more context in the error message
        error_details = f"HTTP Status {e.response.status_code}"
        try:
            # Try to get error details from response if available (might be HTML or JSON)
            error_body = e.response.text
            error_details += f" - Response: {error_body[:200]}"  # Limit response length
        except Exception:
            pass  # Ignore if response body cannot be read
        return f"Error fetching subtitle content: {error_details}"
    except httpx.RequestError as e:
        ctx.log(
            "error",
            f"Network error fetching subtitle content for URL {e.request.url}: {e}",
        )
        return f"Error fetching subtitle content (network issue): {e}"
    except video.VideoInfoNotFoundError:
        ctx.log("error", f"Video info not found for bvid: {bvid}")
        return f"Error: Video with bvid '{bvid}' not found or access denied."
    except video.SubtitleNotFoundError:
        ctx.log("warning", f"Subtitle info not found for cid: {cid}")
        return "Info: No subtitle metadata found for this video part."
    except Exception as e:
        ctx.log("error", f"An unexpected error occurred: {e}")  # Log full traceback
        return f"An unexpected error occurred: {type(e).__name__} - {e}"


def main():
    """
    Main function to run the MCP server with CLI arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Bilibili Subtitle Fetch MCP Server")
    parser.add_argument(
        "--preferred-lang",
        default=os.environ.get("BILIBILI_PREFERRED_LANG", "zh-CN"),
        help="Preferred subtitle language (default: zh-CN)",
    )
    parser.add_argument(
        "--output-format",
        default=os.environ.get("BILIBILI_OUTPUT_FORMAT", "text"),
        choices=["text", "timestamped"],
        help="Subtitle output format (text or timestamped)",
    )

    args = parser.parse_args()

    # Update the tool's default parameters
    get_bilibili_subtitle.__defaults__ = (args.preferred_lang, args.output_format)

    mcp.run()


# Main execution block to run the server
if __name__ == "__main__":
    main()
