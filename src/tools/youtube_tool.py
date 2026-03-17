"""
tools/youtube_tool.py — Search YouTube for educational videos on ML/AI/DS topics.
Uses YouTubeSearch (no API key needed) with optional YouTube Data API v3.
Prioritises channels provided by the user.
"""

from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class YouTubeVideo:
    title: str
    channel: str
    url: str
    description: str
    view_count: str
    duration: str
    published: str
    thumbnail: str
    is_priority_channel: bool = False


# Top ML/AI/DS YouTube channels
KNOWN_ML_CHANNELS = [
    "3Blue1Brown", "Andrej Karpathy", "Yannic Kilcher",
    "Sentdex", "Two Minute Papers", "Lex Fridman",
    "StatQuest with Josh Starmer", "The Math of Intelligence",
    "Deeplearning.ai", "Stanford Online", "MIT OpenCourseWare",
    "Google DeepMind", "AI Coffee Break with Letitia",
    "Arxiv Insights", "Alfredo Canziani",
]


def search_youtube_videos(
    query: str,
    max_results: int = 10,
    priority_channel: Optional[str] = None,
    youtube_api_key: Optional[str] = None,
) -> list[YouTubeVideo]:
    """
    Search YouTube for educational videos.
    Tries YouTube Data API v3 first, falls back to youtubesearchpython.
    """
    videos = []

    if youtube_api_key:
        videos = _search_with_api(query, max_results, priority_channel, youtube_api_key)

    if not videos:
        videos = _search_without_api(query, max_results, priority_channel)

    # Mark priority channel videos
    if priority_channel:
        for v in videos:
            if priority_channel.lower() in v.channel.lower():
                v.is_priority_channel = True

    # Sort: priority channel first, then by view count
    videos.sort(key=lambda v: (v.is_priority_channel, _parse_views(v.view_count)), reverse=True)
    return videos[:max_results]


def _parse_views(view_str: str) -> int:
    """Parse view count string to int for sorting."""
    try:
        clean = view_str.replace(",", "").replace(" views", "").replace("K", "000").replace("M", "000000")
        return int(clean)
    except Exception:
        return 0


def _search_with_api(
    query: str,
    max_results: int,
    priority_channel: Optional[str],
    api_key: str,
) -> list[YouTubeVideo]:
    """Search using YouTube Data API v3."""
    videos = []
    try:
        import requests

        # Priority channel search first
        if priority_channel:
            try:
                channel_resp = requests.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part": "snippet",
                        "q": f"{priority_channel} {query}",
                        "type": "video",
                        "maxResults": min(3, max_results),
                        "key": api_key,
                        "relevanceLanguage": "en",
                    },
                    timeout=15,
                )
                if channel_resp.status_code == 200:
                    for item in channel_resp.json().get("items", []):
                        v = _parse_yt_api_item(item)
                        if v:
                            v.is_priority_channel = True
                            videos.append(v)
            except Exception:
                pass

        # General search
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": f"{query} tutorial machine learning",
                "type": "video",
                "maxResults": min(max_results, 10),
                "key": api_key,
                "relevanceLanguage": "en",
                "order": "relevance",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            for item in resp.json().get("items", []):
                v = _parse_yt_api_item(item)
                if v and not any(ev.url == v.url for ev in videos):
                    videos.append(v)

    except Exception as e:
        print(f"[youtube_tool] API search failed: {e}")
    return videos


def _parse_yt_api_item(item: dict) -> Optional[YouTubeVideo]:
    """Parse a YouTube API search result item."""
    try:
        vid_id = item.get("id", {}).get("videoId", "")
        if not vid_id:
            return None
        snippet = item.get("snippet", {})
        return YouTubeVideo(
            title=snippet.get("title", "Unknown"),
            channel=snippet.get("channelTitle", "Unknown"),
            url=f"https://www.youtube.com/watch?v={vid_id}",
            description=snippet.get("description", "")[:200],
            view_count="N/A",
            duration="N/A",
            published=snippet.get("publishedAt", "")[:10],
            thumbnail=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
        )
    except Exception:
        return None


def _search_without_api(
    query: str,
    max_results: int,
    priority_channel: Optional[str],
) -> list[YouTubeVideo]:
    """Fallback: search using youtubesearchpython (no API key)."""
    videos = []
    try:
        from youtubesearchpython import VideosSearch

        search_query = f"{query} machine learning tutorial"
        if priority_channel:
            search_query = f"{priority_channel} {query}"

        vs = VideosSearch(search_query, limit=min(max_results, 10))
        results = vs.result()

        for item in (results or {}).get("result", [])[:max_results]:
            try:
                vid_id = item.get("id", "")
                thumbnails = item.get("thumbnails", [{}])
                thumb = thumbnails[0].get("url", "") if thumbnails else ""

                videos.append(YouTubeVideo(
                    title=item.get("title", "Unknown"),
                    channel=item.get("channel", {}).get("name", "Unknown"),
                    url=f"https://www.youtube.com/watch?v={vid_id}",
                    description=item.get("descriptionSnippet") or "",
                    view_count=item.get("viewCount", {}).get("short", "N/A"),
                    duration=item.get("duration") or "N/A",
                    published=item.get("publishedTime") or "N/A",
                    thumbnail=thumb,
                ))
            except Exception as e:
                print(f"[youtube_tool] Video parse error: {e}")

    except Exception as e:
        print(f"[youtube_tool] youtubesearchpython failed: {e}")
        # Last resort: construct YouTube search URL
        videos.append(YouTubeVideo(
            title=f"Search YouTube: {query} machine learning",
            channel="YouTube Search",
            url=f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}+machine+learning",
            description="Click to search YouTube directly",
            view_count="N/A",
            duration="N/A",
            published="N/A",
            thumbnail="",
        ))
    return videos
