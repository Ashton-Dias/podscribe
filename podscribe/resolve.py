"""Spotify episode URL -> episode in a show's RSS feed (audio comes from the feed, not Spotify)."""
import html as htmllib
import re
import sys
import time
from dataclasses import dataclass

import feedparser
import requests
from rapidfuzz import fuzz

from . import PodscribeError
from .net import safe_get

OEMBED = "https://open.spotify.com/oembed"
SPOTIFY_RE = re.compile(r"^https?://open\.spotify\.com/(?:intl-[a-z]+/)?episode/[A-Za-z0-9]+")
ITUNES_SEARCH = "https://itunes.apple.com/search"
FEED_SHOW_MATCH = 70  # min similarity between Spotify show name and a feed's own title
SHOW_MATCH = 85  # min similarity between Spotify show name and iTunes collection name
CONFIDENT = 85
MARGIN = 5


@dataclass
class Episode:
    title: str
    show: str
    date: str  # YYYY-MM-DD
    mp3_url: str
    source_url: str
    feed_url: str = ""
    discovered: bool = False  # feed was found by search rather than given with --feed


@dataclass
class Match:
    spotify_title: str
    candidates: list  # [(score, Episode)] best first, max 3
    confident: bool
    show: str | None = None


def is_spotify_episode(url: str) -> bool:
    return bool(SPOTIFY_RE.match(url or ""))


def spotify_title(url: str) -> str:
    if not is_spotify_episode(url):
        raise PodscribeError("Expected a Spotify episode URL (https://open.spotify.com/episode/...).")
    try:
        r = requests.get(OEMBED, params={"url": url}, timeout=15)
        r.raise_for_status()
        return r.json()["title"]
    except (requests.RequestException, KeyError, ValueError) as e:
        raise PodscribeError(f"Could not get episode title from Spotify oEmbed: {e}")


def parse_spotify_page(page: str, title: str) -> str | None:
    """Show name from the public episode page (og:description 'Show · Episode', or '<title> - Show | Podcast on Spotify')."""
    m = re.search(r'<meta property="og:description" content="([^"]*)"', page)
    if m and " · " in m.group(1):
        first = htmllib.unescape(m.group(1)).split(" · ")[0].strip()
        if first and len(first) <= 120:
            return first
    m = re.search(r"<title>(.*?)</title>", page, re.S)
    if m:
        t = re.sub(r"\s*\|\s*Podcast on Spotify\s*$", "", htmllib.unescape(m.group(1)).strip())
        if t.startswith(title + " - ") and t[len(title) + 3:].strip():
            return t[len(title) + 3:].strip()
    return None


def spotify_meta(url: str) -> tuple[str, str | None]:
    """(episode title, show name or None). The show name is best-effort; its absence is not an error."""
    title = spotify_title(url)
    try:
        r = safe_get(SPOTIFY_RE.match(url).group(0))
        r.raise_for_status()
        return title, parse_spotify_page(r.text, title)
    except (requests.RequestException, PodscribeError):
        return title, None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def _mp3_url(entry) -> str | None:
    for enc in entry.get("enclosures", []):
        if enc.get("href") and ("audio" in enc.get("type", "") or enc["href"].split("?")[0].endswith(".mp3")):
            return enc["href"]
    return None


def _date(entry) -> str:
    t = entry.get("published_parsed")
    return time.strftime("%Y-%m-%d", t) if t else "unknown-date"


def load_feed(feed_url: str):
    try:
        r = safe_get(feed_url)
        r.raise_for_status()
    except requests.RequestException as e:
        raise PodscribeError(f"Could not fetch feed {feed_url}: {e}")
    feed = feedparser.parse(r.content)
    if not feed.entries:
        raise PodscribeError(f"Could not read any episodes from feed {feed_url}")
    return feed


def _episode(feed, entry, source_url) -> Episode:
    mp3 = _mp3_url(entry)
    if not mp3:
        raise PodscribeError(f"Episode '{entry.get('title')}' has no audio enclosure in the feed.")
    return Episode(entry.get("title", "untitled"), feed.feed.get("title", "unknown-show"), _date(entry), mp3, source_url)


def _itunes(entity: str, term: str, limit: int) -> list[dict]:
    try:
        r = requests.get(ITUNES_SEARCH, params={"media": "podcast", "entity": entity, "term": term, "limit": limit}, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except (requests.RequestException, ValueError):
        return []


def _sim(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(_norm(a), _norm(b))


def _show_sim(a: str, b: str) -> float:
    """Strict show-name comparison (ignores 'the'/'podcast'/'show'); partial matches must not pass."""
    strip = lambda x: " ".join(w for w in _norm(x).split() if w not in ("the", "podcast", "show"))
    return fuzz.token_sort_ratio(strip(a), strip(b))


def discover_feeds(title: str, show: str | None) -> list[str]:
    """RSS feed URLs (from the free iTunes Search API) that plausibly belong to this show."""
    found: list[tuple[float, str]] = []
    if show:
        for r in _itunes("podcast", show, 5):
            if r.get("feedUrl") and _show_sim(show, r.get("collectionName", "")) >= SHOW_MATCH:
                found.append((_show_sim(show, r["collectionName"]), r["feedUrl"]))
        found.sort(key=lambda x: -x[0])
    for r in _itunes("podcastEpisode", title, 10):
        feed = r.get("feedUrl")
        if not feed:
            continue
        if show and _show_sim(show, r.get("collectionName", "")) >= SHOW_MATCH:
            found.append((50, feed))
        elif not show and _sim(title, r.get("trackName", "")) >= 95:
            found.append((50, feed))
    urls: list[str] = []
    for _, u in found:
        if u not in urls:
            urls.append(u)
    return urls[:4]


def _score(title: str, feed_urls: list[str], discovered: bool, errors: list[str], show: str | None = None) -> list[tuple[float, Episode]]:
    scored = []
    for fu in feed_urls:
        try:
            feed = load_feed(fu)
        except PodscribeError as e:
            errors.append(str(e))
            continue
        # An episode from a different show can never be the answer, however similar the titles look.
        if show and _show_sim(show, feed.feed.get("title", "")) < FEED_SHOW_MATCH:
            continue
        for e in feed.entries:
            if _mp3_url(e):
                ep = _episode(feed, e, "")
                ep.feed_url, ep.discovered = fu, discovered
                scored.append((_sim(title, e.get("title", "")), ep))
    return scored


def find_matches(url: str, feed_urls: list[str] | None = None) -> Match:
    """Match a Spotify episode to an RSS episode. With explicit feed URLs only those are searched; otherwise the
    show's feed is discovered through the iTunes search API. Nothing is remembered between calls."""
    title, show = spotify_meta(url)
    explicit = bool(feed_urls)
    urls = feed_urls or []
    errors: list[str] = []
    scored = _score(title, urls, False, errors, show)
    if not explicit and not _confident(scored):
        new = [u for u in discover_feeds(title, show) if u not in urls]
        scored += _score(title, new, True, errors, show)
    if not scored:
        raise PodscribeError(errors[0] if errors and explicit else _not_found(title, show))
    for _, e in scored:
        e.source_url = url  # remember the Spotify link the user pasted
    scored.sort(key=lambda x: -x[0])
    if scored[0][0] < 50:
        raise PodscribeError(_not_found(title, show))
    top = scored[:3]
    confident = _confident(scored)
    if confident and top[0][1].discovered and not show:
        confident = False  # feed found by episode title alone: make the user confirm
    return Match(title, top, confident, show)


def _confident(scored) -> bool:
    if not scored:
        return False
    s = sorted((x[0] for x in scored), reverse=True)
    return s[0] >= CONFIDENT and (len(s) == 1 or s[0] == 100 or s[0] - s[1] >= MARGIN)


def _not_found(title: str, show: str | None) -> str:
    name = f"'{show}'" if show else f"'{title}'"
    return (f"Couldn't find an RSS feed with a matching episode for {name}. It may be a Spotify-exclusive podcast "
            "(those have no public feed) or the episode is too new; you can add its feed URL in Settings.")


def resolve(url: str, feed_url: str | None = None) -> Episode:
    """CLI resolver: asks on the terminal when the match is uncertain."""
    err = lambda *a, **k: print(*a, file=sys.stderr, **k)
    err("Finding the episode and its RSS feed...")
    m = find_matches(url, [feed_url] if feed_url else None)
    ep = m.candidates[0][1]
    if not m.confident:
        err(f"Spotify: {m.spotify_title}" + (f" ({m.show})" if m.show else "") + "\nUncertain match. Top candidates:")
        for i, (score, e) in enumerate(m.candidates, 1):
            err(f"  {i}. [{score:.0f}] {e.title} — {e.show} ({e.date})\n     feed: {e.feed_url}")
        err("Pick 1-3 (or Enter to cancel): ", end="")
        choice = input().strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(m.candidates):
            raise PodscribeError("No matching episode found (selection cancelled).")
        ep = m.candidates[int(choice) - 1][1]
    return ep
