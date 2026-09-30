import json

import pytest

import podscribe
from podscribe import resolve
from podscribe.resolve import Episode

SPOTIFY = "https://open.spotify.com/episode/abc123"
TITLE = "Episode 27 - Updates on Growth in Business & Life"
SHOW = "The Holistic Hippie Podcast"
FEED = "https://rss.example.com/holistic.rss"
OTHER_FEED = "https://rss.example.com/other.rss"


def ep(title, show, feed, discovered=False):
    return Episode(title, show, "2024-06-11", f"https://cdn.example.com/{abs(hash(title))}.mp3", "", feed, discovered)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Patch network edges: Spotify meta, iTunes search, feed loading."""
    state = {"show": SHOW, "itunes": {}, "feeds": {}, "loaded": []}
    monkeypatch.setattr(resolve, "spotify_meta", lambda url: (TITLE, state["show"]))
    monkeypatch.setattr(resolve, "_itunes", lambda entity, term, limit: state["itunes"].get(entity, []))

    def score(title, urls, discovered, errors, show=None):
        out = []
        for u in urls:
            state["loaded"].append(u)
            for e in state["feeds"].get(u, []):
                e = Episode(**{**e.__dict__, "feed_url": u, "discovered": discovered})
                out.append((resolve._sim(title, e.title), e))
        return out

    monkeypatch.setattr(resolve, "_score", score)
    return state


# ---- page parsing
def test_parse_show_from_og_description():
    page = '<meta property="og:description" content="The Holistic Hippie Podcast · Episode"/>'
    assert resolve.parse_spotify_page(page, TITLE) == SHOW


def test_parse_show_from_title_tag_and_entities():
    page = "<title>Episode 27 - Updates on Growth in Business &amp; Life - The Holistic Hippie Podcast | Podcast on Spotify</title>"
    assert resolve.parse_spotify_page(page, TITLE) == SHOW


@pytest.mark.parametrize("page", ["", "<html></html>", '<meta property="og:description" content="Just a long episode description with no separator"/>',
                                   "<title>Something else entirely</title>"])
def test_parse_show_missing_or_odd_metadata(page):
    assert resolve.parse_spotify_page(page, TITLE) is None


# ---- iTunes filtering
def test_discover_rejects_other_shows(monkeypatch):
    results = {
        "podcast": [{"collectionName": SHOW, "feedUrl": FEED}, {"collectionName": "The Western Hippie", "feedUrl": OTHER_FEED}],
        "podcastEpisode": [{"collectionName": "CREOGs Over Coffee", "trackName": "Episode 129", "feedUrl": "https://x.example/creog.rss"}],
    }
    monkeypatch.setattr(resolve, "_itunes", lambda entity, term, limit: results.get(entity, []))
    assert resolve.discover_feeds(TITLE, SHOW) == [FEED]


def test_discover_without_show_needs_near_exact_title(monkeypatch):
    results = {"podcastEpisode": [{"collectionName": "A", "trackName": TITLE, "feedUrl": FEED},
                                   {"collectionName": "B", "trackName": "Life Update", "feedUrl": OTHER_FEED}]}
    monkeypatch.setattr(resolve, "_itunes", lambda entity, term, limit: results.get(entity, []))
    assert resolve.discover_feeds(TITLE, None) == [FEED]


# ---- resolution order
def test_discovers_feed_for_any_show(fake):
    fake["itunes"]["podcast"] = [{"collectionName": SHOW, "feedUrl": FEED}]
    fake["feeds"][FEED] = [ep(TITLE, SHOW, FEED), ep("Episode 26 - Microdosing", SHOW, FEED)]
    m = resolve.find_matches(SPOTIFY)
    e = m.candidates[0][1]
    assert m.confident and e.discovered and e.feed_url == FEED


def test_explicit_feed_disables_discovery(fake):
    fake["itunes"]["podcast"] = [{"collectionName": SHOW, "feedUrl": FEED}]
    fake["feeds"][FEED] = [ep(TITLE, SHOW, FEED)]
    with pytest.raises(podscribe.PodscribeError):
        resolve.find_matches(SPOTIFY, [OTHER_FEED])


def test_spotify_exclusive_message(fake):
    with pytest.raises(podscribe.PodscribeError) as e:
        resolve.find_matches(SPOTIFY)
    assert "Spotify-exclusive" in str(e.value) and SHOW in str(e.value)


def test_no_show_name_forces_confirmation(fake):
    fake["show"] = None
    fake["itunes"]["podcastEpisode"] = [{"collectionName": SHOW, "trackName": TITLE, "feedUrl": FEED}]
    fake["feeds"][FEED] = [ep(TITLE, SHOW, FEED)]
    m = resolve.find_matches(SPOTIFY)
    assert m.candidates and not m.confident


def test_uncertain_episode_match_not_confident(fake):
    fake["itunes"]["podcast"] = [{"collectionName": SHOW, "feedUrl": FEED}]
    fake["feeds"][FEED] = [ep("Episode 27 - Updates on Growth", SHOW, FEED), ep("Episode 28 - Updates on Business", SHOW, FEED)]
    assert not resolve.find_matches(SPOTIFY).confident


# ---- CLI confirm flow (nothing is remembered)
def test_cli_confirm_flow(fake, monkeypatch):
    fake["itunes"]["podcast"] = [{"collectionName": SHOW, "feedUrl": FEED}]
    fake["feeds"][FEED] = [ep("Episode 27 - Updates on Growth", SHOW, FEED), ep("Episode 28 - Updates on Business", SHOW, FEED)]
    monkeypatch.setattr("builtins.input", lambda *a: "1")
    assert resolve.resolve(SPOTIFY).feed_url == FEED
    assert not hasattr(resolve, "remember_feed")


def test_wrong_show_feed_is_ignored(monkeypatch):
    """Regression: a configured feed for another show must not win on a lucky title score."""
    other = ep("Episode 20 - The 5 Obstacles on The Spiritual Path", "The Holistic Hippie Podcast", OTHER_FEED)
    feed = type("F", (), {"entries": [{"title": other.title, "enclosures": [{"href": other.mp3_url, "type": "audio/mpeg"}]}],
                          "feed": {"title": "The Holistic Hippie Podcast"}})()
    monkeypatch.setattr(resolve, "load_feed", lambda u: feed)
    assert resolve._score("Short Stuff: The Retirement Curse", [OTHER_FEED], False, [], "Stuff You Should Know") == []
    assert resolve._score("Episode 20 - The 5 Obstacles", [OTHER_FEED], False, [], "The Holistic Hippie Podcast")


def test_unrelated_titles_score_low():
    assert resolve._sim("Short Stuff: The Retirement Curse", "Episode 20 - The 5 Obstacles on The Spiritual Path") < 60


def test_source_url_is_the_spotify_link(fake):
    fake["itunes"]["podcast"] = [{"collectionName": SHOW, "feedUrl": FEED}]
    fake["feeds"][FEED] = [ep(TITLE, SHOW, FEED)]
    m = resolve.find_matches(SPOTIFY)
    assert m.candidates[0][1].source_url == SPOTIFY


def test_no_feed_persistence_api():
    import importlib.util
    assert importlib.util.find_spec("podscribe.config") is None
    assert not hasattr(podscribe, "CONFIG_PATH") and not hasattr(podscribe, "OUTPUT_DIR")
