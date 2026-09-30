"""Outbound HTTP with basic SSRF protection: http(s) only, public addresses only, redirects re-checked."""
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

import requests

from . import PodscribeError

HEADERS = {"User-Agent": "Mozilla/5.0 (podscribe)"}


def check_public_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise PodscribeError(f"Only http(s) URLs are allowed: {url[:100]}")
    if os.environ.get("PODSCRIBE_ALLOW_PRIVATE"):
        return
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise PodscribeError(f"Could not resolve host: {p.hostname}")
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
            raise PodscribeError(f"Refusing to fetch a private/local address: {p.hostname}")


def safe_get(url: str, *, stream: bool = False, timeout: int = 30, max_redirects: int = 5) -> requests.Response:
    for _ in range(max_redirects + 1):
        check_public_url(url)
        r = requests.get(url, stream=stream, timeout=timeout, headers=HEADERS, allow_redirects=False)
        if r.is_redirect and r.headers.get("location"):
            url = urljoin(url, r.headers["location"])
            r.close()
            continue
        return r
    raise PodscribeError("Too many redirects.")
