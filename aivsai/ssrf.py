"""SSRF guard — block private / loopback / link-local / metadata destinations.

Pen-testers legitimately need to test internal targets, so this is opt-in:
either set ``AIVSAI_ALLOW_PRIVATE_TARGETS=1`` or pass ``allow_private=True``
per-run from the UI checkbox.

The check happens once at run-start (against the resolved hostname) so a
malicious DNS response can't flip a public hostname to a private IP between
the check and the request. We do a fresh DNS resolve and verify each
returned address against blocklists.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse


# AWS / GCP / Azure metadata service — special blocked addresses
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True   # unparseable → fail closed
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or ip_str in _METADATA_IPS
    )


def check_target_url(url: str, allow_private: bool = False) -> Optional[str]:
    """Return None if the URL is safe to fetch. Otherwise an error string
    explaining why it was blocked."""
    if allow_private:
        return None

    try:
        u = urlparse(url)
    except ValueError:
        return f"Could not parse URL: {url!r}"
    if u.scheme not in ("http", "https"):
        return f"Only http(s) targets allowed, got scheme {u.scheme!r}."

    host = u.hostname
    if not host:
        return "Target URL has no host."

    # Direct-IP destination
    try:
        ipaddress.ip_address(host)
        if _is_private_ip(host):
            return (
                f"Target host {host} is a private/loopback/metadata address. "
                f"If you're intentionally testing an internal target, tick the "
                f"'I'm testing an internal/local target' checkbox or set the "
                f"AIVSAI_ALLOW_PRIVATE_TARGETS=1 environment variable."
            )
        return None
    except ValueError:
        # Not a literal IP — fall through to DNS resolution
        pass

    # Resolve hostname and reject if any A/AAAA record points to a blocked range.
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        return f"DNS resolution failed for {host!r}: {e}"
    for info in infos:
        ip_str = info[4][0]
        if _is_private_ip(ip_str):
            return (
                f"Hostname {host!r} resolves to a private/internal address "
                f"({ip_str}). If you're intentionally testing an internal "
                f"target, tick the 'I'm testing an internal/local target' "
                f"checkbox or set AIVSAI_ALLOW_PRIVATE_TARGETS=1."
            )
    return None
