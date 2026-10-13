#!/usr/bin/env python3
"""Open the Colab MCP connection page through the local Chromium CDP endpoint."""
import json
import sys
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


def open_colab(url, endpoint="http://127.0.0.1:9222"):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "colab.research.google.com", "colab.google.com"
    }:
        raise ValueError("Expected a Colab HTTPS URL.")
    request = Request(endpoint + "/json/new?" + quote(url, safe=""), method="PUT")
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        target = json.load(response)
    if not target.get("id"):
        raise RuntimeError("Chromium did not return a tab ID.")


def main():
    if len(sys.argv) != 2:
        print("Expected one Colab URL.", file=sys.stderr)
        return 2
    try:
        open_colab(sys.argv[1])
    except Exception as exc:
        # A connection URL contains a private token. Do not print the URL or exception text.
        print(f"Cannot open Colab through Chromium on port 9222 ({type(exc).__name__}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
