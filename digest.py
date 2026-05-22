#!/usr/bin/env python3
"""
EMEA Public Sector Daily Digest

Fetches RSS feeds, optionally ranks with Claude, posts a digest to a Slack
workflow webhook configured with a single variable named `text`.

Environment variables:
  SLACK_WEBHOOK_URL  required, Slack workflow trigger URL
  ANTHROPIC_API_KEY  optional, enables Claude ranking and italian summaries
  CLAUDE_MODEL       optional, default claude-sonnet-4-6
  LOOKBACK_HOURS     optional, default 24
  TOP_N              optional, default 12
  DRY_RUN            optional, if set to 1 prints message instead of posting
"""
import json
import os
import sys
import datetime as dt
from xml.etree import ElementTree as ET

import feedparser
import requests

LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "24"))
MAX_ITEMS_PER_FEED = 6
TOP_N = int(os.environ.get("TOP_N", "12"))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DRY_RUN = os.environ.get("DRY_RUN") == "1"

if not SLACK_WEBHOOK_URL and not DRY_RUN:
    print("[error] SLACK_WEBHOOK_URL not set", file=sys.stderr)
    sys.exit(1)


def parse_opml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    feeds = []
    for category in root.find("body"):
        cat_name = category.get("text", "Misc")
        for outline in category.findall("outline"):
            url = outline.get("xmlUrl")
            name = outline.get("text", url)
            if url:
                feeds.append((cat_name, name, url))
    return feeds


def fetch_feed(name, url):
    try:
        d = feedparser.parse(
            url,
            request_headers={"User-Agent": "EMEA-PS-Digest/1.0"},
        )
        return d.entries[:MAX_ITEMS_PER_FEED]
    except Exception as e:
        print(f"[warn] {name} failed: {e}", file=sys.stderr)
        return []


def entry_datetime(entry):
    for key in ("published_parsed", "updated_parsed"):
        v = entry.get(key)
        if v:
            return dt.datetime(*v[:6], tzinfo=dt.timezone.utc)
    return None


def collect_recent(feeds):
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=LOOKBACK_HOURS)
    items = []
    for category, name, url in feeds:
        for entry in fetch_feed(name, url):
            ts = entry_datetime(entry)
            if ts is None or ts >= cutoff:
                items.append({
                    "category": category,
                    "source": name,
                    "title": (entry.get("title") or "").strip(),
                    "link": entry.get("link") or "",
                    "summary": (entry.get("summary") or "").strip()[:600],
                    "ts": ts.isoformat() if ts else "",
                })
    return items


def rank_with_claude(items):
    if not ANTHROPIC_API_KEY or not items:
        return items[:TOP_N], False

    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    items_for_prompt = [
        {"i": i, "src": x["source"], "cat": x["category"],
         "title": x["title"], "summary": x["summary"][:300]}
        for i, x in enumerate(items)
    ]

    system = (
        "Sei un analyst MongoDB EMEA Public Sector. Selezioni e sintetizzi "
        "le notizie più rilevanti per un team enterprise sales che lavora "
        "con la PA europea (Italia, Germania, Francia, UK, Spagna).\n"
        "Sono rilevanti: sovranità digitale, regolamentazione EU (AI Act, "
        "Data Act, NIS2, DORA, EUCS), cloud, AI, data platform, data center, "
        "vendor public sector (AWS, GCP, Azure, Oracle, SAP, IBM, MongoDB), "
        "iniziative nazionali (PNRR/PSN, Germany Stack, France 2030), "
        "competizione, gare, M&A, security.\n"
        "Non sono rilevanti: consumer, gadget, gossip, crypto speculativo."
    )

    user = (
        f"Hai {len(items)} articoli delle ultime ore. "
        f"Seleziona i {TOP_N} più rilevanti per il public sector EMEA, "
        f"ordinati per importanza decrescente.\n"
        f"Per ciascuno produci un riassunto in italiano massimo 2 righe "
        f"(circa 50 parole) che catturi il punto e perché interessa il sales.\n\n"
        f"Rispondi SOLO con un JSON valido in questo formato esatto, niente "
        f"altro testo prima o dopo:\n"
        f'[{{"i": <indice>, "summary_it": "<riassunto in italiano>"}}, ...]\n\n'
        f"Articoli:\n{json.dumps(items_for_prompt, ensure_ascii=False)}"
    )

    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = msg.content[0].text
    except Exception as e:
        print(f"[warn] Claude call failed: {e}", file=sys.stderr)
        return items[:TOP_N], False

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= 0:
        print("[warn] no JSON in Claude response, falling back", file=sys.stderr)
        return items[:TOP_N], False

    try:
        picks = json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        print(f"[warn] JSON parse error: {e}", file=sys.stderr)
        return items[:TOP_N], False

    ranked = []
    for p in picks:
        idx = p.get("i")
        if isinstance(idx, int) and 0 <= idx < len(items):
            it = dict(items[idx])
            it["summary_it"] = p.get("summary_it", "")
            ranked.append(it)
    return ranked, True


def build_message(items, ai_curated):
    if not items:
        return "Oggi nessuna notizia rilevante nei feed monitorati."

    today = dt.datetime.now().strftime("%A %d %B %Y")
    badge = "AI-curated" if ai_curated else "raw feed"
    lines = [f":newspaper: *EMEA PS Daily Digest — {today}*  _{badge}_", ""]

    for i, x in enumerate(items, 1):
        title = (x.get("title") or "")[:200]
        link = x.get("link") or ""
        source = x.get("source") or ""
        summary = x.get("summary_it") or ""
        lines.append(f"*{i}.* <{link}|{title}>")
        lines.append(f"    _{source}_")
        if summary:
            lines.append(f"    {summary}")
        lines.append("")

    text = "\n".join(lines)
    if len(text) > 3500:
        text = text[:3450] + "\n\n_(digest troncato per limite Slack)_"
    return text


def post_to_slack(text):
    if DRY_RUN:
        print("=== DRY RUN ===")
        print(text)
        return
    r = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=30)
    if r.status_code >= 300:
        print(f"[error] Slack returned {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)
    print(f"[ok] posted {len(text)} chars to Slack")


def main():
    feeds = parse_opml("feeds.opml")
    print(f"[info] {len(feeds)} feeds loaded")
    items = collect_recent(feeds)
    print(f"[info] {len(items)} items in last {LOOKBACK_HOURS}h")
    ranked, ai_curated = rank_with_claude(items)
    print(f"[info] {len(ranked)} items in digest, AI={ai_curated}")
    text = build_message(ranked, ai_curated)
    post_to_slack(text)


if __name__ == "__main__":
    main()
