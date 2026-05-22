#!/usr/bin/env python3
"""
EMEA Public Sector Daily Digest v3 (Gemini)

Selects 5 highly relevant news per day across 5 distinct categories
and 5 distinct sources, posts a curated digest to a Slack workflow webhook.

Environment variables:
  SLACK_WEBHOOK_URL  required (unless DRY_RUN), Slack workflow trigger URL
  GEMINI_API_KEY     required for curation
  GEMINI_MODEL       optional, default gemini-2.5-flash
  LOOKBACK_HOURS     optional, default 24
  DRY_RUN            optional, if "1" prints to stdout instead of posting
"""
import json
import os
import sys
import datetime as dt
from xml.etree import ElementTree as ET

import feedparser
import requests

LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "24"))
MAX_ITEMS_PER_FEED = 8
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DRY_RUN = os.environ.get("DRY_RUN") == "1"


CATEGORIES = {
    "ai_data": {
        "label": "AI & Data",
        "icon": ":robot_face:",
        "desc": "AI technology, data platforms, vector DB, RAG, LLM enterprise, ML, MongoDB and data-layer competitors",
    },
    "sovereignty": {
        "label": "Digital Sovereignty",
        "icon": ":shield:",
        "desc": "Sovereign cloud, Germany Stack, EuroStack, Gaia-X, EU strategic autonomy, anti vendor lock-in, sovereign AI",
    },
    "public_sector_emea": {
        "label": "Public Sector EMEA",
        "icon": ":classical_building:",
        "desc": "Public administration digital initiatives in Italy (PNRR, PSN, Sogei, Almaviva, Engineering), France (DINUM, France 2030), Germany (BMDS), UK (GDS), Spain, EU tenders, health-tech, e-government",
    },
    "regulation": {
        "label": "Regulation",
        "icon": ":balance_scale:",
        "desc": "AI Act, Data Act, NIS2, DORA, EUCS, GDPR, DSA, DMA, compliance and enforcement, court decisions, practical impact on cloud and AI vendors",
    },
    "middle_east": {
        "label": "Middle East PS",
        "icon": ":globe_with_meridians:",
        "desc": "Public sector digital initiatives in UAE, Saudi Arabia, Qatar, Turkey: Vision 2030, smart cities, sovereign cloud, sovereign AI, digital ministries, large GovTech tenders",
    },
}


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
            request_headers={"User-Agent": "EMEA-PS-Digest/3.0"},
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


def curate_with_gemini(items):
    """
    Ask Gemini to pick the single best article per target category,
    enforcing five distinct sources and five distinct categories.
    """
    if not GEMINI_API_KEY:
        print("[error] GEMINI_API_KEY is required for curation", file=sys.stderr)
        return []

    if not items:
        return []

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)

    items_for_prompt = [
        {
            "i": i,
            "source": x["source"],
            "feed_category": x["category"],
            "title": x["title"],
            "summary": x["summary"][:400],
        }
        for i, x in enumerate(items)
    ]

    cat_block = "\n".join(
        f'- "{key}" — {meta["label"]}: {meta["desc"]}'
        for key, meta in CATEGORIES.items()
    )

    system_instruction = (
        "You are a senior analyst for MongoDB EMEA Public Sector. "
        "You select only high-signal news for an enterprise sales team working with "
        "public administration, healthcare and government in Europe, Turkey and the Gulf. "
        "Be extremely selective: better to leave a category empty than to include generic news."
    )

    user_prompt = f"""You receive a pool of {len(items)} articles from the last few hours. Select at most 5, one for each target category below.

Target categories (use the exact key in the JSON):
{cat_block}

Strict rules:
- At most one article per target category
- At most one article per source (5 different sources)
- Only articles with high relevance for EMEA public sector sales. If a category has no sufficiently relevant article, omit that category
- No consumer news, gadgets, sports, speculative crypto, gossip
- Prefer: public tenders and procurement, ministry announcements, EU regulation with concrete impact, digital sovereignty moves, vendor moves in public sector, strategic AI initiatives

For each selected article produce:
- "i": index of the article in the pool (integer)
- "category": exact target category key ("ai_data", "sovereignty", "public_sector_emea", "regulation", "middle_east")
- "summary_en": one-line English summary, max 50 words, capturing the key point and why a public sector sales team should care

Reply ONLY with valid JSON in this exact schema:
{{"picks": [{{"i": 12, "category": "ai_data", "summary_en": "..."}}]}}

Articles:
{json.dumps(items_for_prompt, ensure_ascii=False)}
"""

    try:
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=system_instruction,
        )
        response = model.generate_content(
            user_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": 2000,
                "temperature": 0.3,
            },
        )
        raw = response.text
    except Exception as e:
        print(f"[error] Gemini call failed: {e}", file=sys.stderr)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[error] JSON parse error: {e}", file=sys.stderr)
        print(f"[debug] raw: {raw[:500]}", file=sys.stderr)
        return []

    picks = data.get("picks", [])
    seen_sources = set()
    seen_categories = set()
    canonical = []
    for p in picks:
        idx = p.get("i")
        cat = p.get("category")
        if not isinstance(idx, int) or not 0 <= idx < len(items):
            continue
        if cat not in CATEGORIES:
            continue
        if cat in seen_categories:
            continue
        src = items[idx]["source"]
        if src in seen_sources:
            continue
        seen_sources.add(src)
        seen_categories.add(cat)
        it = dict(items[idx])
        it["target_category"] = cat
        it["summary_en"] = p.get("summary_en", "").strip()
        canonical.append(it)

    order = list(CATEGORIES.keys())
    canonical.sort(key=lambda x: order.index(x["target_category"]))
    return canonical


def build_message(items):
    today = dt.datetime.now().strftime("%A %d %B %Y")
    if not items:
        return f":newspaper: *EMEA PS Daily Digest — {today}*\n\n_No relevant news in the monitored feeds today._"

    sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    lines = [
        f":newspaper: *EMEA PS Daily Digest — {today}*",
        f"_{len(items)} curated stories for the EMEA Public Sector team_",
        "",
    ]

    for x in items:
        cat_key = x["target_category"]
        meta = CATEGORIES[cat_key]
        title = (x.get("title") or "")[:200]
        summary = (x.get("summary_en") or "").strip()
        source = x.get("source") or ""
        link = x.get("link") or ""

        lines.append(sep)
        lines.append("")
        lines.append(f"{meta['icon']} *{meta['label'].upper()}*  •  _{source}_")
        lines.append("")
        lines.append(f"*{title}*")
        if summary:
            lines.append(summary)
        lines.append("")
        lines.append(link)
        lines.append("")

    text = "\n".join(lines)
    if len(text) > 3800:
        text = text[:3750] + "\n\n_(truncated due to Slack limit)_"
    return text


def post_to_slack(text):
    if DRY_RUN or not SLACK_WEBHOOK_URL:
        print("=" * 60)
        print("DRY RUN — message that would be posted to Slack:")
        print("=" * 60)
        print(text)
        print("=" * 60)
        return
    r = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=30)
    if r.status_code >= 300:
        print(f"[error] Slack returned {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)
    print(f"[ok] posted {len(text)} chars to Slack")


def main():
    feeds = parse_opml("feeds.opml")
    print(f"[info] {len(feeds)} feeds in OPML")

    items = collect_recent(feeds)
    print(f"[info] {len(items)} items in last {LOOKBACK_HOURS}h")

    curated = curate_with_gemini(items)
    print(f"[info] {len(curated)} curated items")

    text = build_message(curated)
    post_to_slack(text)


if __name__ == "__main__":
    main()
