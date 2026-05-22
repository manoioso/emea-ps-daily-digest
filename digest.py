def curate_with_gemini(items):
    """
    Ask Gemini to pick the single best article per target category,
    enforcing distinct sources and distinct categories.
    Robust against output truncation, JSON errors and oversized prompts.
    """
    if not GEMINI_API_KEY:
        print("[error] GEMINI_API_KEY is required for curation", file=sys.stderr)
        return []
    if not items:
        return []

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)

    # Pre-filter and sort: most recent first, cap at 80 articles to keep
    # prompt size reasonable. Items without a title are useless.
    filtered = [x for x in items if x.get("title")]
    filtered.sort(key=lambda x: x.get("ts", ""), reverse=True)
    pool = filtered[:80]
    if not pool:
        print("[warn] no articles with title in pool", file=sys.stderr)
        return []

    items_for_prompt = [
        {
            "i": i,
            "source": x["source"],
            "feed_category": x["category"],
            "title": x["title"][:200],
            "summary": (x.get("summary") or "")[:300],
        }
        for i, x in enumerate(pool)
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

    user_prompt = f"""You receive a pool of {len(pool)} articles. Select at most 5, one for each target category below.

Target categories (use the exact key in the JSON):
{cat_block}

Strict rules:
- At most one article per target category
- At most one article per source
- Only articles with strong relevance for EMEA public sector enterprise sales
- If a category has no sufficiently relevant article, omit that category
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
                "max_output_tokens": 8192,
                "temperature": 0.3,
            },
        )
    except Exception as e:
        print(f"[error] Gemini call failed: {e}", file=sys.stderr)
        return []

    # Diagnostics: log why Gemini stopped (MAX_TOKENS, SAFETY, STOP, ...)
    try:
        finish = response.candidates[0].finish_reason
        print(f"[info] Gemini finish_reason: {finish}", file=sys.stderr)
    except Exception:
        pass

    raw = ""
    try:
        raw = response.text or ""
    except Exception as e:
        print(f"[error] Gemini response has no text: {e}", file=sys.stderr)
        return []

    if not raw.strip():
        print("[error] Gemini returned empty response", file=sys.stderr)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[error] JSON parse error: {e}", file=sys.stderr)
        print(f"[debug] raw start: {raw[:300]}", file=sys.stderr)
        print(f"[debug] raw end: {raw[-300:]}", file=sys.stderr)
        return []

    if not isinstance(data, dict):
        print(f"[error] unexpected JSON root: {type(data).__name__}", file=sys.stderr)
        return []
    picks = data.get("picks")
    if not isinstance(picks, list):
        print("[error] picks field missing or not a list", file=sys.stderr)
        return []

    seen_sources = set()
    seen_categories = set()
    canonical = []
    for p in picks:
        if not isinstance(p, dict):
            continue
        idx = p.get("i")
        cat = p.get("category")
        summary_en = (p.get("summary_en") or "").strip()
        if not isinstance(idx, int) or not 0 <= idx < len(pool):
            continue
        if cat not in CATEGORIES:
            continue
        if cat in seen_categories:
            continue
        src = pool[idx]["source"]
        if src in seen_sources:
            continue
        if not summary_en:
            continue
        seen_sources.add(src)
        seen_categories.add(cat)
        it = dict(pool[idx])
        it["target_category"] = cat
        it["summary_en"] = summary_en
        canonical.append(it)

    order = list(CATEGORIES.keys())
    canonical.sort(key=lambda x: order.index(x["target_category"]))
    return canonical
