"""
tinyforge/harvest/extractor.py

Pure HTML parsing. Takes already-fetched HTML + its URL, returns
structured fields. No browser, no network access — that's crawler.py's
job (see stealth.py for the optional stealth-fetch path).

Fields: title, headings, text, code, lists, tables, links, metadata,
media, numbers, names, phone_numbers.

NAME EXTRACTION NOTE: `names` uses a zero-dependency heuristic (runs of
capitalized words), not real NER. Fast, no install required, but noisy —
will misfire on headings/brand names and miss lowercase/foreign names.
Swap extract_names() for spaCy if the research needs real entity
extraction; it's isolated for exactly that reason.

PHONE NUMBER NOTE: `phone_numbers` is shape-based (looks like a phone
number, digit-count filtered to 7-15 digits), not validated against
real area/country codes.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urldefrag, urlparse

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer", "header", "aside", "form"}
_BLOCK_TAGS = {"article", "main", "section", "div", "p", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote", "tr"}
_MEDIA_TAGS = {"img", "video", "audio", "source"}

EXTRACTABLE_FIELDS = (
    "title", "headings", "text", "code", "lists", "tables", "links",
    "metadata", "media", "numbers", "names", "phone_numbers",
)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._title_depth = 0
        self._skip_depth = 0
        self._parts: list[str] = []
        self.links: list[str] = []
        self.media: list[dict[str, str]] = []
        self._in_pre = False
        self.headings: list[str] = []
        self.lists: list[str] = []
        self.code: list[str] = []
        self.tables: list[str] = []
        self._current_heading = False
        self._current_list = False
        self._current_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = dict(attrs)
        if tag == "title":
            self._title_depth += 1
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "pre":
            self._in_pre = True
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._current_heading = True
        if tag == "li":
            self._current_list = True
        if tag in {"table", "tr", "td", "th"}:
            self._current_table = True
        if tag == "a":
            href = attr_map.get("href")
            if href:
                self.links.append(href)
        if tag in _MEDIA_TAGS:
            src = attr_map.get("src") or (attr_map.get("srcset") or "").split(",")[0].strip().split(" ")[0]
            if src:
                self.media.append({"type": tag, "src": src, "alt": attr_map.get("alt", "")})
        if not self._skip_depth and tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "pre":
            self._in_pre = False
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._current_heading = False
        if tag == "li":
            self._current_list = False
        if tag == "table":
            self._current_table = False
        if not self._skip_depth and tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self.title += data
        if not self._skip_depth:
            value = data if self._in_pre else " ".join(data.split())
            if self._in_pre:
                self.code.append(value)
            elif self._current_heading:
                self.headings.append(value)
            elif self._current_list:
                self.lists.append(value)
            elif self._current_table:
                self.tables.append(value)
            self._parts.append(value)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line).strip()


# --- numbers -----------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    [$€£¥]?\s?
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        |
        \d+(?:\.\d+)?
    )
    %?
    """,
    re.VERBOSE,
)


def extract_numbers(text: str) -> list[str]:
    """Generic numeric tokens: plain, comma-grouped, decimal, %, currency-
    prefixed. Not phone-aware — see extract_phone_numbers()."""
    found = [m.strip() for m in _NUMBER_RE.findall(text)]
    return [n for n in found if any(ch.isdigit() for ch in n)]


# --- phone numbers -------------------------------------------------------

_PHONE_RE = re.compile(
    r"""
    (?<!\d)
    (?:\+\d{1,3}[-.\s]?)?
    (?:\(\d{2,4}\)[-.\s]?)?
    \d{2,4}
    (?:[-.\s]\d{2,4}){1,3}
    (?:\s?(?:x|ext\.?)\s?\d{1,5})?
    (?!\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_phone_numbers(text: str) -> list[str]:
    """Shape-based phone number detection: (555) 123-4567, 555-123-4567,
    555.123.4567, +1 555 123 4567, +44 20 7946 0958, with optional
    x1234/ext 1234 extensions. Digit-count filtered to 7-15 digits."""
    candidates = [m.strip() for m in _PHONE_RE.findall(text)]
    seen: dict[str, None] = {}
    for c in candidates:
        digit_count = sum(ch.isdigit() for ch in c)
        if 7 <= digit_count <= 15:
            seen.setdefault(c, None)
    return list(seen.keys())


# --- names (heuristic, zero-dependency) ---------------------------------

_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "We", "You",
    "I", "He", "She", "They", "And", "Or", "But", "If", "In", "On", "At",
    "For", "With", "By", "To", "Of", "As", "Is", "Are", "Was", "Were",
}

# Two or more consecutive Capitalized Words on the same line — \s+ would
# also match newlines and falsely join words from separate headings/
# paragraphs into one "name", so this matches literal spaces/tabs only.
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)\b")


def extract_names(text: str) -> list[str]:
    """Heuristic entity extraction: runs of 2+ capitalized words, with
    sentence-initial stopwords filtered. See module docstring for
    accuracy caveats."""
    candidates = _NAME_RE.findall(text)
    seen: dict[str, None] = {}
    for c in candidates:
        words = c.split()
        if words[0] in _STOPWORDS:
            continue
        seen.setdefault(c, None)
    return list(seen.keys())


# --- main entry points ---------------------------------------------------

def extract_page(html: str, url: str, fields: set[str] | None = None) -> dict[str, object]:
    parser = PageParser()
    parser.feed(html)
    parsed = urlparse(url)
    selected = set(fields or EXTRACTABLE_FIELDS)
    links = [urldefrag(urljoin(url, link))[0] for link in parser.links]
    content = parser.text() if "text" in selected else ""
    if "headings" in selected:
        content = "\n".join(parser.headings) + ("\n" + content if content else "")
    if "lists" in selected:
        content += ("\n" if content else "") + "\n".join(f"- {item}" for item in parser.lists)
    if "code" in selected:
        content += ("\n" if content else "") + "\n".join(f"```\n{item}\n```" for item in parser.code)
    if "tables" in selected:
        content += ("\n" if content else "") + "\n".join(parser.tables)
    result: dict[str, object] = {"url": url, "content": content.strip()}
    if "title" in selected:
        result["title"] = " ".join(parser.title.split())
    if "metadata" in selected:
        result.update({"source": parsed.netloc, "language": "en"})
    if "links" in selected:
        result["links"] = links
    if "media" in selected:
        media = []
        for m in parser.media:
            src = urldefrag(urljoin(url, m["src"]))[0] if m["src"] else m["src"]
            media.append({"type": m["type"], "src": src, "alt": m["alt"]})
        result["media"] = media
    if "numbers" in selected:
        base_text = parser.text()
        result["numbers"] = extract_numbers(base_text)
    if "names" in selected:
        base_text = parser.text()
        result["names"] = extract_names(base_text)
    if "phone_numbers" in selected:
        base_text = parser.text()
        result["phone_numbers"] = extract_phone_numbers(base_text)
    return result


def detect_extractable(html: str) -> dict[str, int]:
    parser = PageParser()
    parser.feed(html)
    body_text = parser.text()
    return {
        "title": int(bool(parser.title.strip())),
        "headings": len(parser.headings),
        "text": len(body_text),
        "code": len([item for item in parser.code if item.strip()]),
        "lists": len([item for item in parser.lists if item.strip()]),
        "tables": len([item for item in parser.tables if item.strip()]),
        "links": len(parser.links),
        "metadata": 2,
        "media": len(parser.media),
        "numbers": len(extract_numbers(body_text)),
        "names": len(extract_names(body_text)),
        "phone_numbers": len(extract_phone_numbers(body_text)),
    }