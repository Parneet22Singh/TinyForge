from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin, urldefrag, urlparse


_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer", "header", "aside", "form"}
_BLOCK_TAGS = {"article", "main", "section", "div", "p", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote", "tr"}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._title_depth = 0
        self._skip_depth = 0
        self._parts: list[str] = []
        self.links: list[str] = []
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._title_depth += 1
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "pre":
            self._in_pre = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
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
        if not self._skip_depth and tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self.title += data
        if not self._skip_depth:
            self._parts.append(data if self._in_pre else " ".join(data.split()))

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line).strip()


def extract_page(html: str, url: str) -> dict[str, object]:
    parser = PageParser()
    parser.feed(html)
    parsed = urlparse(url)
    return {
        "url": url,
        "title": " ".join(parser.title.split()),
        "content": parser.text(),
        "source": parsed.netloc,
        "language": "en",
        "links": [urldefrag(urljoin(url, link))[0] for link in parser.links],
    }
