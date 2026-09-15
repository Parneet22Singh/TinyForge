from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

from .extractor import extract_page
from .quality import score_document
from tinyforge.dataset.statistics import calculate_stats


class Crawler:
    def __init__(self, output: Path, max_pages: int = 20, delay: float = 1.0, timeout: int = 15) -> None:
        self.output = output
        self.max_pages = max_pages
        self.delay = delay
        self.timeout = timeout
        self.user_agent = "TinyForge/0.1 (+local research tool)"

    def _fetch(self, url: str) -> tuple[str, str]:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "text/html"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"unsupported content type: {content_type}")
            return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace"), content_type

    def run(self, start_url: str) -> dict[str, object]:
        parsed = urlparse(start_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("URL must be an absolute http(s) URL")
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "raw").mkdir(exist_ok=True)
        (self.output / "cleaned").mkdir(exist_ok=True)
        robots = urllib.robotparser.RobotFileParser()
        robots.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        try:
            robots.read()
        except (OSError, urllib.error.URLError):
            robots = None
        queue = [start_url]
        visited: set[str] = set()
        seen_content: set[str] = set()
        pages: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []
        while queue and len(visited) < self.max_pages:
            url = queue.pop(0)
            if url in visited or urlparse(url).netloc != parsed.netloc:
                continue
            visited.add(url)
            if robots is not None and not robots.can_fetch(self.user_agent, url):
                errors.append({"url": url, "error": "disallowed by robots.txt"})
                continue
            if len(visited) > 1:
                time.sleep(self.delay)
            try:
                html, _ = self._fetch(url)
                page = extract_page(html, url)
                content = str(page["content"])
                digest = hashlib.sha256(" ".join(content.split()).lower().encode()).hexdigest()
                duplicate = digest in seen_content
                seen_content.add(digest)
                quality = score_document(content, len(html), duplicate)
                slug = hashlib.sha1(url.encode()).hexdigest()[:12]
                (self.output / "raw" / f"{slug}.html").write_text(html, encoding="utf-8")
                (self.output / "cleaned" / f"{slug}.md").write_text(content + "\n", encoding="utf-8")
                pages.append({"url": url, "title": page["title"], "content": content, "source": page["source"], "language": page["language"], "quality": quality["score"], "quality_signals": quality["signals"], "duplicate": duplicate})
                for link in page["links"]:
                    link_url = str(link)
                    if link_url not in visited and urlparse(link_url).netloc == parsed.netloc and link_url not in queue:
                        queue.append(link_url)
            except (OSError, ValueError, urllib.error.URLError, UnicodeError) as exc:
                errors.append({"url": url, "error": str(exc)})
        records = [{"text": page["content"], "url": page["url"], "title": page["title"], "source": page["source"], "language": page["language"], "quality": page["quality"], "duplicate": page["duplicate"]} for page in pages if page["content"] and not page["duplicate"]]
        _write_jsonl(self.output / "pages.jsonl", pages)
        _write_jsonl(self.output / "dataset.jsonl", records)
        stats = calculate_stats(records)
        (self.output / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        report = {"start_url": start_url, "pages_attempted": len(visited), "pages_extracted": len(pages), "errors": errors, "dataset": stats}
        (self.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
