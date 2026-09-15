from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .extractor import extract_page
from .quality import score_document
from tinyforge.dataset.statistics import calculate_stats

# Fix #3: cap response size so a single huge page can't hang/blow up a crawl.
# timeout only bounds connection/read *stalls*, not total payload size.
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Stealth browser (optional fetch path — see Crawler.use_stealth below).
#
# Kept in this file rather than a separate module by design constraint.
# Selenium/undetected-chromedriver are only imported lazily inside
# _launch_stealth_driver(), so a Crawler that never sets use_stealth=True
# doesn't need those packages installed at all.
#
# Scope reminder (per upstream github.com/greekr4/playwright-bot-bypass
# README / SECURITY.md): authorized use only — QA, accessibility, and
# research on sites you own or have permission to test. Doesn't defeat
# IP reputation/rate limiting or behavioral gates (Turnstile, DataDome,
# Kasada).
# ---------------------------------------------------------------------------

@dataclass
class StealthConfig:
    headless: bool = False
    window_size: tuple[int, int] = (1366, 768)
    page_load_timeout: int = 30


def _launch_stealth_driver(config: "StealthConfig | None" = None):
    import undetected_chromedriver as uc

    cfg = config or StealthConfig()
    options = uc.ChromeOptions()
    options.add_argument(f"--window-size={cfg.window_size[0]},{cfg.window_size[1]}")
    driver = uc.Chrome(options=options, headless=cfg.headless)
    driver.set_page_load_timeout(cfg.page_load_timeout)
    return driver


def _simulate_mouse_movement(driver, steps: int = 4) -> None:
    """Clamped to the real <body> box, each step wrapped so one
    out-of-bounds move doesn't abort a crawl mid-run."""
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By

    body = driver.find_element(By.TAG_NAME, "body")
    size = body.size
    body_width = max(int(size.get("width", 0)), 1)
    body_height = max(int(size.get("height", 0)), 1)
    for _ in range(steps):
        x = random.randint(0, body_width - 1)
        y = random.randint(0, body_height - 1)
        try:
            actions = ActionChains(driver)
            actions.move_to_element_with_offset(body, x, y)
            actions.pause(random.uniform(0.05, 0.2))
            actions.perform()
        except Exception:
            continue


def _stealth_fetch(driver, url: str, warm_up: bool = False) -> tuple[str, str]:
    """Fetch one page using an already-open stealth driver (reuse across
    a crawl — launching a fresh browser per page would be far too slow).
    warm_up defaults off here since a multi-page crawl already spends
    real wall-clock time per page."""
    driver.get(url)
    if warm_up:
        _simulate_mouse_movement(driver)
    content_type = driver.execute_script("return document.contentType") or "text/html"
    return driver.page_source, content_type


class Crawler:
    def __init__(
        self,
        output: Path,
        max_pages: int = 20,
        delay: float = 1.0,
        timeout: int = 15,
        use_stealth: bool = False,
    ) -> None:
        self.output = output
        self.max_pages = max_pages
        self.delay = delay
        self.timeout = timeout
        self.user_agent = "TinyForge/0.1 (+local research tool)"
        # Default False: the polite urllib path (honest UA, robots.txt,
        # rate-limited) is the crawler's whole design ethos. Stealth is
        # an explicit opt-in for sites that block plain requests —
        # authorized research/QA use only, see module docstring above.
        self.use_stealth = use_stealth
        self._stealth_driver = None  # lazily created, reused across pages

    def _fetch(self, url: str) -> tuple[str, str]:
        if self.use_stealth:
            if self._stealth_driver is None:
                self._stealth_driver = _launch_stealth_driver()
            html, content_type = _stealth_fetch(self._stealth_driver, url)
            if len(html.encode("utf-8", errors="replace")) > _MAX_RESPONSE_BYTES:
                raise ValueError(f"response exceeds {_MAX_RESPONSE_BYTES} byte cap")
            return html, content_type
        return self._fetch_urllib(url)

    def _fetch_urllib(self, url: str) -> tuple[str, str]:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "text/html"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            # Fix #2: urlopen follows redirects transparently. If a
            # same-domain URL redirects off-domain, response.url reflects
            # the final landing URL — check it against the original
            # domain before trusting the content as "same-origin".
            final_netloc = urlparse(response.url).netloc
            if final_netloc and final_netloc != urlparse(url).netloc:
                raise ValueError(f"redirected off-domain to {final_netloc}")
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"unsupported content type: {content_type}")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ValueError(f"response exceeds {_MAX_RESPONSE_BYTES} byte cap")
            return raw.decode(response.headers.get_content_charset() or "utf-8", errors="replace"), content_type

    def _close_stealth(self) -> None:
        if self._stealth_driver is not None:
            self._stealth_driver.quit()
            self._stealth_driver = None

    def run(self, start_url: str, extract_fields: set[str] | None = None) -> dict[str, object]:
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
        try:
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
                    page = extract_page(html, url, (extract_fields or set()) | {"links"})
                    crawl_links = list(page.get("links", []))
                    content = str(page.get("content", ""))
                    digest = hashlib.sha256(" ".join(content.split()).lower().encode()).hexdigest()
                    duplicate = digest in seen_content
                    seen_content.add(digest)
                    quality = score_document(content, len(html), duplicate)
                    slug = hashlib.sha1(url.encode()).hexdigest()[:12]
                    (self.output / "raw" / f"{slug}.html").write_text(html, encoding="utf-8")
                    (self.output / "cleaned" / f"{slug}.md").write_text(content + "\n", encoding="utf-8")
                    page_record = {
                        **page, "content": content, "quality": quality["score"],
                        "quality_signals": quality["signals"], "duplicate": duplicate,
                    }
                    if "links" not in (extract_fields or set()):
                        page_record.pop("links", None)
                    pages.append(page_record)
                    for link in crawl_links:
                        link_url = str(link)
                        if link_url not in visited and urlparse(link_url).netloc == parsed.netloc and link_url not in queue:
                            queue.append(link_url)
                # Fix #1: extract_page (and its numbers/names/phone regex
                # passes) can in principle throw on pathological/malformed
                # markup. Widened from the original 4-exception tuple so
                # one bad page logs an error and the crawl continues,
                # rather than the whole run dying partway through.
                except Exception as exc:
                    errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self._close_stealth()
        records = [{
            "text": page.get("content", ""), "url": page["url"],
            "title": page.get("title", ""), "source": page.get("source", ""),
            "language": page.get("language", "unknown"), "quality": page["quality"],
            "duplicate": page["duplicate"],
        } for page in pages if page.get("content") and not page["duplicate"]]
        _write_jsonl(self.output / "pages.jsonl", pages)
        _write_jsonl(self.output / "dataset.jsonl", records)
        stats = calculate_stats(records)
        (self.output / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        report = {
            "start_url": start_url, "pages_attempted": len(visited), "pages_extracted": len(pages),
            "errors": errors, "dataset": stats,
            "consent": {"approved_fields": sorted(extract_fields or set())},
            "fetch_mode": "stealth" if self.use_stealth else "urllib",
        }
        (self.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")