import json
import tempfile
import unittest
from pathlib import Path

from tinyforge.dataset.statistics import calculate_stats
from tinyforge.harvest.extractor import extract_page
from tinyforge.harvest.quality import score_document


class PipelineTests(unittest.TestCase):
    def test_extractor_removes_non_content_and_resolves_links(self):
        page = extract_page(
            "<html><head><title> Docs </title></head><body>"
            "<nav>Menu</nav><main><h1>Guide</h1><p>Hello <b>world</b>.</p>"
            "<a href='/next#part'>Next</a></main><script>ignore()</script></body></html>",
            "https://example.test/docs",
        )
        self.assertEqual(page["title"], "Docs")
        self.assertIn("Guide", page["content"])
        self.assertNotIn("Menu", page["content"])
        self.assertEqual(page["links"], ["https://example.test/next"])

    def test_quality_score_is_bounded_and_exposes_signals(self):
        result = score_document("A useful document with clear punctuation.", 100)
        self.assertTrue(0 <= result["score"] <= 1)
        self.assertEqual(set(result["signals"]), {
            "content_density", "boilerplate_ratio", "duplication",
            "text_coherence", "length_suitability",
        })

    def test_statistics_are_json_serializable(self):
        stats = calculate_stats([{
            "text": "one two three", "language": "en", "quality": 0.8,
            "duplicate": False,
        }])
        self.assertEqual(stats["examples"], 1)
        json.dumps(stats)
