"""Tests for single-trajectory memory induction."""

from reasoning_bank.core.induction import induce
from reasoning_bank.core.parsing import parse_memory_items, strip_thinking

# ---------------------------------------------------------------------------
# Unit tests for parse_memory_items (shared parser)
# ---------------------------------------------------------------------------


class TestParseMemoryItems:
    """Tests for the shared parse_memory_items function."""

    def test_structured_multiple_items(self):
        raw = "# Memory Item 1\n## Title A\n## Content a\n\n# Memory Item 2\n## Title B\n## Content b"
        items = parse_memory_items(raw)
        assert len(items) == 2

    def test_structured_with_thinking(self):
        raw = (
            "<thinking>\nLet me analyze this trajectory...\n"
            "The agent succeeded by using search effectively.\n"
            "</thinking>\n\n"
            "# Memory Item 1\n## Title Search Strategy\n## Content Use search first\n\n"
            "# Memory Item 2\n## Title Navigation\n## Content Follow links after search"
        )
        items = parse_memory_items(raw)
        assert len(items) == 2
        assert "Search Strategy" in items[0]
        assert "Navigation" in items[1]

    def test_thinking_only_no_items(self):
        raw = "<thinking>\nI need to think about this...\nBut I can't extract anything useful.\n</thinking>"
        items = parse_memory_items(raw)
        assert items == []

    def test_unclosed_thinking_tag(self):
        raw = (
            "<thinking>\nSome reasoning that never closes\n\n"
            "# Memory Item 1\n## Title Oops\n"
            "## Content This gets stripped"
        )
        items = parse_memory_items(raw)
        # Unclosed <thinking> eats everything after it
        assert items == []

    def test_thinking_multiline(self):
        raw = (
            "<thinking>\n"
            "Line 1 of thinking\n"
            "Line 2 of thinking\n"
            "Line 3 of thinking\n"
            "</thinking>\n\n"
            "# Memory Item 1\n## Title Result\n## Content The actual insight"
        )
        items = parse_memory_items(raw)
        assert len(items) == 1
        assert "Result" in items[0]

    def test_fallback_title_headers(self):
        """When no '# Memory Item' headers, fall back to '## Title' headers."""
        raw = "Some preamble text\n## Title First\n## Content first\n## Title Second\n## Content second"
        items = parse_memory_items(raw)
        assert len(items) == 2

    def test_fallback_double_newline(self):
        """When no structured headers at all, fall back to double-newline split."""
        raw = "First item\n\nSecond item\n\nThird item"
        items = parse_memory_items(raw)
        assert len(items) == 3

    def test_empty_string(self):
        items = parse_memory_items("")
        assert items == []

    def test_only_whitespace(self):
        items = parse_memory_items("   \n\n  \n  ")
        assert items == []

    def test_single_item_no_thinking(self):
        raw = "# Memory Item 1\n## Title Test\n## Content A single item"
        items = parse_memory_items(raw)
        assert len(items) == 1
        assert "Test" in items[0]

    def test_thinking_case_insensitive(self):
        raw = "<THINKING>Some thoughts</THINKING>\n\n# Memory Item 1\n## Title X\n## Content Y"
        items = parse_memory_items(raw)
        assert len(items) == 1

    def test_multiple_thinking_blocks(self):
        raw = (
            "<thinking>First thought block</thinking>\n\n"
            "# Memory Item 1\n## Title A\n## Content a\n\n"
            "<thinking>Second thought block</thinking>\n\n"
            "# Memory Item 2\n## Title B\n## Content b"
        )
        items = parse_memory_items(raw)
        assert len(items) == 2

    def test_nested_thinking_tags(self):
        """Nested thinking: iterative removal handles inner blocks,
        orphan cleanup removes leftover closing tags. Memory items survive."""
        raw = (
            "<thinking>Outer\n<thinking>Inner</thinking>\n"
            "Still outer</thinking>\n\n"
            "# Memory Item 1\n## Title Nested\n## Content survived"
        )
        items = parse_memory_items(raw)
        assert any("survived" in item for item in items)
        assert any("Nested" in item for item in items)

    def test_literal_thinking_in_content(self):
        """Literal '<thinking>' in memory content should NOT be stripped."""
        raw = "# Memory Item 1\n## Title Meta\n## Content When discussing <thinking> processes, note this"
        items = parse_memory_items(raw)
        assert len(items) == 1
        assert "<thinking>" in items[0]

    def test_real_world_llm_output(self):
        """Simulate a realistic LLM output with thinking + 2 memory items."""
        raw = (
            "<thinking>\n"
            "The user query is about preference analysis. The trajectory shows\n"
            "explicit statements about dark mode, vim editor, and late-night work.\n"
            "I should extract: 1) UI/theme preferences, 2) tool preferences.\n"
            "These are concrete and actionable.\n"
            "</thinking>\n\n"
            "# Memory Item 1\n"
            "## Title User Preference Extraction\n"
            "## Description Use when building a user profile from conversational data.\n"
            "## Content Extract and categorize explicit statements regarding visual/UI "
            "preferences, preferred software tools, and behavioral habits.\n\n"
            "# Memory Item 2\n"
            "## Title Working Style Detection\n"
            "## Description Use when scheduling tasks or setting expectations.\n"
            "## Content Note explicit mentions of preferred working hours or schedules "
            "to optimize task assignment timing."
        )
        items = parse_memory_items(raw)
        assert len(items) == 2
        assert "User Preference Extraction" in items[0]
        assert "Working Style Detection" in items[1]


# ---------------------------------------------------------------------------
# Unit tests for strip_thinking (isolated)
# ---------------------------------------------------------------------------


class TestStripThinking:
    """Tests for the strip_thinking helper."""

    def test_no_thinking(self):
        assert strip_thinking("Hello world") == "Hello world"

    def test_thinking_removed(self):
        result = strip_thinking("<thinking>inner thoughts</thinking>actual output")
        assert result == "actual output"

    def test_multiline_thinking(self):
        result = strip_thinking("<thinking>\nline1\nline2\n</thinking>\nresult")
        assert result == "result"


# ---------------------------------------------------------------------------
# Integration tests (require real LLM API — skipped if LLM_MODEL not set)
# ---------------------------------------------------------------------------


async def test_induce_success(llm_with_retry):
    items = await induce(
        llm_with_retry,
        query="t1",
        trajectory="go home\nthink...\naction...",
        status="success",
        domain="web",
    )
    assert len(items) >= 1
    assert items[0].status == "success"
    assert items[0].domain == "web"


async def test_induce_fail(llm_with_retry):
    items = await induce(
        llm_with_retry,
        query="t2",
        trajectory="go home\nthink...\naction...",
        status="fail",
        domain="web",
    )
    assert len(items) >= 1
    assert items[0].status == "fail"


async def test_induce_coding_domain(llm_with_retry):
    items = await induce(
        llm_with_retry,
        query="t3",
        trajectory="fix bug\nthink...",
        status="success",
        domain="coding",
    )
    assert len(items) >= 1
    assert items[0].domain == "coding"


async def test_induce_returns_few_items(llm_with_retry):
    """Regression test: induce should return 1-3 items, not dozens."""
    items = await induce(
        llm_with_retry,
        query="t4",
        trajectory="navigate\nthink...\ndo...",
        status="success",
        domain="web",
    )
    assert 1 <= len(items) <= 3, f"Expected 1-3 items, got {len(items)}"
