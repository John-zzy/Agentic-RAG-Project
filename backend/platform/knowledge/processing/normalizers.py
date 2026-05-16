from __future__ import annotations

import re
from html import unescape

_SCRIPT_OR_STYLE_PATTERN = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_URL_ONLY_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:<)?https?://\S+(?:>)?(?:\s+(?:<)?https?://\S+(?:>)?)*\s*$",
    re.IGNORECASE,
)
_MARKDOWN_FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|$)", re.DOTALL)
_MARKDOWN_TOC_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+\.)\s*\[[^]]+\]\(#.+\)\s*$")
_MARKDOWN_BOILERPLATE_LINES = {"[toc]", "toc", "table of contents", "目录"}


def trim_whitespace(content: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def strip_html_tags(content: str) -> str:
    without_scripts = _SCRIPT_OR_STYLE_PATTERN.sub("", content)
    without_tags = _HTML_TAG_PATTERN.sub("", without_scripts)
    return unescape(without_tags)


def remove_markdown_boilerplate(content: str) -> str:
    without_frontmatter = _MARKDOWN_FRONTMATTER_PATTERN.sub("", content)
    kept_lines: list[str] = []
    for line in without_frontmatter.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        normalized_heading = lowered.lstrip("#").strip()
        if lowered in _MARKDOWN_BOILERPLATE_LINES or normalized_heading in _MARKDOWN_BOILERPLATE_LINES:
            continue
        if _MARKDOWN_TOC_ITEM_PATTERN.match(stripped):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def remove_url_lines(content: str) -> str:
    kept_lines = [line for line in content.splitlines() if not _URL_ONLY_LINE_PATTERN.match(line)]
    return "\n".join(kept_lines)
