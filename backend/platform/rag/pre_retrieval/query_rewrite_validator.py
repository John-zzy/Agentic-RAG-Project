from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re

import jieba


jieba.setLogLevel(logging.WARNING)


DEFAULT_PRESERVED_TOKEN_PATTERNS: tuple[str, ...] = (
    r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b",
    r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{3,}\b",
    r"\b\d+(?:\.\d+)+\b",
    r"\b\d{2,}\b",
    r"\b[A-Z]{2,}\b",
    r"\b[A-Z][A-Za-z0-9_+-]*(?=\s+\d+(?:\.\d+)+\b)",
    r"\b(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])[A-Za-z][A-Za-z0-9]{2,}\b",
    r"`([^`]+)`",
)


DEFAULT_CHINESE_QUERY_STOPWORDS: frozenset[str] = frozenset(
    {
        "当前",
        "哪些",
        "什么",
        "怎么",
        "如何",
        "是否",
        "请问",
        "帮我",
        "一下",
        "这个",
        "那个",
        "相关",
        "文档",
        "说明",
        "查询",
        "多久",
        "还有",
        "可以",
        "需要",
        "进行",
        "根据",
        "帮忙",
    }
)


DEFAULT_UNSUPPORTED_GENERIC_EXPANSION_TERMS: tuple[str, ...] = (
    "数据模型",
    "表结构",
    "常见问题",
    "faq",
    "frequently asked questions",
)


@dataclass(frozen=True)
class PreservedTokenExtractionConfig:
    """配置关键 token 提取策略，避免把默认规则写死在流程逻辑里。"""

    regex_patterns: tuple[str, ...] = DEFAULT_PRESERVED_TOKEN_PATTERNS
    chinese_stopwords: frozenset[str] = DEFAULT_CHINESE_QUERY_STOPWORDS
    min_chinese_token_length: int = 2
    use_jieba: bool = True
    jieba_hmm: bool = False


@dataclass(frozen=True)
class QueryRewriteValidationConfig:
    """配置 query rewrite 校验策略。"""

    token_extraction: PreservedTokenExtractionConfig = field(
        default_factory=PreservedTokenExtractionConfig
    )
    unsupported_generic_terms: tuple[str, ...] = DEFAULT_UNSUPPORTED_GENERIC_EXPANSION_TERMS


@dataclass(frozen=True)
class PreservedTokenExtractor:
    """提取 rewrite 必须保留的结构化 token 和中文关键词。"""

    config: PreservedTokenExtractionConfig = field(default_factory=PreservedTokenExtractionConfig)

    def extract(self, *queries: str) -> tuple[str, ...]:
        """从原问题和当前 query 中提取稳定 token，保持首次出现顺序。"""
        tokens: list[str] = []
        seen: set[str] = set()
        for query in queries:
            self._append_regex_tokens(query, tokens=tokens, seen=seen)
            if self.config.use_jieba:
                self._append_jieba_tokens(query, tokens=tokens, seen=seen)
        return tuple(tokens)

    def _append_regex_tokens(self, query: str, *, tokens: list[str], seen: set[str]) -> None:
        """用规则捕获 jieba 不擅长的错误码、版本号、缩写和代码型 token。"""
        for pattern in self.config.regex_patterns:
            for match in re.finditer(pattern, query):
                raw_token = match.group(1) if match.lastindex else match.group(0)
                self._append_token(raw_token, tokens=tokens, seen=seen)

    def _append_jieba_tokens(self, query: str, *, tokens: list[str], seen: set[str]) -> None:
        """用 jieba 提取中文关键词，过滤常见问法词，避免过度保留。"""
        for raw_token in jieba.cut_for_search(query, HMM=self.config.jieba_hmm):
            token = raw_token.strip()
            if not self._is_preservable_chinese_token(token):
                continue
            self._append_token(token, tokens=tokens, seen=seen)

    def _append_token(self, raw_token: str, *, tokens: list[str], seen: set[str]) -> None:
        token = normalize_preserved_token(raw_token)
        token_key = token.lower()
        if not token or token_key in seen:
            return
        tokens.append(token)
        seen.add(token_key)

    def _is_preservable_chinese_token(self, token: str) -> bool:
        if len(token) < self.config.min_chinese_token_length:
            return False
        if token in self.config.chinese_stopwords:
            return False
        return bool(re.search(r"[\u4e00-\u9fff]", token))


@dataclass(frozen=True)
class QueryRewriteValidator:
    """集中校验 LLM 改写后的 query 是否可用于下一轮检索。"""

    config: QueryRewriteValidationConfig = field(default_factory=QueryRewriteValidationConfig)
    token_extractor: PreservedTokenExtractor | None = None

    def __post_init__(self) -> None:
        if self.token_extractor is not None:
            return
        object.__setattr__(
            self,
            "token_extractor",
            PreservedTokenExtractor(config=self.config.token_extraction),
        )

    def extract_preserved_tokens(self, *queries: str) -> tuple[str, ...]:
        if self.token_extractor is None:
            return ()
        return self.token_extractor.extract(*queries)

    def resolve_unsafe_reason(
        self,
        *,
        original_query: str,
        rewritten_query: str,
        preserved_tokens: tuple[str, ...],
    ) -> str | None:
        missing_token = self._first_missing_preserved_token(rewritten_query, preserved_tokens)
        if missing_token is not None:
            return f"missing_preserved_token:{missing_token}"

        added_term = self._first_added_generic_expansion(original_query, rewritten_query)
        if added_term is not None:
            return f"unsupported_generic_expansion:{added_term}"
        return None

    def _first_missing_preserved_token(
        self,
        rewritten_query: str,
        preserved_tokens: tuple[str, ...],
    ) -> str | None:
        normalized_rewrite = normalize_for_token_check(rewritten_query)
        for token in preserved_tokens:
            if normalize_for_token_check(token) not in normalized_rewrite:
                return token
        return None

    def _first_added_generic_expansion(
        self,
        original_query: str,
        rewritten_query: str,
    ) -> str | None:
        normalized_original = normalize_for_token_check(original_query)
        normalized_rewrite = normalize_for_token_check(rewritten_query)
        for term in self.config.unsupported_generic_terms:
            normalized_term = normalize_for_token_check(term)
            if normalized_term in normalized_rewrite and normalized_term not in normalized_original:
                return term
        return None


def normalize_preserved_token(token: str) -> str:
    """归一化关键 token，避免反引号或多余空白影响后续校验。"""
    return re.sub(r"\s+", " ", token.strip("`").strip()).strip()


def normalize_for_token_check(value: str) -> str:
    """统一安全校验的大小写与空白规则。"""
    return re.sub(r"\s+", " ", value).strip().lower()
