from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessingChunkConfig:
    """数据预处理默认切块配置。"""

    chunk_size: int = 500
    chunk_overlap: int = 80


DEFAULT_PROCESSING_CHUNK_CONFIG = ProcessingChunkConfig()
