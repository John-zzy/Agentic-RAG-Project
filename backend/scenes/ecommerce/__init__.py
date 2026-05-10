"""电商场景入口。"""

from backend.scenes.ecommerce.definition import (
    build_ecommerce_scene_definition,
    create_agentic_knowledge_retriever,
)

__all__ = [
    "build_ecommerce_scene_definition",
    "create_agentic_knowledge_retriever",
]
