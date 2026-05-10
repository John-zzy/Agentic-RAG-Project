from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_rag_answer_prompt_template(system_prompt: str | None = None) -> ChatPromptTemplate:
    """构建可按场景定制 system prompt 的 RAG 问答模板。"""
    resolved_system_prompt = system_prompt or (
        "你是一名通用知识助手。请优先依据检索上下文回答，无法确认时明确说明不确定。"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", resolved_system_prompt),
            (
                "human",
                "历史对话：\n{history}\n\n用户问题：\n{input}\n\n检索上下文：\n{context}\n\n请输出最终回答。",
            ),
        ]
    )
