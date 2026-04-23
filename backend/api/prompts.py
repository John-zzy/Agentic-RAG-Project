from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_rag_answer_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名电商客服助手。请优先依据检索上下文回答，无法确认时明确说明不确定。"),
            (
                "human",
                "历史对话：\n{history}\n\n用户问题：\n{input}\n\n检索上下文：\n{context}\n\n请输出最终回答。",
            ),
        ]
    )

