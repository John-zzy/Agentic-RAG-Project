from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_direct_answer_prompt_template(system_prompt: str | None = None) -> ChatPromptTemplate:
    """构建不依赖检索证据的普通对话回答模板。"""
    resolved_system_prompt = system_prompt or (
        "你是一名通用助手。请直接、自然地回答用户问题。"
    )
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"{resolved_system_prompt}\n"
                    "当前问题不要求使用知识库证据，不要编造引用编号。"
                ),
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )


def build_rag_answer_prompt_template(system_prompt: str | None = None) -> ChatPromptTemplate:
    """构建可按场景定制 system prompt 的 RAG 问答模板。"""
    resolved_system_prompt = system_prompt or (
        "你是一名通用知识助手。请优先依据检索上下文回答，无法确认时明确说明不确定。"
    )
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"{resolved_system_prompt}\n"
                    "你会收到已经编号的检索证据块。\n"
                    "只要答案使用了某条证据，就在对应句子末尾标注方括号编号，例如 [1]。\n"
                    "不要编造不存在的编号；如果证据不足，就明确说明不确定。"
                ),
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                (
                    "用户问题：\n{input}\n\n"
                    "检索上下文：\n{context}\n\n"
                    "请输出最终回答。\n"
                    "要求：\n"
                    "1. 回答尽量直接、易懂。\n"
                    "2. 使用到证据的句子后面要带 [n] 编号。\n"
                    "3. 如果没有足够证据，不要硬编。"
                ),
            ),
        ]
    )

