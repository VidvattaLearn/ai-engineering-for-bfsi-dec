from __future__ import annotations

from typing import AsyncIterator, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from .base_agent import BaseAgent


class VoiceAgent(BaseAgent):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._client = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version="2024-06-01",
            temperature=0.3,
            streaming=True,
        )
        self._system_prompt = system_prompt

    def _to_langchain_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        result: List[BaseMessage] = []
        if self._system_prompt:
            result.append(SystemMessage(content=self._system_prompt))
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))
        return result

    async def stream_response(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        lc_messages = self._to_langchain_messages(messages)
        async for chunk in self._client.astream(lc_messages):
            content = getattr(chunk, "content", "")
            if isinstance(content, str):
                if content:
                    yield content
            elif content:
                yield str(content)
