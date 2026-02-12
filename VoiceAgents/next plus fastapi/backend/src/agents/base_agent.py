from __future__ import annotations

from typing import AsyncIterator, List, Dict


class BaseAgent:
    async def stream_response(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        raise NotImplementedError
