from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseModel(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a final text response for a prompt."""

    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream tokens or text chunks as they are generated."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the model backend is reachable."""
