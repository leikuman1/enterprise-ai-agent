# -*- coding: utf-8 -*-
"""通过 Cohere 风格的远程 API 对检索结果重排序。"""

from __future__ import annotations

import math
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.models.schemas import RetrievalResult


class Reranker:
    """使用独立配置调用远程 Rerank API。"""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30,
    ) -> None:
        if not api_base.strip() or not api_key.strip() or not model.strip():
            raise ValueError("RERANK_API_BASE、RERANK_API_KEY 和 RERANK_MODEL 必须配置")
        if timeout_seconds <= 0:
            raise ValueError("RERANK_TIMEOUT_SECONDS 必须大于 0")

        self._url = api_base.rstrip("/") + "/rerank"
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Reranker:
        settings = settings or get_settings()
        return cls(
            api_base=settings.rerank_api_base,
            api_key=settings.rerank_api_key,
            model=settings.rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
        )

    async def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not documents or top_k <= 0:
            return []

        body = {
            "model": self._model,
            "query": query,
            "documents": [document.content for document in documents],
            "top_n": min(top_k, len(documents)),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                )
                response.raise_for_status()
                payload: Any = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Rerank API 请求失败（HTTP {exc.response.status_code}）") from None
        except httpx.TimeoutException:
            raise RuntimeError("Rerank API 请求超时") from None
        except httpx.RequestError:
            raise RuntimeError("Rerank API 连接失败") from None
        except ValueError:
            raise RuntimeError("Rerank API 返回了无效 JSON") from None

        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RuntimeError("Rerank API 响应缺少 results 列表")
        results = payload["results"]
        if len(results) != body["top_n"]:
            raise RuntimeError("Rerank API 返回结果数与 top_n 不匹配")

        ranked: list[RetrievalResult] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("Rerank API 返回了无效结果项")
            index = item.get("index")
            score = item.get("relevance_score")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(documents)
            ):
                raise RuntimeError("Rerank API 返回了无效索引")
            if index in seen:
                raise RuntimeError("Rerank API 返回了重复索引")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                raise RuntimeError("Rerank API 返回了无效相关性分数")
            seen.add(index)
            doc = documents[index].model_copy(deep=True)
            doc.score = float(score)
            doc.metadata = {**doc.metadata, "rerank_score": float(score)}
            ranked.append(doc)

        ranked.sort(key=lambda document: document.score, reverse=True)
        return ranked
