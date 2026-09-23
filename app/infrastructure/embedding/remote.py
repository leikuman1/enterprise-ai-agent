"""OpenAI 兼容的远程 Embedding，提供 LangChain 同步接口。"""

from __future__ import annotations

import math
from typing import Any

from langchain_core.embeddings import Embeddings
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

from app.config import Settings, get_settings


class RemoteEmbedding(Embeddings):
    """使用独立配置调用远程 Embedding API。"""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        dimension: int = 1024,
        timeout_seconds: float = 30,
    ) -> None:
        if not api_base.strip() or not api_key.strip() or not model.strip():
            raise ValueError("EMBEDDING_API_BASE、EMBEDDING_API_KEY 和 EMBEDDING_MODEL 必须配置")
        if dimension != 1024:
            raise ValueError("EMBEDDING_DIMENSION 必须为 1024")
        if timeout_seconds <= 0:
            raise ValueError("EMBEDDING_TIMEOUT_SECONDS 必须大于 0")

        self._model = model
        self._dimension = dimension
        # SDK 仅对连接错误、超时、限流和服务端错误自动重试。
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/") + "/",
            timeout=timeout_seconds,
            max_retries=2,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RemoteEmbedding:
        settings = settings or get_settings()
        return cls(
            api_base=settings.embedding_api_base,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout_seconds=settings.embedding_timeout_seconds,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding 输入必须是非空文本")

        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimension,
                encoding_format="float",
            )
        except APIStatusError as exc:
            raise RuntimeError(f"Embedding API 请求失败（HTTP {exc.status_code}）") from None
        except APITimeoutError:
            raise RuntimeError("Embedding API 请求超时") from None
        except APIConnectionError:
            raise RuntimeError("Embedding API 连接失败") from None
        except APIError:
            raise RuntimeError("Embedding API 请求失败") from None

        ordered: list[list[float] | None] = [None] * len(texts)
        data: Any = getattr(response, "data", None)
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("Embedding API 返回的向量数量不匹配")
        for item in data:
            index = getattr(item, "index", None)
            vector = getattr(item, "embedding", None)
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(texts):
                raise RuntimeError("Embedding API 返回了无效索引")
            if ordered[index] is not None:
                raise RuntimeError("Embedding API 返回了重复索引")
            if not isinstance(vector, list) or len(vector) != self._dimension:
                raise RuntimeError(f"Embedding API 返回的向量维度不是 {self._dimension}")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in vector
            ):
                raise RuntimeError("Embedding API 返回了无效向量值")
            ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise RuntimeError("Embedding API 返回的向量索引不完整")
        return [vector for vector in ordered if vector is not None]
