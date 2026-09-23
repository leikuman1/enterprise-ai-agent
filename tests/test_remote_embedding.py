"""远程向量适配器的离线协议测试。"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
from openai import APIStatusError, APITimeoutError, OpenAI

from app.config import Settings
from app.infrastructure.cache.redis_cache import RedisCache
from app.infrastructure.embedding import RemoteEmbedding


def _item(index: int, dimension: int = 1024, value: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(index=index, embedding=[value] * dimension)


class RemoteEmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = RemoteEmbedding(
            api_base="https://example.test/v1",
            api_key="secret-key",
            model="test-embedding",
        )
        self.create = Mock()
        self.embedding._client = SimpleNamespace(embeddings=SimpleNamespace(create=self.create))

    def test_single_and_batch_restore_index_order(self) -> None:
        self.create.return_value = SimpleNamespace(data=[_item(0)])
        self.assertEqual(len(self.embedding.embed_query("query")), 1024)

        self.create.return_value = SimpleNamespace(data=[_item(1, value=2.0), _item(0)])
        vectors = self.embedding.embed_documents(["first", "second"])
        self.assertEqual([vector[0] for vector in vectors], [1.0, 2.0])
        self.create.assert_called_with(
            model="test-embedding",
            input=["first", "second"],
            dimensions=1024,
            encoding_format="float",
        )

    def test_openai_compatible_wire_format(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={
                "object": "list",
                "model": "test-embedding",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [2.0] * 1024},
                    {"object": "embedding", "index": 0, "embedding": [1.0] * 1024},
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            with OpenAI(
                api_key="secret-key",
                base_url="https://example.test/v1",
                http_client=http_client,
            ) as client:
                self.embedding._client = client
                vectors = self.embedding.embed_documents(["first", "second"])

        self.assertEqual([vector[0] for vector in vectors], [1.0, 2.0])
        self.assertEqual(str(requests[0].url), "https://example.test/v1/embeddings")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer secret-key")
        self.assertEqual(json.loads(requests[0].content)["dimensions"], 1024)

    def test_empty_input_and_invalid_responses(self) -> None:
        self.assertEqual(self.embedding.embed_documents([]), [])
        self.create.assert_not_called()
        with self.assertRaisesRegex(ValueError, "非空文本"):
            self.embedding.embed_query("")

        for items in ([], [_item(0, 10)], [_item(1)], [_item(0), _item(0)]):
            with self.subTest(items=items):
                self.create.return_value = SimpleNamespace(data=items)
                with self.assertRaises(RuntimeError):
                    self.embedding.embed_documents(["one"] if len(items) != 2 else ["one", "two"])

    def test_timeout_and_auth_error_hide_provider_message(self) -> None:
        request = httpx.Request("POST", "https://example.test/v1/embeddings")
        self.create.side_effect = APITimeoutError(request)
        with self.assertRaisesRegex(RuntimeError, "超时"):
            self.embedding.embed_query("private document")

        response = httpx.Response(401, request=request)
        self.create.side_effect = APIStatusError(
            "provider error secret-key private document", response=response, body=None
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            self.embedding.embed_query("private document")
        self.assertNotIn("secret-key", str(caught.exception))
        self.assertNotIn("private document", str(caught.exception))

    def test_config_is_checked_only_when_embedding_is_constructed(self) -> None:
        settings = Settings(
            _env_file=None,
            embedding_api_base="",
            embedding_api_key="",
            embedding_model="",
        )
        self.assertIsInstance(settings.embedding_model, str)
        with self.assertRaisesRegex(ValueError, "EMBEDDING_API_BASE"):
            RemoteEmbedding.from_settings(settings)
        with self.assertRaisesRegex(ValueError, "1024"):
            RemoteEmbedding(
                api_base="https://example.test/v1",
                api_key="key",
                model="model",
                dimension=1536,
            )

    def test_redis_cache_uses_embed_query_and_normalizes(self) -> None:
        embedder = SimpleNamespace(embed_query=Mock(return_value=[3.0, 4.0]))
        cache = object.__new__(RedisCache)
        cache._semantic_embedder = embedder
        cache._embed_lock = asyncio.Lock()

        async def check() -> None:
            vector = await cache._encode_query("  TeSt  ")
            np.testing.assert_allclose(vector, [0.6, 0.8])
            embedder.embed_query.assert_called_once_with("test")

        asyncio.run(check())
