"""远程重排接口的离线协议测试。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from app.config import Settings
from app.core.rag.reranker import Reranker
from app.models.schemas import RetrievalResult


class RemoteRerankerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.reranker = Reranker(
            api_base="https://example.test/v2",
            api_key="secret-key",
            model="test-rerank",
        )
        self.documents = [
            RetrievalResult(id="a", content="first", metadata={"source": "one"}),
            RetrievalResult(id="b", content="second"),
        ]

    async def _call_with_response(self, status: int, payload: object, top_k: int = 2):
        request_seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_seen.append(request)
            return httpx.Response(status, json=payload)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("app.core.rag.reranker.httpx.AsyncClient", return_value=client):
            result = await self.reranker.rerank("query", self.documents, top_k=top_k)
        return result, request_seen[0]

    async def test_ranking_request_and_response(self) -> None:
        result, request = await self._call_with_response(
            200,
            {"results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
            ]},
        )
        self.assertEqual([item.id for item in result], ["b", "a"])
        self.assertEqual(result[0].metadata["rerank_score"], 0.9)
        self.assertEqual(self.documents[0].metadata, {"source": "one"})
        self.assertEqual(str(request.url), "https://example.test/v2/rerank")
        self.assertEqual(request.headers["Authorization"], "Bearer secret-key")
        self.assertEqual(json.loads(request.content), {
            "model": "test-rerank",
            "query": "query",
            "documents": ["first", "second"],
            "top_n": 2,
        })

    async def test_invalid_response_and_http_error(self) -> None:
        for payload in (
            {},
            {"results": []},
            {"results": [{"index": 9, "relevance_score": 0.4}]},
            {"results": [{"index": 0}]},
            {"results": [{"index": 0, "relevance_score": "0.4"}]},
            {"results": [
                {"index": 0, "relevance_score": 0.4},
                {"index": 0, "relevance_score": 0.5},
            ]},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError):
                    result_count = len(payload.get("results", []))
                    await self._call_with_response(200, payload, top_k=max(1, result_count))

        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            await self._call_with_response(401, {"message": "secret-key"})
        self.assertNotIn("secret-key", str(caught.exception))

    async def test_timeout_and_empty_input(self) -> None:
        self.assertEqual(await self.reranker.rerank("query", [], top_k=5), [])

        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("private query", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
        with patch("app.core.rag.reranker.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "超时") as caught:
                await self.reranker.rerank("private query", self.documents)
        self.assertNotIn("private query", str(caught.exception))

    def test_config_is_checked_only_when_reranker_is_constructed(self) -> None:
        settings = Settings(
            _env_file=None,
            rerank_api_base="",
            rerank_api_key="",
            rerank_model="",
        )
        self.assertIsInstance(settings.rerank_model, str)
        with self.assertRaisesRegex(ValueError, "RERANK_API_BASE"):
            Reranker.from_settings(settings)
