"""Tests for the query flow in rag_engine (query_rag / query_rag_stream).

The RAG chain is replaced with a fake so these exercise the real query
logic — the payload shape, result extraction, and streaming chunk handling —
without loading embeddings or hitting Ollama. All tests run in milliseconds.

Run with:  .venv/bin/python -m unittest test_query_flow
"""

import unittest
from unittest import mock

import rag_engine


class FakeChain:
    """Stands in for the RetrievalQA chain returned by get_rag_chain()."""

    def __init__(self, result=None, stream_chunks=None):
        self.result = result
        self.stream_chunks = stream_chunks or []
        self.invoked_with = None
        self.streamed_with = None

    def invoke(self, payload):
        self.invoked_with = payload
        return {"result": self.result}

    def stream(self, payload):
        self.streamed_with = payload
        for chunk in self.stream_chunks:
            yield chunk


class QueryRagTests(unittest.TestCase):
    def test_returns_result_text_and_passes_question_as_query(self):
        fake = FakeChain(result="The penalty is one stroke.")
        with mock.patch.object(rag_engine, "get_rag_chain", return_value=fake):
            answer = rag_engine.query_rag("What is the penalty?")

        self.assertEqual(answer, "The penalty is one stroke.")
        # The question must be passed under the "query" key the chain expects.
        self.assertEqual(fake.invoked_with, {"query": "What is the penalty?"})


class QueryRagStreamTests(unittest.TestCase):
    def test_yields_result_pieces_from_dict_chunks(self):
        chunks = [{"result": "Hello "}, {"result": "world"}]
        fake = FakeChain(stream_chunks=chunks)
        with mock.patch.object(rag_engine, "get_rag_chain", return_value=fake):
            out = list(rag_engine.query_rag_stream("hi"))

        self.assertEqual(out, ["Hello ", "world"])
        self.assertEqual("".join(out), "Hello world")
        self.assertEqual(fake.streamed_with, {"query": "hi"})

    def test_yields_plain_string_chunks_unchanged(self):
        fake = FakeChain(stream_chunks=["Hello ", "world"])
        with mock.patch.object(rag_engine, "get_rag_chain", return_value=fake):
            out = list(rag_engine.query_rag_stream("hi"))

        self.assertEqual(out, ["Hello ", "world"])

    def test_ignores_dict_chunks_without_a_result_key(self):
        # RetrievalQA can emit non-answer dicts (e.g. source documents) while
        # streaming; those must not leak into the answer text.
        chunks = [{"source_documents": ["doc"]}, {"result": "Answer"}]
        fake = FakeChain(stream_chunks=chunks)
        with mock.patch.object(rag_engine, "get_rag_chain", return_value=fake):
            out = list(rag_engine.query_rag_stream("hi"))

        self.assertEqual(out, ["Answer"])


if __name__ == "__main__":
    unittest.main()
