"""Tests for the model warm-up flow in rag_engine (warm_up()).

warm_up() pings the LLM at startup so the first real question doesn't pay the
cold-start load. The LLM is mocked here, so these run instantly with no Ollama:
they check that a ping is actually sent on success and that any failure (e.g.
Ollama not running) is swallowed into a False return rather than propagated.

Run with:  .venv/bin/python -m unittest test_warm_up
"""

import unittest
from unittest import mock

import rag_engine


class WarmUpTests(unittest.TestCase):
    def test_returns_true_and_pings_the_model_on_success(self):
        fake_llm = mock.Mock()
        fake_llm.invoke.return_value = "ok"
        with mock.patch.object(rag_engine, "get_llm", return_value=fake_llm):
            result = rag_engine.warm_up()

        self.assertTrue(result)
        fake_llm.invoke.assert_called_once()  # a ping was actually sent

    def test_returns_false_when_the_model_call_fails(self):
        # Simulates Ollama being unreachable — the error must not propagate.
        fake_llm = mock.Mock()
        fake_llm.invoke.side_effect = ConnectionError("ollama not running")
        with mock.patch.object(rag_engine, "get_llm", return_value=fake_llm):
            result = rag_engine.warm_up()

        self.assertFalse(result)

    def test_returns_false_when_the_client_cannot_be_built(self):
        # A failure constructing the LLM itself must also be handled gracefully.
        with mock.patch.object(rag_engine, "get_llm",
                               side_effect=RuntimeError("boom")):
            result = rag_engine.warm_up()

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
