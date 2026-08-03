"""Tests for the automatic PDF indexing / change-detection logic.

These exercise the *decision* logic in rag_engine — the folder fingerprint
and ensure_indexed()'s branching — while mocking the heavy embedding and
Chroma work, so the suite runs fast and needs no Ollama or model downloads.

Run with:  .venv/bin/python -m unittest test_auto_indexing
"""

import os
import tempfile
import unittest
from unittest import mock

import rag_engine


def write_pdf(folder, name, content=b"%PDF-1.4 fake"):
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(content)
    return path


class FakeCollection:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count


class FakeStore:
    """Stands in for a Chroma vector store with a controllable chunk count."""

    def __init__(self, count):
        self._collection = FakeCollection(count)


# ── Folder fingerprint ─────────────────────────────────────────
class ManifestFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.patch = mock.patch.object(rag_engine, "PDF_FOLDER", self.folder)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_lists_only_pdfs_with_size_and_mtime(self):
        write_pdf(self.folder, "a.pdf")
        write_pdf(self.folder, "b.pdf")
        open(os.path.join(self.folder, "notes.txt"), "w").close()  # ignored

        manifest = rag_engine._pdf_manifest()

        self.assertEqual(set(manifest), {"a.pdf", "b.pdf"})
        self.assertEqual(len(manifest["a.pdf"]), 2)  # [size, mtime]

    def test_detects_add_change_and_remove(self):
        write_pdf(self.folder, "a.pdf", b"%PDF short")
        base = rag_engine._pdf_manifest()

        # Adding a file changes the fingerprint...
        write_pdf(self.folder, "b.pdf")
        self.assertNotEqual(rag_engine._pdf_manifest(), base)

        # ...removing it again restores the original fingerprint.
        os.remove(os.path.join(self.folder, "b.pdf"))
        self.assertEqual(rag_engine._pdf_manifest(), base)

        # Editing a file (different size) changes the fingerprint.
        write_pdf(self.folder, "a.pdf", b"%PDF noticeably longer content")
        self.assertNotEqual(rag_engine._pdf_manifest(), base)


# ── Manifest persistence ───────────────────────────────────────
class ManifestPersistenceTests(unittest.TestCase):
    def test_save_then_load_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            db_dir = os.path.join(d, "chroma")
            path = os.path.join(db_dir, "pdf_manifest.json")
            with mock.patch.object(rag_engine, "CHROMA_DB_PATH", db_dir), \
                 mock.patch.object(rag_engine, "MANIFEST_PATH", path):
                data = {"a.pdf": [10, 123]}
                rag_engine._save_manifest(data)
                self.assertEqual(rag_engine._load_manifest(), data)

    def test_missing_manifest_loads_as_none(self):
        with mock.patch.object(rag_engine, "MANIFEST_PATH", "/no/such/file.json"):
            self.assertIsNone(rag_engine._load_manifest())


# ── ensure_indexed decision logic ──────────────────────────────
class EnsureIndexedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.patches = [
            mock.patch.object(rag_engine, "PDF_FOLDER", self.folder),
            mock.patch.object(rag_engine, "CHROMA_DB_PATH", self.folder),
            mock.patch.object(rag_engine, "MANIFEST_PATH",
                              os.path.join(self.folder, "manifest.json")),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_empty_store_triggers_indexing(self):
        write_pdf(self.folder, "a.pdf")
        with mock.patch.object(rag_engine, "load_vector_store", return_value=FakeStore(0)), \
             mock.patch.object(rag_engine, "index_pdfs", return_value="✅ indexed") as idx:
            result = rag_engine.ensure_indexed()
        idx.assert_called_once()
        self.assertEqual(result, "✅ indexed")

    def test_unchanged_folder_is_a_noop(self):
        write_pdf(self.folder, "a.pdf")
        rag_engine._save_manifest(rag_engine._pdf_manifest())
        with mock.patch.object(rag_engine, "load_vector_store", return_value=FakeStore(5)), \
             mock.patch.object(rag_engine, "index_pdfs") as idx:
            result = rag_engine.ensure_indexed()
        idx.assert_not_called()
        self.assertIn("existing index", result)

    def test_changed_folder_triggers_reindex(self):
        write_pdf(self.folder, "a.pdf")
        rag_engine._save_manifest(rag_engine._pdf_manifest())
        write_pdf(self.folder, "b.pdf")  # a new file appears after indexing
        with mock.patch.object(rag_engine, "load_vector_store", return_value=FakeStore(5)), \
             mock.patch.object(rag_engine, "index_pdfs", return_value="✅ reindexed") as idx:
            result = rag_engine.ensure_indexed()
        idx.assert_called_once()
        self.assertEqual(result, "✅ reindexed")

    def test_no_pdfs_reports_without_indexing(self):
        with mock.patch.object(rag_engine, "load_vector_store", return_value=FakeStore(0)), \
             mock.patch.object(rag_engine, "index_pdfs") as idx:
            result = rag_engine.ensure_indexed()
        idx.assert_not_called()
        self.assertIn("No PDFs", result)


# ── Rebuild is idempotent (duplicate-append regression) ────────
class IndexRebuildTests(unittest.TestCase):
    def test_resets_collection_before_rebuilding_and_saves_manifest(self):
        order = []
        fake_docs = [mock.Mock(metadata={"source": "a.pdf"})]
        with mock.patch.object(rag_engine, "load_pdfs", return_value=fake_docs), \
             mock.patch.object(rag_engine, "split_documents", return_value=["chunk"]), \
             mock.patch.object(rag_engine, "_pdf_manifest", return_value={"a.pdf": [1, 2]}), \
             mock.patch.object(rag_engine, "_reset_vector_store",
                               side_effect=lambda: order.append("reset")), \
             mock.patch.object(rag_engine, "create_vector_store",
                               side_effect=lambda chunks: order.append("create")), \
             mock.patch.object(rag_engine, "_save_manifest") as save:
            result = rag_engine.index_pdfs()

        # The collection must be reset BEFORE it is rebuilt, otherwise
        # Chroma.from_documents would append and duplicate every chunk.
        self.assertEqual(order, ["reset", "create"])
        save.assert_called_once_with({"a.pdf": [1, 2]})
        self.assertIn("✅", result)


if __name__ == "__main__":
    unittest.main()
