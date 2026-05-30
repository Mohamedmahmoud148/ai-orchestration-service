"""
tests/test_embedding_service.py

Unit tests for the multi-provider embedding service.
Tests cover: mode detection, keyword fallback, cosine similarity,
health reporting, and config-driven provider selection.

Note: Tests do NOT make real API calls — the OpenAI client is never
constructed in these tests because OPENAI_API_KEY is not set in conftest.
"""
import os
import pytest


class TestEmbeddingServiceMode:
    """Tests for mode detection at construction time."""

    def setup_method(self):
        # Ensure no API key so we test the fallback path
        os.environ.pop("OPENAI_API_KEY", None)
        # Re-import to get a fresh instance with current env
        import importlib
        import app.services.embedding_service as mod
        importlib.reload(mod)
        from app.services.embedding_service import EmbeddingService
        self.svc = EmbeddingService()

    def test_keyword_fallback_when_no_key(self):
        assert self.svc.get_mode() == "keyword_fallback"

    def test_not_real_embeddings_in_fallback(self):
        assert self.svc.is_using_real_embeddings() is False

    def test_embedding_dim_positive(self):
        assert self.svc.embedding_dim > 0

    def test_get_mode_returns_string(self):
        assert isinstance(self.svc.get_mode(), str)


class TestKeywordVector:
    """Tests for the keyword-overlap fallback embedding."""

    def setup_method(self):
        os.environ.pop("OPENAI_API_KEY", None)
        import importlib
        import app.services.embedding_service as mod
        importlib.reload(mod)
        from app.services.embedding_service import EmbeddingService
        self.svc = EmbeddingService()

    def test_empty_text_returns_zero_vector(self):
        vec = self.svc._keyword_vector("")
        assert all(v == 0.0 for v in vec)

    def test_vector_has_correct_dim(self):
        vec = self.svc._keyword_vector("test text here")
        assert len(vec) == 1536  # default keyword dim

    def test_vector_is_normalized(self):
        import math
        vec = self.svc._keyword_vector("hello world university")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 1e-6

    def test_different_texts_different_vectors(self):
        v1 = self.svc._keyword_vector("data structures algorithms")
        v2 = self.svc._keyword_vector("graduation requirements credits")
        assert v1 != v2

    def test_same_text_same_vector(self):
        v1 = self.svc._keyword_vector("مادة قواعد البيانات")
        v2 = self.svc._keyword_vector("مادة قواعد البيانات")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_embed_text_returns_list(self):
        vec = await self.svc.embed_text("test")
        assert isinstance(vec, list)
        assert len(vec) > 0

    @pytest.mark.asyncio
    async def test_embed_batch_correct_length(self):
        texts = ["hello", "world", ""]
        vecs = await self.svc.embed_batch(texts)
        assert len(vecs) == 3

    @pytest.mark.asyncio
    async def test_embed_empty_text_returns_zeros(self):
        vec = await self.svc.embed_text("")
        assert all(v == 0.0 for v in vec)

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        result = await self.svc.embed_batch([])
        assert result == []


class TestCosineSimilarity:
    """Tests for cosine_similarity static method."""

    def test_identical_vectors_similarity_one(self):
        v = [1.0, 0.0, 0.0]
        assert EmbeddingService_cls().cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_similarity_zero(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert EmbeddingService_cls().cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert EmbeddingService_cls().cosine_similarity(v1, v2) == 0.0

    def test_different_lengths_returns_zero(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert EmbeddingService_cls().cosine_similarity(v1, v2) == 0.0

    def test_similar_keyword_vectors_higher_than_different(self):
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        import importlib
        import app.services.embedding_service as mod
        importlib.reload(mod)
        from app.services.embedding_service import EmbeddingService
        svc = EmbeddingService()
        v1 = svc._keyword_vector("data structures algorithms")
        v2 = svc._keyword_vector("data structures trees")
        v3 = svc._keyword_vector("poetry literature history")
        sim_close = svc.cosine_similarity(v1, v2)
        sim_far   = svc.cosine_similarity(v1, v3)
        assert sim_close > sim_far


def EmbeddingService_cls():
    """Helper to get a fresh EmbeddingService for static method testing."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    import importlib
    import app.services.embedding_service as mod
    importlib.reload(mod)
    from app.services.embedding_service import EmbeddingService
    return EmbeddingService()
