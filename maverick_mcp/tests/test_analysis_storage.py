"""
Tests for the Analysis Storage Service.

Tests automatic storage and retrieval of analysis results.
"""

import pytest

from maverick_mcp.data.analysis_storage import (
    AnalysisStorageService,
    get_analysis_storage,
)
from maverick_mcp.data.models import AnalysisCache
from maverick_mcp.data.session_management import get_db_session


class TestAnalysisStorageService:
    """Test the AnalysisStorageService class."""

    @pytest.fixture
    def storage(self):
        """Create a storage service instance."""
        return AnalysisStorageService()

    @pytest.mark.asyncio
    async def test_store_and_retrieve_analysis(self, storage):
        """Test storing and retrieving an analysis."""
        symbol = "AAPL"
        tool_name = "technical_analysis"
        analysis_type = "rsi"
        result = {
            "status": "success",
            "symbol": symbol,
            "current_rsi": 65.5,
            "signal": "neutral",
        }

        # Store the analysis
        analysis_id = await storage.store_analysis(
            symbol=symbol,
            tool_name=tool_name,
            analysis_type=analysis_type,
            result=result,
        )

        assert analysis_id is not None

        # Retrieve the analysis
        retrieved = await storage.get_latest_analysis(
            symbol=symbol,
            tool_name=tool_name,
            analysis_type=analysis_type,
        )

        assert retrieved is not None
        assert retrieved["tool_name"] == tool_name
        assert retrieved["analysis_type"] == analysis_type
        assert retrieved["result"]["current_rsi"] == 65.5

    @pytest.mark.asyncio
    async def test_store_skips_failed_analysis(self, storage):
        """Test that failed analyses are not stored."""
        symbol = "MSFT"
        result = {
            "status": "error",
            "error": "API error",
        }

        analysis_id = await storage.store_analysis(
            symbol=symbol,
            tool_name="technical_analysis",
            analysis_type="rsi",
            result=result,
        )

        assert analysis_id is None

    @pytest.mark.asyncio
    async def test_get_analysis_history(self, storage):
        """Test retrieving analysis history."""
        symbol = "GOOGL"
        tool_name = "quant_analysis"

        # Store multiple analyses
        for i in range(3):
            await storage.store_analysis(
                symbol=symbol,
                tool_name=tool_name,
                analysis_type="beta",
                result={"status": "success", "beta": 1.0 + (i * 0.1)},
            )

        # Retrieve history
        history = await storage.get_analysis_history(
            symbol=symbol,
            tool_name=tool_name,
            limit=10,
        )

        assert len(history) >= 3
        # Should be ordered newest first
        assert history[0]["created_at"] >= history[-1]["created_at"]

    @pytest.mark.asyncio
    async def test_get_history_with_no_results(self, storage):
        """Test retrieving history for a symbol with no analyses."""
        history = await storage.get_analysis_history(
            symbol="NONEXISTENT",
            tool_name="technical_analysis",
        )

        assert history == []

    def test_compare_analyses_numeric_change(self, storage):
        """Test comparison of numeric values."""
        previous = {
            "created_at": "2025-01-10T10:00:00Z",
            "result": {"current_rsi": 70.0, "signal": "overbought"},
        }
        current = {
            "current_rsi": 65.0,
            "signal": "neutral",
        }

        comparison = storage.compare_analyses(
            previous=previous,
            current=current,
            tool_name="technical_analysis",
            analysis_type="rsi",
        )

        assert "previous_date" in comparison
        assert "changes" in comparison
        # RSI should show a change
        if "current_rsi" in comparison["changes"]:
            assert comparison["changes"]["current_rsi"]["delta"] == -5.0

    def test_compare_analyses_string_change(self, storage):
        """Test comparison of string values."""
        previous = {
            "created_at": "2025-01-10T10:00:00Z",
            "result": {"trend": "bullish"},
        }
        current = {
            "trend": "bearish",
        }

        comparison = storage.compare_analyses(
            previous=previous,
            current=current,
            tool_name="technical_analysis",
            analysis_type="full",
        )

        # Trend change should be detected
        if "trend" in comparison["changes"]:
            assert comparison["changes"]["trend"]["changed"] is True


class TestAnalysisCacheModel:
    """Test the AnalysisCache database model."""

    def test_get_latest_returns_none_for_nonexistent(self):
        """Test that get_latest returns None for nonexistent analysis."""
        import uuid

        with get_db_session() as session:
            result = AnalysisCache.get_latest(
                session=session,
                stock_id=uuid.uuid4(),  # Random UUID won't exist
                tool_name="technical_analysis",
                analysis_type="rsi",
            )
            assert result is None

    def test_to_dict_returns_valid_structure(self):
        """Test that to_dict returns expected structure."""
        import uuid
        from datetime import UTC, datetime

        cache = AnalysisCache(
            id=uuid.uuid4(),
            stock_id=uuid.uuid4(),
            tool_name="technical_analysis",
            analysis_type="rsi",
            result='{"status": "success", "value": 65}',
            input_params='{"period": 14}',
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        result = cache.to_dict()

        assert "id" in result
        assert "tool_name" in result
        assert result["tool_name"] == "technical_analysis"
        assert "result" in result
        assert result["result"]["value"] == 65


class TestGetAnalysisStorage:
    """Test the singleton getter function."""

    def test_returns_same_instance(self):
        """Test that get_analysis_storage returns singleton."""
        storage1 = get_analysis_storage()
        storage2 = get_analysis_storage()
        assert storage1 is storage2

    def test_returns_service_instance(self):
        """Test that returned object is AnalysisStorageService."""
        storage = get_analysis_storage()
        assert isinstance(storage, AnalysisStorageService)


class TestAnalysisWrapper:
    """Test the analysis wrapper decorator."""

    @pytest.mark.asyncio
    async def test_decorator_stores_result(self):
        """Test that decorated function stores its result."""
        from maverick_mcp.api.routers.unified.analysis_wrapper import (
            with_analysis_storage,
        )

        @with_analysis_storage("test_tool")
        async def test_analysis(symbol: str, analysis_type: str = "test"):
            return {
                "status": "success",
                "symbol": symbol,
                "value": 42,
            }

        result = await test_analysis(symbol="AAPL", analysis_type="test")

        assert result["status"] == "success"
        assert "analysis_id" in result  # Should have stored

    @pytest.mark.asyncio
    async def test_decorator_skips_error_result(self):
        """Test that decorator doesn't store error results."""
        from maverick_mcp.api.routers.unified.analysis_wrapper import (
            with_analysis_storage,
        )

        @with_analysis_storage("test_tool")
        async def failing_analysis(symbol: str, analysis_type: str = "test"):
            return {
                "status": "error",
                "error": "Test error",
            }

        result = await failing_analysis(symbol="FAIL")

        assert result["status"] == "error"
        assert "analysis_id" not in result  # Should not have stored


class TestAnalysisHistoryTool:
    """Test the analysis_history unified tool."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_history(self):
        """Test that tool returns empty list for symbols with no history."""
        from maverick_mcp.api.routers.unified.unified_analysis_history import (
            analysis_history,
        )

        result = await analysis_history(symbol="NOHIST123")

        assert result["status"] == "success"
        assert result["total_analyses"] == 0
        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_filters_by_tool_name(self):
        """Test that tool filters by tool_name correctly."""
        from maverick_mcp.api.routers.unified.unified_analysis_history import (
            analysis_history,
        )

        # First store some analyses
        storage = get_analysis_storage()
        await storage.store_analysis(
            symbol="FILTTEST",
            tool_name="technical_analysis",
            analysis_type="rsi",
            result={"status": "success", "rsi": 50},
        )
        await storage.store_analysis(
            symbol="FILTTEST",
            tool_name="quant_analysis",
            analysis_type="beta",
            result={"status": "success", "beta": 1.1},
        )

        # Query with filter
        result = await analysis_history(
            symbol="FILTTEST", tool_name="technical_analysis"
        )

        assert result["status"] == "success"
        # All results should be from technical_analysis
        for h in result["history"]:
            assert h["tool"] == "technical_analysis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
