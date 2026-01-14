"""
Analysis Storage Service.

Provides automatic storage and retrieval of analysis results
to reduce redundant API calls and provide historical context.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from maverick_mcp.data.models import AnalysisCache, Stock
from maverick_mcp.data.session_management import get_db_session

logger = logging.getLogger(__name__)


class AnalysisStorageService:
    """
    Manages storage and retrieval of analysis results.

    This service provides:
    - Automatic storage of analysis results
    - Retrieval of latest analysis for a symbol
    - Analysis history queries
    - Comparison between past and current analyses
    """

    def __init__(self):
        """Initialize the storage service."""
        self._comparison_keys = {
            "technical_analysis": {
                "rsi": ["current_rsi", "signal"],
                "macd": ["macd_value", "signal_value", "histogram", "trend"],
                "full": ["trend", "outlook", "current_price"],
                "bollinger": ["position", "bandwidth"],
            },
            "quant_analysis": {
                "beta": ["beta", "alpha", "correlation"],
                "momentum": ["trend", "momentum_score"],
                "volatility": ["current_regime", "annualized_volatility"],
                "seasonality": ["best_months", "worst_months", "current_pattern"],
            },
            "risk_analysis": {
                "var": ["var_95", "var_99"],
                "comprehensive": ["risk_score", "max_drawdown"],
            },
            "alternative_data": {
                "insider": ["sentiment_score", "net_activity"],
            },
        }

    async def store_analysis(
        self,
        symbol: str,
        tool_name: str,
        analysis_type: str,
        result: dict[str, Any],
        input_params: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Store analysis result in the database.

        Args:
            symbol: Stock ticker symbol
            tool_name: Name of the analysis tool (e.g., 'technical_analysis')
            analysis_type: Type of analysis (e.g., 'rsi', 'full')
            result: The analysis result dictionary
            input_params: Input parameters used for the analysis

        Returns:
            The analysis ID if stored successfully, None otherwise
        """
        # Don't store failed analyses
        if result.get("status") == "error":
            logger.debug(f"Skipping storage of failed analysis for {symbol}")
            return None

        try:
            with get_db_session() as session:
                # Get or create stock
                stock = Stock.get_or_create(session, symbol.upper())

                # Compute hash for deduplication
                result_json = json.dumps(result, sort_keys=True, default=str)
                result_hash = hashlib.sha256(result_json.encode()).hexdigest()

                # Create cache entry
                cache_entry = AnalysisCache(
                    stock_id=stock.stock_id,
                    tool_name=tool_name,
                    analysis_type=analysis_type,
                    result=result_json,
                    result_hash=result_hash,
                    input_params=json.dumps(input_params, default=str)
                    if input_params
                    else None,
                    data_date_range=self._extract_date_range(result),
                )

                session.add(cache_entry)
                session.commit()

                logger.info(
                    f"Stored analysis: {symbol} / {tool_name} / {analysis_type}"
                )
                return str(cache_entry.id)

        except Exception as e:
            logger.error(f"Error storing analysis for {symbol}: {e}")
            return None

    async def get_latest_analysis(
        self,
        symbol: str,
        tool_name: str,
        analysis_type: str,
        max_age_hours: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve the most recent analysis for a symbol.

        Args:
            symbol: Stock ticker symbol
            tool_name: Name of the analysis tool
            analysis_type: Type of analysis
            max_age_hours: Maximum age in hours (None = no limit)

        Returns:
            The cached analysis result or None if not found
        """
        try:
            with get_db_session() as session:
                # Find stock
                stock = (
                    session.query(Stock)
                    .filter(Stock.ticker_symbol == symbol.upper())
                    .first()
                )
                if not stock:
                    return None

                # Get latest analysis
                cached = AnalysisCache.get_latest(
                    session, stock.stock_id, tool_name, analysis_type
                )

                if not cached:
                    return None

                # Check age if max_age_hours is specified
                if max_age_hours is not None:
                    age = datetime.now(UTC) - cached.created_at.replace(tzinfo=UTC)
                    if age > timedelta(hours=max_age_hours):
                        logger.debug(
                            f"Cached analysis for {symbol} is too old "
                            f"({age.total_seconds() / 3600:.1f}h > {max_age_hours}h)"
                        )
                        return None

                return cached.to_dict()

        except Exception as e:
            logger.error(f"Error retrieving analysis for {symbol}: {e}")
            return None

    async def get_analysis_history(
        self,
        symbol: str,
        tool_name: str | None = None,
        days: int = 30,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get analysis history for a symbol.

        Args:
            symbol: Stock ticker symbol
            tool_name: Filter by specific tool (optional)
            days: Maximum age in days
            limit: Maximum number of results

        Returns:
            List of analysis results ordered by date (newest first)
        """
        try:
            with get_db_session() as session:
                # Find stock
                stock = (
                    session.query(Stock)
                    .filter(Stock.ticker_symbol == symbol.upper())
                    .first()
                )
                if not stock:
                    return []

                # Build query
                query = session.query(AnalysisCache).filter(
                    AnalysisCache.stock_id == stock.stock_id,
                    AnalysisCache.created_at
                    >= datetime.now(UTC) - timedelta(days=days),
                )

                if tool_name:
                    query = query.filter(AnalysisCache.tool_name == tool_name)

                results = (
                    query.order_by(AnalysisCache.created_at.desc()).limit(limit).all()
                )

                return [r.to_dict() for r in results]

        except Exception as e:
            logger.error(f"Error getting history for {symbol}: {e}")
            return []

    def compare_analyses(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
        tool_name: str,
        analysis_type: str,
    ) -> dict[str, Any]:
        """
        Compare previous and current analysis to highlight changes.

        Args:
            previous: Previous analysis result
            current: Current analysis result
            tool_name: Name of the tool
            analysis_type: Type of analysis

        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "previous_date": previous.get("created_at"),
            "days_ago": self._calculate_days_ago(previous.get("created_at")),
            "changes": {},
        }

        # Get keys to compare for this tool/type
        prev_result = previous.get("result", {})
        curr_result = current

        keys_to_compare = self._comparison_keys.get(tool_name, {}).get(
            analysis_type, []
        )

        for key in keys_to_compare:
            prev_val = self._extract_nested_value(prev_result, key)
            curr_val = self._extract_nested_value(curr_result, key)

            if prev_val is not None and curr_val is not None:
                change = self._compute_change(prev_val, curr_val, key)
                if change:
                    comparison["changes"][key] = change

        # Add summary
        comparison["summary"] = self._generate_comparison_summary(
            comparison["changes"], tool_name, analysis_type
        )

        return comparison

    def _extract_date_range(self, result: dict[str, Any]) -> str | None:
        """Extract date range from analysis result."""
        # Try common date fields
        start_date = result.get("start_date") or result.get("from_date")
        end_date = result.get("end_date") or result.get("to_date")

        if start_date and end_date:
            return f"{start_date} to {end_date}"

        return None

    def _calculate_days_ago(self, date_str: str | None) -> int | None:
        """Calculate days since a date string."""
        if not date_str:
            return None
        try:
            if isinstance(date_str, str):
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                dt = date_str
            delta = datetime.now(UTC) - dt.replace(tzinfo=UTC)
            return delta.days
        except Exception:
            return None

    def _extract_nested_value(self, data: dict, key: str) -> Any:
        """Extract value from nested dictionary using dot notation."""
        keys = key.split(".")
        value = data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value

    def _compute_change(self, prev: Any, curr: Any, key: str) -> dict | None:
        """Compute change between previous and current values."""
        try:
            if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
                delta = curr - prev
                pct_change = (delta / prev * 100) if prev != 0 else 0
                return {
                    "previous": prev,
                    "current": curr,
                    "delta": round(delta, 4),
                    "percent_change": round(pct_change, 2),
                }
            elif prev != curr:
                return {
                    "previous": prev,
                    "current": curr,
                    "changed": True,
                }
        except Exception:
            pass
        return None

    def _generate_comparison_summary(
        self,
        changes: dict[str, Any],
        tool_name: str,
        analysis_type: str,
    ) -> str:
        """Generate human-readable summary of changes."""
        if not changes:
            return "No significant changes from previous analysis"

        summaries = []
        for key, change in changes.items():
            if "delta" in change:
                direction = "increased" if change["delta"] > 0 else "decreased"
                summaries.append(
                    f"{key} {direction} by {abs(change['delta'])} "
                    f"({change['percent_change']:+.1f}%)"
                )
            elif change.get("changed"):
                summaries.append(
                    f"{key} changed from '{change['previous']}' "
                    f"to '{change['current']}'"
                )

        return "; ".join(summaries) if summaries else "Minor changes detected"


# Singleton instance for convenience
_storage_service: AnalysisStorageService | None = None


def get_analysis_storage() -> AnalysisStorageService:
    """Get or create the singleton storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = AnalysisStorageService()
    return _storage_service
