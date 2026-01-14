"""
Unified Analysis History Tool.

Provides access to stored analysis history for any symbol.
"""

import logging
from typing import Any

from maverick_mcp.data.analysis_storage import get_analysis_storage

logger = logging.getLogger(__name__)


async def analysis_history(
    symbol: str,
    tool_name: str | None = None,
    days: int = 30,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Retrieve analysis history for a symbol.

    This tool provides access to stored analysis results, allowing you to:
    - Review past analyses for a stock
    - Track how metrics have changed over time
    - Filter by specific analysis tools

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        tool_name: Filter by specific tool (optional):
            - 'technical_analysis': RSI, MACD, Bollinger, support/resistance
            - 'quant_analysis': Beta, momentum, volatility, factors
            - 'risk_analysis': VaR, CVaR, drawdown, stress tests
            - 'alternative_data': Insider, institutional, short interest
            - None: Return history from all tools
        days: Maximum age of analyses to retrieve (default: 30)
        limit: Maximum number of analyses to return (default: 20)

    Returns:
        Dictionary containing:
        - symbol: The requested symbol
        - total_analyses: Total number of analyses found
        - history: List of analysis summaries ordered by date (newest first)
        - tools_used: List of unique tools found in history

    Examples:
        # Get all recent analyses for AAPL
        >>> analysis_history("AAPL")

        # Get only technical analysis history
        >>> analysis_history("AAPL", tool_name="technical_analysis")

        # Get last 90 days of quant analysis
        >>> analysis_history("NVDA", tool_name="quant_analysis", days=90)
    """
    symbol = symbol.strip().upper()

    try:
        storage = get_analysis_storage()

        # Get analysis history
        history = await storage.get_analysis_history(
            symbol=symbol,
            tool_name=tool_name,
            days=days,
            limit=limit,
        )

        if not history:
            return {
                "symbol": symbol,
                "status": "success",
                "message": f"No analysis history found for {symbol}",
                "total_analyses": 0,
                "history": [],
                "tools_used": [],
            }

        # Extract unique tools
        tools_used = list({h["tool_name"] for h in history})

        # Format history for response
        formatted_history = []
        for h in history:
            result = h.get("result", {})
            formatted_history.append(
                {
                    "id": h["id"],
                    "date": h["created_at"],
                    "tool": h["tool_name"],
                    "type": h["analysis_type"],
                    "summary": _extract_summary(result, h["tool_name"]),
                }
            )

        return {
            "symbol": symbol,
            "status": "success",
            "total_analyses": len(history),
            "filter": {
                "tool_name": tool_name,
                "days": days,
                "limit": limit,
            },
            "history": formatted_history,
            "tools_used": tools_used,
        }

    except Exception as e:
        logger.error(f"Error retrieving analysis history for {symbol}: {e}")
        return {
            "symbol": symbol,
            "status": "error",
            "error": str(e),
        }


def _extract_summary(result: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Extract key summary fields from an analysis result based on tool type."""
    summary = {}

    # Common fields
    if "status" in result:
        summary["status"] = result["status"]

    # Tool-specific fields
    if tool_name == "technical_analysis":
        for key in ["trend", "outlook", "signal", "current_price", "current_rsi"]:
            if key in result:
                summary[key] = result[key]
        # Check nested analysis
        if "analysis" in result and isinstance(result["analysis"], dict):
            for key in ["signal", "current_rsi", "trend"]:
                if key in result["analysis"]:
                    summary[key] = result["analysis"][key]

    elif tool_name == "quant_analysis":
        for key in ["beta", "alpha", "correlation", "momentum_score", "trend"]:
            if key in result:
                summary[key] = result[key]
        if "beta_analysis" in result and isinstance(result["beta_analysis"], dict):
            for key in ["beta", "alpha"]:
                if key in result["beta_analysis"]:
                    summary[key] = result["beta_analysis"][key]

    elif tool_name == "risk_analysis":
        for key in ["var_95", "var_99", "max_drawdown", "sharpe_ratio", "risk_score"]:
            if key in result:
                summary[key] = result[key]

    elif tool_name == "alternative_data":
        for key in [
            "sentiment_score",
            "net_activity",
            "short_interest_ratio",
            "institutional_ownership",
        ]:
            if key in result:
                summary[key] = result[key]
        if "insider_sentiment_score" in result:
            score_data = result["insider_sentiment_score"]
            if isinstance(score_data, dict):
                summary["insider_score"] = score_data.get("composite_score")
                summary["insider_rating"] = score_data.get("rating")

    return summary


async def clear_analysis_history(
    symbol: str | None = None,
    tool_name: str | None = None,
    older_than_days: int | None = None,
) -> dict[str, Any]:
    """
    Clear stored analysis history.

    Args:
        symbol: Clear history for specific symbol (None = all)
        tool_name: Clear history for specific tool (None = all)
        older_than_days: Clear analyses older than N days (None = all matching)

    Returns:
        Dictionary with count of cleared entries
    """
    # This is a placeholder - implement if needed
    return {
        "status": "not_implemented",
        "message": "Clear functionality not yet implemented",
    }
