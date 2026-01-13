"""
Unified Alternative Data Tool.

Consolidates 5 alternative data tools into 1 unified interface:
- Short interest analysis
- Insider transactions
- Institutional holdings
- Options flow analysis
- Composite sentiment scoring
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def alternative_data(
    symbol: str,
    data_type: str = "sentiment",
    period_days: int = 90,
) -> dict[str, Any]:
    """
    Unified alternative data analysis for market insights.

    Consolidates short interest, insider transactions, institutional holdings,
    options flow, and sentiment analysis into a single tool.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GME')
        data_type: Type of alternative data:
            - 'short_interest': Short interest % and squeeze potential
            - 'insider': Insider buying/selling activity
            - 'institutional': Top institutional holders and flow
            - 'options_flow': Put/call ratio and unusual activity
            - 'sentiment': Composite sentiment score combining all sources (default)
        period_days: Analysis period for insider/sentiment (default: 90)

    Returns:
        Dictionary containing alternative data analysis results.

    Examples:
        # Composite sentiment score
        >>> alternative_data("AAPL")

        # Short squeeze analysis
        >>> alternative_data("GME", data_type="short_interest")

        # Insider transactions
        >>> alternative_data("MSFT", data_type="insider", period_days=60)

        # Institutional holdings
        >>> alternative_data("NVDA", data_type="institutional")

        # Options flow analysis
        >>> alternative_data("TSLA", data_type="options_flow")
    """
    if not symbol:
        return {"error": "Symbol is required", "status": "error"}

    symbol = symbol.strip().upper()
    data_type = data_type.lower().strip()

    valid_types = ["short_interest", "insider", "institutional", "options_flow", "sentiment"]
    if data_type not in valid_types:
        return {
            "error": f"Invalid data_type '{data_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.alternative_data import (
            alt_insider_transactions,
            alt_institutional_holdings,
            alt_options_flow,
            alt_sentiment_composite,
            alt_short_interest,
        )

        if data_type == "short_interest":
            result = await alt_short_interest(symbol=symbol)
            result["data_type"] = "short_interest"
            return result

        elif data_type == "insider":
            result = await alt_insider_transactions(
                symbol=symbol,
                period_days=period_days,
            )
            result["data_type"] = "insider"
            return result

        elif data_type == "institutional":
            result = await alt_institutional_holdings(symbol=symbol)
            result["data_type"] = "institutional"
            return result

        elif data_type == "options_flow":
            result = await alt_options_flow(symbol=symbol)
            result["data_type"] = "options_flow"
            return result

        else:  # sentiment
            result = await alt_sentiment_composite(symbol=symbol)
            result["data_type"] = "sentiment"
            return result

    except Exception as e:
        logger.error(f"Error in alternative_data for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "data_type": data_type,
            "status": "error",
        }
