"""
Unified Technical Analysis Tool.

Consolidates 5 technical analysis tools into 1 unified interface:
- RSI analysis
- MACD analysis
- Bollinger Bands analysis
- Support/Resistance levels
- Full comprehensive analysis
- Chart generation

DISCLAIMER: All technical analysis is for educational purposes only.
Technical indicators do not predict future price movements.
"""

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

AnalysisType = Literal["rsi", "macd", "bollinger", "support_resistance", "full", "chart"]


async def technical_analysis(
    symbol: str,
    analysis_type: str = "full",
    period: int = 14,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    days: int = 365,
    include_chart: bool = False,
) -> dict[str, Any]:
    """
    Unified technical analysis for any indicator or comprehensive analysis.

    Consolidates RSI, MACD, Bollinger Bands, support/resistance, and full
    technical analysis into a single tool with an analysis_type parameter.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        analysis_type: Type of analysis to perform:
            - 'rsi': RSI oscillator analysis (oversold/overbought signals)
            - 'macd': MACD momentum analysis (trend and momentum)
            - 'bollinger': Bollinger Bands volatility analysis
            - 'support_resistance': Key support and resistance price levels
            - 'full': Comprehensive multi-indicator analysis (default)
            - 'chart': Generate visual technical chart
        period: RSI period (default: 14)
        fast_period: MACD fast EMA period (default: 12)
        slow_period: MACD slow EMA period (default: 26)
        signal_period: MACD signal line period (default: 9)
        days: Days of historical data to analyze (default: 365)
        include_chart: Include chart with full analysis (default: False)

    Returns:
        Dictionary containing technical analysis results based on analysis_type.

    Examples:
        # Get RSI analysis
        >>> technical_analysis("AAPL", analysis_type="rsi")

        # Get full analysis with chart
        >>> technical_analysis("MSFT", analysis_type="full", include_chart=True)

        # Get support/resistance levels
        >>> technical_analysis("GOOGL", analysis_type="support_resistance")
    """
    symbol = symbol.strip().upper()
    analysis_type = analysis_type.lower().strip()

    valid_types = ["rsi", "macd", "bollinger", "support_resistance", "full", "chart"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        # Import the underlying implementations
        from maverick_mcp.api.routers.technical import (
            get_full_technical_analysis,
            get_macd_analysis,
            get_rsi_analysis,
            get_stock_chart_analysis,
            get_support_resistance,
        )
        from maverick_mcp.core.technical_analysis import analyze_bollinger_bands
        from maverick_mcp.utils.stock_helpers import get_stock_dataframe_async

        if analysis_type == "rsi":
            result = await get_rsi_analysis(symbol, period=period, days=days)
            result["analysis_type"] = "rsi"
            return result

        elif analysis_type == "macd":
            result = await get_macd_analysis(
                symbol,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                days=days,
            )
            result["analysis_type"] = "macd"
            return result

        elif analysis_type == "bollinger":
            df = await get_stock_dataframe_async(symbol, days)
            bb_analysis = analyze_bollinger_bands(df)
            current_price = float(df["close"].iloc[-1])
            return {
                "symbol": symbol,
                "analysis_type": "bollinger",
                "current_price": current_price,
                "bollinger_bands": bb_analysis,
                "status": "success",
            }

        elif analysis_type == "support_resistance":
            result = await get_support_resistance(symbol, days=days)
            result["analysis_type"] = "support_resistance"
            return result

        elif analysis_type == "chart":
            result = await get_stock_chart_analysis(symbol)
            return result

        else:  # full
            result = await get_full_technical_analysis(symbol, days=days)
            result["analysis_type"] = "full"

            # Optionally include chart
            if include_chart:
                chart_result = await get_stock_chart_analysis(symbol)
                result["chart"] = chart_result

            return result

    except Exception as e:
        logger.error(f"Error in technical_analysis for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }
