"""
Unified Multi-Timeframe Analysis Tool.

Consolidates 4 multi-timeframe tools into 1 unified interface:
- Trend alignment across timeframes
- RSI divergence detection
- Moving average confirmation
- Composite signal scoring
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def multi_timeframe(
    symbol: str,
    analysis_type: str = "signal_score",
    timeframes: list[str] | None = None,
    ma_periods: list[int] | None = None,
    rsi_period: int = 14,
) -> dict[str, Any]:
    """
    Unified multi-timeframe analysis for trend confirmation.

    Consolidates trend, RSI, moving averages, and composite signal
    analysis across daily, weekly, and monthly timeframes.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        analysis_type: Type of multi-timeframe analysis:
            - 'trend': Trend direction alignment across timeframes
            - 'rsi': RSI divergence detection across timeframes
            - 'moving_averages': MA alignment and golden/death cross detection
            - 'signal_score': Composite score (0-100) combining all analyses (default)
        timeframes: Timeframes to analyze (default: ['1d', '1wk', '1mo'])
        ma_periods: Moving average periods (default: [20, 50, 200])
        rsi_period: RSI calculation period (default: 14)

    Returns:
        Dictionary containing multi-timeframe analysis results.

    Examples:
        # Get composite signal score
        >>> multi_timeframe("AAPL")

        # Analyze trend alignment
        >>> multi_timeframe("SPY", analysis_type="trend")

        # RSI across timeframes
        >>> multi_timeframe("MSFT", analysis_type="rsi")

        # Moving average analysis with custom periods
        >>> multi_timeframe("GOOGL", analysis_type="moving_averages", ma_periods=[10, 50, 200])
    """
    if not symbol:
        return {"error": "Symbol is required", "status": "error"}

    symbol = symbol.strip().upper()
    analysis_type = analysis_type.lower().strip()

    valid_types = ["trend", "rsi", "moving_averages", "signal_score"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.multi_timeframe import (
            multi_timeframe_moving_averages,
            multi_timeframe_rsi,
            multi_timeframe_signal_score,
            multi_timeframe_trend,
        )

        if analysis_type == "trend":
            result = await multi_timeframe_trend(
                symbol=symbol,
                timeframes=timeframes,
            )
            result["analysis_type"] = "trend"
            return result

        elif analysis_type == "rsi":
            result = await multi_timeframe_rsi(
                symbol=symbol,
                timeframes=timeframes,
                period=rsi_period,
            )
            result["analysis_type"] = "rsi"
            return result

        elif analysis_type == "moving_averages":
            result = await multi_timeframe_moving_averages(
                symbol=symbol,
                timeframes=timeframes,
                ma_periods=ma_periods,
            )
            result["analysis_type"] = "moving_averages"
            return result

        else:  # signal_score
            result = await multi_timeframe_signal_score(symbol=symbol)
            result["analysis_type"] = "signal_score"
            return result

    except Exception as e:
        logger.error(f"Error in multi_timeframe for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }
