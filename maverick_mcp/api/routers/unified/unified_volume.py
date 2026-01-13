"""
Unified Volume Analysis Tool.

Consolidates 4 volume analysis tools into 1 unified interface:
- Volume profile with POC and value areas
- VWAP with standard deviation bands
- Market profile (TPO) analysis
- Volume footprint (buy/sell pressure)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def volume_analysis(
    symbol: str,
    analysis_type: str = "profile",
    period_days: int = 20,
    num_bins: int = 50,
    vwap_period: str = "1d",
    std_devs: list[float] | None = None,
) -> dict[str, Any]:
    """
    Unified volume analysis for volume distribution and VWAP.

    Consolidates volume profile, VWAP bands, market profile, and
    footprint analysis into a single tool with an analysis_type parameter.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        analysis_type: Type of volume analysis:
            - 'profile': Volume profile with POC, value areas, HVN/LVN (default)
            - 'vwap': VWAP with standard deviation bands
            - 'market_profile': TPO-style time-price opportunity analysis
            - 'footprint': Buy/sell volume delta and pressure analysis
        period_days: Analysis period in trading days (default: 20)
        num_bins: Number of price bins for volume profile (default: 50)
        vwap_period: Period for VWAP ('1d' for intraday, '5d' for week)
        std_devs: Standard deviation levels for VWAP bands (default: [1.0, 2.0, 3.0])

    Returns:
        Dictionary containing volume analysis results.

    Examples:
        # Volume profile with POC and value areas
        >>> volume_analysis("AAPL")

        # VWAP with deviation bands
        >>> volume_analysis("SPY", analysis_type="vwap")

        # Market profile analysis
        >>> volume_analysis("MSFT", analysis_type="market_profile", period_days=5)

        # Buy/sell pressure analysis
        >>> volume_analysis("NVDA", analysis_type="footprint", period_days=30)
    """
    if not symbol:
        return {"error": "Symbol is required", "status": "error"}

    symbol = symbol.strip().upper()
    analysis_type = analysis_type.lower().strip()

    valid_types = ["profile", "vwap", "market_profile", "footprint"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.volume_profile import (
            volume_footprint_analysis,
            volume_market_profile,
            volume_profile_analysis,
            volume_vwap_bands,
        )

        if analysis_type == "profile":
            result = await volume_profile_analysis(
                symbol=symbol,
                period_days=period_days,
                num_bins=num_bins,
            )
            result["analysis_type"] = "profile"
            return result

        elif analysis_type == "vwap":
            result = await volume_vwap_bands(
                symbol=symbol,
                period=vwap_period,
                std_devs=std_devs,
            )
            result["analysis_type"] = "vwap"
            return result

        elif analysis_type == "market_profile":
            result = await volume_market_profile(
                symbol=symbol,
                period_days=period_days,
            )
            result["analysis_type"] = "market_profile"
            return result

        elif analysis_type == "footprint":
            result = await volume_footprint_analysis(
                symbol=symbol,
                lookback_bars=period_days,
            )
            result["analysis_type"] = "footprint"
            return result

        else:
            return {"error": f"Unknown analysis_type: {analysis_type}", "status": "error"}

    except Exception as e:
        logger.error(f"Error in volume_analysis for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }
