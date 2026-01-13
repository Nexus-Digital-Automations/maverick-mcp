"""
Unified Market Breadth Tool.

Consolidates 5 market breadth tools into 1 unified interface:
- Advance/decline analysis
- New highs/lows tracking
- Sector rotation analysis
- Market regime detection
- Breadth divergence identification
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def market_breadth(
    analysis_type: str = "overview",
    index: str = "SP500",
    period_days: int = 30,
) -> dict[str, Any]:
    """
    Unified market breadth analysis for market-wide indicators.

    Consolidates advance/decline, highs/lows, sector analysis, regime detection,
    and divergence analysis into a single tool with an analysis_type parameter.

    Args:
        analysis_type: Type of breadth analysis:
            - 'advance_decline': A/D line, ratio, McClellan Oscillator, thrust
            - 'highs_lows': 52-week new highs vs new lows count
            - 'sector': Sector rotation and momentum analysis
            - 'regime': Bull/bear market regime detection (SPY-based)
            - 'divergence': Price vs breadth divergence detection
            - 'overview': Combined key breadth metrics (default)
        index: Market index to analyze ('SP500', 'SPY' for regime)
        period_days: Analysis period in trading days (default: 30)

    Returns:
        Dictionary containing market breadth analysis results.

    Examples:
        # Get market breadth overview
        >>> market_breadth()

        # Analyze advance/decline metrics
        >>> market_breadth(analysis_type="advance_decline")

        # Check for price/breadth divergences
        >>> market_breadth(analysis_type="divergence", period_days=60)

        # Detect market regime
        >>> market_breadth(analysis_type="regime", index="SPY")
    """
    analysis_type = analysis_type.lower().strip()

    valid_types = ["advance_decline", "highs_lows", "sector", "regime", "divergence", "overview"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.market_breadth import (
            breadth_advance_decline,
            breadth_divergence_check,
            breadth_market_regime,
            breadth_new_highs_lows,
            breadth_sector_analysis,
        )

        if analysis_type == "advance_decline":
            result = await breadth_advance_decline(
                index=index,
                period_days=period_days,
            )
            result["analysis_type"] = "advance_decline"
            return result

        elif analysis_type == "highs_lows":
            result = await breadth_new_highs_lows(index=index)
            result["analysis_type"] = "highs_lows"
            return result

        elif analysis_type == "sector":
            result = await breadth_sector_analysis(period_days=period_days)
            result["analysis_type"] = "sector"
            return result

        elif analysis_type == "regime":
            result = await breadth_market_regime(
                index=index if index != "SP500" else "SPY",
                lookback_days=period_days,
            )
            result["analysis_type"] = "regime"
            return result

        elif analysis_type == "divergence":
            result = await breadth_divergence_check(
                index=index if index != "SP500" else "SPY",
                period_days=period_days,
            )
            result["analysis_type"] = "divergence"
            return result

        else:  # overview
            # Combine key metrics from multiple analyses
            ad_result = await breadth_advance_decline(index=index, period_days=period_days)
            hl_result = await breadth_new_highs_lows(index=index)
            sector_result = await breadth_sector_analysis(period_days=period_days)

            return {
                "index": index,
                "analysis_type": "overview",
                "advance_decline": {
                    "ad_ratio": ad_result.get("metrics", {}).get("ad_ratio", "N/A"),
                    "mcclellan_oscillator": ad_result.get("metrics", {}).get("mcclellan_oscillator", "N/A"),
                    "thrust_pct": ad_result.get("metrics", {}).get("thrust_pct", "N/A"),
                    "signal": ad_result.get("metrics", {}).get("breadth_thrust_signal", "N/A"),
                },
                "highs_lows": {
                    "new_highs": hl_result.get("new_52w_highs", {}).get("count", "N/A"),
                    "new_lows": hl_result.get("new_52w_lows", {}).get("count", "N/A"),
                    "high_low_ratio": hl_result.get("high_low_ratio", "N/A"),
                    "signal": hl_result.get("signal", "N/A"),
                },
                "sector_rotation": {
                    "rotation_signal": sector_result.get("rotation_signal", "N/A"),
                    "leaders": [s["sector"] for s in sector_result.get("leaders", [])[:3]],
                    "laggards": [s["sector"] for s in sector_result.get("laggards", [])[:3]],
                },
                "interpretation": (
                    f"A/D: {ad_result.get('metrics', {}).get('breadth_thrust_signal', 'N/A')}. "
                    f"H/L: {hl_result.get('signal', 'N/A')}. "
                    f"Sectors: {sector_result.get('rotation_signal', 'N/A')}"
                ),
                "status": "success",
            }

    except Exception as e:
        logger.error(f"Error in market_breadth: {e}")
        return {
            "error": str(e),
            "analysis_type": analysis_type,
            "status": "error",
        }
