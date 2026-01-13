"""
Unified Volatility Analysis Tool.

Consolidates 4 VIX/volatility tools into 1 unified interface:
- VIX term structure analysis
- Contango/backwardation tracking
- 3D volatility surface
- Volatility regime detection
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def volatility_analysis(
    analysis_type: str = "term_structure",
    symbol: str | None = None,
    lookback_days: int = 60,
) -> dict[str, Any]:
    """
    Unified volatility analysis for VIX term structure and regimes.

    Consolidates VIX term structure, contango/backwardation tracking,
    volatility surface, and regime detection into a single tool.

    Args:
        analysis_type: Type of volatility analysis:
            - 'term_structure': VIX futures curve and roll yield (default)
            - 'contango': Historical contango/backwardation with trading signals
            - 'surface': 3D volatility surface for options (requires symbol)
            - 'regime': Current volatility regime detection (low/normal/elevated/crisis)
        symbol: Stock ticker for volatility surface analysis (required for 'surface')
        lookback_days: Analysis period for contango/regime analysis (default: 60)

    Returns:
        Dictionary containing volatility analysis results.

    Examples:
        # VIX term structure analysis
        >>> volatility_analysis()

        # Contango/backwardation tracking
        >>> volatility_analysis(analysis_type="contango", lookback_days=30)

        # Volatility surface for specific stock
        >>> volatility_analysis(analysis_type="surface", symbol="AAPL")

        # Volatility regime detection
        >>> volatility_analysis(analysis_type="regime")
    """
    analysis_type = analysis_type.lower().strip()

    valid_types = ["term_structure", "contango", "surface", "regime"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.volatility_term import (
            vix_contango_backwardation,
            vix_term_structure,
            volatility_regime_indicator,
            volatility_surface_3d,
        )

        if analysis_type == "term_structure":
            result = await vix_term_structure()
            result["analysis_type"] = "term_structure"
            return result

        elif analysis_type == "contango":
            result = await vix_contango_backwardation(lookback_days=lookback_days)
            result["analysis_type"] = "contango"
            return result

        elif analysis_type == "surface":
            if not symbol:
                return {
                    "error": "symbol is required for volatility surface analysis",
                    "hint": "Example: volatility_analysis('surface', symbol='AAPL')",
                    "status": "error",
                }
            result = await volatility_surface_3d(symbol=symbol.strip().upper())
            result["analysis_type"] = "surface"
            return result

        elif analysis_type == "regime":
            result = await volatility_regime_indicator(lookback_days=lookback_days)
            result["analysis_type"] = "regime"
            return result

        else:
            return {"error": f"Unknown analysis_type: {analysis_type}", "status": "error"}

    except Exception as e:
        logger.error(f"Error in volatility_analysis: {e}")
        return {
            "error": str(e),
            "analysis_type": analysis_type,
            "status": "error",
        }
