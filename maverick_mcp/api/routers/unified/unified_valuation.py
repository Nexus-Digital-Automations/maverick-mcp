"""
Unified Valuation Models Tool.

Consolidates 5 valuation tools into 1 unified interface:
- DCF (Discounted Cash Flow) valuation
- Valuation multiples (P/E, P/B, EV/EBITDA, etc.)
- Comparable company analysis
- Dividend Discount Model (DDM)
- Blended fair value estimate
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def valuation(
    symbol: str,
    model: str = "fair_value",
    comparables: list[str] | None = None,
    growth_rate: float | None = None,
    discount_rate: float | None = None,
    required_return: float = 0.10,
) -> dict[str, Any]:
    """
    Unified valuation analysis using multiple models.

    Consolidates DCF, multiples, comparable company analysis, DDM,
    and blended fair value into a single tool with a model parameter.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        model: Valuation model to use:
            - 'dcf': Discounted Cash Flow valuation
            - 'multiples': P/E, P/B, P/S, EV/EBITDA ratios
            - 'comps': Comparable company analysis
            - 'ddm': Dividend Discount Model
            - 'fair_value': Blended estimate from all applicable models (default)
        comparables: List of comparable company tickers (for 'comps' model)
        growth_rate: FCF growth rate for DCF (estimated if None)
        discount_rate: WACC for DCF (estimated if None)
        required_return: Required rate of return for DDM (default: 10%)

    Returns:
        Dictionary containing valuation analysis results.

    Examples:
        # Blended fair value estimate
        >>> valuation("AAPL")

        # DCF valuation with custom assumptions
        >>> valuation("MSFT", model="dcf", growth_rate=0.08, discount_rate=0.10)

        # Valuation multiples
        >>> valuation("GOOGL", model="multiples")

        # Comparable company analysis
        >>> valuation("NVDA", model="comps", comparables=["AMD", "INTC", "QCOM"])

        # Dividend discount model
        >>> valuation("JNJ", model="ddm", required_return=0.08)
    """
    if not symbol:
        return {"error": "Symbol is required", "status": "error"}

    symbol = symbol.strip().upper()
    model = model.lower().strip()

    valid_models = ["dcf", "multiples", "comps", "ddm", "fair_value"]
    if model not in valid_models:
        return {
            "error": f"Invalid model '{model}'. Must be one of: {valid_models}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.valuation_models import (
            valuation_comparable_company,
            valuation_dcf,
            valuation_dividend_discount,
            valuation_fair_value_estimate,
            valuation_multiples,
        )

        if model == "dcf":
            result = await valuation_dcf(
                symbol=symbol,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
            )
            result["model"] = "dcf"
            return result

        elif model == "multiples":
            result = await valuation_multiples(symbol=symbol)
            result["model"] = "multiples"
            return result

        elif model == "comps":
            result = await valuation_comparable_company(
                symbol=symbol,
                comparables=comparables,
            )
            result["model"] = "comps"
            return result

        elif model == "ddm":
            result = await valuation_dividend_discount(
                symbol=symbol,
                required_return=required_return,
            )
            result["model"] = "ddm"
            return result

        else:  # fair_value
            result = await valuation_fair_value_estimate(symbol=symbol)
            result["model"] = "fair_value"
            return result

    except Exception as e:
        logger.error(f"Error in valuation for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "model": model,
            "status": "error",
        }
