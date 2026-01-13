"""
Unified Stock Screening Tool.

Consolidates 5 screening tools into 1 unified interface:
- Maverick bullish stocks
- Maverick bearish stocks
- Supply/demand breakouts
- All screening recommendations
- Custom criteria screening

DISCLAIMER: Stock screening is for educational purposes only.
Past performance does not guarantee future results.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def stock_screener(
    strategy: str = "all",
    limit: int = 20,
    filter_moving_averages: bool = False,
    min_momentum_score: float | None = None,
    min_volume: int | None = None,
    max_price: float | None = None,
    sectors: list[str] | None = None,
) -> dict[str, Any]:
    """
    Unified stock screening across all strategies.

    Consolidates Maverick, Bear, Supply/Demand, and custom screening
    into a single tool with a strategy parameter.

    Args:
        strategy: Screening strategy to use:
            - 'maverick': High momentum bullish setups (default for bullish)
            - 'bear': Weak stocks for bearish/short opportunities
            - 'supply_demand': Breakout patterns from accumulation zones
            - 'momentum': Pure momentum-based screen
            - 'value': Value-oriented screen (low P/E, etc.)
            - 'all': Results from all strategies combined (default)
            - 'custom': Custom criteria using filter parameters
        limit: Maximum number of stocks to return (default: 20)
        filter_moving_averages: For supply_demand, filter to stocks above all MAs
        min_momentum_score: Minimum momentum score (0-100) for custom filtering
        min_volume: Minimum average daily volume for custom filtering
        max_price: Maximum stock price for custom filtering
        sectors: List of sectors to filter (e.g., ["Technology", "Healthcare"])

    Returns:
        Dictionary containing screening results.

    Examples:
        # Get Maverick bullish stocks
        >>> stock_screener(strategy="maverick", limit=10)

        # Get all screening recommendations
        >>> stock_screener(strategy="all")

        # Custom screening with criteria
        >>> stock_screener(strategy="custom", min_momentum_score=70, max_price=50)
    """
    strategy = strategy.lower().strip()

    valid_strategies = ["maverick", "bear", "supply_demand", "momentum", "value", "all", "custom"]
    if strategy not in valid_strategies:
        return {
            "error": f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}",
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.screening import (
            get_all_screening_recommendations,
            get_maverick_bear_stocks,
            get_maverick_stocks,
            get_screening_by_criteria,
            get_supply_demand_breakouts,
        )

        if strategy == "maverick":
            result = get_maverick_stocks(limit=limit)
            result["strategy"] = "maverick"
            return result

        elif strategy == "bear":
            result = get_maverick_bear_stocks(limit=limit)
            result["strategy"] = "bear"
            return result

        elif strategy == "supply_demand":
            result = get_supply_demand_breakouts(
                limit=limit, filter_moving_averages=filter_moving_averages
            )
            result["strategy"] = "supply_demand"
            return result

        elif strategy == "all":
            result = get_all_screening_recommendations()
            result["strategy"] = "all"
            return result

        elif strategy in ["momentum", "value", "custom"]:
            # Use custom criteria screening
            result = get_screening_by_criteria(
                min_momentum_score=min_momentum_score,
                min_volume=min_volume,
                max_price=max_price,
                sector=sectors[0] if sectors else None,
                limit=limit,
            )
            result["strategy"] = strategy

            # Add filter info
            result["filters_applied"] = {
                "min_momentum_score": min_momentum_score,
                "min_volume": min_volume,
                "max_price": max_price,
                "sectors": sectors,
            }
            return result

        else:
            return {
                "error": f"Unknown strategy: {strategy}",
                "status": "error",
            }

    except Exception as e:
        logger.error(f"Error in stock_screener: {e}")
        return {
            "error": str(e),
            "strategy": strategy,
            "status": "error",
        }
