"""
Unified Simulation Tool.

Consolidates Monte Carlo and probabilistic analysis tools into 1 unified interface:
- Single asset Monte Carlo simulation
- Portfolio Monte Carlo with correlated assets
- Distribution analysis (normal, t, Laplace fitting)
- VaR/CVaR risk metrics
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def simulation(
    symbol: str | None = None,
    symbols: list[str] | None = None,
    weights: list[float] | None = None,
    simulation_type: str = "monte_carlo",
    num_simulations: int = 10000,
    forecast_days: int = 252,
    initial_value: float = 100000.0,
    scenarios: list[dict] | None = None,
    include_distribution: bool = True,
) -> dict[str, Any]:
    """
    Unified simulation analysis for Monte Carlo and probabilistic forecasting.

    Consolidates asset simulations, portfolio simulations, and distribution
    analysis into a single tool with a simulation_type parameter.

    Args:
        symbol: Single ticker for asset simulation
        symbols: List of tickers for portfolio simulation
        weights: Portfolio weights (must sum to 1.0, equal weight if None)
        simulation_type: Type of simulation:
            - 'monte_carlo': Single asset GBM simulation (default)
            - 'portfolio': Multi-asset portfolio simulation with correlations
        num_simulations: Number of simulation paths (default: 10000)
        forecast_days: Trading days to simulate (default: 252 = 1 year)
        initial_value: Initial portfolio value for portfolio sim (default: $100,000)
        scenarios: Optional scenario definitions for probability analysis
            Example: [{"name": "bull", "return": 0.20}, {"name": "bear", "return": -0.20}]
        include_distribution: Include distribution statistics (default: True)

    Returns:
        Dictionary containing simulation results with percentiles, VaR, and statistics.

    Examples:
        # Single asset Monte Carlo
        >>> simulation(symbol="AAPL")

        # Asset simulation with custom scenarios
        >>> simulation(symbol="SPY", scenarios=[
        ...     {"name": "bull", "return": 0.20},
        ...     {"name": "bear", "return": -0.20}
        ... ])

        # Portfolio simulation with correlations
        >>> simulation(
        ...     symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
        ...     weights=[0.3, 0.3, 0.2, 0.2],
        ...     simulation_type="portfolio",
        ...     initial_value=100000
        ... )

        # Quick simulation with fewer paths
        >>> simulation(symbol="TSLA", num_simulations=1000, forecast_days=63)
    """
    simulation_type = simulation_type.lower().strip()

    valid_types = ["monte_carlo", "portfolio"]
    if simulation_type not in valid_types:
        return {
            "error": f"Invalid simulation_type '{simulation_type}'. Must be one of: {valid_types}",
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.monte_carlo import (
            mc_asset_simulation,
            mc_portfolio_simulation,
        )

        if simulation_type == "monte_carlo":
            if not symbol:
                return {
                    "error": "symbol is required for monte_carlo simulation",
                    "hint": "Example: simulation(symbol='AAPL')",
                    "status": "error",
                }
            result = await mc_asset_simulation(
                symbol=symbol.strip().upper(),
                num_simulations=num_simulations,
                forecast_days=forecast_days,
                scenarios=scenarios,
                include_distribution=include_distribution,
            )
            result["simulation_type"] = "monte_carlo"
            return result

        elif simulation_type == "portfolio":
            if not symbols or len(symbols) < 2:
                return {
                    "error": "At least 2 symbols required for portfolio simulation",
                    "hint": "Example: simulation(symbols=['AAPL', 'MSFT', 'GOOGL'], simulation_type='portfolio')",
                    "status": "error",
                }
            result = await mc_portfolio_simulation(
                symbols=[s.strip().upper() for s in symbols],
                weights=weights,
                initial_value=initial_value,
                num_simulations=min(num_simulations, 5000),  # Cap for portfolio
                forecast_days=forecast_days,
            )
            result["simulation_type"] = "portfolio"
            return result

        else:
            return {"error": f"Unknown simulation_type: {simulation_type}", "status": "error"}

    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        return {
            "error": str(e),
            "simulation_type": simulation_type,
            "status": "error",
        }
