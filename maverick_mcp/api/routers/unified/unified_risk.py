"""
Unified Risk Analysis Tool.

Consolidates 6 risk metrics tools into 1 unified interface:
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Drawdown analysis
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Stress testing
- Portfolio VaR
"""

import logging
from typing import Any

from maverick_mcp.api.routers.unified.analysis_wrapper import with_analysis_storage
from maverick_mcp.api.utils.metric_guides import convert_numpy_types

logger = logging.getLogger(__name__)


@with_analysis_storage("risk_analysis")
async def risk_analysis(
    symbol: str | None = None,
    symbols: list[str] | None = None,
    weights: list[float] | None = None,
    analysis_type: str = "comprehensive",
    position_value: float = 10000.0,
    confidence: float = 0.95,
    holding_period: int = 1,
    var_method: str = "historical",
    benchmark: str = "SPY",
    period_days: int = 252,
    period_years: int = 5,
    stress_scenarios: list[str] | None = None,
) -> dict[str, Any]:
    """
    Unified risk analysis for single assets or portfolios.

    Consolidates VaR, CVaR, drawdown, stress test, and risk-adjusted metrics
    into a single tool with an analysis_type parameter.

    Args:
        symbol: Single ticker symbol (for individual asset analysis)
        symbols: List of ticker symbols (for portfolio analysis)
        weights: Portfolio weights (should sum to 1.0, used with symbols)
        analysis_type: Type of risk analysis:
            - 'var': Value at Risk calculation
            - 'cvar': Conditional VaR (Expected Shortfall)
            - 'drawdown': Maximum drawdown analysis
            - 'stress_test': Historical crisis scenario testing
            - 'risk_adjusted': Sharpe, Sortino, Calmar, Treynor ratios
            - 'portfolio': Portfolio-level VaR with diversification
            - 'comprehensive': All metrics combined (default)
        position_value: Dollar value of position (default: 10000)
        confidence: Confidence level for VaR (0.90, 0.95, 0.99)
        holding_period: Holding period in days for VaR
        var_method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
        benchmark: Benchmark ticker for risk-adjusted metrics (default: SPY)
        period_days: Analysis period in trading days (default: 252)
        period_years: Years of history for drawdown (default: 5)
        stress_scenarios: List of scenarios for stress testing

    Returns:
        Dictionary containing risk analysis results.

    Examples:
        # Calculate VaR for a single stock
        >>> risk_analysis("AAPL", analysis_type="var", confidence=0.99)

        # Comprehensive risk analysis
        >>> risk_analysis("MSFT", analysis_type="comprehensive")

        # Portfolio VaR with diversification
        >>> risk_analysis(symbols=["AAPL", "MSFT", "GOOGL"], weights=[0.4, 0.3, 0.3], analysis_type="portfolio")
    """
    analysis_type = analysis_type.lower().strip()

    valid_types = ["var", "cvar", "drawdown", "stress_test", "risk_adjusted", "portfolio", "comprehensive"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "status": "error",
        }

    # Normalize symbol - convert empty string to None
    if symbol is not None:
        symbol = symbol.strip().upper() if symbol.strip() else None

    # Normalize symbols list - filter out empty strings
    if symbols:
        symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        if not symbols:
            symbols = None

    # Default symbol if none provided
    if symbol is None and symbols is None:
        symbol = "AAPL"
        logger.debug("No symbol provided, defaulting to AAPL")

    try:
        from maverick_mcp.api.routers.risk_metrics import (
            risk_adjusted_returns,
            risk_calculate_cvar,
            risk_calculate_var,
            risk_drawdown_analysis,
            risk_portfolio_var,
            risk_stress_test,
        )

        if analysis_type == "var":
            result = await risk_calculate_var(
                symbol=symbol or (symbols[0] if symbols else "AAPL"),
                position_value=position_value,
                confidence=confidence,
                holding_period=holding_period,
                method=var_method,
            )
            result["analysis_type"] = "var"
            return result

        elif analysis_type == "cvar":
            result = await risk_calculate_cvar(
                symbol=symbol or (symbols[0] if symbols else "AAPL"),
                position_value=position_value,
                confidence=confidence,
            )
            result["analysis_type"] = "cvar"
            return result

        elif analysis_type == "drawdown":
            result = await risk_drawdown_analysis(
                symbol=symbol or (symbols[0] if symbols else "AAPL"),
                period_years=period_years,
            )
            result["analysis_type"] = "drawdown"
            return result

        elif analysis_type == "stress_test":
            result = await risk_stress_test(
                symbol=symbol or (symbols[0] if symbols else "AAPL"),
                position_value=position_value,
                scenarios=stress_scenarios,
            )
            result["analysis_type"] = "stress_test"
            return result

        elif analysis_type == "risk_adjusted":
            result = await risk_adjusted_returns(
                symbol=symbol or (symbols[0] if symbols else "AAPL"),
                benchmark=benchmark,
                period_days=period_days,
            )
            result["analysis_type"] = "risk_adjusted"
            return result

        elif analysis_type == "portfolio":
            result = await risk_portfolio_var(
                symbols=symbols or ([symbol] if symbol else ["AAPL", "MSFT", "GOOGL"]),
                weights=weights,
                total_value=position_value,
                confidence=confidence,
            )
            result["analysis_type"] = "portfolio"
            return result

        else:  # comprehensive
            # Run multiple analyses and combine
            target_symbol = symbol or (symbols[0] if symbols else "AAPL")

            var_result = await risk_calculate_var(
                symbol=target_symbol,
                position_value=position_value,
                confidence=confidence,
                method=var_method,
            )

            cvar_result = await risk_calculate_cvar(
                symbol=target_symbol,
                position_value=position_value,
                confidence=confidence,
            )

            drawdown_result = await risk_drawdown_analysis(
                symbol=target_symbol,
                period_years=min(period_years, 3),  # Limit for speed
            )

            risk_adj_result = await risk_adjusted_returns(
                symbol=target_symbol,
                benchmark=benchmark,
                period_days=period_days,
            )

            # Convert any remaining numpy types for JSON serialization
            return convert_numpy_types({
                "symbol": target_symbol,
                "analysis_type": "comprehensive",
                "position_value": position_value,
                "var": var_result.get("var_analysis", {}),
                "cvar": cvar_result.get("cvar_analysis", {}),
                "drawdown": drawdown_result.get("maximum_drawdown", {}),
                "risk_adjusted_metrics": risk_adj_result.get("risk_adjusted_metrics", {}),
                "volatility": var_result.get("volatility", {}),
                "interpretation": (
                    f"VaR ({confidence*100:.0f}%): ${var_result.get('var_analysis', {}).get('var_dollars', 0):,.2f}. "
                    f"Max drawdown: {drawdown_result.get('maximum_drawdown', {}).get('depth_pct', 0):.1f}%. "
                    f"Sharpe: {risk_adj_result.get('risk_adjusted_metrics', {}).get('sharpe_ratio', 0):.2f}"
                ),
                "status": "success",
            })

    except Exception as e:
        logger.error(f"Error in risk_analysis: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }
