"""
Unified Quantitative Analysis Tool.

Consolidates 6 quant analysis tools into 1 unified interface:
- Beta and alpha calculation
- Factor exposure analysis
- Correlation matrix
- Momentum analysis
- Volatility analysis
- Composite factor scores
"""

import logging
from typing import Any

from maverick_mcp.api.routers.unified.analysis_wrapper import with_analysis_storage

logger = logging.getLogger(__name__)


@with_analysis_storage("quant_analysis")
async def quant_analysis(
    symbol: str | None = None,
    symbols: list[str] | None = None,
    analysis_type: str = "comprehensive",
    benchmark: str = "SPY",
    period_days: int = 252,
    factors: list[str] | None = None,
    momentum_periods: list[int] | None = None,
) -> dict[str, Any]:
    """
    Unified quantitative analysis for factor exposure and statistical metrics.

    Consolidates beta, correlation, factor exposure, momentum, and volatility
    analysis into a single tool with an analysis_type parameter.

    Args:
        symbol: Single ticker symbol (for individual analysis)
        symbols: List of ticker symbols (for correlation matrix)
        analysis_type: Type of quant analysis:
            - 'beta': Beta, alpha, and correlation vs benchmark
            - 'correlation': Correlation matrix across multiple assets
            - 'factors': Factor exposure (value, momentum, quality, size, volatility)
            - 'momentum': Multi-timeframe momentum analysis
            - 'volatility': Volatility regime analysis
            - 'scores': Composite factor scores with recommendation
            - 'comprehensive': All quant metrics combined (default)
        benchmark: Benchmark ticker for beta calculation (default: SPY)
        period_days: Analysis period in trading days (default: 252)
        factors: Specific factors to analyze (value, momentum, quality, size, volatility)
        momentum_periods: List of momentum periods in days (default: [21, 63, 126, 252])

    Returns:
        Dictionary containing quantitative analysis results.

    Examples:
        # Calculate beta vs S&P 500
        >>> quant_analysis("AAPL", analysis_type="beta")

        # Get correlation matrix for multiple stocks
        >>> quant_analysis(symbols=["AAPL", "MSFT", "GOOGL", "AMZN"], analysis_type="correlation")

        # Factor exposure analysis
        >>> quant_analysis("NVDA", analysis_type="factors")
    """
    analysis_type = analysis_type.lower().strip()

    valid_types = ["beta", "correlation", "factors", "momentum", "volatility", "scores", "comprehensive", "seasonality"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "status": "error",
        }

    # Default symbol if none provided
    if symbol is None and symbols is None:
        symbol = "AAPL"

    try:
        from maverick_mcp.api.routers.quant_analysis import (
            quant_calculate_beta,
            quant_correlation_matrix,
            quant_factor_exposure,
            quant_factor_scores,
            quant_momentum_analysis,
            quant_seasonality_analysis,
            quant_volatility_analysis,
        )

        if analysis_type == "beta":
            result = await quant_calculate_beta(
                symbol=symbol or symbols[0] if symbols else "AAPL",
                benchmark=benchmark,
                period_days=period_days,
            )
            result["analysis_type"] = "beta"
            return result

        elif analysis_type == "correlation":
            result = await quant_correlation_matrix(
                symbols=symbols or [symbol, "SPY"] if symbol else ["AAPL", "MSFT", "GOOGL", "AMZN"],
                period_days=period_days,
            )
            result["analysis_type"] = "correlation"
            return result

        elif analysis_type == "factors":
            result = await quant_factor_exposure(
                symbol=symbol or symbols[0] if symbols else "AAPL",
                factors=factors,
            )
            result["analysis_type"] = "factors"
            return result

        elif analysis_type == "momentum":
            result = await quant_momentum_analysis(
                symbol=symbol or symbols[0] if symbols else "AAPL",
                periods=momentum_periods,
            )
            result["analysis_type"] = "momentum"
            return result

        elif analysis_type == "volatility":
            result = await quant_volatility_analysis(
                symbol=symbol or symbols[0] if symbols else "AAPL",
                period_days=period_days,
            )
            result["analysis_type"] = "volatility"
            return result

        elif analysis_type == "scores":
            result = await quant_factor_scores(
                symbol=symbol or symbols[0] if symbols else "AAPL",
            )
            result["analysis_type"] = "scores"
            return result

        elif analysis_type == "seasonality":
            result = await quant_seasonality_analysis(
                symbol=symbol or symbols[0] if symbols else "AAPL",
            )
            result["analysis_type"] = "seasonality"
            return result

        else:  # comprehensive
            target_symbol = symbol or (symbols[0] if symbols else "AAPL")

            beta_result = await quant_calculate_beta(
                symbol=target_symbol,
                benchmark=benchmark,
                period_days=period_days,
            )

            factors_result = await quant_factor_exposure(
                symbol=target_symbol,
                factors=factors,
            )

            momentum_result = await quant_momentum_analysis(
                symbol=target_symbol,
                periods=momentum_periods,
            )

            volatility_result = await quant_volatility_analysis(
                symbol=target_symbol,
                period_days=period_days,
            )

            return {
                "symbol": target_symbol,
                "analysis_type": "comprehensive",
                "benchmark": benchmark,
                "beta_analysis": beta_result.get("beta_analysis", {}),
                "factor_scores": factors_result.get("factor_scores", {}),
                "momentum_summary": momentum_result.get("summary", {}),
                "volatility_regime": volatility_result.get("regime", {}),
                "interpretation": (
                    f"Beta: {beta_result.get('beta_analysis', {}).get('beta', 1):.2f}. "
                    f"Factor composite: {factors_result.get('composite_score', 50):.0f}/100. "
                    f"Trend: {momentum_result.get('summary', {}).get('trend', 'N/A')}. "
                    f"Vol regime: {volatility_result.get('regime', {}).get('current_regime', 'N/A')}"
                ),
                "status": "success",
            }

    except Exception as e:
        logger.error(f"Error in quant_analysis: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }
