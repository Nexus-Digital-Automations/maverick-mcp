"""
Unified Portfolio Tools.

Consolidates 10 portfolio tools into 2 unified interfaces:
- portfolio_manage: Add, remove, view, clear positions
- portfolio_analyze: Risk analysis, correlation, comparison, attribution
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def portfolio_manage(
    action: str,
    ticker: str | None = None,
    shares: float | None = None,
    purchase_price: float | None = None,
    purchase_date: str | None = None,
    notes: str | None = None,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
    confirm: bool = False,
) -> dict[str, Any]:
    """
    Unified portfolio management for adding, removing, and viewing positions.

    Consolidates portfolio CRUD operations into a single tool with
    an action parameter.

    Args:
        action: Portfolio action to perform:
            - 'add': Add or update a position (requires ticker, shares, purchase_price)
            - 'remove': Remove shares or entire position (requires ticker)
            - 'view': Get complete portfolio with all positions and P&L
            - 'clear': Clear all positions (requires confirm=True)
        ticker: Stock ticker symbol (required for add/remove)
        shares: Number of shares to add/remove
        purchase_price: Price per share at purchase (for add)
        purchase_date: Purchase date YYYY-MM-DD format (optional for add)
        notes: Optional notes about position (for add)
        user_id: User identifier (default: "default")
        portfolio_name: Portfolio name (default: "My Portfolio")
        confirm: Safety confirmation for clear action

    Returns:
        Dictionary containing action result.

    Examples:
        # Add a position
        >>> portfolio_manage("add", ticker="AAPL", shares=10, purchase_price=150.50)

        # View portfolio
        >>> portfolio_manage("view")

        # Remove 5 shares
        >>> portfolio_manage("remove", ticker="AAPL", shares=5)

        # Remove entire position
        >>> portfolio_manage("remove", ticker="AAPL")

        # Clear all positions
        >>> portfolio_manage("clear", confirm=True)
    """
    action = action.lower().strip()

    valid_actions = ["add", "remove", "view", "clear"]
    if action not in valid_actions:
        return {
            "error": f"Invalid action '{action}'. Must be one of: {valid_actions}",
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.portfolio import (
            add_portfolio_position,
            clear_my_portfolio,
            get_my_portfolio,
            remove_portfolio_position,
        )

        if action == "add":
            if not ticker or shares is None or purchase_price is None:
                return {
                    "error": "ticker, shares, and purchase_price are required for add action",
                    "hint": "Example: portfolio_manage('add', ticker='AAPL', shares=10, purchase_price=150.50)",
                    "status": "error",
                }
            result = add_portfolio_position(
                ticker=ticker,
                shares=shares,
                purchase_price=purchase_price,
                purchase_date=purchase_date,
                notes=notes,
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["action"] = "add"
            return result

        elif action == "remove":
            if not ticker:
                return {
                    "error": "ticker is required for remove action",
                    "hint": "Example: portfolio_manage('remove', ticker='AAPL', shares=5)",
                    "status": "error",
                }
            result = remove_portfolio_position(
                ticker=ticker,
                shares=shares,  # None means remove entire position
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["action"] = "remove"
            return result

        elif action == "view":
            result = get_my_portfolio(
                user_id=user_id,
                portfolio_name=portfolio_name,
                include_current_prices=True,
            )
            result["action"] = "view"
            return result

        elif action == "clear":
            result = clear_my_portfolio(
                user_id=user_id,
                portfolio_name=portfolio_name,
                confirm=confirm,
            )
            result["action"] = "clear"
            return result

        else:
            return {"error": f"Unknown action: {action}", "status": "error"}

    except Exception as e:
        logger.error(f"Error in portfolio_manage: {e}")
        return {
            "error": str(e),
            "action": action,
            "status": "error",
        }


def portfolio_analyze(
    analysis_type: str = "summary",
    ticker: str | None = None,
    tickers: list[str] | None = None,
    peers: list[str] | None = None,
    days: int = 252,
    years: int = 10,
    peer_count: int = 10,
    risk_level: float = 50.0,
    benchmark: str = "SPY",
    attribution_period: str = "1Y",
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """
    Unified portfolio analysis for risk, correlation, comparison, dividends, peers, and attribution.

    Consolidates portfolio analytics into a single tool with an analysis_type
    parameter. Automatically uses portfolio holdings when no tickers provided.

    Args:
        analysis_type: Type of portfolio analysis:
            - 'summary': Complete portfolio summary with P&L (same as view)
            - 'risk': Risk-adjusted analysis for a specific ticker
            - 'correlation': Correlation matrix of holdings
            - 'compare': Side-by-side comparison of tickers/holdings
            - 'dividend': Dividend analysis (yield, history, growth rates)
            - 'dividend_safety': Dividend safety scoring
            - 'peers': Find and compare to peer companies
            - 'peer_valuation': Detailed valuation comparison vs peers
            - 'attribution': Return attribution (allocation vs selection effect)
            - 'factor_exposure': Portfolio factor loadings (market, size, value, momentum)
            - 'style_analysis': Value/Growth/Size style classification
        ticker: Single ticker for risk/dividend/peer analysis
        tickers: List of tickers for correlation/comparison (uses portfolio if None)
        peers: Optional list of peer tickers for peer_valuation
        days: Analysis period in trading days (default: 252)
        years: Years of history for dividend analysis (default: 10)
        peer_count: Number of peers to compare (default: 10)
        risk_level: Risk tolerance 0-100 for risk analysis (default: 50)
        benchmark: Benchmark for attribution analysis (default: "SPY")
        attribution_period: Period for attribution: 1M, 3M, 6M, 1Y, YTD (default: "1Y")
        user_id: User identifier (default: "default")
        portfolio_name: Portfolio name (default: "My Portfolio")

    Returns:
        Dictionary containing portfolio analysis results.

    Examples:
        # Get portfolio summary
        >>> portfolio_analyze()

        # Risk analysis for specific stock
        >>> portfolio_analyze(analysis_type="risk", ticker="AAPL", risk_level=70)

        # Correlation analysis of portfolio holdings
        >>> portfolio_analyze(analysis_type="correlation")

        # Compare specific tickers
        >>> portfolio_analyze(analysis_type="compare", tickers=["AAPL", "MSFT", "GOOGL"])

        # Dividend analysis
        >>> portfolio_analyze(analysis_type="dividend", ticker="JNJ", years=10)

        # Dividend safety score
        >>> portfolio_analyze(analysis_type="dividend_safety", ticker="KO")

        # Find and compare to peers
        >>> portfolio_analyze(analysis_type="peers", ticker="AAPL")

        # Detailed peer valuation comparison
        >>> portfolio_analyze(analysis_type="peer_valuation", ticker="NVDA", peer_count=10)

        # Return attribution vs SPY
        >>> portfolio_analyze(analysis_type="attribution", benchmark="SPY", attribution_period="1Y")

        # Factor exposure analysis
        >>> portfolio_analyze(analysis_type="factor_exposure")

        # Style analysis
        >>> portfolio_analyze(analysis_type="style_analysis")
    """
    analysis_type = analysis_type.lower().strip()

    valid_types = [
        "summary",
        "risk",
        "correlation",
        "compare",
        "dividend",
        "dividend_safety",
        "peers",
        "peer_valuation",
        "attribution",
        "factor_exposure",
        "style_analysis",
    ]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.portfolio import (
            compare_tickers,
            get_my_portfolio,
            portfolio_correlation_analysis,
            risk_adjusted_analysis,
        )

        if analysis_type == "summary":
            result = get_my_portfolio(
                user_id=user_id,
                portfolio_name=portfolio_name,
                include_current_prices=True,
            )
            result["analysis_type"] = "summary"
            return result

        elif analysis_type == "risk":
            if not ticker:
                return {
                    "error": "ticker is required for risk analysis",
                    "hint": "Example: portfolio_analyze('risk', ticker='AAPL')",
                    "status": "error",
                }
            result = risk_adjusted_analysis(
                ticker=ticker,
                risk_level=risk_level,
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["analysis_type"] = "risk"
            return result

        elif analysis_type == "correlation":
            result = portfolio_correlation_analysis(
                tickers=tickers,  # Uses portfolio if None
                days=days,
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["analysis_type"] = "correlation"
            return result

        elif analysis_type == "compare":
            result = compare_tickers(
                tickers=tickers,  # Uses portfolio if None
                days=min(days, 90),  # Comparison typically shorter period
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["analysis_type"] = "compare"
            return result

        elif analysis_type == "dividend":
            if not ticker:
                return {
                    "error": "ticker is required for dividend analysis",
                    "hint": "Example: portfolio_analyze('dividend', ticker='JNJ')",
                    "status": "error",
                }
            from maverick_mcp.domain.portfolio import get_dividend_analysis

            result = get_dividend_analysis(ticker=ticker, years=years)
            result["analysis_type"] = "dividend"
            return result

        elif analysis_type == "dividend_safety":
            if not ticker:
                return {
                    "error": "ticker is required for dividend safety analysis",
                    "hint": "Example: portfolio_analyze('dividend_safety', ticker='KO')",
                    "status": "error",
                }
            from maverick_mcp.domain.portfolio import get_dividend_safety_score

            result = get_dividend_safety_score(ticker=ticker)
            result["analysis_type"] = "dividend_safety"
            return result

        elif analysis_type == "peers":
            if not ticker:
                return {
                    "error": "ticker is required for peer analysis",
                    "hint": "Example: portfolio_analyze('peers', ticker='AAPL')",
                    "status": "error",
                }
            from maverick_mcp.domain.portfolio import find_peers

            result = find_peers(ticker=ticker, peer_count=peer_count)
            result["analysis_type"] = "peers"
            return result

        elif analysis_type == "peer_valuation":
            if not ticker:
                return {
                    "error": "ticker is required for peer valuation analysis",
                    "hint": "Example: portfolio_analyze('peer_valuation', ticker='NVDA')",
                    "status": "error",
                }
            from maverick_mcp.domain.portfolio import get_peer_comparison

            result = get_peer_comparison(
                ticker=ticker, peers=peers, peer_count=peer_count
            )
            result["analysis_type"] = "peer_valuation"
            return result

        elif analysis_type == "attribution":
            result = _calculate_attribution(
                tickers=tickers,
                benchmark=benchmark,
                attribution_period=attribution_period,
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["analysis_type"] = "attribution"
            return result

        elif analysis_type == "factor_exposure":
            result = _calculate_factor_exposure(
                tickers=tickers,
                days=days,
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["analysis_type"] = "factor_exposure"
            return result

        elif analysis_type == "style_analysis":
            result = _calculate_style_analysis(
                tickers=tickers,
                user_id=user_id,
                portfolio_name=portfolio_name,
            )
            result["analysis_type"] = "style_analysis"
            return result

        else:
            return {"error": f"Unknown analysis_type: {analysis_type}", "status": "error"}

    except Exception as e:
        logger.error(f"Error in portfolio_analyze: {e}")
        return {
            "error": str(e),
            "analysis_type": analysis_type,
            "status": "error",
        }


def _get_portfolio_holdings(
    user_id: str, portfolio_name: str, tickers: list[str] | None
) -> tuple[list[str], dict[str, float]]:
    """Get portfolio holdings and weights, or use provided tickers with equal weights."""
    import yfinance as yf

    from maverick_mcp.api.routers.portfolio import get_my_portfolio

    if tickers:
        # Use provided tickers with equal weights
        weights = {t.upper(): 1.0 / len(tickers) for t in tickers}
        return [t.upper() for t in tickers], weights

    # Get portfolio holdings
    portfolio = get_my_portfolio(
        user_id=user_id,
        portfolio_name=portfolio_name,
        include_current_prices=True,
    )

    holdings = portfolio.get("positions", [])
    if not holdings:
        return [], {}

    # Calculate weights based on current value
    total_value = sum(p.get("current_value", 0) for p in holdings)
    if total_value == 0:
        # Equal weight if no values
        weights = {p["ticker"]: 1.0 / len(holdings) for p in holdings}
    else:
        weights = {
            p["ticker"]: p.get("current_value", 0) / total_value for p in holdings
        }

    tickers_list = [p["ticker"] for p in holdings]
    return tickers_list, weights


def _calculate_attribution(
    tickers: list[str] | None,
    benchmark: str,
    attribution_period: str,
    user_id: str,
    portfolio_name: str,
) -> dict[str, Any]:
    """Calculate return attribution vs benchmark."""
    import yfinance as yf

    try:
        # Get holdings
        holding_tickers, weights = _get_portfolio_holdings(
            user_id, portfolio_name, tickers
        )

        if not holding_tickers:
            return {
                "error": "No portfolio holdings found. Add positions or provide tickers.",
                "status": "error",
            }

        # Determine period
        period_map = {
            "1M": 21,
            "3M": 63,
            "6M": 126,
            "1Y": 252,
            "YTD": None,  # Calculate dynamically
        }

        if attribution_period.upper() == "YTD":
            year_start = datetime(datetime.now().year, 1, 1)
            days = (datetime.now() - year_start).days
        else:
            days = period_map.get(attribution_period.upper(), 252)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer

        # Download data
        all_tickers = holding_tickers + [benchmark]
        data = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
        )

        if data.empty:
            return {"error": "Could not fetch price data", "status": "error"}

        # Handle single vs multiple ticker response
        if len(all_tickers) == 2:
            # Single holding + benchmark
            prices = data["Close"]
        else:
            prices = data["Close"]

        # Calculate returns
        returns = prices.pct_change().dropna().tail(days)

        # Portfolio return
        portfolio_return = 0.0
        position_returns = {}

        for ticker, weight in weights.items():
            if ticker in returns.columns:
                ticker_total_return = (1 + returns[ticker]).prod() - 1
                weighted_return = ticker_total_return * weight
                portfolio_return += weighted_return
                position_returns[ticker] = {
                    "weight": round(weight * 100, 2),
                    "return": round(ticker_total_return * 100, 2),
                    "contribution": round(weighted_return * 100, 2),
                }

        # Benchmark return
        benchmark_return = 0.0
        if benchmark in returns.columns:
            benchmark_return = (1 + returns[benchmark]).prod() - 1

        # Calculate attribution effects
        excess_return = portfolio_return - benchmark_return

        # Simplified attribution: allocation + selection = excess
        # Allocation effect: being in different sectors than benchmark
        # Selection effect: picking better stocks within sectors

        # Without sector data, estimate based on correlation
        if benchmark in returns.columns:
            portfolio_returns_series = sum(
                returns[t] * weights.get(t, 0)
                for t in holding_tickers
                if t in returns.columns
            )

            if hasattr(portfolio_returns_series, "corr"):
                correlation = portfolio_returns_series.corr(returns[benchmark])
            else:
                correlation = 0.8  # Default estimate

            # Higher correlation = more selection effect (same sectors, better picks)
            # Lower correlation = more allocation effect (different sectors)
            selection_estimate = excess_return * correlation
            allocation_estimate = excess_return * (1 - correlation)
        else:
            selection_estimate = excess_return * 0.5
            allocation_estimate = excess_return * 0.5

        return {
            "portfolio_return": round(portfolio_return * 100, 2),
            "benchmark_return": round(benchmark_return * 100, 2),
            "excess_return": round(excess_return * 100, 2),
            "benchmark": benchmark,
            "period": attribution_period,
            "attribution": {
                "allocation_effect": round(allocation_estimate * 100, 2),
                "selection_effect": round(selection_estimate * 100, 2),
            },
            "position_contributions": position_returns,
            "top_contributors": sorted(
                position_returns.items(),
                key=lambda x: x[1]["contribution"],
                reverse=True,
            )[:5],
            "interpretation": (
                f"Portfolio returned {portfolio_return*100:.1f}% vs {benchmark} {benchmark_return*100:.1f}% "
                f"({excess_return*100:+.1f}% excess). "
                f"Allocation effect: {allocation_estimate*100:+.1f}%, "
                f"Selection effect: {selection_estimate*100:+.1f}%."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error calculating attribution: {e}")
        return {"error": str(e), "status": "error"}


def _calculate_factor_exposure(
    tickers: list[str] | None,
    days: int,
    user_id: str,
    portfolio_name: str,
) -> dict[str, Any]:
    """Calculate portfolio factor exposures (market, size, value, momentum)."""
    import yfinance as yf

    try:
        # Get holdings
        holding_tickers, weights = _get_portfolio_holdings(
            user_id, portfolio_name, tickers
        )

        if not holding_tickers:
            return {
                "error": "No portfolio holdings found. Add positions or provide tickers.",
                "status": "error",
            }

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)

        # Factor proxies (ETFs)
        factor_proxies = {
            "market": "SPY",    # Market factor
            "size": "IWM",      # Small cap (vs SPY for size factor)
            "value": "IWD",     # Large cap value
            "growth": "IWF",    # Large cap growth
            "momentum": "MTUM", # Momentum factor
            "quality": "QUAL",  # Quality factor
        }

        # Download portfolio and factor data
        all_tickers = holding_tickers + list(factor_proxies.values())
        data = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
        )

        if data.empty:
            return {"error": "Could not fetch price data", "status": "error"}

        prices = data["Close"]
        returns = prices.pct_change().dropna().tail(days)

        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for ticker, weight in weights.items():
            if ticker in returns.columns:
                portfolio_returns += returns[ticker] * weight

        # Calculate factor betas using simple regression
        factor_exposures = {}

        for factor_name, factor_ticker in factor_proxies.items():
            if factor_ticker in returns.columns:
                factor_returns = returns[factor_ticker]

                # Simple beta calculation
                covariance = portfolio_returns.cov(factor_returns)
                variance = factor_returns.var()

                if variance > 0:
                    beta = covariance / variance
                    correlation = portfolio_returns.corr(factor_returns)

                    factor_exposures[factor_name] = {
                        "beta": round(beta, 3),
                        "correlation": round(correlation, 3),
                        "proxy": factor_ticker,
                    }

        # Determine dominant factor
        dominant_factor = max(
            factor_exposures.items(),
            key=lambda x: abs(x[1].get("beta", 0)),
        ) if factor_exposures else ("unknown", {"beta": 0})

        # Calculate factor tilts
        value_tilt = None
        size_tilt = None

        if "value" in factor_exposures and "growth" in factor_exposures:
            value_beta = factor_exposures["value"]["beta"]
            growth_beta = factor_exposures["growth"]["beta"]
            if value_beta > growth_beta + 0.1:
                value_tilt = "value"
            elif growth_beta > value_beta + 0.1:
                value_tilt = "growth"
            else:
                value_tilt = "blend"

        if "size" in factor_exposures:
            size_beta = factor_exposures["size"]["beta"]
            if size_beta > 1.1:
                size_tilt = "small_cap"
            elif size_beta < 0.9:
                size_tilt = "large_cap"
            else:
                size_tilt = "mid_cap"

        return {
            "holdings_count": len(holding_tickers),
            "analysis_period_days": days,
            "factor_exposures": factor_exposures,
            "factor_tilts": {
                "value_growth": value_tilt,
                "size": size_tilt,
            },
            "dominant_factor": {
                "name": dominant_factor[0],
                "beta": dominant_factor[1].get("beta", 0),
            },
            "interpretation": (
                f"Portfolio has {dominant_factor[0]} beta of {dominant_factor[1].get('beta', 0):.2f}. "
                f"Style tilt: {value_tilt or 'unknown'}, Size tilt: {size_tilt or 'unknown'}."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error calculating factor exposure: {e}")
        return {"error": str(e), "status": "error"}


def _calculate_style_analysis(
    tickers: list[str] | None,
    user_id: str,
    portfolio_name: str,
) -> dict[str, Any]:
    """Analyze portfolio style characteristics (value/growth/size)."""
    import yfinance as yf

    try:
        # Get holdings
        holding_tickers, weights = _get_portfolio_holdings(
            user_id, portfolio_name, tickers
        )

        if not holding_tickers:
            return {
                "error": "No portfolio holdings found. Add positions or provide tickers.",
                "status": "error",
            }

        # Get fundamental data for each holding
        style_metrics = []

        for ticker in holding_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                pe_ratio = info.get("forwardPE") or info.get("trailingPE")
                pb_ratio = info.get("priceToBook")
                market_cap = info.get("marketCap")
                earnings_growth = info.get("earningsGrowth")
                revenue_growth = info.get("revenueGrowth")
                dividend_yield = info.get("dividendYield")

                style_metrics.append({
                    "ticker": ticker,
                    "weight": weights.get(ticker, 0),
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "market_cap": market_cap,
                    "earnings_growth": earnings_growth,
                    "revenue_growth": revenue_growth,
                    "dividend_yield": dividend_yield,
                })
            except Exception as e:
                logger.warning(f"Could not get style metrics for {ticker}: {e}")

        if not style_metrics:
            return {
                "error": "Could not retrieve style metrics for any holdings",
                "status": "error",
            }

        # Calculate weighted averages
        total_weight = sum(m["weight"] for m in style_metrics)
        if total_weight == 0:
            total_weight = 1

        weighted_pe = sum(
            (m["pe_ratio"] or 0) * m["weight"]
            for m in style_metrics
            if m["pe_ratio"]
        ) / sum(m["weight"] for m in style_metrics if m["pe_ratio"]) if any(m["pe_ratio"] for m in style_metrics) else None

        weighted_pb = sum(
            (m["pb_ratio"] or 0) * m["weight"]
            for m in style_metrics
            if m["pb_ratio"]
        ) / sum(m["weight"] for m in style_metrics if m["pb_ratio"]) if any(m["pb_ratio"] for m in style_metrics) else None

        weighted_growth = sum(
            (m["earnings_growth"] or 0) * m["weight"]
            for m in style_metrics
            if m["earnings_growth"]
        ) / sum(m["weight"] for m in style_metrics if m["earnings_growth"]) if any(m["earnings_growth"] for m in style_metrics) else None

        # Classify market cap exposure
        cap_exposure = {"large": 0, "mid": 0, "small": 0}
        for m in style_metrics:
            mc = m.get("market_cap")
            w = m.get("weight", 0)
            if mc:
                if mc >= 10_000_000_000:  # $10B+
                    cap_exposure["large"] += w
                elif mc >= 2_000_000_000:  # $2B-$10B
                    cap_exposure["mid"] += w
                else:
                    cap_exposure["small"] += w

        # Determine style classification
        value_growth_style = "blend"
        if weighted_pe and weighted_growth:
            if weighted_pe < 15 and (weighted_growth or 0) < 0.15:
                value_growth_style = "deep_value"
            elif weighted_pe < 20:
                value_growth_style = "value"
            elif weighted_pe > 30 and (weighted_growth or 0) > 0.2:
                value_growth_style = "aggressive_growth"
            elif weighted_pe > 25:
                value_growth_style = "growth"
            else:
                value_growth_style = "blend"

        # Determine size classification
        size_style = "large_cap"
        if cap_exposure["large"] >= 0.7:
            size_style = "large_cap"
        elif cap_exposure["small"] >= 0.5:
            size_style = "small_cap"
        elif cap_exposure["mid"] >= 0.4:
            size_style = "mid_cap"
        else:
            size_style = "all_cap"

        # Style box position (1-9 grid)
        style_box = _get_style_box_position(value_growth_style, size_style)

        return {
            "holdings_analyzed": len(style_metrics),
            "portfolio_metrics": {
                "weighted_pe": round(weighted_pe, 2) if weighted_pe else None,
                "weighted_pb": round(weighted_pb, 2) if weighted_pb else None,
                "weighted_earnings_growth": (
                    round(weighted_growth * 100, 1) if weighted_growth else None
                ),
            },
            "style_classification": {
                "value_growth": value_growth_style,
                "size": size_style,
                "style_box": style_box,
            },
            "cap_exposure": {
                "large_cap_pct": round(cap_exposure["large"] * 100, 1),
                "mid_cap_pct": round(cap_exposure["mid"] * 100, 1),
                "small_cap_pct": round(cap_exposure["small"] * 100, 1),
            },
            "holdings_detail": sorted(
                style_metrics,
                key=lambda x: x["weight"],
                reverse=True,
            )[:10],
            "interpretation": (
                f"Portfolio style: {size_style.replace('_', ' ')} {value_growth_style}. "
                f"Weighted P/E: {weighted_pe:.1f}x, "
                f"Growth: {weighted_growth*100:.1f}%."
                if weighted_pe and weighted_growth
                else f"Portfolio style: {size_style.replace('_', ' ')} {value_growth_style}."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error calculating style analysis: {e}")
        return {"error": str(e), "status": "error"}


def _get_style_box_position(value_growth: str, size: str) -> dict[str, Any]:
    """Map style to Morningstar-style 3x3 grid position."""
    # Value/Growth axis (1=Value, 2=Blend, 3=Growth)
    vg_map = {
        "deep_value": 1,
        "value": 1,
        "blend": 2,
        "growth": 3,
        "aggressive_growth": 3,
    }

    # Size axis (1=Large, 2=Mid, 3=Small)
    size_map = {
        "large_cap": 1,
        "mid_cap": 2,
        "small_cap": 3,
        "all_cap": 2,
    }

    vg_pos = vg_map.get(value_growth, 2)
    size_pos = size_map.get(size, 1)

    # Grid position (1-9)
    grid_position = (size_pos - 1) * 3 + vg_pos

    labels = {
        1: "Large Value", 2: "Large Blend", 3: "Large Growth",
        4: "Mid Value", 5: "Mid Blend", 6: "Mid Growth",
        7: "Small Value", 8: "Small Blend", 9: "Small Growth",
    }

    return {
        "row": size_pos,
        "column": vg_pos,
        "position": grid_position,
        "label": labels.get(grid_position, "Unknown"),
    }
