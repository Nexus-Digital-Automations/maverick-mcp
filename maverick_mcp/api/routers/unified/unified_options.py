"""
Unified Options Analysis Tool.

Consolidates 7 options analysis tools into 1 unified interface:
- Greeks calculation (delta, gamma, theta, vega, rho)
- Implied volatility surface
- IV percentile/rank
- Volatility skew
- Put/call ratio
- Strategy analysis
- Strategy templates (covered call, iron condor, spreads, etc.)
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Strategy template definitions
STRATEGY_TEMPLATES = {
    "covered_call": {
        "description": "Long stock + sell OTM call for income",
        "legs": [
            {"type": "stock", "action": "buy", "strike_offset": 0},
            {"type": "call", "action": "sell", "strike_offset": 0.05},  # 5% OTM
        ],
        "risk_profile": "limited_upside",
    },
    "protective_put": {
        "description": "Long stock + buy OTM put for protection",
        "legs": [
            {"type": "stock", "action": "buy", "strike_offset": 0},
            {"type": "put", "action": "buy", "strike_offset": -0.05},  # 5% OTM
        ],
        "risk_profile": "limited_downside",
    },
    "collar": {
        "description": "Long stock + buy put + sell call (range-bound protection)",
        "legs": [
            {"type": "stock", "action": "buy", "strike_offset": 0},
            {"type": "put", "action": "buy", "strike_offset": -0.05},
            {"type": "call", "action": "sell", "strike_offset": 0.05},
        ],
        "risk_profile": "range_bound",
    },
    "bull_call_spread": {
        "description": "Buy lower strike call + sell higher strike call",
        "legs": [
            {"type": "call", "action": "buy", "strike_offset": -0.02},  # ITM/ATM
            {"type": "call", "action": "sell", "strike_offset": 0.05},  # OTM
        ],
        "risk_profile": "limited_risk_reward",
    },
    "bear_put_spread": {
        "description": "Buy higher strike put + sell lower strike put",
        "legs": [
            {"type": "put", "action": "buy", "strike_offset": 0.02},  # ITM/ATM
            {"type": "put", "action": "sell", "strike_offset": -0.05},  # OTM
        ],
        "risk_profile": "limited_risk_reward",
    },
    "iron_condor": {
        "description": "Sell OTM put spread + sell OTM call spread",
        "legs": [
            {"type": "put", "action": "buy", "strike_offset": -0.10},  # Far OTM put
            {"type": "put", "action": "sell", "strike_offset": -0.05},  # Near OTM put
            {"type": "call", "action": "sell", "strike_offset": 0.05},  # Near OTM call
            {"type": "call", "action": "buy", "strike_offset": 0.10},  # Far OTM call
        ],
        "risk_profile": "range_bound",
    },
    "iron_butterfly": {
        "description": "Sell ATM straddle + buy OTM strangle (tighter range)",
        "legs": [
            {"type": "put", "action": "buy", "strike_offset": -0.05},
            {"type": "put", "action": "sell", "strike_offset": 0},
            {"type": "call", "action": "sell", "strike_offset": 0},
            {"type": "call", "action": "buy", "strike_offset": 0.05},
        ],
        "risk_profile": "range_bound",
    },
    "straddle": {
        "description": "Buy ATM call + buy ATM put (volatility play)",
        "legs": [
            {"type": "call", "action": "buy", "strike_offset": 0},
            {"type": "put", "action": "buy", "strike_offset": 0},
        ],
        "risk_profile": "unlimited_reward",
    },
    "strangle": {
        "description": "Buy OTM call + buy OTM put (cheaper volatility play)",
        "legs": [
            {"type": "call", "action": "buy", "strike_offset": 0.05},
            {"type": "put", "action": "buy", "strike_offset": -0.05},
        ],
        "risk_profile": "unlimited_reward",
    },
    "calendar_spread": {
        "description": "Sell near-term call + buy far-term call at same strike",
        "legs": [
            {"type": "call", "action": "sell", "strike_offset": 0, "expiry": "near"},
            {"type": "call", "action": "buy", "strike_offset": 0, "expiry": "far"},
        ],
        "risk_profile": "time_decay",
    },
}


async def options_analysis(
    symbol: str,
    analysis_type: str = "overview",
    expiration_date: str | None = None,
    strike: float | None = None,
    option_type: str = "call",
    strategy_legs: list[dict] | None = None,
    strategy_name: str | None = None,
    target_days_to_expiry: int = 30,
    price_range_pct: float = 10.0,
) -> dict[str, Any]:
    """
    Unified options analytics including Greeks, IV, strategies, and templates.

    Consolidates Greeks calculation, IV surface, IV percentile, skew analysis,
    put/call ratio, strategy analysis, and strategy templates into a single tool.

    Args:
        symbol: Underlying ticker symbol (e.g., 'AAPL', 'SPY')
        analysis_type: Type of options analysis:
            - 'greeks': Delta, gamma, theta, vega, rho calculation
            - 'iv_surface': 3D implied volatility surface visualization
            - 'iv_percentile': Historical IV rank/percentile
            - 'skew': Put/call IV skew analysis
            - 'put_call_ratio': Market sentiment indicator from P/C ratio
            - 'strategy': Multi-leg strategy P&L analysis
            - 'strategy_template': Build a predefined strategy (iron condor, etc.)
            - 'overview': Summary of key options metrics (default)
        expiration_date: Expiration date for Greeks (YYYY-MM-DD format)
        strike: Strike price for Greeks calculation
        option_type: Option type for Greeks ('call' or 'put')
        strategy_legs: List of option legs for strategy analysis
            Each leg: {"type": "call"|"put", "strike": float, "expiration": str, "action": "buy"|"sell"}
        strategy_name: For strategy_template, one of:
            - 'covered_call': Long stock + sell OTM call
            - 'protective_put': Long stock + buy put for protection
            - 'collar': Long stock + buy put + sell call
            - 'bull_call_spread': Buy lower strike call + sell higher strike call
            - 'bear_put_spread': Buy higher strike put + sell lower strike put
            - 'iron_condor': Sell OTM put spread + sell OTM call spread
            - 'iron_butterfly': Sell ATM straddle + buy OTM strangle
            - 'straddle': Buy ATM call + buy ATM put
            - 'strangle': Buy OTM call + buy OTM put
            - 'calendar_spread': Sell near-term + buy far-term at same strike
        target_days_to_expiry: Target days to expiration (default 30)
        price_range_pct: P&L grid range as percentage (default 10%)

    Returns:
        Dictionary containing options analysis results.

    Examples:
        # Get Greeks for a specific option
        >>> options_analysis("AAPL", analysis_type="greeks", expiration_date="2024-12-20", strike=180)

        # IV surface analysis
        >>> options_analysis("SPY", analysis_type="iv_surface")

        # Build an iron condor on SPY
        >>> options_analysis("SPY", analysis_type="strategy_template", strategy_name="iron_condor")

        # Build a covered call with 45 DTE
        >>> options_analysis("AAPL", analysis_type="strategy_template", strategy_name="covered_call", target_days_to_expiry=45)
    """
    symbol = symbol.strip().upper()
    analysis_type = analysis_type.lower().strip()

    valid_types = [
        "greeks", "iv_surface", "iv_percentile", "skew",
        "put_call_ratio", "strategy", "strategy_template", "overview"
    ]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        from maverick_mcp.api.routers.options_analysis import (
            options_analyze_iv_surface,
            options_analyze_skew,
            options_calculate_greeks,
            options_calculate_iv_percentile,
            options_calculate_put_call_ratio,
            options_strategy_analyzer,
        )

        if analysis_type == "greeks":
            if not expiration_date or not strike:
                return {
                    "error": "expiration_date and strike are required for Greeks calculation",
                    "hint": "Use analysis_type='overview' to see available expirations first",
                    "symbol": symbol,
                    "status": "error",
                }
            result = await options_calculate_greeks(
                symbol=symbol,
                expiration_date=expiration_date,
                strike=strike,
                option_type=option_type,
            )
            result["analysis_type"] = "greeks"
            return result

        elif analysis_type == "iv_surface":
            result = await options_analyze_iv_surface(symbol=symbol)
            result["analysis_type"] = "iv_surface"
            return result

        elif analysis_type == "iv_percentile":
            result = await options_calculate_iv_percentile(symbol=symbol)
            result["analysis_type"] = "iv_percentile"
            return result

        elif analysis_type == "skew":
            result = await options_analyze_skew(symbol=symbol)
            result["analysis_type"] = "skew"
            return result

        elif analysis_type == "put_call_ratio":
            result = await options_calculate_put_call_ratio(symbol=symbol)
            result["analysis_type"] = "put_call_ratio"
            return result

        elif analysis_type == "strategy":
            if not strategy_legs:
                return {
                    "error": "strategy_legs is required for strategy analysis",
                    "hint": "Provide list of legs: [{'type': 'call', 'strike': 180, 'expiration': '2024-12-20', 'action': 'buy'}]",
                    "symbol": symbol,
                    "status": "error",
                }
            result = await options_strategy_analyzer(
                symbol=symbol,
                legs=strategy_legs,
            )
            result["analysis_type"] = "strategy"
            return result

        elif analysis_type == "strategy_template":
            # Build predefined strategy from template
            result = await _build_strategy_template(
                symbol=symbol,
                strategy_name=strategy_name,
                target_days_to_expiry=target_days_to_expiry,
                price_range_pct=price_range_pct,
            )
            return result

        else:  # overview
            # Combine key metrics
            iv_result = await options_calculate_iv_percentile(symbol=symbol)
            pcr_result = await options_calculate_put_call_ratio(symbol=symbol)
            skew_result = await options_analyze_skew(symbol=symbol)

            return {
                "symbol": symbol,
                "analysis_type": "overview",
                "iv_metrics": {
                    "current_iv": iv_result.get("current_iv", "N/A"),
                    "iv_percentile": iv_result.get("iv_percentile", "N/A"),
                    "iv_rank": iv_result.get("iv_rank", "N/A"),
                },
                "sentiment": {
                    "put_call_ratio": pcr_result.get("put_call_ratio", "N/A"),
                    "interpretation": pcr_result.get("interpretation", "N/A"),
                },
                "skew": {
                    "skew_value": skew_result.get("skew", "N/A"),
                    "interpretation": skew_result.get("interpretation", "N/A"),
                },
                "interpretation": (
                    f"IV at {iv_result.get('iv_percentile', 'N/A')} percentile. "
                    f"P/C ratio: {pcr_result.get('put_call_ratio', 'N/A')}. "
                    f"Skew: {skew_result.get('interpretation', 'N/A')}"
                ),
                "status": "success",
            }

    except Exception as e:
        logger.error(f"Error in options_analysis for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }


async def _build_strategy_template(
    symbol: str,
    strategy_name: str | None,
    target_days_to_expiry: int,
    price_range_pct: float,
) -> dict[str, Any]:
    """Build a predefined options strategy from template."""
    import yfinance as yf
    from scipy.stats import norm

    # Validate strategy name
    if not strategy_name:
        return {
            "error": "strategy_name is required for strategy_template",
            "available_strategies": list(STRATEGY_TEMPLATES.keys()),
            "hint": "Example: options_analysis('SPY', 'strategy_template', strategy_name='iron_condor')",
            "symbol": symbol,
            "status": "error",
        }

    strategy_name = strategy_name.lower().strip()
    if strategy_name not in STRATEGY_TEMPLATES:
        return {
            "error": f"Unknown strategy '{strategy_name}'",
            "available_strategies": list(STRATEGY_TEMPLATES.keys()),
            "symbol": symbol,
            "status": "error",
        }

    template = STRATEGY_TEMPLATES[strategy_name]

    try:
        # Get stock data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            return {
                "error": f"Could not get current price for {symbol}",
                "symbol": symbol,
                "status": "error",
            }

        current_price = float(hist["Close"].iloc[-1])

        # Get options chain and find suitable expiration
        try:
            expirations = ticker.options
            if not expirations:
                return {
                    "error": f"No options available for {symbol}",
                    "symbol": symbol,
                    "status": "error",
                }
        except Exception:
            return {
                "error": f"Could not get options chain for {symbol}",
                "symbol": symbol,
                "status": "error",
            }

        # Find expiration closest to target DTE
        target_date = datetime.now() + timedelta(days=target_days_to_expiry)
        near_exp = None
        far_exp = None
        min_diff = float("inf")

        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            diff = abs((exp_date - target_date).days)
            if diff < min_diff:
                min_diff = diff
                near_exp = exp

        # For calendar spreads, find a farther expiration
        if strategy_name == "calendar_spread":
            far_target = datetime.now() + timedelta(days=target_days_to_expiry * 2)
            min_diff = float("inf")
            for exp in expirations:
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                diff = abs((exp_date - far_target).days)
                if diff < min_diff and exp != near_exp:
                    min_diff = diff
                    far_exp = exp

        if not near_exp:
            return {
                "error": "Could not find suitable expiration date",
                "symbol": symbol,
                "status": "error",
            }

        # Get IV estimate from options chain
        try:
            chain = ticker.option_chain(near_exp)
            atm_calls = chain.calls[
                abs(chain.calls["strike"] - current_price) < current_price * 0.05
            ]
            if not atm_calls.empty and "impliedVolatility" in atm_calls.columns:
                iv = float(atm_calls["impliedVolatility"].mean())
            else:
                iv = 0.25  # Default IV estimate
        except Exception:
            iv = 0.25

        # Calculate days to expiration
        exp_date = datetime.strptime(near_exp, "%Y-%m-%d")
        days_to_exp = max((exp_date - datetime.now()).days, 1)
        time_to_exp = days_to_exp / 365.0
        risk_free_rate = 0.05  # Assume 5% risk-free rate

        # Build strategy legs
        built_legs = []
        total_premium = 0.0
        total_cost = 0.0

        for leg_template in template["legs"]:
            leg_type = leg_template["type"]
            action = leg_template["action"]
            strike_offset = leg_template.get("strike_offset", 0)
            expiry_type = leg_template.get("expiry", "near")

            # Calculate strike price
            strike_price = round(current_price * (1 + strike_offset), 2)

            # Find nearest available strike
            try:
                exp_to_use = far_exp if expiry_type == "far" and far_exp else near_exp
                chain = ticker.option_chain(exp_to_use)
                if leg_type == "call":
                    available_strikes = chain.calls["strike"].values
                elif leg_type == "put":
                    available_strikes = chain.puts["strike"].values
                else:  # stock
                    available_strikes = [current_price]

                if len(available_strikes) > 0:
                    strike_price = float(
                        available_strikes[
                            np.argmin(np.abs(available_strikes - strike_price))
                        ]
                    )
            except Exception:
                pass  # Use calculated strike if chain unavailable

            # Calculate theoretical option price using simplified Black-Scholes
            if leg_type in ["call", "put"]:
                d1 = (
                    np.log(current_price / strike_price)
                    + (risk_free_rate + 0.5 * iv**2) * time_to_exp
                ) / (iv * np.sqrt(time_to_exp))
                d2 = d1 - iv * np.sqrt(time_to_exp)

                if leg_type == "call":
                    price = current_price * norm.cdf(d1) - strike_price * np.exp(
                        -risk_free_rate * time_to_exp
                    ) * norm.cdf(d2)
                    delta = norm.cdf(d1)
                else:  # put
                    price = strike_price * np.exp(
                        -risk_free_rate * time_to_exp
                    ) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
                    delta = norm.cdf(d1) - 1

                price = max(price, 0.01)  # Minimum option price
            else:  # stock
                price = current_price
                delta = 1.0

            # Track premium/cost
            if action == "buy":
                total_cost += price * 100  # Options in 100-share lots
                total_premium -= price
            else:
                total_cost -= price * 100
                total_premium += price

            built_legs.append({
                "type": leg_type,
                "action": action,
                "strike": strike_price,
                "expiration": exp_to_use if leg_type != "stock" else None,
                "theoretical_price": round(price, 2),
                "delta": round(delta, 3) if leg_type != "stock" else 1.0,
            })

        # Calculate P&L at various price points
        price_points = np.linspace(
            current_price * (1 - price_range_pct / 100),
            current_price * (1 + price_range_pct / 100),
            21,
        )

        pl_grid = []
        for price_point in price_points:
            pl = 0.0
            for leg in built_legs:
                leg_type = leg["type"]
                action = leg["action"]
                strike = leg["strike"]
                premium = leg["theoretical_price"]

                if leg_type == "call":
                    intrinsic = max(price_point - strike, 0)
                    if action == "buy":
                        pl += (intrinsic - premium) * 100
                    else:
                        pl += (premium - intrinsic) * 100
                elif leg_type == "put":
                    intrinsic = max(strike - price_point, 0)
                    if action == "buy":
                        pl += (intrinsic - premium) * 100
                    else:
                        pl += (premium - intrinsic) * 100
                else:  # stock
                    if action == "buy":
                        pl += (price_point - premium) * 100
                    else:
                        pl += (premium - price_point) * 100

            pl_grid.append({
                "underlying_price": round(price_point, 2),
                "profit_loss": round(pl, 2),
            })

        # Calculate max profit, max loss, breakevens
        pls = [p["profit_loss"] for p in pl_grid]
        max_profit = max(pls)
        max_loss = min(pls)

        # Find breakeven points (where P&L crosses zero)
        breakevens = []
        for i in range(len(pl_grid) - 1):
            if (pl_grid[i]["profit_loss"] * pl_grid[i + 1]["profit_loss"]) < 0:
                # Linear interpolation to find zero crossing
                p1, pl1 = pl_grid[i]["underlying_price"], pl_grid[i]["profit_loss"]
                p2, pl2 = pl_grid[i + 1]["underlying_price"], pl_grid[i + 1]["profit_loss"]
                be = p1 - pl1 * (p2 - p1) / (pl2 - pl1)
                breakevens.append(round(be, 2))

        # Estimate probability of profit (simplified using normal distribution)
        prob_of_profit = 0.5  # Default
        if breakevens:
            # Calculate probability of being within profitable range
            if max_profit > 0 and max_loss < 0:
                # Use first and last breakeven for range-bound strategies
                if len(breakevens) >= 2:
                    lower_be, upper_be = min(breakevens), max(breakevens)
                    # Standard normal probability within range
                    annual_std = current_price * iv
                    daily_std = annual_std / np.sqrt(252) * np.sqrt(days_to_exp)
                    z_lower = (lower_be - current_price) / daily_std
                    z_upper = (upper_be - current_price) / daily_std
                    prob_of_profit = float(norm.cdf(z_upper) - norm.cdf(z_lower))
                elif len(breakevens) == 1:
                    be = breakevens[0]
                    annual_std = current_price * iv
                    daily_std = annual_std / np.sqrt(252) * np.sqrt(days_to_exp)
                    z = (be - current_price) / daily_std
                    # Determine direction based on P&L slope
                    if pl_grid[-1]["profit_loss"] > pl_grid[0]["profit_loss"]:
                        prob_of_profit = 1 - float(norm.cdf(z))
                    else:
                        prob_of_profit = float(norm.cdf(z))

        return {
            "symbol": symbol,
            "analysis_type": "strategy_template",
            "strategy_name": strategy_name,
            "description": template["description"],
            "risk_profile": template["risk_profile"],
            "current_price": round(current_price, 2),
            "expiration": near_exp,
            "days_to_expiry": days_to_exp,
            "implied_volatility": round(iv * 100, 1),
            "legs": built_legs,
            "summary": {
                "net_premium": round(total_premium * 100, 2),
                "max_profit": round(max_profit, 2),
                "max_loss": round(max_loss, 2),
                "breakeven_points": breakevens,
                "probability_of_profit": round(prob_of_profit * 100, 1),
                "risk_reward_ratio": (
                    round(abs(max_profit / max_loss), 2)
                    if max_loss != 0
                    else float("inf")
                ),
            },
            "pl_grid": pl_grid,
            "interpretation": (
                f"{strategy_name.replace('_', ' ').title()}: "
                f"Max profit ${max_profit:.0f}, Max loss ${max_loss:.0f}. "
                f"Breakeven(s): {breakevens}. "
                f"Estimated {prob_of_profit*100:.0f}% probability of profit."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error building strategy template for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "strategy_name": strategy_name,
            "status": "error",
        }
