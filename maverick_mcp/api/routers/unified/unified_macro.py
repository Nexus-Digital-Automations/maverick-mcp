"""
Unified Macro Analysis Tool.

Provides comprehensive macroeconomic analysis:
- yield_curve: Treasury curve shape, inversion detection
- fed_funds: Market-implied Fed rate expectations
- macro_regime: Bull/bear/recession regime classification
- market_cycle: Credit/business cycle indicators
- comprehensive: Full macro dashboard (default)
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(val) -> float:
    """Safely extract scalar float from pandas/numpy objects."""
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


# FRED series IDs for economic data
FRED_SERIES = {
    "T10Y2Y": "10-Year minus 2-Year Treasury Spread",
    "T10Y3M": "10-Year minus 3-Month Treasury Spread",
    "DGS2": "2-Year Treasury Rate",
    "DGS10": "10-Year Treasury Rate",
    "DGS30": "30-Year Treasury Rate",
    "FEDFUNDS": "Federal Funds Rate",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "INDPRO": "Industrial Production Index",
    "PAYEMS": "Non-Farm Payrolls",
    "UMCSENT": "Consumer Sentiment",
    "BAMLH0A0HYM2": "High Yield Spread",
}


async def macro_analysis(
    analysis_type: str = "comprehensive",
    lookback_days: int = 252,
    include_forecast: bool = True,
) -> dict[str, Any]:
    """
    Comprehensive macroeconomic analysis for market context.

    Provides yield curve analysis, Fed funds expectations, macro regime
    classification, and business cycle indicators.

    Args:
        analysis_type: Type of macro analysis:
            - 'yield_curve': Treasury curve shape, inversion detection
            - 'fed_funds': Market-implied Fed rate expectations
            - 'macro_regime': Bull/bear/recession regime classification
            - 'market_cycle': Credit/business cycle indicators
            - 'comprehensive': Full macro dashboard (default)
        lookback_days: Days of historical data to analyze (default 252)
        include_forecast: Include forward-looking estimates (default True)

    Returns:
        Dictionary containing macro analysis results.

    Examples:
        # Get comprehensive macro analysis
        >>> macro_analysis()

        # Check if yield curve is inverted
        >>> macro_analysis(analysis_type="yield_curve")

        # Get market regime classification
        >>> macro_analysis(analysis_type="macro_regime")
    """
    analysis_type = analysis_type.lower().strip()

    valid_types = ["yield_curve", "fed_funds", "macro_regime", "market_cycle", "comprehensive"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "status": "error",
        }

    try:
        if analysis_type == "yield_curve":
            return await _analyze_yield_curve(lookback_days)

        elif analysis_type == "fed_funds":
            return await _analyze_fed_funds(lookback_days, include_forecast)

        elif analysis_type == "macro_regime":
            return await _analyze_macro_regime(lookback_days)

        elif analysis_type == "market_cycle":
            return await _analyze_market_cycle(lookback_days)

        else:  # comprehensive
            yield_curve = await _analyze_yield_curve(lookback_days)
            fed_funds = await _analyze_fed_funds(lookback_days, include_forecast)
            regime = await _analyze_macro_regime(lookback_days)
            cycle = await _analyze_market_cycle(lookback_days)

            # Determine overall macro environment
            signals = []
            if yield_curve.get("is_inverted"):
                signals.append("yield_curve_inverted")
            if regime.get("regime") == "risk_off":
                signals.append("risk_off_regime")
            if cycle.get("cycle_phase") == "contraction":
                signals.append("contracting_cycle")

            overall_outlook = "neutral"
            if len(signals) >= 2:
                overall_outlook = "cautious"
            elif len(signals) == 0:
                overall_outlook = "favorable"

            return {
                "analysis_type": "comprehensive",
                "timestamp": datetime.now().isoformat(),
                "yield_curve": {
                    "spread_10y2y": yield_curve.get("spread_10y2y"),
                    "spread_10y3m": yield_curve.get("spread_10y3m"),
                    "is_inverted": yield_curve.get("is_inverted"),
                    "curve_shape": yield_curve.get("curve_shape"),
                },
                "fed_funds": {
                    "current_rate": fed_funds.get("current_rate"),
                    "rate_trend": fed_funds.get("rate_trend"),
                    "expected_direction": fed_funds.get("expected_direction"),
                },
                "macro_regime": {
                    "regime": regime.get("regime"),
                    "confidence": regime.get("confidence"),
                    "vix_level": regime.get("vix_level"),
                },
                "market_cycle": {
                    "cycle_phase": cycle.get("cycle_phase"),
                    "leading_indicators": cycle.get("leading_indicators"),
                },
                "overall": {
                    "outlook": overall_outlook,
                    "warning_signals": signals,
                },
                "interpretation": _generate_macro_interpretation(
                    yield_curve, fed_funds, regime, cycle, overall_outlook
                ),
                "status": "success",
            }

    except Exception as e:
        logger.error(f"Error in macro_analysis: {e}")
        return {
            "error": str(e),
            "analysis_type": analysis_type,
            "status": "error",
        }


async def _analyze_yield_curve(lookback_days: int) -> dict[str, Any]:
    """Analyze Treasury yield curve shape and inversions."""
    import yfinance as yf

    try:
        # Get Treasury yields from yfinance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Treasury ETFs/proxies
        tickers = {
            "^TNX": "10Y_yield",  # 10-year Treasury yield
            "^FVX": "5Y_yield",   # 5-year Treasury yield
            "^IRX": "3M_yield",   # 3-month Treasury yield
        }

        yields_data = {}
        current_yields = {}

        for ticker, name in tickers.items():
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                )
                if not data.empty:
                    # Squeeze to handle multi-level column index from yfinance
                    close_series = data["Close"].squeeze()
                    if len(close_series) > 0:
                        yields_data[name] = close_series
                        # Extract last value as scalar safely
                        last_val = close_series.iloc[-1]
                        if isinstance(last_val, (pd.Series, np.ndarray)):
                            current_yields[name] = float(last_val.values[0])
                        else:
                            current_yields[name] = _safe_float(last_val)
            except Exception as e:
                logger.warning(f"Could not get {ticker}: {e}")

        # Calculate spreads
        spread_10y3m = None
        spread_10y2y = None  # Approximate with 5Y as proxy

        if "10Y_yield" in current_yields and "3M_yield" in current_yields:
            spread_10y3m = round(
                current_yields["10Y_yield"] - current_yields["3M_yield"], 3
            )

        if "10Y_yield" in current_yields and "5Y_yield" in current_yields:
            # Using 5Y as proxy for 2Y
            spread_10y2y = round(
                current_yields["10Y_yield"] - current_yields["5Y_yield"], 3
            )

        # Determine if inverted
        is_inverted = False
        if spread_10y3m is not None and spread_10y3m < 0:
            is_inverted = True
        if spread_10y2y is not None and spread_10y2y < 0:
            is_inverted = True

        # Determine curve shape
        curve_shape = "normal"
        if is_inverted:
            curve_shape = "inverted"
        elif spread_10y3m is not None and spread_10y3m < 0.5:
            curve_shape = "flat"
        elif spread_10y3m is not None and spread_10y3m > 2.0:
            curve_shape = "steep"

        # Historical analysis
        spread_history = []
        if "10Y_yield" in yields_data and "3M_yield" in yields_data:
            spread_series = yields_data["10Y_yield"] - yields_data["3M_yield"]
            for date, spread in spread_series.tail(20).items():
                if pd.notna(spread):
                    spread_history.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "spread": round(float(spread), 3),
                    })

        # Inversion duration
        inversion_days = 0
        if "10Y_yield" in yields_data and "3M_yield" in yields_data:
            spread_series = yields_data["10Y_yield"] - yields_data["3M_yield"]
            inverted_mask = spread_series < 0
            if inverted_mask.any():
                # Count consecutive inverted days from most recent
                recent_inverted = inverted_mask.iloc[::-1]
                for val in recent_inverted:
                    if val:
                        inversion_days += 1
                    else:
                        break

        return {
            "analysis_type": "yield_curve",
            "current_yields": current_yields,
            "spread_10y3m": spread_10y3m,
            "spread_10y2y": spread_10y2y,
            "is_inverted": is_inverted,
            "curve_shape": curve_shape,
            "inversion_days": inversion_days if is_inverted else 0,
            "spread_history": spread_history[-10:],  # Last 10 data points
            "interpretation": (
                f"Yield curve is {curve_shape}. "
                f"10Y-3M spread: {spread_10y3m:.2f}%. "
                + (f"Inverted for {inversion_days} days." if is_inverted else "No inversion.")
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error analyzing yield curve: {e}")
        return {
            "analysis_type": "yield_curve",
            "error": str(e),
            "status": "error",
        }


async def _analyze_fed_funds(lookback_days: int, include_forecast: bool) -> dict[str, Any]:
    """Analyze Federal Funds rate and expectations."""
    import yfinance as yf

    try:
        # Get Fed Funds proxy data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Use short-term Treasury as Fed Funds proxy
        irx = yf.download("^IRX", start=start_date, end=end_date, progress=False)

        current_rate = None
        rate_trend = "stable"
        rate_history = []

        if not irx.empty:
            # Squeeze to handle multi-level column index
            close_series = irx["Close"].squeeze()
            if len(close_series) > 0:
                last_val = close_series.iloc[-1]
                current_rate = round(_safe_float(last_val), 2)

                # Calculate trend
                if len(close_series) >= 20:
                    recent_avg = float(close_series.tail(20).mean())
                    older_avg = float(close_series.head(20).mean())

                    if recent_avg > older_avg + 0.25:
                        rate_trend = "rising"
                    elif recent_avg < older_avg - 0.25:
                        rate_trend = "falling"

                # Build rate history
                for date, val in close_series.tail(30).items():
                    if pd.notna(val):
                        rate_history.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "rate": round(_safe_float(val), 2),
                        })

        # Expected direction based on trend and market indicators
        expected_direction = "hold"
        if rate_trend == "rising":
            expected_direction = "hike"
        elif rate_trend == "falling":
            expected_direction = "cut"

        # Rate change probabilities (simplified heuristic)
        prob_hike = 0.3
        prob_cut = 0.3
        prob_hold = 0.4

        if rate_trend == "rising":
            prob_hike = 0.5
            prob_cut = 0.1
            prob_hold = 0.4
        elif rate_trend == "falling":
            prob_hike = 0.1
            prob_cut = 0.5
            prob_hold = 0.4

        return {
            "analysis_type": "fed_funds",
            "current_rate": current_rate,
            "rate_trend": rate_trend,
            "expected_direction": expected_direction,
            "rate_history": rate_history[-10:],
            "probabilities": {
                "hike": round(prob_hike * 100, 1),
                "hold": round(prob_hold * 100, 1),
                "cut": round(prob_cut * 100, 1),
            } if include_forecast else None,
            "interpretation": (
                f"Fed Funds proxy at {current_rate}%. "
                f"Trend: {rate_trend}. "
                f"Expected next move: {expected_direction}."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error analyzing Fed Funds: {e}")
        return {
            "analysis_type": "fed_funds",
            "error": str(e),
            "status": "error",
        }


async def _analyze_macro_regime(lookback_days: int) -> dict[str, Any]:
    """Classify current macro regime (risk-on/risk-off/neutral)."""
    import yfinance as yf

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Get VIX for volatility regime
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

        # Get S&P 500 for trend
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)

        # Gold (GLD) could be used for safe haven analysis in future

        vix_level = None
        vix_percentile = None
        spy_trend = "neutral"
        regime = "neutral"
        confidence = 0.5

        # VIX analysis
        if not vix.empty:
            vix_close = vix["Close"].squeeze()
            if len(vix_close) > 0:
                last_vix = vix_close.iloc[-1]
                vix_level = round(_safe_float(last_vix), 2)
                vix_min = float(vix_close.min())
                vix_max = float(vix_close.max())
                if vix_max > vix_min:
                    vix_percentile = round((vix_level - vix_min) / (vix_max - vix_min) * 100, 1)

        # SPY trend analysis
        if not spy.empty:
            spy_close = spy["Close"].squeeze()
            if len(spy_close) >= 50:
                sma_50 = float(spy_close.rolling(50).mean().iloc[-1])
                if len(spy_close) >= 200:
                    sma_200 = float(spy_close.rolling(200).mean().iloc[-1])
                else:
                    sma_200 = sma_50
                last_spy = spy_close.iloc[-1]
                current_price = _safe_float(last_spy)

                if current_price > sma_50 > sma_200:
                    spy_trend = "bullish"
                elif current_price < sma_50 < sma_200:
                    spy_trend = "bearish"
                elif current_price > sma_50:
                    spy_trend = "neutral_bullish"
                else:
                    spy_trend = "neutral_bearish"

        # Regime classification
        if vix_level is not None:
            if vix_level < 15 and spy_trend in ["bullish", "neutral_bullish"]:
                regime = "risk_on"
                confidence = 0.8 if spy_trend == "bullish" else 0.6
            elif vix_level > 25 or spy_trend in ["bearish", "neutral_bearish"]:
                regime = "risk_off"
                confidence = 0.8 if vix_level > 30 else 0.6
            else:
                regime = "neutral"
                confidence = 0.5

        # Risk indicators
        risk_indicators = {
            "vix_elevated": vix_level > 20 if vix_level else None,
            "equity_downtrend": spy_trend in ["bearish", "neutral_bearish"],
            "high_volatility": vix_level > 25 if vix_level else None,
        }

        return {
            "analysis_type": "macro_regime",
            "regime": regime,
            "confidence": round(confidence, 2),
            "vix_level": vix_level,
            "vix_percentile": vix_percentile,
            "spy_trend": spy_trend,
            "risk_indicators": risk_indicators,
            "interpretation": (
                f"Macro regime: {regime.upper()} (confidence: {confidence:.0%}). "
                f"VIX at {vix_level}, {spy_trend} equity trend."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error analyzing macro regime: {e}")
        return {
            "analysis_type": "macro_regime",
            "error": str(e),
            "status": "error",
        }


async def _analyze_market_cycle(lookback_days: int) -> dict[str, Any]:
    """Analyze business/credit cycle indicators."""
    import yfinance as yf

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Get sector ETFs for rotation analysis
        sectors = {
            "XLY": "Consumer Discretionary",
            "XLK": "Technology",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLU": "Utilities",
            "XLP": "Consumer Staples",
        }

        sector_performance = {}
        for ticker, name in sectors.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and len(data) >= 20:
                    close_series = data["Close"].squeeze()
                    last_val = close_series.iloc[-1]
                    prev_val = close_series.iloc[-20]
                    last_price = _safe_float(last_val)
                    prev_price = _safe_float(prev_val)
                    returns_1m = (last_price / prev_price - 1) * 100
                    sector_performance[name] = round(returns_1m, 2)
            except Exception:
                pass

        # Determine cycle phase based on sector leadership
        cycle_phase = "mid_cycle"
        leading_sectors = []
        lagging_sectors = []

        if sector_performance:
            sorted_sectors = sorted(
                sector_performance.items(), key=lambda x: x[1], reverse=True
            )
            leading_sectors = [s[0] for s in sorted_sectors[:2]]
            lagging_sectors = [s[0] for s in sorted_sectors[-2:]]

            # Simplified cycle classification
            # Early cycle: Financials, Consumer Discretionary lead
            # Mid cycle: Technology, Industrials lead
            # Late cycle: Energy, Materials lead
            # Recession: Utilities, Consumer Staples lead

            if "Utilities" in leading_sectors or "Consumer Staples" in leading_sectors:
                cycle_phase = "late_cycle" if "Energy" in leading_sectors else "defensive"
            elif "Technology" in leading_sectors or "Consumer Discretionary" in leading_sectors:
                cycle_phase = "expansion"
            elif "Financials" in leading_sectors:
                cycle_phase = "early_cycle"
            elif "Energy" in leading_sectors:
                cycle_phase = "late_cycle"

        # Get credit spreads proxy (HYG vs LQD)
        credit_spread = None
        try:
            hyg = yf.download("HYG", start=start_date, end=end_date, progress=False)
            lqd = yf.download("LQD", start=start_date, end=end_date, progress=False)

            if not hyg.empty and not lqd.empty:
                hyg_close = hyg["Close"].squeeze()
                lqd_close = lqd["Close"].squeeze()
                if len(hyg_close) > 0 and len(lqd_close) > 0:
                    hyg_last = hyg_close.iloc[-1]
                    lqd_last = lqd_close.iloc[-1]
                    hyg_price = _safe_float(hyg_last)
                    lqd_price = _safe_float(lqd_last)
                    hyg_yield = (1 / hyg_price) * 100  # Simplified
                    lqd_yield = (1 / lqd_price) * 100
                    credit_spread = round(hyg_yield - lqd_yield, 2)
        except Exception:
            pass

        # Leading indicators summary
        leading_indicators = {
            "sector_rotation": cycle_phase,
            "leading_sectors": leading_sectors,
            "lagging_sectors": lagging_sectors,
            "credit_spread": credit_spread,
        }

        return {
            "analysis_type": "market_cycle",
            "cycle_phase": cycle_phase,
            "sector_performance": sector_performance,
            "leading_indicators": leading_indicators,
            "interpretation": (
                f"Market cycle phase: {cycle_phase.replace('_', ' ')}. "
                f"Leading sectors: {', '.join(leading_sectors)}. "
                f"Lagging sectors: {', '.join(lagging_sectors)}."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error analyzing market cycle: {e}")
        return {
            "analysis_type": "market_cycle",
            "error": str(e),
            "status": "error",
        }


def _generate_macro_interpretation(
    yield_curve: dict,
    fed_funds: dict,
    regime: dict,
    cycle: dict,
    overall_outlook: str,
) -> str:
    """Generate overall macro interpretation."""
    parts = []

    # Yield curve
    if yield_curve.get("is_inverted"):
        parts.append("Yield curve inverted (recession warning)")
    elif yield_curve.get("curve_shape") == "flat":
        parts.append("Yield curve flat (late cycle)")
    else:
        parts.append(f"Yield curve {yield_curve.get('curve_shape', 'normal')}")

    # Fed funds
    if fed_funds.get("expected_direction"):
        parts.append(f"Fed likely to {fed_funds['expected_direction']}")

    # Regime
    if regime.get("regime"):
        parts.append(f"{regime['regime'].replace('_', ' ')} environment")

    # Cycle
    if cycle.get("cycle_phase"):
        parts.append(f"{cycle['cycle_phase'].replace('_', ' ')} cycle phase")

    # Overall
    outlook_text = {
        "favorable": "Macro backdrop supportive for risk assets.",
        "neutral": "Mixed macro signals, selective positioning advised.",
        "cautious": "Elevated macro risks, defensive positioning recommended.",
    }

    interpretation = ". ".join(parts)
    interpretation += f" {outlook_text.get(overall_outlook, '')}"

    return interpretation
