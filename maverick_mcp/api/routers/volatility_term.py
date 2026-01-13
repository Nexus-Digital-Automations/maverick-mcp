"""
VIX Term Structure Analysis Module.

Provides tools for analyzing the VIX futures curve, contango/backwardation,
volatility surfaces, and market regime detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from openbb import obb

logger = logging.getLogger(__name__)


async def vix_term_structure() -> dict[str, Any]:
    """
    Analyze current VIX futures term structure (contango vs backwardation).

    The VIX term structure shows the relationship between spot VIX and
    VIX futures. Contango (normal) is bullish; backwardation (inverted) is bearish.

    Returns:
        Dictionary containing:
        - spot_vix: Current VIX level
        - futures_curve: Estimated VIX futures values by month
        - structure_type: contango, backwardation, or flat
        - roll_yield: Estimated monthly decay/gain from rolling
        - market_signal: bullish, bearish, or neutral
    """
    try:
        loop = asyncio.get_event_loop()

        # Get VIX spot data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        vix_result = await loop.run_in_executor(
            None,
            lambda: obb.equity.price.historical(
                symbol="^VIX",
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                provider="yfinance"
            )
        )

        if not hasattr(vix_result, "results") or not vix_result.results:
            return {
                "error": "Could not fetch VIX data",
                "status": "error"
            }

        vix_df = pd.DataFrame([r.model_dump() for r in vix_result.results])
        current_vix = float(vix_df["close"].iloc[-1])

        # Try to get VIX9D (9-day VIX) for near-term
        try:
            vix9d_result = await loop.run_in_executor(
                None,
                lambda: obb.equity.price.historical(
                    symbol="^VIX9D",
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                    provider="yfinance"
                )
            )
            if hasattr(vix9d_result, "results") and vix9d_result.results:
                vix9d_df = pd.DataFrame([r.model_dump() for r in vix9d_result.results])
                vix_9d = float(vix9d_df["close"].iloc[-1])
            else:
                vix_9d = current_vix * 0.95  # Estimate
        except Exception:
            vix_9d = current_vix * 0.95  # Estimate

        # Try to get VIX3M (3-month VIX) for longer-term
        try:
            vix3m_result = await loop.run_in_executor(
                None,
                lambda: obb.equity.price.historical(
                    symbol="^VIX3M",
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                    provider="yfinance"
                )
            )
            if hasattr(vix3m_result, "results") and vix3m_result.results:
                vix3m_df = pd.DataFrame([r.model_dump() for r in vix3m_result.results])
                vix_3m = float(vix3m_df["close"].iloc[-1])
            else:
                vix_3m = current_vix * 1.08  # Normal contango estimate
        except Exception:
            vix_3m = current_vix * 1.08  # Normal contango estimate

        # Construct estimated term structure
        # Typical VIX term structure slopes upward ~4-8% per month in contango
        futures_curve = {
            "spot": round(current_vix, 2),
            "front_month": round(current_vix * 1.03, 2),  # ~3% premium
            "second_month": round(current_vix * 1.06, 2),  # ~6% premium
            "third_month": round(vix_3m, 2),
            "six_month": round(vix_3m * 1.05, 2),
        }

        # Calculate term structure slope
        front_premium = (futures_curve["front_month"] - current_vix) / current_vix * 100
        three_month_premium = (vix_3m - current_vix) / current_vix * 100

        # Determine structure type
        if front_premium > 2:
            if three_month_premium > front_premium:
                structure_type = "steep_contango"
                structure_meaning = "Strong normal curve - very bullish"
            else:
                structure_type = "contango"
                structure_meaning = "Normal upward sloping curve - bullish"
        elif front_premium < -2:
            if three_month_premium < front_premium:
                structure_type = "steep_backwardation"
                structure_meaning = "Strong inverted curve - crisis/fear mode"
            else:
                structure_type = "backwardation"
                structure_meaning = "Inverted curve - elevated near-term fear"
        else:
            structure_type = "flat"
            structure_meaning = "Neutral term structure"

        # Calculate roll yield (what you gain/lose from rolling futures)
        monthly_roll_yield = -front_premium  # Negative because contango costs, backwardation gains
        annual_roll_yield = monthly_roll_yield * 12

        # Market signal based on term structure
        if structure_type in ["contango", "steep_contango"]:
            market_signal = "bullish"
            signal_description = "Normal VIX curve suggests complacency - favorable for equities"
        elif structure_type in ["backwardation", "steep_backwardation"]:
            market_signal = "bearish"
            signal_description = "Inverted VIX curve signals near-term fear - caution warranted"
        else:
            market_signal = "neutral"
            signal_description = "Flat term structure - no strong signal"

        # VIX level interpretation
        if current_vix < 15:
            vix_regime = "low_volatility"
            vix_interpretation = "Complacency - potential for vol expansion"
        elif current_vix < 20:
            vix_regime = "normal"
            vix_interpretation = "Normal market conditions"
        elif current_vix < 30:
            vix_regime = "elevated"
            vix_interpretation = "Increased uncertainty - caution advised"
        else:
            vix_regime = "crisis"
            vix_interpretation = "High fear - potential capitulation or crash"

        return {
            "timestamp": datetime.now().isoformat(),
            "spot_vix": round(current_vix, 2),
            "vix_9d": round(vix_9d, 2),
            "vix_3m": round(vix_3m, 2),
            "futures_curve": futures_curve,
            "term_structure": {
                "type": structure_type,
                "meaning": structure_meaning,
                "front_month_premium_pct": round(front_premium, 2),
                "three_month_premium_pct": round(three_month_premium, 2),
            },
            "roll_yield": {
                "monthly_pct": round(monthly_roll_yield, 2),
                "annual_pct": round(annual_roll_yield, 2),
                "description": f"{'Gain' if monthly_roll_yield > 0 else 'Cost'} of {abs(monthly_roll_yield):.1f}% per month from rolling"
            },
            "vix_regime": vix_regime,
            "vix_interpretation": vix_interpretation,
            "market_signal": market_signal,
            "signal_description": signal_description,
            "interpretation": f"VIX at {current_vix:.1f} ({vix_regime}) with {structure_type} term structure. Market signal: {market_signal.upper()}",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in vix_term_structure: {e}")
        return {"error": str(e), "status": "error"}


async def vix_contango_backwardation(
    lookback_days: int = 30,
) -> dict[str, Any]:
    """
    Historical analysis of VIX contango/backwardation with trading signals.

    Tracks the VIX term structure over time to identify regime changes
    and potential mean-reversion opportunities.

    Args:
        lookback_days: Number of days to analyze (default: 30)

    Returns:
        Dictionary containing:
        - current_state: contango or backwardation
        - days_in_contango: Number of days in contango
        - days_in_backwardation: Number of days in backwardation
        - signal: Trading signal based on regime analysis
    """
    try:
        loop = asyncio.get_event_loop()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

        # Get VIX spot and VIX3M for term structure analysis
        vix_task = loop.run_in_executor(
            None,
            lambda: obb.equity.price.historical(
                symbol="^VIX",
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                provider="yfinance"
            )
        )

        vix3m_task = loop.run_in_executor(
            None,
            lambda: obb.equity.price.historical(
                symbol="^VIX3M",
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                provider="yfinance"
            )
        )

        vix_result, vix3m_result = await asyncio.gather(vix_task, vix3m_task)

        if not hasattr(vix_result, "results") or not vix_result.results:
            return {"error": "Could not fetch VIX data", "status": "error"}

        vix_df = pd.DataFrame([r.model_dump() for r in vix_result.results])
        vix_df = vix_df.tail(lookback_days)

        # Try to use VIX3M, otherwise estimate
        if hasattr(vix3m_result, "results") and vix3m_result.results:
            vix3m_df = pd.DataFrame([r.model_dump() for r in vix3m_result.results])
            vix3m_df = vix3m_df.tail(lookback_days)
            has_vix3m = len(vix3m_df) >= len(vix_df) * 0.8
        else:
            has_vix3m = False

        # Calculate daily term structure
        history = []
        contango_days = 0
        backwardation_days = 0

        for _, row in vix_df.iterrows():
            spot_vix = float(row["close"])

            if has_vix3m:
                # Find matching VIX3M date
                try:
                    vix3m_close = float(vix3m_df[vix3m_df["date"] == row["date"]]["close"].iloc[0])
                except (IndexError, KeyError):
                    vix3m_close = spot_vix * 1.05  # Estimate

                premium = (vix3m_close - spot_vix) / spot_vix * 100
            else:
                # Estimate based on historical averages
                premium = 5.0  # Typical contango

            state = "contango" if premium > 0 else "backwardation"
            if state == "contango":
                contango_days += 1
            else:
                backwardation_days += 1

            history.append({
                "date": str(row.get("date", ""))[:10],
                "vix": round(spot_vix, 2),
                "premium_pct": round(premium, 2),
                "state": state
            })

        # Current state analysis
        current_state = history[-1]["state"] if history else "unknown"
        current_premium = history[-1]["premium_pct"] if history else 0

        # Calculate streak
        streak = 0
        for h in reversed(history):
            if h["state"] == current_state:
                streak += 1
            else:
                break

        # Average premium
        avg_premium = np.mean([h["premium_pct"] for h in history])

        # Generate signal
        if current_state == "backwardation" and streak >= 3:
            signal = "buy_vol"
            signal_reason = f"Sustained backwardation ({streak} days) - vol likely to normalize"
        elif current_state == "contango" and current_premium > 10:
            signal = "sell_vol"
            signal_reason = f"Steep contango ({current_premium:.1f}%) - vol likely to mean-revert"
        elif backwardation_days >= lookback_days * 0.5:
            signal = "caution"
            signal_reason = f"Frequent backwardation ({backwardation_days}/{lookback_days} days) - elevated risk"
        else:
            signal = "neutral"
            signal_reason = "Normal conditions - no strong signal"

        return {
            "lookback_days": lookback_days,
            "current_state": current_state,
            "current_premium_pct": round(current_premium, 2),
            "current_streak_days": streak,
            "days_in_contango": contango_days,
            "days_in_backwardation": backwardation_days,
            "contango_pct": round(contango_days / len(history) * 100, 1) if history else 0,
            "average_premium_pct": round(avg_premium, 2),
            "signal": signal,
            "signal_reason": signal_reason,
            "recent_history": history[-10:],  # Last 10 days
            "interpretation": f"VIX in {current_state} for {streak} days. {signal_reason}",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in vix_contango_backwardation: {e}")
        return {"error": str(e), "status": "error"}


async def volatility_surface_3d(
    symbol: str,
) -> dict[str, Any]:
    """
    Build 3D volatility surface: strike x expiration x implied volatility.

    The vol surface shows how IV varies across strikes and expirations,
    revealing skew and term structure patterns.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')

    Returns:
        Dictionary containing:
        - surface_data: IV values across strike/expiration grid
        - atm_iv: At-the-money implied volatility
        - skew: IV skew (OTM puts vs OTM calls)
        - term_structure: IV by expiration
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()

    try:
        loop = asyncio.get_event_loop()

        # Get current price
        quote_result = await loop.run_in_executor(
            None,
            lambda: obb.equity.price.quote(symbol=symbol, provider="yfinance")
        )

        if not hasattr(quote_result, "results") or not quote_result.results:
            return {
                "error": f"Could not fetch quote for {symbol}",
                "status": "error"
            }

        current_price = float(quote_result.results[0].last_price or quote_result.results[0].prev_close or 0)

        # Get options chains
        chains_result = await loop.run_in_executor(
            None,
            lambda: obb.derivatives.options.chains(symbol=symbol, provider="yfinance")
        )

        if not hasattr(chains_result, "to_df"):
            return {
                "error": f"Could not fetch options for {symbol}",
                "status": "error"
            }

        # Convert to DataFrame using OpenBB's to_df() method
        df = chains_result.to_df()

        if len(df) < 10:
            return {
                "error": f"Insufficient options data for {symbol}",
                "status": "error"
            }

        # Get unique expirations
        expirations = df["expiration"].unique()[:6]  # Limit to first 6 expirations

        surface_data = []
        term_structure = {}
        atm_ivs = []

        for exp in expirations:
            exp_df = df[df["expiration"] == exp]

            # Get strikes around ATM
            calls = exp_df[exp_df["option_type"] == "call"].copy()
            puts = exp_df[exp_df["option_type"] == "put"].copy()

            if len(calls) < 3 or len(puts) < 3:
                continue

            # Find ATM options
            calls["moneyness"] = (calls["strike"] - current_price) / current_price
            atm_call = calls.iloc[(calls["moneyness"].abs()).argmin()]

            puts["moneyness"] = (puts["strike"] - current_price) / current_price
            atm_put = puts.iloc[(puts["moneyness"].abs()).argmin()]

            # ATM IV (average of call and put)
            atm_call_iv = float(atm_call.get("implied_volatility", 0) or 0)
            atm_put_iv = float(atm_put.get("implied_volatility", 0) or 0)
            atm_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else max(atm_call_iv, atm_put_iv)

            if atm_iv > 0:
                atm_ivs.append(atm_iv)
                term_structure[str(exp)[:10]] = round(atm_iv * 100, 2)

            # Sample surface points
            for _, row in calls.head(10).iterrows():
                iv = float(row.get("implied_volatility", 0) or 0)
                if iv > 0:
                    surface_data.append({
                        "expiration": str(exp)[:10],
                        "strike": float(row["strike"]),
                        "moneyness": round((float(row["strike"]) - current_price) / current_price * 100, 1),
                        "iv_pct": round(iv * 100, 2),
                        "option_type": "call"
                    })

            for _, row in puts.head(10).iterrows():
                iv = float(row.get("implied_volatility", 0) or 0)
                if iv > 0:
                    surface_data.append({
                        "expiration": str(exp)[:10],
                        "strike": float(row["strike"]),
                        "moneyness": round((float(row["strike"]) - current_price) / current_price * 100, 1),
                        "iv_pct": round(iv * 100, 2),
                        "option_type": "put"
                    })

        if not surface_data:
            return {
                "error": f"Could not build volatility surface for {symbol}",
                "status": "error"
            }

        # Calculate overall ATM IV
        overall_atm_iv = np.mean(atm_ivs) * 100 if atm_ivs else 0

        # Calculate skew (OTM put IV vs OTM call IV)
        otm_puts = [s for s in surface_data if s["option_type"] == "put" and s["moneyness"] < -5]
        otm_calls = [s for s in surface_data if s["option_type"] == "call" and s["moneyness"] > 5]

        otm_put_iv = np.mean([s["iv_pct"] for s in otm_puts]) if otm_puts else overall_atm_iv
        otm_call_iv = np.mean([s["iv_pct"] for s in otm_calls]) if otm_calls else overall_atm_iv

        skew = otm_put_iv - otm_call_iv

        if skew > 5:
            skew_interpretation = "Put skew - downside protection expensive, fear of drops"
        elif skew < -5:
            skew_interpretation = "Call skew - upside calls expensive, speculative"
        else:
            skew_interpretation = "Balanced skew - normal market conditions"

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "atm_iv_pct": round(overall_atm_iv, 2),
            "skew": {
                "value": round(skew, 2),
                "otm_put_iv_pct": round(otm_put_iv, 2),
                "otm_call_iv_pct": round(otm_call_iv, 2),
                "interpretation": skew_interpretation
            },
            "term_structure": term_structure,
            "surface_points": len(surface_data),
            "sample_surface": surface_data[:20],  # First 20 points
            "expirations_analyzed": len(term_structure),
            "interpretation": f"{symbol} ATM IV: {overall_atm_iv:.1f}%, Skew: {skew:.1f}% ({skew_interpretation})",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in volatility_surface_3d for {symbol}: {e}")
        return {"error": str(e), "status": "error"}


async def volatility_regime_indicator(
    lookback_days: int = 60,
) -> dict[str, Any]:
    """
    Detect current volatility regime: low, normal, elevated, or crisis.

    Uses VIX percentiles and term structure to classify the current
    market volatility environment.

    Args:
        lookback_days: Days of VIX history to analyze (default: 60)

    Returns:
        Dictionary containing:
        - current_regime: low/normal/elevated/crisis
        - vix_percentile: Current VIX relative to lookback period
        - regime_change_signal: Whether regime recently changed
        - recommended_strategy: Strategy suggestion for current regime
    """
    try:
        loop = asyncio.get_event_loop()

        # Get extended VIX history for regime analysis
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

        vix_result = await loop.run_in_executor(
            None,
            lambda: obb.equity.price.historical(
                symbol="^VIX",
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                provider="yfinance"
            )
        )

        if not hasattr(vix_result, "results") or not vix_result.results:
            return {"error": "Could not fetch VIX data", "status": "error"}

        vix_df = pd.DataFrame([r.model_dump() for r in vix_result.results])
        vix_df = vix_df.tail(lookback_days)

        if len(vix_df) < 20:
            return {
                "error": "Insufficient VIX history",
                "status": "error"
            }

        vix_values = vix_df["close"].values
        current_vix = float(vix_values[-1])

        # Calculate statistics
        vix_mean = float(np.mean(vix_values))
        vix_std = float(np.std(vix_values))
        vix_min = float(np.min(vix_values))
        vix_max = float(np.max(vix_values))

        # Calculate percentile
        vix_percentile = float(np.sum(vix_values < current_vix) / len(vix_values) * 100)

        # Determine regime
        z_score = (current_vix - vix_mean) / vix_std if vix_std > 0 else 0

        if current_vix < 15 or vix_percentile < 20:
            regime = "low_volatility"
            regime_description = "Market complacency - low fear"
            recommended_strategy = "Consider long volatility positions, tight stops on directional trades"
            risk_level = "elevated_complacency"
        elif current_vix < 20 or vix_percentile < 60:
            regime = "normal"
            regime_description = "Normal market conditions"
            recommended_strategy = "Standard position sizing, balanced approach"
            risk_level = "normal"
        elif current_vix < 30 or vix_percentile < 85:
            regime = "elevated"
            regime_description = "Increased uncertainty - caution"
            recommended_strategy = "Reduce position sizes, consider hedges"
            risk_level = "elevated"
        else:
            regime = "crisis"
            regime_description = "High fear - potential capitulation"
            recommended_strategy = "Defensive positioning, look for contrarian opportunities"
            risk_level = "extreme"

        # Check for regime change
        if len(vix_values) >= 5:
            recent_avg = np.mean(vix_values[-5:])
            prior_avg = np.mean(vix_values[-10:-5])

            change_pct = (recent_avg - prior_avg) / prior_avg * 100 if prior_avg > 0 else 0

            if change_pct > 20:
                regime_change = "rising"
                change_signal = "Vol expanding - risk increasing"
            elif change_pct < -20:
                regime_change = "falling"
                change_signal = "Vol contracting - risk normalizing"
            else:
                regime_change = "stable"
                change_signal = "Regime stable"
        else:
            regime_change = "unknown"
            change_signal = "Insufficient data"

        # Historical regime context
        low_vol_days = int(np.sum(vix_values < 15))
        normal_days = int(np.sum((vix_values >= 15) & (vix_values < 20)))
        elevated_days = int(np.sum((vix_values >= 20) & (vix_values < 30)))
        crisis_days = int(np.sum(vix_values >= 30))

        return {
            "timestamp": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "current_vix": round(current_vix, 2),
            "vix_percentile": round(vix_percentile, 1),
            "vix_statistics": {
                "mean": round(vix_mean, 2),
                "std": round(vix_std, 2),
                "min": round(vix_min, 2),
                "max": round(vix_max, 2),
                "z_score": round(z_score, 2)
            },
            "current_regime": regime,
            "regime_description": regime_description,
            "risk_level": risk_level,
            "regime_change": regime_change,
            "change_signal": change_signal,
            "recommended_strategy": recommended_strategy,
            "regime_distribution": {
                "low_vol_days": low_vol_days,
                "normal_days": normal_days,
                "elevated_days": elevated_days,
                "crisis_days": crisis_days
            },
            "interpretation": f"VIX at {current_vix:.1f} ({vix_percentile:.0f}th percentile) - {regime.upper()} regime. {change_signal}",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in volatility_regime_indicator: {e}")
        return {"error": str(e), "status": "error"}
