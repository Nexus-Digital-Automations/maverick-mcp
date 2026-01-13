"""
Volume Profile Analysis Module.

Provides tools for analyzing volume distribution across price levels,
including Point of Control (POC), Value Areas, and VWAP bands.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from openbb import obb

logger = logging.getLogger(__name__)


async def volume_profile_analysis(
    symbol: str,
    period_days: int = 20,
    num_bins: int = 50,
) -> dict[str, Any]:
    """
    Calculate volume profile with POC, value areas, and high/low volume nodes.

    Volume Profile shows volume distribution at different price levels, helping
    identify support/resistance and fair value.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        period_days: Number of trading days to analyze (default: 20)
        num_bins: Number of price bins for volume distribution (default: 50)

    Returns:
        Dictionary containing:
        - poc: Point of Control (price with highest volume)
        - value_area_high: Upper bound of value area (70% volume)
        - value_area_low: Lower bound of value area (70% volume)
        - high_volume_nodes: Price levels with concentrated volume
        - low_volume_nodes: Price levels with thin volume (potential breakout zones)
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()

    try:
        loop = asyncio.get_event_loop()

        # Calculate start date
        start_date = (datetime.now() - timedelta(days=period_days * 2)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch daily OHLCV data
        result = await loop.run_in_executor(
            None,
            lambda: obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                provider="yfinance"
            )
        )

        if not hasattr(result, "results") or not result.results:
            return {
                "error": f"Could not fetch data for {symbol}",
                "status": "error"
            }

        df = pd.DataFrame([r.model_dump() for r in result.results])
        df = df.tail(period_days)  # Use last N days

        if len(df) < 5:
            return {
                "error": f"Insufficient data for {symbol}",
                "status": "error"
            }

        # Get price range
        price_min = df["low"].min()
        price_max = df["high"].max()
        current_price = float(df["close"].iloc[-1])

        # Create price bins
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Distribute volume across price bins for each day
        volume_profile = np.zeros(num_bins)

        for _, row in df.iterrows():
            low, high, volume = row["low"], row["high"], row["volume"]

            # Find which bins this candle spans
            bin_low = np.searchsorted(bins, low, side="right") - 1
            bin_high = np.searchsorted(bins, high, side="left")

            bin_low = max(0, bin_low)
            bin_high = min(num_bins - 1, bin_high)

            # Distribute volume evenly across spanned bins
            num_spanned = bin_high - bin_low + 1
            if num_spanned > 0:
                volume_per_bin = volume / num_spanned
                volume_profile[bin_low:bin_high + 1] += volume_per_bin

        # Find POC (Point of Control) - price with highest volume
        poc_idx = np.argmax(volume_profile)
        poc_price = float(bin_centers[poc_idx])
        poc_volume = float(volume_profile[poc_idx])

        # Calculate Value Area (70% of total volume)
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * 0.70

        # Expand from POC until we capture 70% volume
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        accumulated_volume = volume_profile[poc_idx]

        while accumulated_volume < target_volume:
            expand_low = va_low_idx > 0
            expand_high = va_high_idx < num_bins - 1

            if not expand_low and not expand_high:
                break

            vol_low = volume_profile[va_low_idx - 1] if expand_low else 0
            vol_high = volume_profile[va_high_idx + 1] if expand_high else 0

            if expand_low and (not expand_high or vol_low >= vol_high):
                va_low_idx -= 1
                accumulated_volume += vol_low
            elif expand_high:
                va_high_idx += 1
                accumulated_volume += vol_high

        value_area_low = float(bin_centers[va_low_idx])
        value_area_high = float(bin_centers[va_high_idx])

        # Find High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        avg_volume = np.mean(volume_profile)
        std_volume = np.std(volume_profile)

        hvn_threshold = avg_volume + std_volume
        lvn_threshold = avg_volume - std_volume * 0.5

        high_volume_nodes = []
        low_volume_nodes = []

        for i, vol in enumerate(volume_profile):
            price = float(bin_centers[i])
            if vol > hvn_threshold:
                high_volume_nodes.append({
                    "price": round(price, 2),
                    "volume": float(vol),
                    "type": "resistance" if price > current_price else "support"
                })
            elif vol < lvn_threshold and vol > 0:
                low_volume_nodes.append({
                    "price": round(price, 2),
                    "volume": float(vol),
                    "type": "potential_breakout"
                })

        # Sort by proximity to current price
        high_volume_nodes.sort(key=lambda x: abs(x["price"] - current_price))
        low_volume_nodes.sort(key=lambda x: abs(x["price"] - current_price))

        # Price position relative to value area
        if current_price > value_area_high:
            price_position = "above_value"
            interpretation = f"Price above value area - potential resistance at ${value_area_high:.2f}"
        elif current_price < value_area_low:
            price_position = "below_value"
            interpretation = f"Price below value area - potential support at ${value_area_low:.2f}"
        else:
            price_position = "inside_value"
            interpretation = f"Price inside value area (${value_area_low:.2f} - ${value_area_high:.2f}) - fair value zone"

        return {
            "symbol": symbol,
            "period_days": period_days,
            "current_price": round(current_price, 2),
            "poc": {
                "price": round(poc_price, 2),
                "volume": round(poc_volume, 0),
                "description": "Price with highest traded volume"
            },
            "value_area": {
                "high": round(value_area_high, 2),
                "low": round(value_area_low, 2),
                "range_pct": round((value_area_high - value_area_low) / current_price * 100, 2),
                "description": "70% of volume traded within this range"
            },
            "price_position": price_position,
            "high_volume_nodes": high_volume_nodes[:5],  # Top 5
            "low_volume_nodes": low_volume_nodes[:5],  # Top 5
            "total_volume_analyzed": float(total_volume),
            "interpretation": interpretation,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in volume_profile_analysis for {symbol}: {e}")
        return {"error": str(e), "status": "error"}


async def volume_vwap_bands(
    symbol: str,
    period: str = "1d",
    std_devs: list[float] | None = None,
) -> dict[str, Any]:
    """
    Calculate VWAP with standard deviation bands.

    VWAP (Volume Weighted Average Price) is the benchmark price for the day,
    with deviation bands showing overbought/oversold levels.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        period: Time period ('1d' for intraday, '5d' for week)
        std_devs: Standard deviation levels for bands (default: [1.0, 2.0, 3.0])

    Returns:
        Dictionary containing:
        - vwap: Current VWAP value
        - upper_bands: Price levels at +1, +2, +3 std dev
        - lower_bands: Price levels at -1, -2, -3 std dev
        - current_position: Price relative to VWAP bands
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()
    std_devs = std_devs or [1.0, 2.0, 3.0]

    try:
        loop = asyncio.get_event_loop()

        # Determine interval and lookback
        if period == "1d":
            interval = "1h"  # Hourly for intraday
            days_back = 5
        else:
            interval = "1d"
            days_back = 30

        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        result = await loop.run_in_executor(
            None,
            lambda: obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                provider="yfinance"
            )
        )

        if not hasattr(result, "results") or not result.results:
            return {
                "error": f"Could not fetch data for {symbol}",
                "status": "error"
            }

        df = pd.DataFrame([r.model_dump() for r in result.results])

        if len(df) < 5:
            return {
                "error": f"Insufficient data for {symbol}",
                "status": "error"
            }

        # Calculate VWAP
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["volume_price"] = df["typical_price"] * df["volume"]
        df["cumulative_vp"] = df["volume_price"].cumsum()
        df["cumulative_vol"] = df["volume"].cumsum()
        df["vwap"] = df["cumulative_vp"] / df["cumulative_vol"]

        # Calculate standard deviation of price from VWAP
        df["vwap_dev"] = df["typical_price"] - df["vwap"]
        df["vwap_dev_sq"] = df["vwap_dev"] ** 2
        df["cumulative_dev_sq"] = df["vwap_dev_sq"].cumsum()
        df["vwap_std"] = np.sqrt(df["cumulative_dev_sq"] / df["cumulative_vol"])

        current_vwap = float(df["vwap"].iloc[-1])
        current_std = float(df["vwap_std"].iloc[-1])
        current_price = float(df["close"].iloc[-1])

        # Calculate bands
        upper_bands = {}
        lower_bands = {}

        for std in std_devs:
            upper_bands[f"+{std}σ"] = round(current_vwap + (std * current_std), 2)
            lower_bands[f"-{std}σ"] = round(current_vwap - (std * current_std), 2)

        # Determine price position
        z_score = (current_price - current_vwap) / current_std if current_std > 0 else 0

        if z_score >= 2:
            position = "overbought_extreme"
            signal = "strong_sell"
        elif z_score >= 1:
            position = "overbought"
            signal = "lean_sell"
        elif z_score <= -2:
            position = "oversold_extreme"
            signal = "strong_buy"
        elif z_score <= -1:
            position = "oversold"
            signal = "lean_buy"
        else:
            position = "fair_value"
            signal = "neutral"

        return {
            "symbol": symbol,
            "period": period,
            "current_price": round(current_price, 2),
            "vwap": round(current_vwap, 2),
            "vwap_std": round(current_std, 2),
            "upper_bands": upper_bands,
            "lower_bands": lower_bands,
            "z_score": round(z_score, 2),
            "position": position,
            "signal": signal,
            "price_vs_vwap_pct": round((current_price - current_vwap) / current_vwap * 100, 2),
            "interpretation": f"{symbol} at ${current_price:.2f} is {position} (z-score: {z_score:.2f}), VWAP: ${current_vwap:.2f}",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in volume_vwap_bands for {symbol}: {e}")
        return {"error": str(e), "status": "error"}


async def volume_market_profile(
    symbol: str,
    period_days: int = 5,
) -> dict[str, Any]:
    """
    Generate TPO (Time Price Opportunity) style market profile.

    Shows how much time price spent at each level, identifying balance areas
    and potential breakout zones.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        period_days: Number of days to analyze (default: 5)

    Returns:
        Dictionary containing:
        - tpo_profile: Time spent at each price level
        - balance_area: Price range where most time was spent
        - initial_balance: First hour's range (if intraday data available)
        - profile_type: normal, p-shaped (selling), b-shaped (buying)
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()

    try:
        loop = asyncio.get_event_loop()

        start_date = (datetime.now() - timedelta(days=period_days * 2)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Try to get intraday data for better TPO resolution
        try:
            result = await loop.run_in_executor(
                None,
                lambda: obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1h",
                    provider="yfinance"
                )
            )
        except Exception:
            # Fallback to daily
            result = await loop.run_in_executor(
                None,
                lambda: obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                    provider="yfinance"
                )
            )

        if not hasattr(result, "results") or not result.results:
            return {
                "error": f"Could not fetch data for {symbol}",
                "status": "error"
            }

        df = pd.DataFrame([r.model_dump() for r in result.results])

        # Limit to requested days
        df = df.tail(period_days * 8)  # Approximate hours in trading days

        if len(df) < 5:
            return {
                "error": f"Insufficient data for {symbol}",
                "status": "error"
            }

        # Calculate price range and bins
        price_min = df["low"].min()
        price_max = df["high"].max()
        current_price = float(df["close"].iloc[-1])

        # Create price bins (finer resolution than volume profile)
        num_bins = 30
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Count time (periods) spent at each price level
        tpo_counts = np.zeros(num_bins)

        for _, row in df.iterrows():
            low, high = row["low"], row["high"]

            # Find which bins this candle spans
            bin_low = np.searchsorted(bins, low, side="right") - 1
            bin_high = np.searchsorted(bins, high, side="left")

            bin_low = max(0, bin_low)
            bin_high = min(num_bins - 1, bin_high)

            # Add 1 TPO for each bin touched
            tpo_counts[bin_low:bin_high + 1] += 1

        # Find value area (70% of TPOs)
        total_tpos = np.sum(tpo_counts)
        poc_idx = np.argmax(tpo_counts)
        poc_price = float(bin_centers[poc_idx])

        # Calculate balance area
        target_tpos = total_tpos * 0.70
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        accumulated_tpos = tpo_counts[poc_idx]

        while accumulated_tpos < target_tpos:
            expand_low = va_low_idx > 0
            expand_high = va_high_idx < num_bins - 1

            if not expand_low and not expand_high:
                break

            tpo_low = tpo_counts[va_low_idx - 1] if expand_low else 0
            tpo_high = tpo_counts[va_high_idx + 1] if expand_high else 0

            if expand_low and (not expand_high or tpo_low >= tpo_high):
                va_low_idx -= 1
                accumulated_tpos += tpo_low
            elif expand_high:
                va_high_idx += 1
                accumulated_tpos += tpo_high

        balance_low = float(bin_centers[va_low_idx])
        balance_high = float(bin_centers[va_high_idx])

        # Determine profile type
        upper_tpos = np.sum(tpo_counts[poc_idx:])
        lower_tpos = np.sum(tpo_counts[:poc_idx + 1])

        if upper_tpos > lower_tpos * 1.3:
            profile_type = "p-shaped"
            profile_meaning = "Selling tail - potential reversal down"
        elif lower_tpos > upper_tpos * 1.3:
            profile_type = "b-shaped"
            profile_meaning = "Buying tail - potential reversal up"
        else:
            profile_type = "normal"
            profile_meaning = "Balanced distribution - range-bound"

        # Generate TPO visualization data
        tpo_profile = []
        for i in range(num_bins):
            if tpo_counts[i] > 0:
                tpo_profile.append({
                    "price": round(float(bin_centers[i]), 2),
                    "tpos": int(tpo_counts[i]),
                    "is_poc": i == poc_idx,
                    "in_value_area": va_low_idx <= i <= va_high_idx
                })

        return {
            "symbol": symbol,
            "period_days": period_days,
            "current_price": round(current_price, 2),
            "poc": {
                "price": round(poc_price, 2),
                "tpos": int(tpo_counts[poc_idx]),
            },
            "balance_area": {
                "high": round(balance_high, 2),
                "low": round(balance_low, 2),
            },
            "profile_type": profile_type,
            "profile_meaning": profile_meaning,
            "total_tpos": int(total_tpos),
            "tpo_distribution": tpo_profile,
            "price_position": "above_balance" if current_price > balance_high else ("below_balance" if current_price < balance_low else "inside_balance"),
            "interpretation": f"{symbol} {profile_type} profile with POC at ${poc_price:.2f}. Balance area: ${balance_low:.2f}-${balance_high:.2f}",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in volume_market_profile for {symbol}: {e}")
        return {"error": str(e), "status": "error"}


async def volume_footprint_analysis(
    symbol: str,
    lookback_bars: int = 20,
) -> dict[str, Any]:
    """
    Analyze volume delta and buying/selling pressure.

    Shows the balance between buying and selling volume to identify
    accumulation or distribution.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        lookback_bars: Number of bars to analyze (default: 20)

    Returns:
        Dictionary containing:
        - cumulative_delta: Running sum of (buy - sell) volume
        - delta_trend: Rising (accumulation) or falling (distribution)
        - buy_sell_ratio: Ratio of buying to selling volume
        - pressure: Current buying/selling pressure indicator
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()

    try:
        loop = asyncio.get_event_loop()

        start_date = (datetime.now() - timedelta(days=lookback_bars * 2)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        result = await loop.run_in_executor(
            None,
            lambda: obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                provider="yfinance"
            )
        )

        if not hasattr(result, "results") or not result.results:
            return {
                "error": f"Could not fetch data for {symbol}",
                "status": "error"
            }

        df = pd.DataFrame([r.model_dump() for r in result.results])
        df = df.tail(lookback_bars)

        if len(df) < 5:
            return {
                "error": f"Insufficient data for {symbol}",
                "status": "error"
            }

        # Estimate buy/sell volume based on price action
        # If close > open, assign more volume to buying; vice versa
        deltas = []
        buy_volumes = []
        sell_volumes = []

        for _, row in df.iterrows():
            open_price, close, high, low, volume = row["open"], row["close"], row["high"], row["low"], row["volume"]

            # Price range
            total_range = high - low if high > low else 0.01

            # Calculate buying vs selling ratio based on candle position
            if close >= open_price:
                # Bullish candle - estimate buying pressure
                buy_ratio = (close - low) / total_range
                sell_ratio = (high - close) / total_range
            else:
                # Bearish candle - estimate selling pressure
                buy_ratio = (high - close) / total_range
                sell_ratio = (close - low) / total_range

            # Normalize ratios
            total_ratio = buy_ratio + sell_ratio
            if total_ratio > 0:
                buy_ratio /= total_ratio
                sell_ratio /= total_ratio
            else:
                buy_ratio = sell_ratio = 0.5

            buy_vol = float(volume * buy_ratio)
            sell_vol = float(volume * sell_ratio)

            buy_volumes.append(buy_vol)
            sell_volumes.append(sell_vol)
            deltas.append(buy_vol - sell_vol)

        # Calculate cumulative delta
        cumulative_delta = np.cumsum(deltas)
        current_cum_delta = float(cumulative_delta[-1])

        # Calculate delta trend (is it rising or falling?)
        if len(cumulative_delta) >= 5:
            recent_delta = cumulative_delta[-5:]
            delta_slope = (recent_delta[-1] - recent_delta[0]) / 5
            if delta_slope > 0:
                delta_trend = "rising"
                trend_meaning = "Accumulation - buyers in control"
            elif delta_slope < 0:
                delta_trend = "falling"
                trend_meaning = "Distribution - sellers in control"
            else:
                delta_trend = "flat"
                trend_meaning = "Neutral - balanced volume"
        else:
            delta_trend = "insufficient_data"
            trend_meaning = "Not enough data for trend"

        # Calculate overall buy/sell ratio
        total_buy = sum(buy_volumes)
        total_sell = sum(sell_volumes)
        buy_sell_ratio = total_buy / total_sell if total_sell > 0 else 1.0

        # Determine pressure
        if buy_sell_ratio >= 1.3:
            pressure = "strong_buying"
            pressure_signal = "bullish"
        elif buy_sell_ratio >= 1.1:
            pressure = "moderate_buying"
            pressure_signal = "lean_bullish"
        elif buy_sell_ratio <= 0.7:
            pressure = "strong_selling"
            pressure_signal = "bearish"
        elif buy_sell_ratio <= 0.9:
            pressure = "moderate_selling"
            pressure_signal = "lean_bearish"
        else:
            pressure = "neutral"
            pressure_signal = "neutral"

        # Recent bar analysis
        recent_deltas = deltas[-5:]
        positive_deltas = sum(1 for d in recent_deltas if d > 0)
        negative_deltas = sum(1 for d in recent_deltas if d < 0)

        return {
            "symbol": symbol,
            "lookback_bars": lookback_bars,
            "current_price": round(float(df["close"].iloc[-1]), 2),
            "cumulative_delta": round(current_cum_delta, 0),
            "delta_trend": delta_trend,
            "trend_meaning": trend_meaning,
            "buy_sell_ratio": round(buy_sell_ratio, 2),
            "total_buy_volume": round(total_buy, 0),
            "total_sell_volume": round(total_sell, 0),
            "pressure": pressure,
            "pressure_signal": pressure_signal,
            "recent_analysis": {
                "positive_delta_bars": positive_deltas,
                "negative_delta_bars": negative_deltas,
                "recent_bias": "buying" if positive_deltas > negative_deltas else ("selling" if negative_deltas > positive_deltas else "neutral")
            },
            "interpretation": f"{symbol} shows {pressure} pressure with {delta_trend} cumulative delta. Buy/Sell ratio: {buy_sell_ratio:.2f}",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in volume_footprint_analysis for {symbol}: {e}")
        return {"error": str(e), "status": "error"}
