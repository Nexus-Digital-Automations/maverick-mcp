"""
Multi-Timeframe Confirmation Analysis Module.

Provides tools for analyzing trend alignment and signal confirmation
across multiple timeframes (daily, weekly, monthly).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from openbb import obb

logger = logging.getLogger(__name__)


async def multi_timeframe_trend(
    symbol: str,
    timeframes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze trend direction across multiple timeframes with alignment score.

    Checks if daily, weekly, and monthly trends are aligned for stronger signals.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        timeframes: List of timeframes to analyze (default: ['1d', '1wk', '1mo'])

    Returns:
        Dictionary containing:
        - trends: Trend direction per timeframe (bullish/bearish/neutral)
        - alignment_score: Percentage of timeframes aligned (0-100)
        - dominant_trend: Overall trend based on majority
        - signal_strength: weak/moderate/strong based on alignment
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()
    timeframes = timeframes or ["1d", "1wk", "1mo"]

    try:
        loop = asyncio.get_event_loop()
        trends = {}
        prices_by_tf = {}

        # Fetch data for each timeframe
        for tf in timeframes:
            try:
                # Determine lookback period based on timeframe
                if tf == "1d":
                    days_back = 60
                elif tf == "1wk":
                    days_back = 365
                else:  # 1mo
                    days_back = 730

                start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                result = await loop.run_in_executor(
                    None,
                    lambda tf=tf, start=start_date, end=end_date: obb.equity.price.historical(
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        interval=tf,
                        provider="yfinance"
                    )
                )

                if hasattr(result, "results") and result.results:
                    df = pd.DataFrame([r.model_dump() for r in result.results])
                    if len(df) >= 10:
                        prices_by_tf[tf] = df
            except Exception as e:
                logger.debug(f"Could not fetch {tf} data for {symbol}: {e}")

        if not prices_by_tf:
            return {
                "error": f"Could not fetch data for {symbol}",
                "status": "error"
            }

        # Analyze trend for each timeframe
        for tf, df in prices_by_tf.items():
            close = df["close"].values

            # Calculate SMAs
            sma_20 = np.mean(close[-20:]) if len(close) >= 20 else np.mean(close)
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)

            # Current price vs SMAs
            current_price = close[-1]

            # Trend determination
            price_above_sma20 = current_price > sma_20
            price_above_sma50 = current_price > sma_50
            sma20_above_sma50 = sma_20 > sma_50

            # Calculate price change
            if len(close) >= 20:
                price_change = (close[-1] - close[-20]) / close[-20] * 100
            else:
                price_change = (close[-1] - close[0]) / close[0] * 100

            # Determine trend direction
            bullish_signals = sum([price_above_sma20, price_above_sma50, sma20_above_sma50, price_change > 0])

            if bullish_signals >= 3:
                trend = "bullish"
            elif bullish_signals <= 1:
                trend = "bearish"
            else:
                trend = "neutral"

            trends[tf] = {
                "trend": trend,
                "price": float(current_price),
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "price_change_pct": float(round(price_change, 2)),
                "price_above_sma20": price_above_sma20,
                "price_above_sma50": price_above_sma50,
            }

        # Calculate alignment
        trend_values = [t["trend"] for t in trends.values()]
        bullish_count = trend_values.count("bullish")
        bearish_count = trend_values.count("bearish")
        neutral_count = trend_values.count("neutral")

        total_tf = len(trend_values)

        # Determine dominant trend
        if bullish_count > bearish_count:
            dominant_trend = "bullish"
            aligned_count = bullish_count
        elif bearish_count > bullish_count:
            dominant_trend = "bearish"
            aligned_count = bearish_count
        else:
            dominant_trend = "neutral"
            aligned_count = max(bullish_count, bearish_count)

        alignment_score = round((aligned_count / total_tf) * 100, 1)

        # Signal strength
        if alignment_score >= 100:
            signal_strength = "strong"
        elif alignment_score >= 66:
            signal_strength = "moderate"
        else:
            signal_strength = "weak"

        return {
            "symbol": symbol,
            "timeframes_analyzed": list(trends.keys()),
            "trends": trends,
            "alignment_score": alignment_score,
            "dominant_trend": dominant_trend,
            "signal_strength": signal_strength,
            "bullish_timeframes": bullish_count,
            "bearish_timeframes": bearish_count,
            "neutral_timeframes": neutral_count,
            "interpretation": f"{symbol} shows {dominant_trend} trend across {aligned_count}/{total_tf} timeframes ({signal_strength} signal)",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in multi_timeframe_trend for {symbol}: {e}")
        return {"error": str(e), "status": "error"}


async def multi_timeframe_rsi(
    symbol: str,
    timeframes: list[str] | None = None,
    period: int = 14,
) -> dict[str, Any]:
    """
    Calculate RSI across multiple timeframes with divergence detection.

    Useful for identifying overbought/oversold conditions confirmed by multiple timeframes.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        timeframes: List of timeframes to analyze (default: ['1d', '1wk', '1mo'])
        period: RSI period (default: 14)

    Returns:
        Dictionary containing:
        - rsi_values: RSI for each timeframe
        - overbought_count: Number of timeframes showing overbought (>70)
        - oversold_count: Number of timeframes showing oversold (<30)
        - divergence: Detected divergences between timeframes
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()
    timeframes = timeframes or ["1d", "1wk", "1mo"]

    try:
        loop = asyncio.get_event_loop()
        rsi_data = {}

        def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
            """Calculate RSI from price array."""
            if len(prices) < period + 1:
                return 50.0  # Return neutral if not enough data

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)

        # Fetch data and calculate RSI for each timeframe
        for tf in timeframes:
            try:
                if tf == "1d":
                    days_back = 60
                elif tf == "1wk":
                    days_back = 365
                else:
                    days_back = 730

                start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                result = await loop.run_in_executor(
                    None,
                    lambda tf=tf, start=start_date, end=end_date: obb.equity.price.historical(
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        interval=tf,
                        provider="yfinance"
                    )
                )

                if hasattr(result, "results") and result.results:
                    df = pd.DataFrame([r.model_dump() for r in result.results])
                    if len(df) >= period + 1:
                        close = df["close"].values
                        rsi_value = calculate_rsi(close, period)

                        # Determine condition
                        if rsi_value >= 70:
                            condition = "overbought"
                        elif rsi_value <= 30:
                            condition = "oversold"
                        elif rsi_value >= 60:
                            condition = "bullish"
                        elif rsi_value <= 40:
                            condition = "bearish"
                        else:
                            condition = "neutral"

                        rsi_data[tf] = {
                            "rsi": round(rsi_value, 2),
                            "condition": condition,
                            "period": period,
                        }
            except Exception as e:
                logger.debug(f"Could not calculate RSI for {tf}: {e}")

        if not rsi_data:
            return {
                "error": f"Could not calculate RSI for {symbol}",
                "status": "error"
            }

        # Count conditions
        overbought_count = sum(1 for d in rsi_data.values() if d["condition"] == "overbought")
        oversold_count = sum(1 for d in rsi_data.values() if d["condition"] == "oversold")
        bullish_count = sum(1 for d in rsi_data.values() if d["condition"] in ["bullish", "overbought"])
        bearish_count = sum(1 for d in rsi_data.values() if d["condition"] in ["bearish", "oversold"])

        # Check for divergences
        rsi_values = [d["rsi"] for d in rsi_data.values()]
        divergence = None

        if len(rsi_values) >= 2:
            rsi_range = max(rsi_values) - min(rsi_values)
            if rsi_range > 20:  # Significant divergence
                if "1d" in rsi_data and "1wk" in rsi_data:
                    daily_rsi = rsi_data["1d"]["rsi"]
                    weekly_rsi = rsi_data["1wk"]["rsi"]

                    if daily_rsi > 60 and weekly_rsi < 40:
                        divergence = {
                            "type": "bearish_divergence",
                            "description": "Daily RSI bullish but weekly RSI bearish - potential reversal"
                        }
                    elif daily_rsi < 40 and weekly_rsi > 60:
                        divergence = {
                            "type": "bullish_divergence",
                            "description": "Daily RSI bearish but weekly RSI bullish - potential bounce"
                        }

        # Overall signal
        if overbought_count >= len(rsi_data) / 2:
            overall_signal = "overbought"
        elif oversold_count >= len(rsi_data) / 2:
            overall_signal = "oversold"
        elif bullish_count > bearish_count:
            overall_signal = "bullish"
        elif bearish_count > bullish_count:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        return {
            "symbol": symbol,
            "rsi_period": period,
            "timeframes_analyzed": list(rsi_data.keys()),
            "rsi_data": rsi_data,
            "overbought_count": overbought_count,
            "oversold_count": oversold_count,
            "overall_signal": overall_signal,
            "divergence": divergence,
            "interpretation": f"{symbol} RSI shows {overall_signal} across {len(rsi_data)} timeframes",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in multi_timeframe_rsi for {symbol}: {e}")
        return {"error": str(e), "status": "error"}


async def multi_timeframe_moving_averages(
    symbol: str,
    timeframes: list[str] | None = None,
    ma_periods: list[int] | None = None,
) -> dict[str, Any]:
    """
    Analyze moving average alignment across timeframes.

    Detects golden cross / death cross patterns confirmed across multiple timeframes.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        timeframes: List of timeframes to analyze (default: ['1d', '1wk', '1mo'])
        ma_periods: Moving average periods to use (default: [20, 50, 200])

    Returns:
        Dictionary containing:
        - ma_data: MA values and alignment for each timeframe
        - golden_cross_count: Timeframes with golden cross (50 > 200)
        - death_cross_count: Timeframes with death cross (50 < 200)
        - alignment_signal: Overall MA alignment signal
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()
    timeframes = timeframes or ["1d", "1wk", "1mo"]
    ma_periods = ma_periods or [20, 50, 200]

    try:
        loop = asyncio.get_event_loop()
        ma_data = {}

        for tf in timeframes:
            try:
                if tf == "1d":
                    days_back = 400  # Need enough for 200 SMA
                elif tf == "1wk":
                    days_back = 1500
                else:
                    days_back = 2500

                start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                result = await loop.run_in_executor(
                    None,
                    lambda tf=tf, start=start_date, end=end_date: obb.equity.price.historical(
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        interval=tf,
                        provider="yfinance"
                    )
                )

                if hasattr(result, "results") and result.results:
                    df = pd.DataFrame([r.model_dump() for r in result.results])
                    close = df["close"].values
                    current_price = float(close[-1])

                    mas = {}
                    for period in ma_periods:
                        if len(close) >= period:
                            mas[f"sma_{period}"] = float(round(np.mean(close[-period:]), 2))
                        else:
                            mas[f"sma_{period}"] = None

                    # Calculate crossovers
                    crossovers = {}
                    if mas.get("sma_50") and mas.get("sma_200"):
                        if mas["sma_50"] > mas["sma_200"]:
                            crossovers["50_200"] = "golden_cross"
                        else:
                            crossovers["50_200"] = "death_cross"

                    if mas.get("sma_20") and mas.get("sma_50"):
                        if mas["sma_20"] > mas["sma_50"]:
                            crossovers["20_50"] = "bullish"
                        else:
                            crossovers["20_50"] = "bearish"

                    # Price position
                    price_above_all = all(
                        current_price > ma for ma in mas.values() if ma is not None
                    )
                    price_below_all = all(
                        current_price < ma for ma in mas.values() if ma is not None
                    )

                    ma_data[tf] = {
                        "current_price": current_price,
                        "moving_averages": mas,
                        "crossovers": crossovers,
                        "price_above_all_mas": price_above_all,
                        "price_below_all_mas": price_below_all,
                    }
            except Exception as e:
                logger.debug(f"Could not calculate MAs for {tf}: {e}")

        if not ma_data:
            return {
                "error": f"Could not calculate moving averages for {symbol}",
                "status": "error"
            }

        # Count signals
        golden_cross_count = sum(
            1 for d in ma_data.values()
            if d.get("crossovers", {}).get("50_200") == "golden_cross"
        )
        death_cross_count = sum(
            1 for d in ma_data.values()
            if d.get("crossovers", {}).get("50_200") == "death_cross"
        )
        price_above_count = sum(
            1 for d in ma_data.values() if d.get("price_above_all_mas")
        )
        price_below_count = sum(
            1 for d in ma_data.values() if d.get("price_below_all_mas")
        )

        # Overall signal
        total_tf = len(ma_data)
        if golden_cross_count > death_cross_count and price_above_count >= total_tf / 2:
            alignment_signal = "strongly_bullish"
        elif golden_cross_count > death_cross_count:
            alignment_signal = "bullish"
        elif death_cross_count > golden_cross_count and price_below_count >= total_tf / 2:
            alignment_signal = "strongly_bearish"
        elif death_cross_count > golden_cross_count:
            alignment_signal = "bearish"
        else:
            alignment_signal = "mixed"

        return {
            "symbol": symbol,
            "ma_periods": ma_periods,
            "timeframes_analyzed": list(ma_data.keys()),
            "ma_data": ma_data,
            "golden_cross_count": golden_cross_count,
            "death_cross_count": death_cross_count,
            "price_above_all_mas_count": price_above_count,
            "price_below_all_mas_count": price_below_count,
            "alignment_signal": alignment_signal,
            "interpretation": f"{symbol} MA alignment is {alignment_signal} with {golden_cross_count} golden crosses, {death_cross_count} death crosses",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in multi_timeframe_moving_averages for {symbol}: {e}")
        return {"error": str(e), "status": "error"}


async def multi_timeframe_signal_score(
    symbol: str,
) -> dict[str, Any]:
    """
    Calculate composite multi-timeframe signal score (0-100).

    Combines trend, RSI, and MA analysis into a single actionable score.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')

    Returns:
        Dictionary containing:
        - composite_score: Overall score 0-100 (higher = more bullish)
        - component_scores: Individual scores from each analysis
        - signal: buy/sell/hold recommendation
        - confidence: low/medium/high based on alignment
    """
    if not symbol:
        return {"error": "Symbol required", "status": "error"}

    symbol = symbol.upper()

    try:
        # Run all analyses in parallel
        trend_task = multi_timeframe_trend(symbol)
        rsi_task = multi_timeframe_rsi(symbol)
        ma_task = multi_timeframe_moving_averages(symbol)

        trend_result, rsi_result, ma_result = await asyncio.gather(
            trend_task, rsi_task, ma_task
        )

        scores = {}
        valid_analyses = 0

        # Score from trend analysis (0-100)
        if trend_result.get("status") == "success":
            trend_score = 50  # Base neutral
            if trend_result.get("dominant_trend") == "bullish":
                trend_score = 50 + (trend_result.get("alignment_score", 0) / 2)
            elif trend_result.get("dominant_trend") == "bearish":
                trend_score = 50 - (trend_result.get("alignment_score", 0) / 2)
            scores["trend"] = round(trend_score, 1)
            valid_analyses += 1

        # Score from RSI analysis (0-100)
        if rsi_result.get("status") == "success":
            rsi_score = 50
            overall_signal = rsi_result.get("overall_signal", "neutral")
            if overall_signal == "oversold":
                rsi_score = 80  # Contrarian bullish
            elif overall_signal == "overbought":
                rsi_score = 20  # Contrarian bearish
            elif overall_signal == "bullish":
                rsi_score = 65
            elif overall_signal == "bearish":
                rsi_score = 35
            scores["rsi"] = rsi_score
            valid_analyses += 1

        # Score from MA analysis (0-100)
        if ma_result.get("status") == "success":
            ma_score = 50
            alignment = ma_result.get("alignment_signal", "mixed")
            if alignment == "strongly_bullish":
                ma_score = 90
            elif alignment == "bullish":
                ma_score = 70
            elif alignment == "strongly_bearish":
                ma_score = 10
            elif alignment == "bearish":
                ma_score = 30
            scores["ma"] = ma_score
            valid_analyses += 1

        if valid_analyses == 0:
            return {
                "error": f"Could not complete analysis for {symbol}",
                "status": "error"
            }

        # Calculate composite score (weighted average)
        weights = {"trend": 0.35, "rsi": 0.25, "ma": 0.40}
        composite_score = 0
        total_weight = 0

        for key, score in scores.items():
            weight = weights.get(key, 0.33)
            composite_score += score * weight
            total_weight += weight

        composite_score = round(composite_score / total_weight, 1) if total_weight > 0 else 50

        # Determine signal
        if composite_score >= 70:
            signal = "buy"
        elif composite_score >= 55:
            signal = "lean_bullish"
        elif composite_score <= 30:
            signal = "sell"
        elif composite_score <= 45:
            signal = "lean_bearish"
        else:
            signal = "hold"

        # Determine confidence based on score agreement
        score_values = list(scores.values())
        score_range = max(score_values) - min(score_values) if score_values else 0

        if score_range <= 15:
            confidence = "high"
        elif score_range <= 30:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "symbol": symbol,
            "composite_score": composite_score,
            "component_scores": scores,
            "signal": signal,
            "confidence": confidence,
            "analyses_completed": valid_analyses,
            "trend_summary": trend_result.get("interpretation", "N/A"),
            "rsi_summary": rsi_result.get("interpretation", "N/A"),
            "ma_summary": ma_result.get("interpretation", "N/A"),
            "interpretation": f"{symbol} composite MTF score: {composite_score}/100 - Signal: {signal.upper()} ({confidence} confidence)",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in multi_timeframe_signal_score for {symbol}: {e}")
        return {"error": str(e), "status": "error"}
