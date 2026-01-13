"""
Unified ML Predictions Tool.

Provides machine learning based market predictions:
- price_forecast: Multi-day price prediction with confidence intervals
- pattern_recognition: Technical patterns (H&S, triangles, flags, etc.)
- regime_prediction: Market regime classification
- trend_prediction: Trend direction probability
- ensemble: Combined predictions (default)
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Technical pattern definitions
PATTERN_DEFINITIONS = {
    "head_and_shoulders": {
        "description": "Bearish reversal pattern with three peaks",
        "signal": "bearish",
    },
    "inverse_head_and_shoulders": {
        "description": "Bullish reversal pattern with three troughs",
        "signal": "bullish",
    },
    "double_top": {
        "description": "Bearish reversal with two similar highs",
        "signal": "bearish",
    },
    "double_bottom": {
        "description": "Bullish reversal with two similar lows",
        "signal": "bullish",
    },
    "ascending_triangle": {
        "description": "Bullish continuation with flat top, rising bottom",
        "signal": "bullish",
    },
    "descending_triangle": {
        "description": "Bearish continuation with flat bottom, falling top",
        "signal": "bearish",
    },
    "symmetrical_triangle": {
        "description": "Consolidation with converging trend lines",
        "signal": "neutral",
    },
    "bull_flag": {
        "description": "Bullish continuation after sharp rise",
        "signal": "bullish",
    },
    "bear_flag": {
        "description": "Bearish continuation after sharp decline",
        "signal": "bearish",
    },
    "cup_and_handle": {
        "description": "Bullish continuation with U-shaped base",
        "signal": "bullish",
    },
}


async def ml_predictions(
    symbol: str,
    analysis_type: str = "ensemble",
    forecast_days: int = 30,
    lookback_days: int = 252,
) -> dict[str, Any]:
    """
    Machine learning based stock predictions and pattern recognition.

    Provides price forecasts, pattern detection, regime classification,
    and trend predictions using statistical and ML techniques.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'NVDA')
        analysis_type: Type of ML analysis:
            - 'price_forecast': Multi-day price prediction with confidence intervals
            - 'pattern_recognition': Detect technical chart patterns
            - 'regime_prediction': Classify current market regime
            - 'trend_prediction': Predict trend direction probability
            - 'ensemble': Combined predictions from all models (default)
        forecast_days: Number of days to forecast (default 30)
        lookback_days: Historical data for training/analysis (default 252)

    Returns:
        Dictionary containing ML prediction results.

    Examples:
        # Get ensemble predictions
        >>> ml_predictions("AAPL")

        # Get price forecast for 30 days
        >>> ml_predictions("NVDA", analysis_type="price_forecast", forecast_days=30)

        # Detect chart patterns
        >>> ml_predictions("TSLA", analysis_type="pattern_recognition")

        # Get regime classification
        >>> ml_predictions("SPY", analysis_type="regime_prediction")
    """
    import yfinance as yf

    symbol = symbol.strip().upper()
    analysis_type = analysis_type.lower().strip()

    valid_types = [
        "price_forecast", "pattern_recognition",
        "regime_prediction", "trend_prediction", "ensemble"
    ]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 50)

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty or len(hist) < 60:
            return {
                "error": f"Insufficient historical data for {symbol}. Need at least 60 days.",
                "symbol": symbol,
                "status": "error",
            }

        if analysis_type == "price_forecast":
            return await _price_forecast(symbol, hist, forecast_days)

        elif analysis_type == "pattern_recognition":
            return await _pattern_recognition(symbol, hist)

        elif analysis_type == "regime_prediction":
            return await _regime_prediction(symbol, hist)

        elif analysis_type == "trend_prediction":
            return await _trend_prediction(symbol, hist, forecast_days)

        else:  # ensemble
            price_forecast = await _price_forecast(symbol, hist, forecast_days)
            patterns = await _pattern_recognition(symbol, hist)
            regime = await _regime_prediction(symbol, hist)
            trend = await _trend_prediction(symbol, hist, forecast_days)

            # Combine signals
            signals = []
            if price_forecast.get("forecast", {}).get("expected_return", 0) > 0:
                signals.append("bullish_forecast")
            else:
                signals.append("bearish_forecast")

            if patterns.get("detected_patterns"):
                for p in patterns["detected_patterns"]:
                    if p.get("signal") == "bullish":
                        signals.append("bullish_pattern")
                    elif p.get("signal") == "bearish":
                        signals.append("bearish_pattern")

            if trend.get("trend_probability", {}).get("up", 0) > 0.6:
                signals.append("bullish_trend")
            elif trend.get("trend_probability", {}).get("down", 0) > 0.6:
                signals.append("bearish_trend")

            # Overall signal
            bullish_count = sum(1 for s in signals if "bullish" in s)
            bearish_count = sum(1 for s in signals if "bearish" in s)

            if bullish_count > bearish_count + 1:
                overall_signal = "bullish"
                confidence = min(0.9, 0.5 + (bullish_count - bearish_count) * 0.1)
            elif bearish_count > bullish_count + 1:
                overall_signal = "bearish"
                confidence = min(0.9, 0.5 + (bearish_count - bullish_count) * 0.1)
            else:
                overall_signal = "neutral"
                confidence = 0.5

            return {
                "symbol": symbol,
                "analysis_type": "ensemble",
                "current_price": round(float(hist["Close"].iloc[-1]), 2),
                "forecast_summary": {
                    "target_price": price_forecast.get("forecast", {}).get("target_price"),
                    "expected_return": price_forecast.get("forecast", {}).get("expected_return"),
                    "confidence_range": price_forecast.get("forecast", {}).get("confidence_interval"),
                },
                "patterns_detected": [
                    p["pattern"] for p in patterns.get("detected_patterns", [])
                ],
                "regime": regime.get("regime"),
                "trend_probability": trend.get("trend_probability"),
                "ensemble_signal": {
                    "direction": overall_signal,
                    "confidence": round(confidence, 2),
                    "component_signals": signals,
                },
                "interpretation": (
                    f"Ensemble signal: {overall_signal.upper()} ({confidence:.0%} confidence). "
                    f"Regime: {regime.get('regime', 'unknown')}. "
                    f"Patterns: {', '.join([p['pattern'] for p in patterns.get('detected_patterns', [])[:2]]) or 'None detected'}."
                ),
                "status": "success",
            }

    except Exception as e:
        logger.error(f"Error in ml_predictions for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }


async def _price_forecast(
    symbol: str, hist: pd.DataFrame, forecast_days: int
) -> dict[str, Any]:
    """Generate price forecast using statistical methods."""
    try:
        closes = hist["Close"].values
        returns = np.diff(np.log(closes))

        current_price = closes[-1]

        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Monte Carlo simulation
        n_simulations = 1000
        simulated_returns = np.random.normal(
            mean_return, std_return, (n_simulations, forecast_days)
        )
        simulated_prices = current_price * np.exp(
            np.cumsum(simulated_returns, axis=1)
        )

        # Get forecast statistics
        final_prices = simulated_prices[:, -1]
        target_price = np.median(final_prices)
        lower_bound = np.percentile(final_prices, 10)
        upper_bound = np.percentile(final_prices, 90)

        expected_return = (target_price / current_price - 1) * 100

        # Probability of profit
        prob_profit = np.mean(final_prices > current_price)

        # Path forecasts (5th, 50th, 95th percentiles)
        path_low = np.percentile(simulated_prices, 5, axis=0)
        path_median = np.percentile(simulated_prices, 50, axis=0)
        path_high = np.percentile(simulated_prices, 95, axis=0)

        forecast_path = []
        for i in range(0, forecast_days, max(1, forecast_days // 10)):
            forecast_path.append({
                "day": i + 1,
                "low": round(path_low[i], 2),
                "median": round(path_median[i], 2),
                "high": round(path_high[i], 2),
            })

        return {
            "symbol": symbol,
            "analysis_type": "price_forecast",
            "current_price": round(current_price, 2),
            "forecast_days": forecast_days,
            "forecast": {
                "target_price": round(target_price, 2),
                "expected_return": round(expected_return, 2),
                "confidence_interval": {
                    "lower_10pct": round(lower_bound, 2),
                    "upper_90pct": round(upper_bound, 2),
                },
                "probability_of_profit": round(prob_profit * 100, 1),
            },
            "forecast_path": forecast_path,
            "model_parameters": {
                "daily_mean_return": round(mean_return * 100, 4),
                "daily_volatility": round(std_return * 100, 2),
                "annualized_volatility": round(std_return * np.sqrt(252) * 100, 1),
            },
            "interpretation": (
                f"Target price ${target_price:.2f} ({expected_return:+.1f}%) in {forecast_days} days. "
                f"80% confidence range: ${lower_bound:.2f} - ${upper_bound:.2f}. "
                f"{prob_profit*100:.0f}% probability of profit."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in price forecast: {e}")
        return {"error": str(e), "symbol": symbol, "status": "error"}


async def _pattern_recognition(
    symbol: str, hist: pd.DataFrame
) -> dict[str, Any]:
    """Detect technical chart patterns."""
    try:
        closes = hist["Close"].values
        highs = hist["High"].values
        lows = hist["Low"].values

        detected_patterns = []

        # Use last 60 days for pattern detection
        window = min(60, len(closes) - 1)
        recent_closes = closes[-window:]
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]

        # Detect patterns using simplified heuristics

        # 1. Double Top/Bottom
        peaks, troughs = _find_peaks_troughs(recent_closes)

        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if abs(recent_highs[last_two_peaks[0]] - recent_highs[last_two_peaks[1]]) < recent_highs[last_two_peaks[0]] * 0.02:
                detected_patterns.append({
                    "pattern": "double_top",
                    "confidence": 0.7,
                    "signal": "bearish",
                    "description": PATTERN_DEFINITIONS["double_top"]["description"],
                })

        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            if abs(recent_lows[last_two_troughs[0]] - recent_lows[last_two_troughs[1]]) < recent_lows[last_two_troughs[0]] * 0.02:
                detected_patterns.append({
                    "pattern": "double_bottom",
                    "confidence": 0.7,
                    "signal": "bullish",
                    "description": PATTERN_DEFINITIONS["double_bottom"]["description"],
                })

        # 2. Triangle Patterns
        triangle = _detect_triangle(recent_closes, recent_highs, recent_lows)
        if triangle:
            detected_patterns.append(triangle)

        # 3. Flag Patterns
        flag = _detect_flag(recent_closes)
        if flag:
            detected_patterns.append(flag)

        # 4. Trend Channels
        trend_channel = _detect_trend_channel(recent_closes)
        if trend_channel:
            detected_patterns.append(trend_channel)

        # Sort by confidence
        detected_patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # Overall pattern signal
        overall_signal = "neutral"
        if detected_patterns:
            bullish = sum(1 for p in detected_patterns if p.get("signal") == "bullish")
            bearish = sum(1 for p in detected_patterns if p.get("signal") == "bearish")
            if bullish > bearish:
                overall_signal = "bullish"
            elif bearish > bullish:
                overall_signal = "bearish"

        return {
            "symbol": symbol,
            "analysis_type": "pattern_recognition",
            "current_price": round(closes[-1], 2),
            "detected_patterns": detected_patterns[:5],  # Top 5 patterns
            "pattern_count": len(detected_patterns),
            "overall_signal": overall_signal,
            "interpretation": (
                f"Detected {len(detected_patterns)} pattern(s). "
                f"Overall signal: {overall_signal.upper()}. "
                + (f"Primary pattern: {detected_patterns[0]['pattern']}." if detected_patterns else "No strong patterns detected.")
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in pattern recognition: {e}")
        return {"error": str(e), "symbol": symbol, "status": "error"}


async def _regime_prediction(
    symbol: str, hist: pd.DataFrame
) -> dict[str, Any]:
    """Classify current market regime."""
    try:
        closes = hist["Close"].values
        volumes = hist["Volume"].values
        returns = np.diff(np.log(closes))

        # Calculate regime indicators
        # 1. Volatility regime
        recent_vol = np.std(returns[-20:]) * np.sqrt(252)
        historical_vol = np.std(returns) * np.sqrt(252)

        vol_regime = "normal"
        if recent_vol > historical_vol * 1.5:
            vol_regime = "high_volatility"
        elif recent_vol < historical_vol * 0.5:
            vol_regime = "low_volatility"

        # 2. Trend regime (using moving averages)
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else sma_50

        current_price = closes[-1]

        trend_regime = "neutral"
        if current_price > sma_20 > sma_50 > sma_200:
            trend_regime = "strong_uptrend"
        elif current_price > sma_50:
            trend_regime = "uptrend"
        elif current_price < sma_20 < sma_50 < sma_200:
            trend_regime = "strong_downtrend"
        elif current_price < sma_50:
            trend_regime = "downtrend"
        else:
            trend_regime = "sideways"

        # 3. Momentum regime
        roc_20 = (closes[-1] / closes[-20] - 1) * 100
        roc_50 = (closes[-1] / closes[-50] - 1) * 100

        momentum_regime = "neutral"
        if roc_20 > 10 and roc_50 > 15:
            momentum_regime = "strong_momentum"
        elif roc_20 > 5:
            momentum_regime = "positive_momentum"
        elif roc_20 < -10 and roc_50 < -15:
            momentum_regime = "strong_negative_momentum"
        elif roc_20 < -5:
            momentum_regime = "negative_momentum"

        # 4. Volume regime
        avg_volume = np.mean(volumes[-50:])
        recent_volume = np.mean(volumes[-5:])

        volume_regime = "normal"
        if recent_volume > avg_volume * 1.5:
            volume_regime = "high_volume"
        elif recent_volume < avg_volume * 0.5:
            volume_regime = "low_volume"

        # Overall regime classification
        if trend_regime in ["strong_uptrend", "uptrend"] and vol_regime != "high_volatility":
            overall_regime = "bull_market"
            confidence = 0.8 if trend_regime == "strong_uptrend" else 0.6
        elif trend_regime in ["strong_downtrend", "downtrend"]:
            overall_regime = "bear_market"
            confidence = 0.8 if trend_regime == "strong_downtrend" else 0.6
        elif vol_regime == "high_volatility":
            overall_regime = "high_volatility"
            confidence = 0.7
        elif trend_regime == "sideways" and vol_regime == "low_volatility":
            overall_regime = "consolidation"
            confidence = 0.6
        else:
            overall_regime = "transitional"
            confidence = 0.5

        return {
            "symbol": symbol,
            "analysis_type": "regime_prediction",
            "current_price": round(current_price, 2),
            "regime": overall_regime,
            "confidence": round(confidence, 2),
            "regime_components": {
                "volatility": vol_regime,
                "trend": trend_regime,
                "momentum": momentum_regime,
                "volume": volume_regime,
            },
            "metrics": {
                "recent_volatility": round(recent_vol * 100, 1),
                "historical_volatility": round(historical_vol * 100, 1),
                "roc_20d": round(roc_20, 2),
                "roc_50d": round(roc_50, 2),
                "price_vs_sma50": round((current_price / sma_50 - 1) * 100, 2),
            },
            "interpretation": (
                f"Current regime: {overall_regime.replace('_', ' ').upper()} "
                f"({confidence:.0%} confidence). "
                f"Trend: {trend_regime}, Volatility: {vol_regime}."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in regime prediction: {e}")
        return {"error": str(e), "symbol": symbol, "status": "error"}


async def _trend_prediction(
    symbol: str, hist: pd.DataFrame, forecast_days: int
) -> dict[str, Any]:
    """Predict trend direction probability."""
    try:
        closes = hist["Close"].values
        returns = np.diff(np.log(closes))

        # Calculate features for trend prediction
        # 1. Momentum indicators
        rsi = _calculate_rsi(closes, 14)
        macd, signal = _calculate_macd(closes)

        # 2. Moving average signals
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])

        current_price = closes[-1]

        # 3. Historical trend analysis
        positive_returns = np.sum(returns[-forecast_days:] > 0) / forecast_days

        # Calculate trend probabilities
        # Base probability from historical returns
        base_prob_up = positive_returns

        # Adjust based on indicators
        adjustments = 0.0

        # RSI adjustment
        if rsi < 30:  # Oversold - bullish
            adjustments += 0.1
        elif rsi > 70:  # Overbought - bearish
            adjustments -= 0.1

        # MACD adjustment
        if macd > signal:  # Bullish crossover
            adjustments += 0.05
        else:  # Bearish
            adjustments -= 0.05

        # Moving average adjustment
        if current_price > sma_20 > sma_50:  # Uptrend
            adjustments += 0.1
        elif current_price < sma_20 < sma_50:  # Downtrend
            adjustments -= 0.1

        # Final probabilities
        prob_up = min(0.9, max(0.1, base_prob_up + adjustments))
        prob_down = min(0.9, max(0.1, 1 - prob_up - 0.1))  # Leave 10% for sideways
        prob_sideways = 1 - prob_up - prob_down

        # Trend strength
        trend_strength = abs(prob_up - prob_down)
        if trend_strength > 0.4:
            strength_label = "strong"
        elif trend_strength > 0.2:
            strength_label = "moderate"
        else:
            strength_label = "weak"

        # Direction
        if prob_up > prob_down + 0.1:
            direction = "up"
        elif prob_down > prob_up + 0.1:
            direction = "down"
        else:
            direction = "sideways"

        return {
            "symbol": symbol,
            "analysis_type": "trend_prediction",
            "current_price": round(current_price, 2),
            "forecast_days": forecast_days,
            "trend_probability": {
                "up": round(prob_up, 2),
                "down": round(prob_down, 2),
                "sideways": round(prob_sideways, 2),
            },
            "prediction": {
                "direction": direction,
                "strength": strength_label,
                "confidence": round(max(prob_up, prob_down, prob_sideways), 2),
            },
            "indicators": {
                "rsi_14": round(rsi, 1),
                "macd_signal": "bullish" if macd > signal else "bearish",
                "price_vs_sma20": round((current_price / sma_20 - 1) * 100, 2),
                "price_vs_sma50": round((current_price / sma_50 - 1) * 100, 2),
            },
            "interpretation": (
                f"Trend prediction: {direction.upper()} ({strength_label} signal). "
                f"Probabilities: {prob_up:.0%} up, {prob_down:.0%} down, {prob_sideways:.0%} sideways. "
                f"RSI: {rsi:.0f}, MACD: {'bullish' if macd > signal else 'bearish'}."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in trend prediction: {e}")
        return {"error": str(e), "symbol": symbol, "status": "error"}


# Helper functions

def _find_peaks_troughs(prices: np.ndarray, order: int = 5) -> tuple[list, list]:
    """Find local peaks and troughs in price series."""
    peaks = []
    troughs = []

    for i in range(order, len(prices) - order):
        if all(prices[i] > prices[i - j] for j in range(1, order + 1)) and \
           all(prices[i] > prices[i + j] for j in range(1, order + 1)):
            peaks.append(i)
        if all(prices[i] < prices[i - j] for j in range(1, order + 1)) and \
           all(prices[i] < prices[i + j] for j in range(1, order + 1)):
            troughs.append(i)

    return peaks, troughs


def _detect_triangle(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> dict | None:
    """Detect triangle patterns."""
    window = len(closes)

    # Calculate trend lines
    high_slope = (highs[-1] - highs[0]) / window
    low_slope = (lows[-1] - lows[0]) / window

    # Ascending triangle: flat top, rising bottom
    if abs(high_slope) < 0.001 and low_slope > 0.002:
        return {
            "pattern": "ascending_triangle",
            "confidence": 0.6,
            "signal": "bullish",
            "description": PATTERN_DEFINITIONS["ascending_triangle"]["description"],
        }

    # Descending triangle: falling top, flat bottom
    if high_slope < -0.002 and abs(low_slope) < 0.001:
        return {
            "pattern": "descending_triangle",
            "confidence": 0.6,
            "signal": "bearish",
            "description": PATTERN_DEFINITIONS["descending_triangle"]["description"],
        }

    # Symmetrical triangle: converging
    if high_slope < -0.001 and low_slope > 0.001:
        return {
            "pattern": "symmetrical_triangle",
            "confidence": 0.5,
            "signal": "neutral",
            "description": PATTERN_DEFINITIONS["symmetrical_triangle"]["description"],
        }

    return None


def _detect_flag(closes: np.ndarray) -> dict | None:
    """Detect flag patterns."""
    if len(closes) < 30:
        return None

    # Look for strong move followed by consolidation
    initial_move = (closes[10] - closes[0]) / closes[0]
    consolidation = (closes[-1] - closes[10]) / closes[10]

    # Bull flag: strong up, slight pullback
    if initial_move > 0.1 and -0.05 < consolidation < 0.02:
        return {
            "pattern": "bull_flag",
            "confidence": 0.65,
            "signal": "bullish",
            "description": PATTERN_DEFINITIONS["bull_flag"]["description"],
        }

    # Bear flag: strong down, slight rally
    if initial_move < -0.1 and -0.02 < consolidation < 0.05:
        return {
            "pattern": "bear_flag",
            "confidence": 0.65,
            "signal": "bearish",
            "description": PATTERN_DEFINITIONS["bear_flag"]["description"],
        }

    return None


def _detect_trend_channel(closes: np.ndarray) -> dict | None:
    """Detect trend channel."""
    window = len(closes)
    x = np.arange(window)

    # Linear regression
    slope = np.polyfit(x, closes, 1)[0]
    daily_pct = slope / closes[0]

    if daily_pct > 0.002:  # Strong uptrend
        return {
            "pattern": "uptrend_channel",
            "confidence": 0.7,
            "signal": "bullish",
            "description": "Price in established uptrend channel",
        }
    elif daily_pct < -0.002:  # Strong downtrend
        return {
            "pattern": "downtrend_channel",
            "confidence": 0.7,
            "signal": "bearish",
            "description": "Price in established downtrend channel",
        }

    return None


def _calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Calculate RSI indicator."""
    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)


def _calculate_macd(closes: np.ndarray) -> tuple[float, float]:
    """Calculate MACD and signal line."""
    def ema(data: np.ndarray, span: int) -> np.ndarray:
        alpha = 2 / (span + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    ema_12 = ema(closes, 12)
    ema_26 = ema(closes, 26)
    macd_line = ema_12 - ema_26
    signal_line = ema(macd_line, 9)

    return float(macd_line[-1]), float(signal_line[-1])
