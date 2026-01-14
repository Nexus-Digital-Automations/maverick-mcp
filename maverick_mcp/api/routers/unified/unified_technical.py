"""
Unified Technical Analysis Tool.

Consolidates technical analysis tools into 1 unified interface:
- RSI analysis
- MACD analysis
- Bollinger Bands analysis
- Support/Resistance levels
- Full comprehensive analysis
- Chart generation
- Relative strength vs benchmark
- RS line analysis
- Sector relative performance

DISCLAIMER: All technical analysis is for educational purposes only.
Technical indicators do not predict future price movements.
"""

import logging
from typing import Any, Literal

import numpy as np

from maverick_mcp.api.routers.unified.analysis_wrapper import with_analysis_storage

logger = logging.getLogger(__name__)

AnalysisType = Literal[
    "rsi", "macd", "bollinger", "support_resistance", "full", "chart",
    "relative_strength", "rs_line", "sector_relative"
]

# Sector to ETF mapping for sector_relative analysis
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
    "Financial": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
}


@with_analysis_storage("technical_analysis")
async def technical_analysis(
    symbol: str,
    analysis_type: str = "full",
    period: int = 14,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    days: int = 365,
    include_chart: bool = False,
) -> dict[str, Any]:
    """
    Unified technical analysis for any indicator or comprehensive analysis.

    Consolidates RSI, MACD, Bollinger Bands, support/resistance, and full
    technical analysis into a single tool with an analysis_type parameter.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        analysis_type: Type of analysis to perform:
            - 'rsi': RSI oscillator analysis (oversold/overbought signals)
            - 'macd': MACD momentum analysis (trend and momentum)
            - 'bollinger': Bollinger Bands volatility analysis
            - 'support_resistance': Key support and resistance price levels
            - 'full': Comprehensive multi-indicator analysis (default)
            - 'chart': Generate visual technical chart
            - 'relative_strength': Performance vs benchmark (SPY)
            - 'rs_line': RS line ratio with trend analysis
            - 'sector_relative': Performance vs sector ETF
        period: RSI period (default: 14)
        fast_period: MACD fast EMA period (default: 12)
        slow_period: MACD slow EMA period (default: 26)
        signal_period: MACD signal line period (default: 9)
        days: Days of historical data to analyze (default: 365)
        include_chart: Include chart with full analysis (default: False)

    Returns:
        Dictionary containing technical analysis results based on analysis_type.

    Examples:
        # Get RSI analysis
        >>> technical_analysis("AAPL", analysis_type="rsi")

        # Get full analysis with chart
        >>> technical_analysis("MSFT", analysis_type="full", include_chart=True)

        # Get support/resistance levels
        >>> technical_analysis("GOOGL", analysis_type="support_resistance")
    """
    symbol = symbol.strip().upper()
    analysis_type = analysis_type.lower().strip()

    valid_types = [
        "rsi", "macd", "bollinger", "support_resistance", "full", "chart",
        "relative_strength", "rs_line", "sector_relative"
    ]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        # Import the underlying implementations
        from maverick_mcp.api.routers.technical import (
            get_full_technical_analysis,
            get_macd_analysis,
            get_rsi_analysis,
            get_stock_chart_analysis,
            get_support_resistance,
        )
        from maverick_mcp.core.technical_analysis import analyze_bollinger_bands
        from maverick_mcp.utils.stock_helpers import get_stock_dataframe_async

        if analysis_type == "rsi":
            result = await get_rsi_analysis(symbol, period=period, days=days)
            result["analysis_type"] = "rsi"
            return result

        elif analysis_type == "macd":
            result = await get_macd_analysis(
                symbol,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                days=days,
            )
            result["analysis_type"] = "macd"
            return result

        elif analysis_type == "bollinger":
            df = await get_stock_dataframe_async(symbol, days)
            bb_analysis = analyze_bollinger_bands(df)
            current_price = float(df["close"].iloc[-1])
            return {
                "symbol": symbol,
                "analysis_type": "bollinger",
                "current_price": current_price,
                "bollinger_bands": bb_analysis,
                "status": "success",
            }

        elif analysis_type == "support_resistance":
            result = await get_support_resistance(symbol, days=days)
            result["analysis_type"] = "support_resistance"
            return result

        elif analysis_type == "chart":
            result = await get_stock_chart_analysis(symbol)
            return result

        elif analysis_type == "relative_strength":
            return await _get_relative_strength(symbol, days=days)

        elif analysis_type == "rs_line":
            return await _get_rs_line(symbol, days=days)

        elif analysis_type == "sector_relative":
            return await _get_sector_relative(symbol, days=days)

        else:  # full
            result = await get_full_technical_analysis(symbol, days=days)
            result["analysis_type"] = "full"

            # Optionally include chart
            if include_chart:
                chart_result = await get_stock_chart_analysis(symbol)
                result["chart"] = chart_result

            return result

    except Exception as e:
        logger.error(f"Error in technical_analysis for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }


async def _get_relative_strength(
    symbol: str,
    benchmark: str = "SPY",
    days: int = 252,
) -> dict[str, Any]:
    """Calculate relative strength of stock vs benchmark."""
    from maverick_mcp.utils.stock_helpers import get_stock_dataframe_async

    try:
        # Get price data for both symbol and benchmark
        stock_df = await get_stock_dataframe_async(symbol, days)
        bench_df = await get_stock_dataframe_async(benchmark, days)

        if stock_df.empty or bench_df.empty:
            return {
                "error": "Unable to fetch price data",
                "symbol": symbol,
                "status": "error",
            }

        # Align dates
        stock_prices = stock_df["close"]
        bench_prices = bench_df["close"]

        # Calculate returns for different periods
        periods = {
            "1m": 21,
            "3m": 63,
            "6m": 126,
            "12m": 252,
        }

        relative_strength = {}
        for period_name, period_days in periods.items():
            if len(stock_prices) >= period_days and len(bench_prices) >= period_days:
                stock_return = (stock_prices.iloc[-1] / stock_prices.iloc[-period_days] - 1) * 100
                bench_return = (bench_prices.iloc[-1] / bench_prices.iloc[-period_days] - 1) * 100
                relative_strength[f"rs_{period_name}"] = round(stock_return - bench_return, 2)
                relative_strength[f"stock_return_{period_name}"] = round(stock_return, 2)
                relative_strength[f"benchmark_return_{period_name}"] = round(bench_return, 2)

        # Determine trend
        rs_values = [v for k, v in relative_strength.items() if k.startswith("rs_")]
        if rs_values:
            avg_rs = sum(rs_values) / len(rs_values)
            if avg_rs > 5:
                trend = "strongly_outperforming"
            elif avg_rs > 0:
                trend = "outperforming"
            elif avg_rs > -5:
                trend = "underperforming"
            else:
                trend = "strongly_underperforming"
        else:
            trend = "unknown"

        current_price = float(stock_prices.iloc[-1])
        benchmark_price = float(bench_prices.iloc[-1])

        return {
            "symbol": symbol,
            "analysis_type": "relative_strength",
            "benchmark": benchmark,
            "current_price": round(current_price, 2),
            "benchmark_price": round(benchmark_price, 2),
            "relative_strength": relative_strength,
            "trend": trend,
            "interpretation": (
                f"{symbol} is {trend.replace('_', ' ')} vs {benchmark}. "
                f"3M excess return: {relative_strength.get('rs_3m', 'N/A')}%"
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in relative_strength for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": "relative_strength",
            "status": "error",
        }


async def _get_rs_line(
    symbol: str,
    benchmark: str = "SPY",
    days: int = 252,
) -> dict[str, Any]:
    """Calculate RS line (stock/benchmark price ratio) with trend analysis."""
    from maverick_mcp.utils.stock_helpers import get_stock_dataframe_async

    try:
        stock_df = await get_stock_dataframe_async(symbol, days)
        bench_df = await get_stock_dataframe_async(benchmark, days)

        if stock_df.empty or bench_df.empty:
            return {
                "error": "Unable to fetch price data",
                "symbol": symbol,
                "status": "error",
            }

        # Align by taking common dates
        stock_prices = stock_df["close"]
        bench_prices = bench_df["close"]

        # Calculate RS ratio (stock price / benchmark price)
        min_len = min(len(stock_prices), len(bench_prices))
        stock_prices = stock_prices.iloc[-min_len:]
        bench_prices = bench_prices.iloc[-min_len:]

        rs_ratio = stock_prices.values / bench_prices.values

        current_ratio = rs_ratio[-1]
        ratio_52w_high = np.max(rs_ratio)
        ratio_52w_low = np.min(rs_ratio)

        # Calculate SMA of RS line
        if len(rs_ratio) >= 20:
            rs_sma_20 = np.mean(rs_ratio[-20:])
            rs_sma_20_prev = np.mean(rs_ratio[-25:-5]) if len(rs_ratio) >= 25 else rs_sma_20
            sma_20_slope = (rs_sma_20 - rs_sma_20_prev) / rs_sma_20_prev * 100
        else:
            rs_sma_20 = current_ratio
            sma_20_slope = 0

        if len(rs_ratio) >= 50:
            rs_sma_50 = np.mean(rs_ratio[-50:])
        else:
            rs_sma_50 = current_ratio

        # Determine trend direction
        if sma_20_slope > 0.5:
            direction = "rising"
            strength = "strong" if sma_20_slope > 2 else "moderate"
        elif sma_20_slope < -0.5:
            direction = "falling"
            strength = "strong" if sma_20_slope < -2 else "moderate"
        else:
            direction = "flat"
            strength = "weak"

        # Signals
        new_52w_high = current_ratio >= ratio_52w_high * 0.99
        breaking_above_sma = current_ratio > rs_sma_20 and rs_ratio[-2] <= np.mean(rs_ratio[-21:-1])

        # Percentile rank
        percentile_rank = int(np.sum(rs_ratio < current_ratio) / len(rs_ratio) * 100)

        return {
            "symbol": symbol,
            "analysis_type": "rs_line",
            "benchmark": benchmark,
            "rs_line": {
                "current_ratio": round(current_ratio, 4),
                "ratio_52w_high": round(ratio_52w_high, 4),
                "ratio_52w_low": round(ratio_52w_low, 4),
                "pct_from_52w_high": round((current_ratio / ratio_52w_high - 1) * 100, 2),
                "percentile_rank": percentile_rank,
            },
            "rs_line_trend": {
                "sma_20": round(rs_sma_20, 4),
                "sma_50": round(rs_sma_50, 4),
                "sma_20_slope_pct": round(sma_20_slope, 2),
                "direction": direction,
                "strength": strength,
            },
            "signals": {
                "new_52w_high": new_52w_high,
                "breaking_above_sma": breaking_above_sma,
            },
            "interpretation": (
                f"RS line {direction} with {strength} momentum. "
                f"Currently at {percentile_rank}th percentile of 52-week range."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in rs_line for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": "rs_line",
            "status": "error",
        }


async def _get_sector_relative(
    symbol: str,
    sector_etf: str | None = None,
    days: int = 252,
) -> dict[str, Any]:
    """Calculate relative strength vs sector ETF."""
    import yfinance as yf

    from maverick_mcp.utils.stock_helpers import get_stock_dataframe_async

    try:
        # Get stock info to determine sector
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get("sector", "Unknown")

        # Auto-detect sector ETF if not provided
        if sector_etf is None:
            sector_etf = SECTOR_ETF_MAP.get(sector, "SPY")

        stock_df = await get_stock_dataframe_async(symbol, days)
        sector_df = await get_stock_dataframe_async(sector_etf, days)
        market_df = await get_stock_dataframe_async("SPY", days)

        if stock_df.empty or sector_df.empty:
            return {
                "error": "Unable to fetch price data",
                "symbol": symbol,
                "status": "error",
            }

        stock_prices = stock_df["close"]
        sector_prices = sector_df["close"]
        market_prices = market_df["close"]

        # Calculate returns for different periods
        periods = {"1m": 21, "3m": 63, "6m": 126}
        relative_performance = {}

        for period_name, period_days in periods.items():
            if len(stock_prices) >= period_days:
                stock_return = (stock_prices.iloc[-1] / stock_prices.iloc[-period_days] - 1) * 100
                sector_return = (sector_prices.iloc[-1] / sector_prices.iloc[-period_days] - 1) * 100
                market_return = (market_prices.iloc[-1] / market_prices.iloc[-period_days] - 1) * 100

                relative_performance[f"vs_sector_{period_name}"] = round(stock_return - sector_return, 2)
                relative_performance[f"vs_market_{period_name}"] = round(stock_return - market_return, 2)

        # Calculate sector alpha (excess return vs sector)
        stock_return_3m = relative_performance.get("vs_sector_3m", 0) + (
            (sector_prices.iloc[-1] / sector_prices.iloc[-63] - 1) * 100 if len(sector_prices) >= 63 else 0
        )

        return {
            "symbol": symbol,
            "analysis_type": "sector_relative",
            "sector": sector,
            "sector_etf": sector_etf,
            "relative_performance": relative_performance,
            "dual_comparison": {
                "stock_return_3m": round(
                    (stock_prices.iloc[-1] / stock_prices.iloc[-63] - 1) * 100, 2
                ) if len(stock_prices) >= 63 else None,
                "sector_return_3m": round(
                    (sector_prices.iloc[-1] / sector_prices.iloc[-63] - 1) * 100, 2
                ) if len(sector_prices) >= 63 else None,
                "market_return_3m": round(
                    (market_prices.iloc[-1] / market_prices.iloc[-63] - 1) * 100, 2
                ) if len(market_prices) >= 63 else None,
            },
            "interpretation": (
                f"{symbol} ({sector}) vs {sector_etf}: "
                f"{relative_performance.get('vs_sector_3m', 'N/A')}% excess (3M). "
                f"vs SPY: {relative_performance.get('vs_market_3m', 'N/A')}% excess (3M)."
            ),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in sector_relative for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": "sector_relative",
            "status": "error",
        }
