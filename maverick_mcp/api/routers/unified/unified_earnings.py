"""
Unified Earnings Analysis Tool.

Provides comprehensive earnings analysis:
- calendar: Upcoming earnings dates
- surprise: EPS surprise history (actual vs. estimated)
- trend: Multi-quarter EPS/revenue trend
- guidance: Forward guidance tracking
- comprehensive: All metrics combined (default)
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


async def earnings_analysis(
    symbol: str,
    analysis_type: str = "comprehensive",
    quarters: int = 8,
    days_ahead: int = 30,
) -> dict[str, Any]:
    """
    Comprehensive earnings analysis including calendar, surprises, and trends.

    Provides earnings calendar, historical EPS surprises, revenue/EPS trends,
    and forward guidance tracking.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        analysis_type: Type of earnings analysis:
            - 'calendar': Upcoming earnings dates
            - 'surprise': Historical EPS surprise (actual vs. estimated)
            - 'trend': Multi-quarter EPS and revenue trend
            - 'guidance': Forward guidance and analyst estimates
            - 'comprehensive': Full earnings dashboard (default)
        quarters: Number of quarters to analyze (default 8)
        days_ahead: Days to look ahead for calendar (default 30)

    Returns:
        Dictionary containing earnings analysis results.

    Examples:
        # Get comprehensive earnings analysis
        >>> earnings_analysis("AAPL")

        # Check upcoming earnings date
        >>> earnings_analysis("MSFT", analysis_type="calendar")

        # Get EPS surprise history
        >>> earnings_analysis("GOOGL", analysis_type="surprise", quarters=12)
    """
    import yfinance as yf

    symbol = symbol.strip().upper()
    analysis_type = analysis_type.lower().strip()

    valid_types = ["calendar", "surprise", "trend", "guidance", "comprehensive"]
    if analysis_type not in valid_types:
        return {
            "error": f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}",
            "symbol": symbol,
            "status": "error",
        }

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if analysis_type == "calendar":
            return await _get_earnings_calendar(ticker, symbol, days_ahead, info)

        elif analysis_type == "surprise":
            return await _get_earnings_surprise(ticker, symbol, quarters, info)

        elif analysis_type == "trend":
            return await _get_earnings_trend(ticker, symbol, quarters, info)

        elif analysis_type == "guidance":
            return await _get_earnings_guidance(ticker, symbol, info)

        else:  # comprehensive
            calendar = await _get_earnings_calendar(ticker, symbol, days_ahead, info)
            surprise = await _get_earnings_surprise(ticker, symbol, quarters, info)
            trend = await _get_earnings_trend(ticker, symbol, quarters, info)
            guidance = await _get_earnings_guidance(ticker, symbol, info)

            return {
                "symbol": symbol,
                "analysis_type": "comprehensive",
                "company_name": info.get("shortName", symbol),
                "calendar": calendar.get("next_earnings"),
                "surprise_summary": {
                    "avg_surprise_pct": surprise.get("summary", {}).get(
                        "avg_surprise_pct"
                    ),
                    "beat_rate": surprise.get("summary", {}).get("beat_rate"),
                    "last_surprise": surprise.get("history", [{}])[0]
                    if surprise.get("history")
                    else None,
                },
                "trend_summary": {
                    "eps_growth_yoy": trend.get("eps_growth_yoy"),
                    "revenue_growth_yoy": trend.get("revenue_growth_yoy"),
                    "eps_trend": trend.get("eps_trend"),
                },
                "guidance_summary": {
                    "forward_pe": guidance.get("forward_pe"),
                    "peg_ratio": guidance.get("peg_ratio"),
                    "analyst_recommendation": guidance.get("recommendation"),
                },
                "interpretation": _generate_earnings_interpretation(
                    calendar, surprise, trend, guidance
                ),
                "status": "success",
            }

    except Exception as e:
        logger.error(f"Error in earnings_analysis for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "analysis_type": analysis_type,
            "status": "error",
        }


async def _get_earnings_calendar(
    ticker: Any, symbol: str, days_ahead: int, info: dict
) -> dict[str, Any]:
    """Get upcoming earnings dates."""
    try:
        # Try to get earnings dates
        earnings_dates = None
        try:
            earnings_dates = ticker.earnings_dates
        except Exception:
            pass

        next_earnings = None
        upcoming_earnings = []

        if earnings_dates is not None and not earnings_dates.empty:
            now = pd.Timestamp.now(tz=earnings_dates.index.tz)
            future_dates = earnings_dates[
                earnings_dates.index > now
            ].head(4)

            if not future_dates.empty:
                next_date = future_dates.index[0]
                next_earnings = {
                    "date": next_date.strftime("%Y-%m-%d"),
                    "days_until": (next_date - pd.Timestamp(now)).days,
                    "eps_estimate": (
                        float(future_dates.iloc[0].get("EPS Estimate", 0))
                        if "EPS Estimate" in future_dates.columns
                        and pd.notna(future_dates.iloc[0].get("EPS Estimate"))
                        else None
                    ),
                }

                for idx, row in future_dates.iterrows():
                    upcoming_earnings.append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "days_until": (idx - pd.Timestamp(now)).days,
                        "eps_estimate": (
                            float(row.get("EPS Estimate", 0))
                            if "EPS Estimate" in future_dates.columns
                            and pd.notna(row.get("EPS Estimate"))
                            else None
                        ),
                    })

        # Fallback to info data
        if not next_earnings:
            earnings_timestamp = info.get("mostRecentQuarter")
            if earnings_timestamp:
                # Most recent quarter is past, estimate next
                last_quarter = datetime.fromtimestamp(earnings_timestamp)
                # Assume ~90 days between quarters
                estimated_next = last_quarter + timedelta(days=90)
                if estimated_next > datetime.now():
                    next_earnings = {
                        "date": estimated_next.strftime("%Y-%m-%d"),
                        "days_until": (estimated_next - datetime.now()).days,
                        "eps_estimate": None,
                        "note": "Estimated based on typical quarterly schedule",
                    }

        return {
            "symbol": symbol,
            "analysis_type": "calendar",
            "next_earnings": next_earnings,
            "upcoming_earnings": upcoming_earnings,
            "fiscal_year_end": info.get("fiscalYearEnd"),
            "status": "success",
        }

    except Exception as e:
        logger.warning(f"Error getting earnings calendar for {symbol}: {e}")
        return {
            "symbol": symbol,
            "analysis_type": "calendar",
            "next_earnings": None,
            "error": str(e),
            "status": "partial",
        }


async def _get_earnings_surprise(
    ticker: Any, symbol: str, quarters: int, info: dict
) -> dict[str, Any]:
    """Get historical EPS surprise data."""
    try:
        earnings_dates = None
        try:
            earnings_dates = ticker.earnings_dates
        except Exception:
            pass

        history = []
        total_surprise = 0.0
        beats = 0
        misses = 0

        if earnings_dates is not None and not earnings_dates.empty:
            now = pd.Timestamp.now(tz=earnings_dates.index.tz)
            past_dates = earnings_dates[
                earnings_dates.index <= now
            ].head(quarters)

            for idx, row in past_dates.iterrows():
                eps_estimate = row.get("EPS Estimate")
                eps_actual = row.get("Reported EPS")
                surprise_pct = row.get("Surprise(%)")

                if pd.notna(eps_actual):
                    entry = {
                        "date": idx.strftime("%Y-%m-%d"),
                        "eps_estimate": (
                            float(eps_estimate) if pd.notna(eps_estimate) else None
                        ),
                        "eps_actual": float(eps_actual),
                        "surprise_pct": (
                            float(surprise_pct) if pd.notna(surprise_pct) else None
                        ),
                    }

                    # Calculate surprise if not provided
                    if entry["surprise_pct"] is None and entry["eps_estimate"]:
                        if entry["eps_estimate"] != 0:
                            entry["surprise_pct"] = round(
                                ((entry["eps_actual"] - entry["eps_estimate"])
                                 / abs(entry["eps_estimate"])) * 100,
                                2,
                            )

                    if entry["surprise_pct"] is not None:
                        total_surprise += entry["surprise_pct"]
                        if entry["surprise_pct"] > 0:
                            beats += 1
                        elif entry["surprise_pct"] < 0:
                            misses += 1

                        entry["result"] = (
                            "beat" if entry["surprise_pct"] > 0
                            else "miss" if entry["surprise_pct"] < 0
                            else "inline"
                        )

                    history.append(entry)

        total_reports = beats + misses
        avg_surprise = total_surprise / len(history) if history else 0

        return {
            "symbol": symbol,
            "analysis_type": "surprise",
            "history": history,
            "summary": {
                "total_quarters": len(history),
                "beats": beats,
                "misses": misses,
                "beat_rate": (
                    round(beats / total_reports * 100, 1) if total_reports > 0 else None
                ),
                "avg_surprise_pct": round(avg_surprise, 2) if history else None,
            },
            "interpretation": (
                f"{symbol} has beaten estimates {beats}/{total_reports} times "
                f"({beats/total_reports*100:.0f}%) with avg surprise of {avg_surprise:.1f}%"
                if total_reports > 0
                else f"No earnings surprise data available for {symbol}"
            ),
            "status": "success",
        }

    except Exception as e:
        logger.warning(f"Error getting earnings surprise for {symbol}: {e}")
        return {
            "symbol": symbol,
            "analysis_type": "surprise",
            "history": [],
            "error": str(e),
            "status": "partial",
        }


async def _get_earnings_trend(
    ticker: Any, symbol: str, quarters: int, info: dict
) -> dict[str, Any]:
    """Get EPS and revenue trend data."""
    try:
        # Get quarterly financials
        quarterly_earnings = None
        quarterly_financials = None

        try:
            quarterly_earnings = ticker.quarterly_earnings
        except Exception:
            pass

        try:
            quarterly_financials = ticker.quarterly_financials
        except Exception:
            pass

        eps_history = []
        revenue_history = []

        # Parse earnings data
        if quarterly_earnings is not None and not quarterly_earnings.empty:
            for idx, row in quarterly_earnings.head(quarters).iterrows():
                period = idx if isinstance(idx, str) else str(idx)
                eps_history.append({
                    "period": period,
                    "eps": float(row.get("Earnings", 0)) if pd.notna(row.get("Earnings")) else None,
                    "revenue": float(row.get("Revenue", 0)) if pd.notna(row.get("Revenue")) else None,
                })

        # Parse financials for revenue if not in earnings
        if quarterly_financials is not None and not quarterly_financials.empty:
            if "Total Revenue" in quarterly_financials.index:
                for col in quarterly_financials.columns[:quarters]:
                    rev = quarterly_financials.loc["Total Revenue", col]
                    if pd.notna(rev):
                        period = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
                        # Check if period already exists
                        existing = next((r for r in revenue_history if r["period"] == period), None)
                        if not existing:
                            revenue_history.append({
                                "period": period,
                                "revenue": float(rev),
                            })

        # Calculate growth rates
        eps_growth_yoy = None
        revenue_growth_yoy = None
        eps_trend = "unknown"

        if len(eps_history) >= 5:
            recent_eps = [e["eps"] for e in eps_history[:4] if e["eps"] is not None]
            older_eps = [e["eps"] for e in eps_history[4:8] if e["eps"] is not None]

            if recent_eps and older_eps:
                avg_recent = sum(recent_eps) / len(recent_eps)
                avg_older = sum(older_eps) / len(older_eps)
                if avg_older != 0:
                    eps_growth_yoy = round(((avg_recent - avg_older) / abs(avg_older)) * 100, 1)

        if len(eps_history) >= 2:
            recent = [e["eps"] for e in eps_history[:4] if e["eps"] is not None]
            if len(recent) >= 2:
                if all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                    eps_trend = "accelerating"
                elif all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                    eps_trend = "decelerating"
                else:
                    eps_trend = "mixed"

        # Use info for additional growth metrics
        earnings_growth = info.get("earningsGrowth")
        revenue_growth = info.get("revenueGrowth")

        if earnings_growth and eps_growth_yoy is None:
            eps_growth_yoy = round(earnings_growth * 100, 1)
        if revenue_growth and revenue_growth_yoy is None:
            revenue_growth_yoy = round(revenue_growth * 100, 1)

        return {
            "symbol": symbol,
            "analysis_type": "trend",
            "eps_history": eps_history[:quarters],
            "revenue_history": revenue_history[:quarters] if revenue_history else None,
            "eps_growth_yoy": eps_growth_yoy,
            "revenue_growth_yoy": revenue_growth_yoy,
            "eps_trend": eps_trend,
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "interpretation": (
                f"EPS trend: {eps_trend}. "
                f"YoY EPS growth: {eps_growth_yoy}%. "
                f"YoY Revenue growth: {revenue_growth_yoy}%."
                if eps_growth_yoy is not None
                else f"Limited trend data available for {symbol}"
            ),
            "status": "success",
        }

    except Exception as e:
        logger.warning(f"Error getting earnings trend for {symbol}: {e}")
        return {
            "symbol": symbol,
            "analysis_type": "trend",
            "eps_history": [],
            "error": str(e),
            "status": "partial",
        }


async def _get_earnings_guidance(
    ticker: Any, symbol: str, info: dict
) -> dict[str, Any]:
    """Get forward guidance and analyst estimates."""
    try:
        # Get analyst recommendations
        recommendations = None
        try:
            recommendations = ticker.recommendations
        except Exception:
            pass

        recent_recs = []
        if recommendations is not None and not recommendations.empty:
            for idx, row in recommendations.tail(5).iterrows():
                recent_recs.append({
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    "firm": row.get("Firm", "Unknown"),
                    "action": row.get("To Grade", row.get("Action", "N/A")),
                    "from_grade": row.get("From Grade"),
                })

        # Extract guidance metrics from info
        forward_pe = info.get("forwardPE")
        trailing_pe = info.get("trailingPE")
        peg_ratio = info.get("pegRatio")
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        recommendation = info.get("recommendationKey")
        num_analysts = info.get("numberOfAnalystOpinions")

        # Calculate upside/downside
        upside_pct = None
        if target_mean and current_price:
            upside_pct = round(((target_mean - current_price) / current_price) * 100, 1)

        return {
            "symbol": symbol,
            "analysis_type": "guidance",
            "forward_pe": round(forward_pe, 2) if forward_pe else None,
            "trailing_pe": round(trailing_pe, 2) if trailing_pe else None,
            "peg_ratio": round(peg_ratio, 2) if peg_ratio else None,
            "price_targets": {
                "current_price": round(current_price, 2) if current_price else None,
                "target_mean": round(target_mean, 2) if target_mean else None,
                "target_high": round(target_high, 2) if target_high else None,
                "target_low": round(target_low, 2) if target_low else None,
                "upside_pct": upside_pct,
            },
            "analyst_consensus": {
                "recommendation": recommendation,
                "num_analysts": num_analysts,
            },
            "recent_recommendations": recent_recs,
            "interpretation": (
                f"Analyst consensus: {recommendation}. "
                f"Mean target ${target_mean:.2f} ({upside_pct:+.1f}% from current). "
                f"Forward P/E: {forward_pe:.1f}x, PEG: {peg_ratio:.2f}."
                if all([recommendation, target_mean, upside_pct, forward_pe, peg_ratio])
                else f"Limited guidance data available for {symbol}"
            ),
            "status": "success",
        }

    except Exception as e:
        logger.warning(f"Error getting earnings guidance for {symbol}: {e}")
        return {
            "symbol": symbol,
            "analysis_type": "guidance",
            "error": str(e),
            "status": "partial",
        }


def _generate_earnings_interpretation(
    calendar: dict, surprise: dict, trend: dict, guidance: dict
) -> str:
    """Generate overall earnings interpretation."""
    parts = []

    # Calendar
    if calendar.get("next_earnings"):
        days = calendar["next_earnings"].get("days_until", "?")
        parts.append(f"Next earnings in {days} days")

    # Surprise
    if surprise.get("summary", {}).get("beat_rate"):
        rate = surprise["summary"]["beat_rate"]
        parts.append(f"beats estimates {rate:.0f}% of the time")

    # Trend
    if trend.get("eps_growth_yoy") is not None:
        growth = trend["eps_growth_yoy"]
        direction = "growing" if growth > 0 else "declining"
        parts.append(f"EPS {direction} {abs(growth):.1f}% YoY")

    # Guidance
    if guidance.get("analyst_consensus", {}).get("recommendation"):
        rec = guidance["analyst_consensus"]["recommendation"]
        parts.append(f"analyst consensus: {rec}")

    if parts:
        return f"{parts[0].capitalize()}, " + ", ".join(parts[1:]) + "."
    return "Limited earnings data available."
