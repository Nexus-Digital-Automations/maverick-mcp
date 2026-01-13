"""
Unified Router Modules for MaverickMCP.

This package contains consolidated tool implementations that combine multiple
related functions into single, parameter-driven unified tools.

Tool Consolidation Summary:
    132+ individual tools → ~30 unified tools

Pattern: Instead of multiple tools (rsi_analysis, macd_analysis, etc.),
we have one tool with an `analysis_type` parameter.

Example:
    # Old: 5 separate tools
    technical_get_rsi_analysis(ticker)
    technical_get_macd_analysis(ticker)
    technical_get_support_resistance(ticker)
    technical_get_full_technical_analysis(ticker)
    technical_get_stock_chart_analysis(ticker)

    # New: 1 unified tool
    technical_analysis(symbol, analysis_type='rsi')
    technical_analysis(symbol, analysis_type='full')
"""

from maverick_mcp.api.routers.unified.unified_alternative import alternative_data
from maverick_mcp.api.routers.unified.unified_breadth import market_breadth
from maverick_mcp.api.routers.unified.unified_options import options_analysis
from maverick_mcp.api.routers.unified.unified_portfolio import (
    portfolio_analyze,
    portfolio_manage,
)
from maverick_mcp.api.routers.unified.unified_quant import quant_analysis
from maverick_mcp.api.routers.unified.unified_risk import risk_analysis
from maverick_mcp.api.routers.unified.unified_screening import stock_screener
from maverick_mcp.api.routers.unified.unified_simulation import simulation
from maverick_mcp.api.routers.unified.unified_technical import technical_analysis
from maverick_mcp.api.routers.unified.unified_timeframe import multi_timeframe
from maverick_mcp.api.routers.unified.unified_valuation import valuation
from maverick_mcp.api.routers.unified.unified_volatility import volatility_analysis
from maverick_mcp.api.routers.unified.unified_volume import volume_analysis

__all__ = [
    # Core Analysis (5 tools)
    "technical_analysis",
    "stock_screener",
    "risk_analysis",
    "quant_analysis",
    "options_analysis",
    # Market Analysis (4 tools)
    "market_breadth",
    "multi_timeframe",
    "volume_analysis",
    "volatility_analysis",
    # Fundamental Analysis (2 tools)
    "valuation",
    "alternative_data",
    # Portfolio (2 tools)
    "portfolio_manage",
    "portfolio_analyze",
    # Simulation (1 tool)
    "simulation",
]
