"""
Tool Registry for MaverickMCP.

Streamlined tool registration using unified interfaces.
Consolidates 132+ individual tools → ~39 unified tools.

Tool Categories:
    - Unified Analysis (20 tools): Consolidated analysis tools with parameter-based dispatch
    - OpenBB Data (9 tools): Primary data provider for all asset classes
    - Yahoo Finance (2 tools): Unique data (holders, recommendations)
    - Research Tools (2 tools): AI-powered research (comprehensive, company)
    - System Tools (2 tools): Health and performance monitoring
    - Backtesting (5 tools): Strategy testing and optimization
"""

import logging

from fastmcp import FastMCP

from maverick_mcp.config.settings import settings

logger = logging.getLogger(__name__)


def _get_tool_tags(tool_name: str) -> set[str] | None:
    """Return tags for tool based on mode configuration.

    In full mode, returns None (no filtering).
    In simple mode, essential tools get {"essential"} tag, others get {"advanced"}.
    Advanced tools are hidden from listing but remain callable by name.
    """
    if settings.tool_mode.mode == "full":
        return None  # No filtering in full mode
    if tool_name in settings.tool_mode.essential_tools:
        return {"essential"}
    return {"advanced"}  # Hidden but callable


# =============================================================================
# UNIFIED ANALYSIS TOOLS (20 tools)
# =============================================================================


def register_unified_tools(mcp: FastMCP) -> None:
    """Register unified analysis tools.

    These tools consolidate 60+ individual analysis tools into 20 unified interfaces
    with parameter-based dispatch.
    """
    from maverick_mcp.api.routers.unified import (
        alternative_data,
        batch_stock_analysis,
        comprehensive_stock_analysis,
        market_breadth,
        multi_timeframe,
        options_analysis,
        portfolio_analyze,
        portfolio_manage,
        quant_analysis,
        risk_analysis,
        simulation,
        stock_screener,
        technical_analysis,
        valuation,
        volatility_analysis,
        volume_analysis,
    )
    from maverick_mcp.api.routers.unified.unified_analysis_history import (
        analysis_history,
    )
    from maverick_mcp.api.routers.unified.unified_earnings import earnings_analysis
    from maverick_mcp.api.routers.unified.unified_macro import macro_analysis
    from maverick_mcp.api.routers.unified.unified_ml_predictions import ml_predictions
    from maverick_mcp.api.routers.unified.unified_watchlist import watchlist_manage

    # Core Analysis Tools
    mcp.tool(name="technical_analysis", tags=_get_tool_tags("technical_analysis"))(
        technical_analysis
    )
    mcp.tool(name="stock_screener", tags=_get_tool_tags("stock_screener"))(
        stock_screener
    )
    mcp.tool(name="risk_analysis", tags=_get_tool_tags("risk_analysis"))(risk_analysis)
    mcp.tool(name="quant_analysis", tags=_get_tool_tags("quant_analysis"))(
        quant_analysis
    )
    mcp.tool(name="options_analysis", tags=_get_tool_tags("options_analysis"))(
        options_analysis
    )

    # Market Analysis Tools
    mcp.tool(name="market_breadth", tags=_get_tool_tags("market_breadth"))(
        market_breadth
    )
    mcp.tool(name="multi_timeframe", tags=_get_tool_tags("multi_timeframe"))(
        multi_timeframe
    )
    mcp.tool(name="volume_analysis", tags=_get_tool_tags("volume_analysis"))(
        volume_analysis
    )
    mcp.tool(name="volatility_analysis", tags=_get_tool_tags("volatility_analysis"))(
        volatility_analysis
    )

    # Fundamental Analysis Tools
    mcp.tool(name="valuation", tags=_get_tool_tags("valuation"))(valuation)
    mcp.tool(name="alternative_data", tags=_get_tool_tags("alternative_data"))(
        alternative_data
    )
    mcp.tool(name="earnings_analysis", tags=_get_tool_tags("earnings_analysis"))(
        earnings_analysis
    )

    # Macro/Market Tools
    mcp.tool(name="macro_analysis", tags=_get_tool_tags("macro_analysis"))(
        macro_analysis
    )

    # Portfolio Tools
    mcp.tool(name="portfolio_manage", tags=_get_tool_tags("portfolio_manage"))(
        portfolio_manage
    )
    mcp.tool(name="portfolio_analyze", tags=_get_tool_tags("portfolio_analyze"))(
        portfolio_analyze
    )

    # Watchlist Tools
    mcp.tool(name="watchlist_manage", tags=_get_tool_tags("watchlist_manage"))(
        watchlist_manage
    )

    # Simulation Tools
    mcp.tool(name="simulation", tags=_get_tool_tags("simulation"))(simulation)

    # ML/Prediction Tools
    mcp.tool(name="ml_predictions", tags=_get_tool_tags("ml_predictions"))(
        ml_predictions
    )

    # Analysis History Tool
    mcp.tool(name="analysis_history", tags=_get_tool_tags("analysis_history"))(
        analysis_history
    )

    # Comprehensive Analysis Tools (parallel execution)
    mcp.tool(
        name="comprehensive_stock_analysis",
        tags=_get_tool_tags("comprehensive_stock_analysis"),
    )(comprehensive_stock_analysis)
    mcp.tool(name="batch_stock_analysis", tags=_get_tool_tags("batch_stock_analysis"))(
        batch_stock_analysis
    )

    logger.info("✓ Unified analysis tools registered (21 tools)")


# =============================================================================
# OPENBB DATA TOOLS (9 tools)
# =============================================================================


def register_openbb_tools(mcp: FastMCP) -> None:
    """Register OpenBB data tools.

    OpenBB is the PRIMARY data provider for all asset classes:
    - Unified historical data (equity, crypto, forex, futures)
    - Unified economic indicators (CPI, GDP, unemployment, FRED)
    - Unified search (equity, crypto, FRED)
    - Individual tools for unique functionality
    """
    from maverick_mcp.api.routers.openbb_data import (
        get_company_news,
        get_cpi_data,
        get_crypto_historical,
        get_currency_historical,
        get_economic_calendar,
        get_equity_historical,
        get_equity_info,
        get_equity_quote,
        get_financial_statements,
        get_fred_series,
        get_futures_historical,
        get_gdp_data,
        get_interest_rates,
        get_treasury_rates,
        get_unemployment_data,
        search_crypto,
        search_equity,
        search_fred_series,
    )

    # Unified historical data tool
    @mcp.tool(
        name="openbb_get_historical", tags=_get_tool_tags("openbb_get_historical")
    )
    async def openbb_get_historical(
        symbol: str,
        asset_class: str = "equity",
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        provider: str = "yfinance",
    ):
        """
        Get historical OHLCV price data for ANY asset class.

        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTCUSD', 'EURUSD', 'ES')
            asset_class: 'equity' | 'crypto' | 'forex' | 'futures'
            start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            interval: Data interval ('1d', '1h', '1m', '1wk', '1mo')
            provider: Data provider ('yfinance', 'fmp', 'polygon', 'tiingo')

        Returns:
            Historical OHLCV data for the asset.
        """
        if asset_class == "equity":
            return await get_equity_historical(
                symbol, start_date, end_date, interval, provider
            )
        elif asset_class == "crypto":
            return await get_crypto_historical(
                symbol, start_date, end_date, interval, provider
            )
        elif asset_class == "forex":
            return await get_currency_historical(
                symbol, start_date, end_date, interval, provider
            )
        elif asset_class == "futures":
            return await get_futures_historical(
                symbol, start_date, end_date, interval, provider
            )
        else:
            return {
                "error": f"Invalid asset_class '{asset_class}'. Use 'equity', 'crypto', 'forex', or 'futures'",
                "symbol": symbol,
            }

    # Unified economic indicator tool
    @mcp.tool(
        name="openbb_get_economic_indicator",
        tags=_get_tool_tags("openbb_get_economic_indicator"),
    )
    async def openbb_get_economic_indicator(
        indicator: str = "cpi",
        country: str = "united_states",
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "month",
        fred_symbol: str | None = None,
        provider: str = "fred",
    ):
        """
        Get economic indicator data (CPI, GDP, unemployment, interest rates, FRED series).

        Args:
            indicator: 'cpi' | 'gdp' | 'unemployment' | 'interest_rates' | 'fred'
            country: Country ('united_states', 'japan', 'germany', 'euro_area')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: 'month' | 'quarter' | 'annual'
            fred_symbol: FRED series ID when indicator='fred' (e.g., 'FEDFUNDS')
            provider: Data provider ('fred', 'oecd')

        Returns:
            Economic indicator time series data.
        """
        if indicator == "cpi":
            return await get_cpi_data(country, start_date, end_date, provider)
        elif indicator == "gdp":
            return await get_gdp_data(country, frequency, provider)
        elif indicator == "unemployment":
            return await get_unemployment_data(country, frequency, provider)
        elif indicator == "interest_rates":
            return await get_interest_rates(country, start_date, end_date, provider)
        elif indicator == "fred":
            if not fred_symbol:
                return {
                    "error": "fred_symbol is required when indicator='fred'",
                    "hint": "Common series: FEDFUNDS, UNRATE, GDP, CPIAUCSL, M2SL",
                }
            return await get_fred_series(fred_symbol, start_date, end_date)
        else:
            return {
                "error": f"Invalid indicator '{indicator}'. Use 'cpi', 'gdp', 'unemployment', 'interest_rates', or 'fred'",
            }

    # Unified search tool
    @mcp.tool(name="openbb_search", tags=_get_tool_tags("openbb_search"))
    async def openbb_search(query: str, search_type: str = "equity"):
        """
        Search for symbols across asset classes.

        Args:
            query: Search term (e.g., 'apple', 'bitcoin', 'inflation')
            search_type: 'equity' | 'crypto' | 'fred'

        Returns:
            List of matching symbols with descriptions.
        """
        if search_type == "equity":
            return await search_equity(query)
        elif search_type == "crypto":
            return await search_crypto(query)
        elif search_type == "fred":
            return await search_fred_series(query)
        else:
            return {
                "error": f"Invalid search_type '{search_type}'. Use 'equity', 'crypto', or 'fred'",
            }

    # Individual tools for unique functionality
    @mcp.tool(
        name="openbb_get_equity_quote", tags=_get_tool_tags("openbb_get_equity_quote")
    )
    async def openbb_equity_quote(symbol: str = "AAPL", provider: str = "yfinance"):
        """Get real-time stock quote with price, volume, and change."""
        return await get_equity_quote(symbol, provider)

    @mcp.tool(
        name="openbb_get_equity_info", tags=_get_tool_tags("openbb_get_equity_info")
    )
    async def openbb_equity_info(symbol: str = "AAPL", provider: str = "yfinance"):
        """Get comprehensive company information (sector, industry, description)."""
        return await get_equity_info(symbol, provider)

    # Note: openbb_get_options_chains removed - use options_analysis(symbol, 'chain') instead

    @mcp.tool(
        name="openbb_get_company_news", tags=_get_tool_tags("openbb_get_company_news")
    )
    async def openbb_company_news(
        symbol: str = "AAPL", limit: int = 20, provider: str = "fmp"
    ):
        """Get recent news articles for a company."""
        return await get_company_news(symbol, limit, provider)

    @mcp.tool(
        name="openbb_get_financial_statements",
        tags=_get_tool_tags("openbb_get_financial_statements"),
    )
    async def openbb_financial_statements(
        symbol: str = "AAPL",
        statement_type: str = "income",
        period: str = "annual",
        limit: int = 5,
        provider: str = "yfinance",
    ):
        """Get financial statements (income, balance, cash flow)."""
        return await get_financial_statements(
            symbol, statement_type, period, limit, provider
        )

    @mcp.tool(
        name="openbb_get_treasury_rates",
        tags=_get_tool_tags("openbb_get_treasury_rates"),
    )
    async def openbb_treasury_rates(
        start_date: str | None = None,
        end_date: str | None = None,
        maturity: str | None = None,
        provider: str = "fred",
    ):
        """Get US Treasury interest rates."""
        return await get_treasury_rates(start_date, end_date, maturity, provider)

    @mcp.tool(
        name="openbb_get_economic_calendar",
        tags=_get_tool_tags("openbb_get_economic_calendar"),
    )
    async def openbb_economic_calendar(
        start_date: str | None = None,
        end_date: str | None = None,
        provider: str = "fmp",
    ):
        """Get upcoming economic events calendar."""
        return await get_economic_calendar(start_date, end_date, provider)

    logger.info("✓ OpenBB data tools registered (9 tools)")


# =============================================================================
# YAHOO FINANCE TOOLS (2 tools - unique data only)
# =============================================================================


def register_yahoo_tools(mcp: FastMCP) -> None:
    """Register Yahoo Finance tools for UNIQUE data not available in OpenBB.

    Yahoo Finance is used only for:
    - Holder information (institutional, insider, major holders)
    - Analyst recommendations (granular upgrades/downgrades)

    Note: Options tools removed - use options_analysis(symbol, 'expirations'/'chain') instead.
    """
    from maverick_mcp.api.routers.yahoo_finance import (
        get_yahoo_holder_info,
        get_yahoo_recommendations,
    )

    @mcp.tool(
        name="yahoo_get_holder_info", tags=_get_tool_tags("yahoo_get_holder_info")
    )
    async def yahoo_holder_info(ticker: str, holder_type: str):
        """
        Get holder/ownership information (UNIQUE to Yahoo Finance).

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            holder_type: Type of holder info:
                - 'major_holders': Major shareholders summary
                - 'institutional_holders': Institutional ownership
                - 'mutualfund_holders': Mutual fund ownership
                - 'insider_transactions': Recent insider trades
                - 'insider_purchases': Insider buying activity
                - 'insider_roster_holders': List of insiders

        Returns:
            Dictionary with holder/ownership data.
        """
        return await get_yahoo_holder_info(ticker, holder_type)

    @mcp.tool(
        name="yahoo_get_recommendations",
        tags=_get_tool_tags("yahoo_get_recommendations"),
    )
    async def yahoo_recommendations(
        ticker: str,
        recommendation_type: str = "recommendations",
        months_back: int = 12,
    ):
        """
        Get analyst recommendations with firm-level detail.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            recommendation_type: 'recommendations' | 'upgrades_downgrades'
            months_back: Months of history for upgrades_downgrades (default 12)

        Returns:
            Dictionary with analyst recommendations data.
        """
        return await get_yahoo_recommendations(ticker, recommendation_type, months_back)

    # Note: yahoo_options removed - use options_analysis(symbol, 'expirations'/'chain') instead

    logger.info("✓ Yahoo Finance tools registered (2 tools)")


# =============================================================================
# RESEARCH TOOLS (2 tools)
# =============================================================================


def register_research_tools(mcp: FastMCP) -> None:
    """Register AI-powered research tools."""
    try:
        from maverick_mcp.api.routers.research import (
            company_comprehensive_research,
            comprehensive_research,
        )

        @mcp.tool(
            name="research_comprehensive", tags=_get_tool_tags("research_comprehensive")
        )
        async def research_comprehensive_tool(
            query: str,
            persona: str = "moderate",
            research_scope: str = "standard",
            max_sources: int = 10,
            timeframe: str = "1m",
        ) -> dict:
            """
            Comprehensive research on any financial topic using web search and AI.

            Args:
                query: Research topic (stock, sector, trend, etc.)
                persona: Research style ('conservative', 'moderate', 'aggressive')
                research_scope: Depth ('basic', 'standard', 'comprehensive', 'exhaustive')
                max_sources: Maximum sources to analyze
                timeframe: Time range for research

            Returns:
                Comprehensive research findings with analysis.
            """
            return await comprehensive_research(
                query=query,
                persona=persona,
                research_scope=research_scope,
                max_sources=min(max_sources, 25),
                timeframe=timeframe,
            )

        @mcp.tool(name="research_company", tags=_get_tool_tags("research_company"))
        async def research_company_tool(
            symbol: str,
            include_competitive_analysis: bool = False,
            persona: str = "moderate",
        ) -> dict:
            """
            Comprehensive company research and fundamental analysis.

            Args:
                symbol: Stock ticker symbol
                include_competitive_analysis: Include competitor comparison
                persona: Research style

            Returns:
                Company research with financial analysis.
            """
            return await company_comprehensive_research(
                symbol=symbol,
                include_competitive_analysis=include_competitive_analysis,
                persona=persona,
            )

        logger.info("✓ Research tools registered (2 tools)")

    except ImportError as e:
        logger.warning(f"Research module not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register research tools: {e}")


# =============================================================================
# SYSTEM TOOLS (2 tools)
# =============================================================================


def register_system_tools(mcp: FastMCP) -> None:
    """Register health and performance monitoring tools."""
    from maverick_mcp.api.routers.performance import (
        get_system_performance_health,
    )

    mcp.tool(name="system_health", tags=_get_tool_tags("system_health"))(
        get_system_performance_health
    )

    try:
        from maverick_mcp.api.routers.health_tools import register_health_tools

        register_health_tools(mcp)
        logger.info("✓ System tools registered (2+ tools)")
    except ImportError:
        logger.info("✓ System tools registered (1 tool)")


# =============================================================================
# BACKTESTING TOOLS (5 tools)
# =============================================================================


def register_backtesting_tools(mcp: FastMCP) -> None:
    """Register VectorBT backtesting tools if available."""
    try:
        from maverick_mcp.api.routers.backtesting import setup_backtesting_tools

        setup_backtesting_tools(mcp)
        logger.info("✓ Backtesting tools registered (5 tools)")
    except ImportError:
        logger.warning(
            "Backtesting module not available - VectorBT may not be installed"
        )
    except Exception as e:
        logger.error(f"Failed to register backtesting tools: {e}")


# =============================================================================
# MCP PROMPTS AND RESOURCES
# =============================================================================


def register_mcp_prompts_and_resources(mcp: FastMCP) -> None:
    """Register MCP prompts and resources for client introspection."""
    try:
        from maverick_mcp.api.routers.mcp_prompts import register_mcp_prompts

        register_mcp_prompts(mcp)
        logger.info("✓ MCP prompts registered")
    except ImportError:
        logger.warning("MCP prompts module not available")
    except Exception as e:
        logger.error(f"Failed to register MCP prompts: {e}")

    try:
        from maverick_mcp.api.routers.introspection import register_introspection_tools

        register_introspection_tools(mcp)
        logger.info("✓ Introspection tools registered")
    except ImportError:
        logger.warning("Introspection module not available")
    except Exception as e:
        logger.error(f"Failed to register introspection tools: {e}")


# =============================================================================
# MAIN REGISTRATION FUNCTION
# =============================================================================


def register_all_router_tools(mcp: FastMCP) -> None:
    """Register all tools on the main server.

    Tool Summary (~39 tools):
        - Unified Analysis: 20 tools (consolidated from 60+)
        - OpenBB Data: 9 tools (equity, crypto, forex, futures, economy)
        - Yahoo Finance: 2 tools (holders, recommendations)
        - Research: 2 tools (comprehensive, company)
        - System: 2 tools (health, performance)
        - Backtesting: 5 tools (backtest, compare, optimize, analyze, report)
    """
    logger.info("=" * 60)
    logger.info("Starting MaverickMCP tool registration...")
    logger.info("=" * 60)

    # Register unified analysis tools (18 tools)
    try:
        register_unified_tools(mcp)
    except Exception as e:
        logger.error(f"✗ Failed to register unified tools: {e}")

    # Register OpenBB data tools (9 tools)
    try:
        register_openbb_tools(mcp)
    except Exception as e:
        logger.error(f"✗ Failed to register OpenBB tools: {e}")

    # Register Yahoo Finance tools (2 tools)
    try:
        register_yahoo_tools(mcp)
    except Exception as e:
        logger.error(f"✗ Failed to register Yahoo tools: {e}")

    # Register research tools (2 tools)
    try:
        register_research_tools(mcp)
    except Exception as e:
        logger.error(f"✗ Failed to register research tools: {e}")

    # Register system tools (2 tools)
    try:
        register_system_tools(mcp)
    except Exception as e:
        logger.error(f"✗ Failed to register system tools: {e}")

    # Register backtesting tools (5 tools)
    try:
        register_backtesting_tools(mcp)
    except Exception as e:
        logger.error(f"✗ Failed to register backtesting tools: {e}")

    # Register MCP prompts and resources
    try:
        register_mcp_prompts_and_resources(mcp)
    except Exception as e:
        logger.error(f"✗ Failed to register MCP prompts: {e}")

    logger.info("=" * 60)
    logger.info("Tool registration complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("📊 UNIFIED ANALYSIS TOOLS (20):")
    logger.info(
        "   • technical_analysis - RSI, MACD, Bollinger, support/resistance, full"
    )
    logger.info("   • stock_screener - maverick, bear, momentum, value, supply_demand")
    logger.info("   • risk_analysis - VaR, CVaR, drawdown, stress test, comprehensive")
    logger.info(
        "   • quant_analysis - beta, correlation, factors, momentum, volatility"
    )
    logger.info(
        "   • options_analysis - Greeks, IV, skew, strategies, strategy_template"
    )
    logger.info("   • market_breadth - advance/decline, highs/lows, sector, regime")
    logger.info("   • multi_timeframe - trend, RSI, MA alignment, signal score")
    logger.info("   • volume_analysis - profile, VWAP, market profile, footprint")
    logger.info(
        "   • volatility_analysis - VIX term structure, contango, surface, regime"
    )
    logger.info("   • valuation - DCF, multiples, comps, DDM, fair value")
    logger.info(
        "   • alternative_data - short interest, insider, institutional, options flow"
    )
    logger.info(
        "   • earnings_analysis - calendar, surprise, trend, guidance, comprehensive"
    )
    logger.info("   • macro_analysis - yield curve, fed funds, regime, market cycle")
    logger.info("   • portfolio_manage - add, remove, view, clear positions")
    logger.info(
        "   • portfolio_analyze - summary, risk, correlation, attribution, factor, style"
    )
    logger.info(
        "   • watchlist_manage - create, list, view, add, remove, delete, performance"
    )
    logger.info("   • simulation - Monte Carlo asset/portfolio simulation")
    logger.info(
        "   • ml_predictions - price forecast, patterns, regime, trend, ensemble"
    )
    logger.info("   • comprehensive_stock_analysis - parallel analysis (4-5x faster)")
    logger.info("   • batch_stock_analysis - multi-symbol parallel analysis")
    logger.info("")
    logger.info("📈 DATA TOOLS (11):")
    logger.info("   • OpenBB: historical, economic indicator, search, quote, info,")
    logger.info("     news, financials, treasury, calendar")
    logger.info("   • Yahoo: holder info, recommendations")
    logger.info("")
    logger.info("🔬 RESEARCH & SYSTEM (9):")
    logger.info("   • Research: comprehensive, company")
    logger.info("   • System: health, performance monitoring")
    logger.info("   • Backtesting: backtest, compare, optimize, analyze, report")
    logger.info("")
    logger.info("Total: ~39 tools (consolidated from 132+)")
    logger.info("=" * 60)
