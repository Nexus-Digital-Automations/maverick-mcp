"""
Tool registry to register router tools directly on main server.
This avoids Claude Desktop's issue with mounted router tool names.
"""

import logging
from datetime import datetime

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_technical_tools(mcp: FastMCP) -> None:
    """Register technical analysis tools directly on main server"""
    from maverick_mcp.api.routers.technical import (
        get_macd_analysis,
        get_rsi_analysis,
        get_support_resistance,
    )

    # Import enhanced versions with proper timeout handling and logging
    from maverick_mcp.api.routers.technical_enhanced import (
        get_full_technical_analysis_enhanced,
        get_stock_chart_analysis_enhanced,
    )
    from maverick_mcp.validation.technical import TechnicalAnalysisRequest

    # Register with prefixed names to maintain organization
    mcp.tool(name="technical_get_rsi_analysis")(get_rsi_analysis)
    mcp.tool(name="technical_get_macd_analysis")(get_macd_analysis)
    mcp.tool(name="technical_get_support_resistance")(get_support_resistance)

    # Use enhanced versions with timeout handling and comprehensive logging
    @mcp.tool(name="technical_get_full_technical_analysis")
    async def technical_get_full_technical_analysis(ticker: str, days: int = 365):
        """
        Get comprehensive technical analysis for a given ticker with enhanced logging and timeout handling.

        This enhanced version provides:
        - Step-by-step logging for debugging
        - 25-second timeout to prevent hangs
        - Comprehensive error handling
        - Guaranteed JSON-RPC responses

        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data to analyze (default: 365)

        Returns:
            Dictionary containing complete technical analysis or error information
        """
        request = TechnicalAnalysisRequest(ticker=ticker, days=days)
        return await get_full_technical_analysis_enhanced(request)

    @mcp.tool(name="technical_get_stock_chart_analysis")
    async def technical_get_stock_chart_analysis(ticker: str):
        """
        Generate a comprehensive technical analysis chart with enhanced error handling.

        This enhanced version provides:
        - 15-second timeout for chart generation
        - Progressive chart sizing for Claude Desktop compatibility
        - Detailed logging for debugging
        - Graceful fallback on errors

        Args:
            ticker: The ticker symbol of the stock to analyze

        Returns:
            Dictionary containing chart data or error information
        """
        return await get_stock_chart_analysis_enhanced(ticker)


def register_screening_tools(mcp: FastMCP) -> None:
    """Register screening tools directly on main server"""
    from maverick_mcp.api.routers.screening import (
        get_all_screening_recommendations,
        get_maverick_bear_stocks,
        get_maverick_stocks,
        get_screening_by_criteria,
        get_supply_demand_breakouts,
    )

    mcp.tool(name="screening_get_maverick_stocks")(get_maverick_stocks)
    mcp.tool(name="screening_get_maverick_bear_stocks")(get_maverick_bear_stocks)
    mcp.tool(name="screening_get_supply_demand_breakouts")(get_supply_demand_breakouts)
    mcp.tool(name="screening_get_all_screening_recommendations")(
        get_all_screening_recommendations
    )
    mcp.tool(name="screening_get_screening_by_criteria")(get_screening_by_criteria)


def register_portfolio_tools(mcp: FastMCP) -> None:
    """Register portfolio tools directly on main server"""
    from maverick_mcp.api.routers.portfolio import (
        add_portfolio_position,
        clear_my_portfolio,
        compare_tickers,
        get_my_portfolio,
        portfolio_correlation_analysis,
        remove_portfolio_position,
        risk_adjusted_analysis,
    )

    # Portfolio management tools
    mcp.tool(name="portfolio_add_position")(add_portfolio_position)
    mcp.tool(name="portfolio_get_my_portfolio")(get_my_portfolio)
    mcp.tool(name="portfolio_remove_position")(remove_portfolio_position)
    mcp.tool(name="portfolio_clear_portfolio")(clear_my_portfolio)

    # Portfolio analysis tools
    mcp.tool(name="portfolio_risk_adjusted_analysis")(risk_adjusted_analysis)
    mcp.tool(name="portfolio_compare_tickers")(compare_tickers)
    mcp.tool(name="portfolio_portfolio_correlation_analysis")(
        portfolio_correlation_analysis
    )


def register_data_tools(mcp: FastMCP) -> None:
    """Register data tools directly on main server"""
    from maverick_mcp.api.routers.data import (
        clear_cache,
        fetch_stock_data,
        fetch_stock_data_batch,
        get_cached_price_data,
        get_chart_links,
        get_stock_info,
    )

    # Import enhanced news sentiment that uses Tiingo or LLM
    from maverick_mcp.api.routers.news_sentiment_enhanced import (
        get_news_sentiment_enhanced,
    )

    mcp.tool(name="data_fetch_stock_data")(fetch_stock_data)
    mcp.tool(name="data_fetch_stock_data_batch")(fetch_stock_data_batch)
    mcp.tool(name="data_get_stock_info")(get_stock_info)

    # Use enhanced news sentiment that doesn't rely on EXTERNAL_DATA_API_KEY
    @mcp.tool(name="data_get_news_sentiment")
    async def get_news_sentiment(ticker: str, timeframe: str = "7d", limit: int = 10):
        """
        Get news sentiment analysis for a stock using Tiingo News API or LLM analysis.

        This enhanced tool provides reliable sentiment analysis by:
        - Using Tiingo's news API if available (requires paid plan)
        - Analyzing sentiment with LLM (Claude/GPT)
        - Falling back to research-based sentiment
        - Never failing due to missing EXTERNAL_DATA_API_KEY

        Args:
            ticker: Stock ticker symbol
            timeframe: Time frame for news (1d, 7d, 30d, etc.)
            limit: Maximum number of news articles to analyze

        Returns:
            Dictionary containing sentiment analysis with confidence scores
        """
        return await get_news_sentiment_enhanced(ticker, timeframe, limit)

    mcp.tool(name="data_get_cached_price_data")(get_cached_price_data)
    mcp.tool(name="data_get_chart_links")(get_chart_links)
    mcp.tool(name="data_clear_cache")(clear_cache)


def register_performance_tools(mcp: FastMCP) -> None:
    """Register performance tools directly on main server"""
    from maverick_mcp.api.routers.performance import (
        analyze_database_index_usage,
        clear_system_caches,
        get_cache_performance_status,
        get_database_performance_status,
        get_redis_health_status,
        get_system_performance_health,
        optimize_cache_configuration,
    )

    mcp.tool(name="performance_get_system_performance_health")(
        get_system_performance_health
    )
    mcp.tool(name="performance_get_redis_health_status")(get_redis_health_status)
    mcp.tool(name="performance_get_cache_performance_status")(
        get_cache_performance_status
    )
    mcp.tool(name="performance_get_database_performance_status")(
        get_database_performance_status
    )
    mcp.tool(name="performance_analyze_database_index_usage")(
        analyze_database_index_usage
    )
    mcp.tool(name="performance_optimize_cache_configuration")(
        optimize_cache_configuration
    )
    mcp.tool(name="performance_clear_system_caches")(clear_system_caches)


def register_agent_tools(mcp: FastMCP) -> None:
    """Register agent tools directly on main server if available"""
    try:
        from maverick_mcp.api.routers.agents import (
            analyze_market_with_agent,
            compare_multi_agent_analysis,
            compare_personas_analysis,
            deep_research_financial,
            get_agent_streaming_analysis,
            list_available_agents,
            orchestrated_analysis,
        )

        # Original agent tools
        mcp.tool(name="agents_analyze_market_with_agent")(analyze_market_with_agent)
        mcp.tool(name="agents_get_agent_streaming_analysis")(
            get_agent_streaming_analysis
        )
        mcp.tool(name="agents_list_available_agents")(list_available_agents)
        mcp.tool(name="agents_compare_personas_analysis")(compare_personas_analysis)

        # New orchestration tools
        mcp.tool(name="agents_orchestrated_analysis")(orchestrated_analysis)
        mcp.tool(name="agents_deep_research_financial")(deep_research_financial)
        mcp.tool(name="agents_compare_multi_agent_analysis")(
            compare_multi_agent_analysis
        )
    except ImportError:
        # Agents module not available
        pass


def register_research_tools(mcp: FastMCP) -> None:
    """Register deep research tools directly on main server"""
    try:
        # Import all research tools from the consolidated research module
        from maverick_mcp.api.routers.research import (
            analyze_market_sentiment,
            company_comprehensive_research,
            comprehensive_research,
            get_research_agent,
        )

        # Register comprehensive research tool with all enhanced features
        @mcp.tool(name="research_comprehensive_research")
        async def research_comprehensive(
            query: str,
            persona: str | None = "moderate",
            research_scope: str | None = "standard",
            max_sources: int | None = 10,
            timeframe: str | None = "1m",
        ) -> dict:
            """
            Perform comprehensive research on any financial topic using web search and AI analysis.

            Enhanced version with:
            - Adaptive timeout based on research scope (basic: 15s, standard: 30s, comprehensive: 60s, exhaustive: 90s)
            - Step-by-step logging for debugging
            - Guaranteed responses to Claude Desktop
            - Optimized parallel execution for faster results

            Perfect for researching stocks, sectors, market trends, company analysis.
            """
            return await comprehensive_research(
                query=query,
                persona=persona or "moderate",
                research_scope=research_scope or "standard",
                max_sources=min(
                    max_sources or 25, 25
                ),  # Increased cap due to adaptive timeout
                timeframe=timeframe or "1m",
            )

        # Enhanced sentiment analysis (imported above)
        @mcp.tool(name="research_analyze_market_sentiment")
        async def analyze_market_sentiment_tool(
            topic: str,
            timeframe: str | None = "1w",
            persona: str | None = "moderate",
        ) -> dict:
            """
            Analyze market sentiment for stocks, sectors, or market trends.

            Enhanced version with:
            - 20-second timeout protection
            - Streamlined execution for speed
            - Step-by-step logging for debugging
            - Guaranteed responses
            """
            return await analyze_market_sentiment(
                topic=topic,
                timeframe=timeframe or "1w",
                persona=persona or "moderate",
            )

        # Enhanced company research (imported above)

        @mcp.tool(name="research_company_comprehensive")
        async def research_company_comprehensive(
            symbol: str,
            include_competitive_analysis: bool = False,
            persona: str | None = "moderate",
        ) -> dict:
            """
            Perform comprehensive company research and fundamental analysis.

            Enhanced version with:
            - 20-second timeout protection to prevent hanging
            - Streamlined analysis for faster execution
            - Step-by-step logging for debugging
            - Focus on core financial metrics
            - Guaranteed responses to Claude Desktop
            """
            return await company_comprehensive_research(
                symbol=symbol,
                include_competitive_analysis=include_competitive_analysis or False,
                persona=persona or "moderate",
            )

        @mcp.tool(name="research_search_financial_news")
        async def search_financial_news(
            query: str,
            timeframe: str = "1w",
            max_results: int = 20,
            persona: str = "moderate",
        ) -> dict:
            """Search for recent financial news and analysis on any topic."""
            agent = get_research_agent()

            # Use basic research for news search
            result = await agent.research_topic(
                query=f"{query} news",
                session_id=f"news_{datetime.now().timestamp()}",
                research_scope="basic",
                max_sources=max_results,
                timeframe=timeframe,
            )

            return {
                "success": True,
                "query": query,
                "news_results": result.get("processed_sources", [])[:max_results],
                "total_found": len(result.get("processed_sources", [])),
                "timeframe": timeframe,
                "persona": persona,
            }

        logger.info("Successfully registered 4 research tools directly")

    except ImportError as e:
        logger.warning(f"Research module not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register research tools: {e}")
        # Don't raise - allow server to continue without research tools


def register_backtesting_tools(mcp: FastMCP) -> None:
    """Register VectorBT backtesting tools directly on main server"""
    try:
        from maverick_mcp.api.routers.backtesting import setup_backtesting_tools

        setup_backtesting_tools(mcp)
        logger.info("✓ Backtesting tools registered successfully")
    except ImportError:
        logger.warning(
            "Backtesting module not available - VectorBT may not be installed"
        )
    except Exception as e:
        logger.error(f"✗ Failed to register backtesting tools: {e}")


def register_yahoo_finance_tools(mcp: FastMCP) -> None:
    """Register Yahoo Finance tools directly on main server.

    These tools provide comprehensive financial data from Yahoo Finance including:
    - Historical prices
    - Financial statements
    - Holder information
    - Options data
    - Analyst recommendations
    - Dividends and splits
    """
    from maverick_mcp.api.routers.yahoo_finance import (
        get_yahoo_dividends,
        get_yahoo_financial_statement,
        get_yahoo_historical_prices,
        get_yahoo_holder_info,
        get_yahoo_news,
        get_yahoo_options_chain,
        get_yahoo_options_expirations,
        get_yahoo_recommendations,
        get_yahoo_splits,
        get_yahoo_stock_actions,
        get_yahoo_stock_info,
    )

    # Historical price data
    @mcp.tool(name="yahoo_get_historical_prices")
    async def yahoo_historical_prices(
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ):
        """
        Get historical OHLCV stock prices from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
            period: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Data interval - 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

        Returns:
            Dictionary with OHLCV data, ticker info, and record count
        """
        return await get_yahoo_historical_prices(ticker, period, interval)

    # Stock info
    @mcp.tool(name="yahoo_get_stock_info")
    async def yahoo_stock_info(ticker: str):
        """
        Get comprehensive stock information from Yahoo Finance.

        Includes: price data, company info, financial metrics, earnings,
        margins, dividends, balance sheet, ownership, analyst coverage, risk metrics.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with detailed stock information
        """
        return await get_yahoo_stock_info(ticker)

    # News
    @mcp.tool(name="yahoo_get_news")
    async def yahoo_news(ticker: str):
        """
        Get recent news articles for a stock from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with news articles including title, summary, URL
        """
        return await get_yahoo_news(ticker)

    # Stock actions (dividends + splits combined)
    @mcp.tool(name="yahoo_get_stock_actions")
    async def yahoo_stock_actions(ticker: str):
        """
        Get stock dividends and splits history combined from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with dividend and split events
        """
        return await get_yahoo_stock_actions(ticker)

    # Financial statements
    @mcp.tool(name="yahoo_get_financial_statement")
    async def yahoo_financial_statement(ticker: str, statement_type: str):
        """
        Get financial statements from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            statement_type: Type of statement:
                - income_stmt: Annual income statement
                - quarterly_income_stmt: Quarterly income statement
                - balance_sheet: Annual balance sheet
                - quarterly_balance_sheet: Quarterly balance sheet
                - cashflow: Annual cash flow statement
                - quarterly_cashflow: Quarterly cash flow statement

        Returns:
            Dictionary with financial statement data by period
        """
        return await get_yahoo_financial_statement(ticker, statement_type)

    # Holder information
    @mcp.tool(name="yahoo_get_holder_info")
    async def yahoo_holder_info(ticker: str, holder_type: str):
        """
        Get holder/ownership information from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            holder_type: Type of holder info:
                - major_holders: Major shareholders summary
                - institutional_holders: Institutional ownership
                - mutualfund_holders: Mutual fund ownership
                - insider_transactions: Recent insider trades
                - insider_purchases: Insider buying activity
                - insider_roster_holders: List of insiders

        Returns:
            Dictionary with holder/ownership data
        """
        return await get_yahoo_holder_info(ticker, holder_type)

    # Options expirations
    @mcp.tool(name="yahoo_get_options_expirations")
    async def yahoo_options_expirations(ticker: str):
        """
        Get available options expiration dates from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with list of expiration dates
        """
        return await get_yahoo_options_expirations(ticker)

    # Options chain
    @mcp.tool(name="yahoo_get_options_chain")
    async def yahoo_options_chain(
        ticker: str,
        expiration_date: str,
        option_type: str = "calls",
    ):
        """
        Get options chain data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            expiration_date: Expiration date (YYYY-MM-DD format)
            option_type: "calls" or "puts"

        Returns:
            Dictionary with option contracts data
        """
        return await get_yahoo_options_chain(ticker, expiration_date, option_type)

    # Recommendations
    @mcp.tool(name="yahoo_get_recommendations")
    async def yahoo_recommendations(
        ticker: str,
        recommendation_type: str = "recommendations",
        months_back: int = 12,
    ):
        """
        Get analyst recommendations from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            recommendation_type: "recommendations" or "upgrades_downgrades"
            months_back: For upgrades_downgrades, months of history (default 12)

        Returns:
            Dictionary with analyst recommendations data
        """
        return await get_yahoo_recommendations(ticker, recommendation_type, months_back)

    # Dividends
    @mcp.tool(name="yahoo_get_dividends")
    async def yahoo_dividends(ticker: str):
        """
        Get dividend payment history from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with dividend payment history
        """
        return await get_yahoo_dividends(ticker)

    # Splits
    @mcp.tool(name="yahoo_get_splits")
    async def yahoo_splits(ticker: str):
        """
        Get stock split history from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with stock split history
        """
        return await get_yahoo_splits(ticker)


def register_openbb_tools(mcp: FastMCP) -> None:
    """Register OpenBB Platform tools directly on main server.

    OpenBB provides unified access to ALL asset classes through a single API:
    - Equity/Stocks: Historical prices, quotes, company info, search
    - Options: Full options chains with greeks
    - News: Company-specific news articles
    - Fundamentals: Financial statements, analyst estimates, dividends
    - Cryptocurrency: Historical prices, search
    - Forex/Currency: Historical, snapshots
    - Futures: Historical, term structure curves
    - Fixed Income: Treasury rates, corporate bonds
    - Economy/Macro: CPI, GDP, unemployment, FRED data

    Uses yfinance as default provider (free, no API key required).
    """
    from maverick_mcp.api.routers.openbb_data import (
        # Equity tools
        get_analyst_estimates,
        get_company_news,
        # Fixed income tools
        get_corporate_bond_spreads,
        # Economy tools
        get_cpi_data,
        # Crypto tools
        get_crypto_historical,
        # Forex tools
        get_currency_historical,
        get_currency_snapshots,
        get_dividends,
        get_economic_calendar,
        get_equity_historical,
        get_equity_info,
        get_equity_quote,
        get_financial_statements,
        get_fred_series,
        get_futures_curve,
        get_futures_historical,
        get_gdp_data,
        get_interest_rates,
        get_options_chains,
        get_treasury_rates,
        get_unemployment_data,
        search_crypto,
        search_equity,
        search_fred_series,
    )

    # ==========================================================================
    # EQUITY / STOCK TOOLS
    # ==========================================================================

    @mcp.tool(name="openbb_get_equity_historical")
    async def openbb_equity_historical(
        symbol: str = "AAPL",
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        provider: str = "yfinance",
    ):
        """
        Get historical stock price data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
            start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            interval: Data interval ('1d', '1h', '1m', '1wk', '1mo')
            provider: Data provider ('yfinance', 'fmp', 'polygon', 'tiingo')

        Returns:
            Historical OHLCV data for the stock.
        """
        return await get_equity_historical(symbol, start_date, end_date, interval, provider)

    @mcp.tool(name="openbb_get_equity_quote")
    async def openbb_equity_quote(
        symbol: str = "AAPL",
        provider: str = "yfinance",
    ):
        """
        Get real-time stock quote data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            provider: Data provider ('yfinance', 'fmp', 'intrinio')

        Returns:
            Current quote with price, volume, change, and market data.
        """
        return await get_equity_quote(symbol, provider)

    @mcp.tool(name="openbb_get_equity_info")
    async def openbb_equity_info(
        symbol: str = "AAPL",
        provider: str = "yfinance",
    ):
        """
        Get comprehensive company/stock information.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            provider: Data provider ('yfinance', 'fmp', 'intrinio')

        Returns:
            Company information including sector, industry, description,
            market cap, employees, and other fundamental data.
        """
        return await get_equity_info(symbol, provider)

    @mcp.tool(name="openbb_search_equity")
    async def openbb_equity_search(query: str = "apple"):
        """
        Search for stock/equity symbols.

        Args:
            query: Search term (e.g., 'apple', 'microsoft', 'tesla')

        Returns:
            List of matching stock symbols with company names.
        """
        return await search_equity(query)

    @mcp.tool(name="openbb_get_options_chains")
    async def openbb_options_chains(
        symbol: str = "AAPL",
        provider: str = "yfinance",
    ):
        """
        Get options chain data for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'SPY')
            provider: Data provider ('yfinance', 'intrinio', 'cboe')

        Returns:
            Options chains including calls and puts with strikes,
            expirations, greeks, and pricing data.
        """
        return await get_options_chains(symbol, provider)

    @mcp.tool(name="openbb_get_company_news")
    async def openbb_company_news(
        symbol: str = "AAPL",
        limit: int = 20,
        provider: str = "fmp",
    ):
        """
        Get recent news articles for a company.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            limit: Maximum number of articles to return (default 20)
            provider: Data provider ('fmp', 'benzinga', 'polygon', 'yfinance')

        Returns:
            Recent news articles with titles, summaries, and URLs.
        """
        return await get_company_news(symbol, limit, provider)

    @mcp.tool(name="openbb_get_financial_statements")
    async def openbb_financial_statements(
        symbol: str = "AAPL",
        statement_type: str = "income",
        period: str = "annual",
        limit: int = 5,
        provider: str = "yfinance",
    ):
        """
        Get financial statements for a company.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            statement_type: Type of statement:
                - 'income': Income statement
                - 'balance': Balance sheet
                - 'cash': Cash flow statement
            period: 'annual' or 'quarter'
            limit: Number of periods to return (default 5)
            provider: Data provider ('yfinance', 'fmp', 'polygon', 'intrinio')

        Returns:
            Financial statement data for the specified periods.
        """
        return await get_financial_statements(symbol, statement_type, period, limit, provider)

    @mcp.tool(name="openbb_get_analyst_estimates")
    async def openbb_analyst_estimates(
        symbol: str = "AAPL",
        provider: str = "fmp",
    ):
        """
        Get analyst estimates and price targets for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            provider: Data provider ('fmp', 'intrinio', 'yfinance')

        Returns:
            Analyst estimates including EPS forecasts, revenue estimates,
            and price targets.
        """
        return await get_analyst_estimates(symbol, provider)

    @mcp.tool(name="openbb_get_dividends")
    async def openbb_dividends(
        symbol: str = "AAPL",
        provider: str = "yfinance",
    ):
        """
        Get dividend history for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            provider: Data provider ('yfinance', 'fmp', 'intrinio')

        Returns:
            Dividend payment history with dates and amounts.
        """
        return await get_dividends(symbol, provider)

    # ==========================================================================
    # CRYPTOCURRENCY TOOLS
    # ==========================================================================

    @mcp.tool(name="openbb_get_crypto_historical")
    async def openbb_crypto_historical(
        symbol: str = "BTCUSD",
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        provider: str = "yfinance",
    ):
        """
        Get historical cryptocurrency price data.

        Args:
            symbol: Crypto trading pair (e.g., 'BTCUSD', 'ETHUSD', 'BTC-USD')
            start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            interval: Data interval ('1d', '1h', '1m', '1wk', '1mo')
            provider: Data provider ('yfinance', 'fmp', 'polygon')

        Returns:
            Historical OHLCV data for the cryptocurrency.
        """
        return await get_crypto_historical(symbol, start_date, end_date, interval, provider)

    @mcp.tool(name="openbb_search_crypto")
    async def openbb_search_crypto(query: str = "bitcoin"):
        """
        Search for cryptocurrency symbols.

        Args:
            query: Search term (e.g., 'bitcoin', 'ethereum', 'solana')

        Returns:
            List of matching cryptocurrency symbols and info.
        """
        return await search_crypto(query)

    # ==========================================================================
    # FOREX / CURRENCY TOOLS
    # ==========================================================================

    @mcp.tool(name="openbb_get_currency_historical")
    async def openbb_currency_historical(
        symbol: str = "EURUSD",
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        provider: str = "yfinance",
    ):
        """
        Get historical forex/currency pair data.

        Args:
            symbol: Currency pair (e.g., 'EURUSD', 'GBPUSD', 'USDJPY')
            start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            interval: Data interval ('1d', '1h', '1m')
            provider: Data provider ('yfinance', 'fmp', 'polygon')

        Returns:
            Historical OHLCV data for the currency pair.
        """
        return await get_currency_historical(symbol, start_date, end_date, interval, provider)

    @mcp.tool(name="openbb_get_currency_snapshots")
    async def openbb_currency_snapshots(
        base: str = "USD",
        counter_currencies: str = "EUR,GBP,JPY,CHF,CAD,AUD",
        provider: str = "fmp",
    ):
        """
        Get current forex exchange rate snapshots.

        Args:
            base: Base currency (e.g., 'USD', 'EUR', 'GBP')
            counter_currencies: Comma-separated counter currencies
            provider: Data provider ('fmp')

        Returns:
            Current exchange rates for the specified currency pairs.
        """
        return await get_currency_snapshots(base, counter_currencies, provider)

    # ==========================================================================
    # FUTURES TOOLS
    # ==========================================================================

    @mcp.tool(name="openbb_get_futures_historical")
    async def openbb_futures_historical(
        symbol: str = "ES",
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        provider: str = "yfinance",
    ):
        """
        Get historical futures price data.

        Args:
            symbol: Futures symbol (e.g., 'ES' S&P 500, 'NQ' Nasdaq, 'CL' Crude Oil,
                    'GC' Gold, 'SI' Silver, 'ZB' T-Bond)
            start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            interval: Data interval ('1d', '1h', '1m')
            provider: Data provider ('yfinance', 'cboe')

        Returns:
            Historical OHLCV data for the futures contract.
        """
        return await get_futures_historical(symbol, start_date, end_date, interval, provider)

    @mcp.tool(name="openbb_get_futures_curve")
    async def openbb_futures_curve(
        symbol: str = "CL",
        provider: str = "yfinance",
        date: str | None = None,
    ):
        """
        Get futures term structure (curve) data.

        Args:
            symbol: Futures symbol (e.g., 'CL' Crude Oil, 'NG' Natural Gas)
            provider: Data provider ('yfinance', 'cboe')
            date: Specific date for curve (YYYY-MM-DD). Defaults to current.

        Returns:
            Futures curve showing prices across different expiration dates.
        """
        return await get_futures_curve(symbol, provider, date)

    # ==========================================================================
    # FIXED INCOME TOOLS
    # ==========================================================================

    @mcp.tool(name="openbb_get_treasury_rates")
    async def openbb_treasury_rates(
        start_date: str | None = None,
        end_date: str | None = None,
        maturity: str | None = None,
        provider: str = "fred",
    ):
        """
        Get US Treasury interest rates.

        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to 90 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            maturity: Specific maturity ('1m', '3m', '6m', '1y', '2y', '5y', '10y', '30y')
            provider: Data provider ('fred', 'fmp')

        Returns:
            Treasury rates for various maturities.
        """
        return await get_treasury_rates(start_date, end_date, maturity, provider)

    @mcp.tool(name="openbb_get_corporate_bond_spreads")
    async def openbb_corporate_bonds(
        category: str = "aaa",
        start_date: str | None = None,
        end_date: str | None = None,
        provider: str = "fred",
    ):
        """
        Get corporate bond spreads/yields.

        Args:
            category: Bond category ('aaa', 'baa', 'high_yield')
            start_date: Start date (YYYY-MM-DD). Defaults to 90 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            provider: Data provider ('fred')

        Returns:
            Corporate bond spread data.
        """
        return await get_corporate_bond_spreads(category, start_date, end_date, provider)

    # ==========================================================================
    # ECONOMY / MACRO TOOLS
    # ==========================================================================

    @mcp.tool(name="openbb_get_economic_calendar")
    async def openbb_economic_calendar(
        start_date: str | None = None,
        end_date: str | None = None,
        provider: str = "fmp",
    ):
        """
        Get upcoming economic events calendar.

        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to today.
            end_date: End date (YYYY-MM-DD). Defaults to 30 days from now.
            provider: Data provider ('fmp', 'nasdaq', 'tradingeconomics')

        Returns:
            Economic events with dates, countries, and expected impacts.
        """
        return await get_economic_calendar(start_date, end_date, provider)

    @mcp.tool(name="openbb_get_cpi_data")
    async def openbb_cpi_data(
        country: str = "united_states",
        start_date: str | None = None,
        end_date: str | None = None,
        provider: str = "fred",
    ):
        """
        Get Consumer Price Index (CPI) inflation data.

        Args:
            country: Country code or name (e.g., 'united_states', 'japan', 'germany')
            start_date: Start date (YYYY-MM-DD). Defaults to 1 year ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            provider: Data provider ('fred', 'oecd')

        Returns:
            CPI data showing inflation trends.
        """
        return await get_cpi_data(country, start_date, end_date, provider)

    @mcp.tool(name="openbb_get_gdp_data")
    async def openbb_gdp_data(
        country: str = "united_states",
        frequency: str = "quarter",
        provider: str = "oecd",
    ):
        """
        Get Gross Domestic Product (GDP) data.

        Args:
            country: Country code or name (e.g., 'united_states', 'germany', 'japan')
            frequency: Data frequency ('quarter', 'annual')
            provider: Data provider ('oecd', 'econdb')

        Returns:
            GDP data with growth rates.
        """
        return await get_gdp_data(country, frequency, provider)

    @mcp.tool(name="openbb_get_unemployment_data")
    async def openbb_unemployment_data(
        country: str = "united_states",
        frequency: str = "month",
        provider: str = "oecd",
    ):
        """
        Get unemployment rate data.

        Args:
            country: Country code or name (e.g., 'united_states', 'all')
            frequency: Data frequency ('month', 'quarter')
            provider: Data provider ('oecd', 'fred')

        Returns:
            Unemployment rate data.
        """
        return await get_unemployment_data(country, frequency, provider)

    @mcp.tool(name="openbb_get_fred_series")
    async def openbb_fred_series(
        symbol: str = "FEDFUNDS",
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Get any FRED economic data series.

        Args:
            symbol: FRED series ID (e.g., 'FEDFUNDS', 'UNRATE', 'GDP', 'CPIAUCSL')
            start_date: Start date (YYYY-MM-DD). Defaults to 1 year ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.

        Common FRED series:
            - FEDFUNDS: Federal Funds Rate
            - UNRATE: Unemployment Rate
            - CPIAUCSL: Consumer Price Index
            - GDP: Gross Domestic Product
            - M2SL: M2 Money Supply
            - T10Y2Y: 10-Year Treasury Minus 2-Year (yield curve)

        Returns:
            Time series data from FRED.
        """
        return await get_fred_series(symbol, start_date, end_date)

    @mcp.tool(name="openbb_search_fred_series")
    async def openbb_search_fred(query: str = "inflation"):
        """
        Search for FRED economic data series.

        Args:
            query: Search term (e.g., 'inflation', 'unemployment', 'GDP')

        Returns:
            List of matching FRED series with descriptions.
        """
        return await search_fred_series(query)

    @mcp.tool(name="openbb_get_interest_rates")
    async def openbb_interest_rates(
        country: str = "united_states",
        start_date: str | None = None,
        end_date: str | None = None,
        provider: str = "oecd",
    ):
        """
        Get central bank interest rates.

        Args:
            country: Country code or name (e.g., 'united_states', 'euro_area', 'japan')
            start_date: Start date (YYYY-MM-DD). Defaults to 1 year ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            provider: Data provider ('oecd', 'fred')

        Returns:
            Central bank interest rate data.
        """
        return await get_interest_rates(country, start_date, end_date, provider)


def register_options_tools(mcp: FastMCP) -> None:
    """Register options analysis tools directly on main server.

    Options analysis provides Greeks calculation, IV analysis, and strategy evaluation.
    """
    from maverick_mcp.api.routers.options_analysis import (
        options_analyze_iv_surface,
        options_analyze_skew,
        options_calculate_greeks,
        options_calculate_iv_percentile,
        options_calculate_put_call_ratio,
        options_strategy_analyzer,
    )

    mcp.tool(name="options_calculate_greeks")(options_calculate_greeks)
    mcp.tool(name="options_analyze_iv_surface")(options_analyze_iv_surface)
    mcp.tool(name="options_calculate_iv_percentile")(options_calculate_iv_percentile)
    mcp.tool(name="options_analyze_skew")(options_analyze_skew)
    mcp.tool(name="options_calculate_put_call_ratio")(options_calculate_put_call_ratio)
    mcp.tool(name="options_strategy_analyzer")(options_strategy_analyzer)


def register_risk_tools(mcp: FastMCP) -> None:
    """Register risk metrics tools directly on main server.

    Risk metrics provide VaR, CVaR, drawdown analysis, and stress testing.
    """
    from maverick_mcp.api.routers.risk_metrics import (
        risk_adjusted_returns,
        risk_calculate_cvar,
        risk_calculate_var,
        risk_drawdown_analysis,
        risk_portfolio_var,
        risk_stress_test,
    )

    mcp.tool(name="risk_calculate_var")(risk_calculate_var)
    mcp.tool(name="risk_calculate_cvar")(risk_calculate_cvar)
    mcp.tool(name="risk_drawdown_analysis")(risk_drawdown_analysis)
    mcp.tool(name="risk_adjusted_returns")(risk_adjusted_returns)
    mcp.tool(name="risk_stress_test")(risk_stress_test)
    mcp.tool(name="risk_portfolio_var")(risk_portfolio_var)


def register_quant_tools(mcp: FastMCP) -> None:
    """Register quantitative analysis tools directly on main server.

    Quant analysis provides beta calculation, factor exposure, and correlation analysis.
    """
    from maverick_mcp.api.routers.quant_analysis import (
        quant_calculate_beta,
        quant_correlation_matrix,
        quant_factor_exposure,
        quant_factor_scores,
        quant_momentum_analysis,
        quant_volatility_analysis,
    )

    mcp.tool(name="quant_calculate_beta")(quant_calculate_beta)
    mcp.tool(name="quant_factor_exposure")(quant_factor_exposure)
    mcp.tool(name="quant_correlation_matrix")(quant_correlation_matrix)
    mcp.tool(name="quant_momentum_analysis")(quant_momentum_analysis)
    mcp.tool(name="quant_volatility_analysis")(quant_volatility_analysis)
    mcp.tool(name="quant_factor_scores")(quant_factor_scores)


def register_valuation_tools(mcp: FastMCP) -> None:
    """Register valuation model tools directly on main server.

    Valuation models provide DCF, multiples, comparable company, and fair value analysis.
    """
    from maverick_mcp.api.routers.valuation_models import (
        valuation_comparable_company,
        valuation_dcf,
        valuation_dividend_discount,
        valuation_fair_value_estimate,
        valuation_multiples,
    )

    mcp.tool(name="valuation_dcf")(valuation_dcf)
    mcp.tool(name="valuation_multiples")(valuation_multiples)
    mcp.tool(name="valuation_comparable_company")(valuation_comparable_company)
    mcp.tool(name="valuation_dividend_discount")(valuation_dividend_discount)
    mcp.tool(name="valuation_fair_value_estimate")(valuation_fair_value_estimate)


def register_breadth_tools(mcp: FastMCP) -> None:
    """Register market breadth tools directly on main server.

    Market breadth provides advance/decline, new highs/lows, sector analysis, and regime detection.
    """
    from maverick_mcp.api.routers.market_breadth import (
        breadth_advance_decline,
        breadth_divergence_check,
        breadth_market_regime,
        breadth_new_highs_lows,
        breadth_sector_analysis,
    )

    mcp.tool(name="breadth_advance_decline")(breadth_advance_decline)
    mcp.tool(name="breadth_new_highs_lows")(breadth_new_highs_lows)
    mcp.tool(name="breadth_sector_analysis")(breadth_sector_analysis)
    mcp.tool(name="breadth_market_regime")(breadth_market_regime)
    mcp.tool(name="breadth_divergence_check")(breadth_divergence_check)


def register_alternative_tools(mcp: FastMCP) -> None:
    """Register alternative data tools directly on main server.

    Alternative data provides short interest, insider transactions, institutional holdings,
    options flow, and composite sentiment analysis.
    """
    from maverick_mcp.api.routers.alternative_data import (
        alt_insider_transactions,
        alt_institutional_holdings,
        alt_options_flow,
        alt_sentiment_composite,
        alt_short_interest,
    )

    mcp.tool(name="alt_short_interest")(alt_short_interest)
    mcp.tool(name="alt_insider_transactions")(alt_insider_transactions)
    mcp.tool(name="alt_institutional_holdings")(alt_institutional_holdings)
    mcp.tool(name="alt_options_flow")(alt_options_flow)
    mcp.tool(name="alt_sentiment_composite")(alt_sentiment_composite)


def register_multi_timeframe_tools(mcp: FastMCP) -> None:
    """Register multi-timeframe analysis tools directly on main server.

    Multi-timeframe analysis provides trend alignment, RSI, and MA confirmation
    across daily, weekly, and monthly timeframes.
    """
    from maverick_mcp.api.routers.multi_timeframe import (
        multi_timeframe_moving_averages,
        multi_timeframe_rsi,
        multi_timeframe_signal_score,
        multi_timeframe_trend,
    )

    mcp.tool(name="mtf_trend_analysis")(multi_timeframe_trend)
    mcp.tool(name="mtf_rsi_analysis")(multi_timeframe_rsi)
    mcp.tool(name="mtf_moving_averages")(multi_timeframe_moving_averages)
    mcp.tool(name="mtf_signal_score")(multi_timeframe_signal_score)


def register_volume_profile_tools(mcp: FastMCP) -> None:
    """Register volume profile analysis tools directly on main server.

    Volume profile provides POC, value areas, VWAP bands, and delta analysis
    for understanding volume distribution across price levels.
    """
    from maverick_mcp.api.routers.volume_profile import (
        volume_footprint_analysis,
        volume_market_profile,
        volume_profile_analysis,
        volume_vwap_bands,
    )

    mcp.tool(name="volume_profile")(volume_profile_analysis)
    mcp.tool(name="volume_vwap_bands")(volume_vwap_bands)
    mcp.tool(name="volume_market_profile")(volume_market_profile)
    mcp.tool(name="volume_footprint")(volume_footprint_analysis)


def register_volatility_term_tools(mcp: FastMCP) -> None:
    """Register VIX term structure and volatility analysis tools.

    Volatility term structure provides VIX curve analysis, contango/backwardation,
    volatility surfaces, and market regime detection.
    """
    from maverick_mcp.api.routers.volatility_term import (
        vix_contango_backwardation,
        vix_term_structure,
        volatility_regime_indicator,
        volatility_surface_3d,
    )

    mcp.tool(name="vix_term_structure")(vix_term_structure)
    mcp.tool(name="vix_contango_backwardation")(vix_contango_backwardation)
    mcp.tool(name="volatility_surface")(volatility_surface_3d)
    mcp.tool(name="volatility_regime")(volatility_regime_indicator)


def register_mcp_prompts_and_resources(mcp: FastMCP) -> None:
    """Register MCP prompts and resources for better client introspection"""
    try:
        from maverick_mcp.api.routers.mcp_prompts import register_mcp_prompts

        register_mcp_prompts(mcp)
        logger.info("✓ MCP prompts registered successfully")
    except ImportError:
        logger.warning("MCP prompts module not available")
    except Exception as e:
        logger.error(f"✗ Failed to register MCP prompts: {e}")

    # Register introspection tools
    try:
        from maverick_mcp.api.routers.introspection import register_introspection_tools

        register_introspection_tools(mcp)
        logger.info("✓ Introspection tools registered successfully")
    except ImportError:
        logger.warning("Introspection module not available")
    except Exception as e:
        logger.error(f"✗ Failed to register introspection tools: {e}")


def register_all_router_tools(mcp: FastMCP) -> None:
    """Register all router tools directly on the main server"""
    logger.info("Starting tool registration process...")

    try:
        register_technical_tools(mcp)
        logger.info("✓ Technical tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register technical tools: {e}")

    try:
        register_screening_tools(mcp)
        logger.info("✓ Screening tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register screening tools: {e}")

    try:
        register_portfolio_tools(mcp)
        logger.info("✓ Portfolio tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register portfolio tools: {e}")

    try:
        register_performance_tools(mcp)
        logger.info("✓ Performance tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register performance tools: {e}")

    try:
        register_agent_tools(mcp)
        logger.info("✓ Agent tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register agent tools: {e}")

    try:
        # Import and register research tools on the main MCP instance
        from maverick_mcp.api.routers.research import create_research_router

        # Pass the main MCP instance to register tools directly on it
        create_research_router(mcp)
        logger.info("✓ Research tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register research tools: {e}")

    try:
        # Import and register health monitoring tools
        from maverick_mcp.api.routers.health_tools import register_health_tools

        register_health_tools(mcp)
        logger.info("✓ Health monitoring tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register health monitoring tools: {e}")

    # Register backtesting tools
    register_backtesting_tools(mcp)

    # Register OpenBB tools (unified multi-asset data: equity, options, news,
    # fundamentals, crypto, forex, futures, fixed income, economy)
    try:
        register_openbb_tools(mcp)
        logger.info("✓ OpenBB tools registered successfully (unified multi-asset data)")
    except Exception as e:
        logger.error(f"✗ Failed to register OpenBB tools: {e}")

    # Register MCP prompts and resources for introspection
    register_mcp_prompts_and_resources(mcp)

    # Register advanced analysis modules (33 new tools)
    try:
        register_options_tools(mcp)
        logger.info("✓ Options analysis tools registered successfully (6 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register options tools: {e}")

    try:
        register_risk_tools(mcp)
        logger.info("✓ Risk metrics tools registered successfully (6 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register risk tools: {e}")

    try:
        register_quant_tools(mcp)
        logger.info("✓ Quantitative analysis tools registered successfully (6 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register quant tools: {e}")

    try:
        register_valuation_tools(mcp)
        logger.info("✓ Valuation model tools registered successfully (5 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register valuation tools: {e}")

    try:
        register_breadth_tools(mcp)
        logger.info("✓ Market breadth tools registered successfully (5 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register breadth tools: {e}")

    try:
        register_alternative_tools(mcp)
        logger.info("✓ Alternative data tools registered successfully (5 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register alternative tools: {e}")

    # Register new analysis modules (12 new tools)
    try:
        register_multi_timeframe_tools(mcp)
        logger.info("✓ Multi-timeframe analysis tools registered successfully (4 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register multi-timeframe tools: {e}")

    try:
        register_volume_profile_tools(mcp)
        logger.info("✓ Volume profile tools registered successfully (4 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register volume profile tools: {e}")

    try:
        register_volatility_term_tools(mcp)
        logger.info("✓ Volatility term structure tools registered successfully (4 tools)")
    except Exception as e:
        logger.error(f"✗ Failed to register volatility term tools: {e}")

    logger.info("Tool registration process completed")
    logger.info("📋 All tools registered:")
    logger.info("   • Technical analysis tools")
    logger.info("   • Stock screening tools")
    logger.info("   • Portfolio analysis tools")
    logger.info("   • Performance monitoring tools")
    logger.info("   • Agent orchestration tools")
    logger.info("   • Research and analysis tools")
    logger.info("   • Health monitoring tools")
    logger.info("   • Backtesting system tools")
    logger.info("   • OpenBB unified tools:")
    logger.info("     - Equity: historical, quotes, info, search")
    logger.info("     - Options: full chains with greeks")
    logger.info("     - News: company-specific articles")
    logger.info("     - Fundamentals: financials, estimates, dividends")
    logger.info("     - Crypto: historical, search")
    logger.info("     - Forex: historical, snapshots")
    logger.info("     - Futures: historical, curves")
    logger.info("     - Fixed Income: treasury, corporate bonds")
    logger.info("     - Economy: CPI, GDP, FRED, calendar")
    logger.info("   • MCP prompts for introspection")
    logger.info("   • Introspection and discovery tools")
    logger.info("   • Advanced analysis modules (45 total tools):")
    logger.info("     - Options: Greeks, IV surface, skew, put/call ratio, strategies")
    logger.info("     - Risk: VaR, CVaR, drawdown, stress testing, risk-adjusted returns")
    logger.info("     - Quant: Beta, factors, correlation, momentum, volatility")
    logger.info("     - Valuation: DCF, multiples, comparables, DDM, fair value")
    logger.info("     - Breadth: Advance/decline, highs/lows, sector rotation, regime")
    logger.info("     - Alternative: Short interest, insider, institutional, options flow")
    logger.info("     - Multi-Timeframe: Trend, RSI, MA alignment, composite score")
    logger.info("     - Volume Profile: POC, value areas, VWAP bands, footprint")
    logger.info("     - Volatility Term: VIX curve, contango/backwardation, regime")
