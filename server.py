import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import (
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_SSE,
    StreamableHTTPServerTransport,
)
from mcp.types import ToolAnnotations
from pydantic import Field


# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


# Initialize FastMCP server
# Relax Accept header requirements for Streamable HTTP to support clients that
# send only wildcard or partial Accept values (e.g. ChatGPT Team connectors).
def _relaxed_check_accept_headers(self, request):
    accept_header = request.headers.get("accept")
    if not accept_header:
        return True, True

    accept_types = [media_type.strip() for media_type in accept_header.split(",") if media_type.strip()]
    wildcard = any(media_type in {"*/*"} for media_type in accept_types)

    has_json = any(media_type.startswith(CONTENT_TYPE_JSON) for media_type in accept_types) or wildcard
    has_sse = any(media_type.startswith(CONTENT_TYPE_SSE) for media_type in accept_types) or wildcard

    return has_json, has_sse


StreamableHTTPServerTransport._check_accept_headers = _relaxed_check_accept_headers  # type: ignore[assignment]

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


CONNECTOR_SAFE_MODE = _env_flag("YFINANCE_CONNECTOR_SAFE_MODE", True)

if CONNECTOR_SAFE_MODE:
    instructions_text = """
# Yahoo Finance MCP Server

This server is used to get information about a given ticker symbol from yahoo finance.

Connector-safe mode is enabled. Only discovery-focused tools are registered for use with clients such as ChatGPT Team connectors.

Available tools:
- search: Search Yahoo Finance for ticker symbols and related news.

(The upcoming `fetch` tool will be exposed automatically once available.)

Set the environment variable `YFINANCE_CONNECTOR_SAFE_MODE=0` before starting the server to expose the full research toolset.
"""
else:
    instructions_text = """
# Yahoo Finance MCP Server

This server is used to get information about a given ticker symbol from yahoo finance.

Connector-safe mode is disabled. All research tools are registered.

Available tools:
- search: Search Yahoo Finance for ticker symbols and related news.
- get_historical_stock_prices: Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
- get_stock_info: Get stock information for a given ticker symbol from yahoo finance. Include the following information: Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.
- get_yahoo_finance_news: Get news for a given ticker symbol from yahoo finance.
- get_stock_actions: Get stock dividends and stock splits for a given ticker symbol from yahoo finance.
- get_financial_statement: Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
- get_holder_info: Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
- get_option_expiration_dates: Fetch the available options expiration dates for a given ticker symbol.
- get_option_chain: Fetch the option chain for a given ticker symbol, expiration date, and option type.
- get_recommendations: Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.
"""


yfinance_server = FastMCP(
    "yfinance",
    host="0.0.0.0",
    port=8090,
    sse_path="/sse",
    message_path="/messages/",
    streamable_http_path="/mcp",
    instructions=instructions_text,
)


def register_tool(*tool_args, safe_only: bool = False, **tool_kwargs):
    """Conditionally register tools based on connector-safe mode.

    Tools that should remain available to connector clients (e.g., search or
    the upcoming fetch tool) should leave ``safe_only`` as ``False``. Tools
    meant for the full research experience must set ``safe_only=True`` so they
    are skipped when ``CONNECTOR_SAFE_MODE`` is enabled.
    """

    def decorator(func):
        if CONNECTOR_SAFE_MODE and safe_only:
            return func
        return yfinance_server.tool(*tool_args, **tool_kwargs)(func)

    return decorator


# Search tool to satisfy ChatGPT search action requirement
@register_tool(
    name="search",
    description="""Search Yahoo Finance for companies, ticker symbols, and related news.

Args:
    query: str
        Free-form text describing the company, ticker, or topic to search for.
    quote_count: int
        Maximum number of ticker matches to return (default 5).
    news_count: int
        Maximum number of news articles to include (default 5).
""",
    annotations=ToolAnnotations(
        title="Search Yahoo Finance",
        readOnlyHint=True,
        openWorldHint=True,
    ),
)
async def search(
    query: Annotated[str, Field(min_length=1, description="Ticker symbol, company name, or topic to search for.")],
    quote_count: Annotated[int, Field(ge=1, le=25, description="Maximum number of ticker matches to include.")] = 5,
    news_count: Annotated[int, Field(ge=0, le=25, description="Maximum number of news articles to include.")] = 5,
) -> str:
    """Search Yahoo Finance for tickers and related news."""

    try:
        search_client = yf.Search(query, max_results=quote_count, news_count=news_count)
    except Exception as error:  # pragma: no cover - remote service errors
        return json.dumps({"query": query, "error": f"Error performing search: {error}"})

    def _convert_timestamp(value: int | None) -> str | None:
        if not value:
            return None
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
        except (TypeError, ValueError):
            return None

    results: list[dict[str, object]] = []
    quotes: list[dict[str, object]] = []
    for quote in search_client.quotes[:quote_count]:
        entry = {
            "type": quote.get("quoteType"),
            "symbol": quote.get("symbol"),
            "name": quote.get("shortname") or quote.get("longname"),
            "exchange": quote.get("exchDisp"),
            "score": quote.get("score"),
            "exchangeDelay": quote.get("exchangeDelay"),
            "currency": quote.get("currency"),
            "url": f"https://finance.yahoo.com/quote/{quote.get('symbol')}" if quote.get("symbol") else None,
        }
        quotes.append(entry)
        results.append(
            {
                "type": "quote",
                "title": entry.get("name") or entry.get("symbol"),
                "url": entry.get("url"),
                "snippet": f"Symbol {entry.get('symbol')} on {entry.get('exchange')} ({entry.get('currency') or 'currency N/A'})",
                "score": entry.get("score"),
                "symbol": entry.get("symbol"),
            }
        )

    news: list[dict[str, object]] = []
    for article in search_client.news[:news_count]:
        entry = {
            "type": "news",
            "title": article.get("title"),
            "publisher": article.get("publisher"),
            "summary": article.get("summary"),
            "url": article.get("link"),
            "publishedAt": _convert_timestamp(article.get("providerPublishTime")),
        }
        news.append(entry)
        results.append(
            {
                "type": "news",
                "title": entry.get("title"),
                "url": entry.get("url"),
                "snippet": entry.get("summary"),
                "publisher": entry.get("publisher"),
                "publishedAt": entry.get("publishedAt"),
            }
        )

    return json.dumps(
        {
            "query": query,
            "results": results,
            "quotes": quotes,
            "news": news,
        }
    )


@register_tool(
    name="get_historical_stock_prices",
    description="""Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
Args:
    ticker: str
        The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        Either Use period parameter or use start and end
        Default is "1mo"
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days
        Default is "1d"
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_historical_stock_prices(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> str:
    """Get historical stock prices for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
            Default is "1mo"
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is "1d"
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting historical stock prices for {ticker}: {e}")
        return f"Error: getting historical stock prices for {ticker}: {e}"

    # If the company is found, get the historical data
    hist_data = company.history(period=period, interval=interval)
    hist_data = hist_data.reset_index(names="Date")
    hist_data = hist_data.to_json(orient="records", date_format="iso")
    return hist_data


@register_tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance. Include the following information:
Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.

Args:
    ticker: str
        The ticker symbol of the stock to get information for, e.g. "AAPL"
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_stock_info(ticker: str) -> str:
    """Get stock information for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting stock information for {ticker}: {e}")
        return f"Error: getting stock information for {ticker}: {e}"
    info = company.info
    return json.dumps(info)


@register_tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get news for, e.g. "AAPL"
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_yahoo_finance_news(ticker: str) -> str:
    """Get news for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get news for, e.g. "AAPL"
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting news for {ticker}: {e}")
        return f"Error: getting news for {ticker}: {e}"

    # If the company is found, get the news
    try:
        news = company.news
    except Exception as e:
        print(f"Error: getting news for {ticker}: {e}")
        return f"Error: getting news for {ticker}: {e}"

    news_list = []
    for news in company.news:
        if news.get("content", {}).get("contentType", "") == "STORY":
            title = news.get("content", {}).get("title", "")
            summary = news.get("content", {}).get("summary", "")
            description = news.get("content", {}).get("description", "")
            url = news.get("content", {}).get("canonicalUrl", {}).get("url", "")
            news_list.append(
                f"Title: {title}\nSummary: {summary}\nDescription: {description}\nURL: {url}"
            )
    if not news_list:
        print(f"No news found for company that searched with {ticker} ticker.")
        return f"No news found for company that searched with {ticker} ticker."
    return "\n\n".join(news_list)


@register_tool(
    name="get_stock_actions",
    description="""Get stock dividends and stock splits for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get stock actions for, e.g. "AAPL"
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_stock_actions(ticker: str) -> str:
    """Get stock dividends and stock splits for a given ticker symbol"""
    try:
        company = yf.Ticker(ticker)
    except Exception as e:
        print(f"Error: getting stock actions for {ticker}: {e}")
        return f"Error: getting stock actions for {ticker}: {e}"
    actions_df = company.actions
    actions_df = actions_df.reset_index(names="Date")
    return actions_df.to_json(orient="records", date_format="iso")


@register_tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.

Args:
    ticker: str
        The ticker symbol of the stock to get financial statement for, e.g. "AAPL"
    financial_type: str
        The type of financial statement to get. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    """Get financial statement for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting financial statement for {ticker}: {e}")
        return f"Error: getting financial statement for {ticker}: {e}"

    if financial_type == FinancialType.income_stmt:
        financial_statement = company.income_stmt
    elif financial_type == FinancialType.quarterly_income_stmt:
        financial_statement = company.quarterly_income_stmt
    elif financial_type == FinancialType.balance_sheet:
        financial_statement = company.balance_sheet
    elif financial_type == FinancialType.quarterly_balance_sheet:
        financial_statement = company.quarterly_balance_sheet
    elif financial_type == FinancialType.cashflow:
        financial_statement = company.cashflow
    elif financial_type == FinancialType.quarterly_cashflow:
        financial_statement = company.quarterly_cashflow
    else:
        return f"Error: invalid financial type {financial_type}. Please use one of the following: {FinancialType.income_stmt}, {FinancialType.quarterly_income_stmt}, {FinancialType.balance_sheet}, {FinancialType.quarterly_balance_sheet}, {FinancialType.cashflow}, {FinancialType.quarterly_cashflow}."

    # Create a list to store all the json objects
    result = []

    # Loop through each column (date)
    for column in financial_statement.columns:
        if isinstance(column, pd.Timestamp):
            date_str = column.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
        else:
            date_str = str(column)

        # Create a dictionary for each date
        date_obj = {"date": date_str}

        # Add each metric as a key-value pair
        for index, value in financial_statement[column].items():
            # Add the value, handling NaN values
            date_obj[index] = None if pd.isna(value) else value

        result.append(date_obj)

    return json.dumps(result)


@register_tool(
    name="get_holder_info",
    description="""Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.

Args:
    ticker: str
        The ticker symbol of the stock to get holder information for, e.g. "AAPL"
    holder_type: str
        The type of holder information to get. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    """Get holder information for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting holder info for {ticker}: {e}")
        return f"Error: getting holder info for {ticker}: {e}"

    if holder_type == HolderType.major_holders:
        return company.major_holders.reset_index(names="metric").to_json(orient="records")
    elif holder_type == HolderType.institutional_holders:
        return company.institutional_holders.to_json(orient="records")
    elif holder_type == HolderType.mutualfund_holders:
        return company.mutualfund_holders.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_transactions:
        return company.insider_transactions.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_purchases:
        return company.insider_purchases.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_roster_holders:
        return company.insider_roster_holders.to_json(orient="records", date_format="iso")
    else:
        return f"Error: invalid holder type {holder_type}. Please use one of the following: {HolderType.major_holders}, {HolderType.institutional_holders}, {HolderType.mutualfund_holders}, {HolderType.insider_transactions}, {HolderType.insider_purchases}, {HolderType.insider_roster_holders}."


@register_tool(
    name="get_option_expiration_dates",
    description="""Fetch the available options expiration dates for a given ticker symbol.

Args:
    ticker: str
        The ticker symbol of the stock to get option expiration dates for, e.g. "AAPL"
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_option_expiration_dates(ticker: str) -> str:
    """Fetch the available options expiration dates for a given ticker symbol."""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting option expiration dates for {ticker}: {e}")
        return f"Error: getting option expiration dates for {ticker}: {e}"
    return json.dumps(company.options)


@register_tool(
    name="get_option_chain",
    description="""Fetch the option chain for a given ticker symbol, expiration date, and option type.

Args:
    ticker: str
        The ticker symbol of the stock to get option chain for, e.g. "AAPL"
    expiration_date: str
        The expiration date for the options chain (format: 'YYYY-MM-DD')
    option_type: str
        The type of option to fetch ('calls' or 'puts')
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
    """Fetch the option chain for a given ticker symbol, expiration date, and option type.

    Args:
        ticker: The ticker symbol of the stock
        expiration_date: The expiration date for the options chain (format: 'YYYY-MM-DD')
        option_type: The type of option to fetch ('calls' or 'puts')

    Returns:
        str: JSON string containing the option chain data
    """

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting option chain for {ticker}: {e}")
        return f"Error: getting option chain for {ticker}: {e}"

    # Check if the expiration date is valid
    if expiration_date not in company.options:
        return f"Error: No options available for the date {expiration_date}. You can use `get_option_expiration_dates` to get the available expiration dates."

    # Check if the option type is valid
    if option_type not in ["calls", "puts"]:
        return "Error: Invalid option type. Please use 'calls' or 'puts'."

    # Get the option chain
    option_chain = company.option_chain(expiration_date)
    if option_type == "calls":
        return option_chain.calls.to_json(orient="records", date_format="iso")
    elif option_type == "puts":
        return option_chain.puts.to_json(orient="records", date_format="iso")
    else:
        return f"Error: invalid option type {option_type}. Please use one of the following: calls, puts."


@register_tool(
    name="get_recommendations",
    description="""Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.

Args:
    ticker: str
        The ticker symbol of the stock to get recommendations for, e.g. "AAPL"
    recommendation_type: str
        The type of recommendation to get. You can choose from the following recommendation types: recommendations, upgrades_downgrades.
    months_back: int
        The number of months back to get upgrades/downgrades for, default is 12.
""",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    safe_only=True,
)
async def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    """Get recommendations or upgrades/downgrades for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting recommendations for {ticker}: {e}")
        return f"Error: getting recommendations for {ticker}: {e}"
    try:
        if recommendation_type == RecommendationType.recommendations:
            return company.recommendations.to_json(orient="records")
        elif recommendation_type == RecommendationType.upgrades_downgrades:
            # Get the upgrades/downgrades based on the cutoff date
            upgrades_downgrades = company.upgrades_downgrades.reset_index()
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[
                upgrades_downgrades["GradeDate"] >= cutoff_date
            ]
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            # Get the first occurrence (most recent) for each firm
            latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
            return latest_by_firm.to_json(orient="records", date_format="iso")
    except Exception as e:
        print(f"Error: getting recommendations for {ticker}: {e}")
        return f"Error: getting recommendations for {ticker}: {e}"


if __name__ == "__main__":
    # Initialize and run the server
    transport = os.getenv("YFINANCE_MCP_TRANSPORT", "streamable-http")
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "Unsupported transport set via YFINANCE_MCP_TRANSPORT. "
            "Valid options are: stdio, sse, streamable-http."
        )

    mode_label = "connector-safe" if CONNECTOR_SAFE_MODE else "full"
    print(
        "Starting Yahoo Finance MCP server using "
        f"{transport} transport in {mode_label} mode..."
    )
    yfinance_server.run(transport=transport)
