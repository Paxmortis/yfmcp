import json
import math
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, NamedTuple

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


# Relax Accept header requirements for Streamable HTTP to support clients with
# minimal Accept values (for example ChatGPT Team connectors).
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


yfinance_server = FastMCP(
    "yfinance",
    host="0.0.0.0",
    port=8090,
    sse_path="/sse",
    message_path="/messages/",
    streamable_http_path="/mcp",
    instructions="""
Yahoo Finance MCP Server

This server exposes two MCP tools compatible with ChatGPT connectors:

- search: Look up ticker symbols and company names on Yahoo Finance. The tool
  returns a list of results with IDs shaped like `ticker:<SYMBOL>:summary`.
- fetch: Retrieve structured Yahoo Finance data for an ID returned by the search
  tool. Each response includes a plain-text overview plus structured metadata
  covering prices, financial statements, holders, actions, news, and analyst
  activity.

Workflow: call `search` with a company or ticker query, choose a result, and
invoke `fetch` with its `id` (for example `ticker:AAPL:summary`).
""",
)


DEFAULT_SEARCH_QUOTE_COUNT = 5
SUMMARY_PERIOD = "1mo"
SUMMARY_INTERVAL = "1d"
NEWS_LIMIT = 10
RECOMMENDATION_MONTHS = 12


class ResourceDescriptor(NamedTuple):
    raw: str
    category: str
    symbol: str
    section: str
    extra: tuple[str, ...]


def _convert_timestamp(value: object) -> str | None:
    if value in {None, "", 0}:
        return None
    try:
        epoch = int(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except (OverflowError, ValueError):
        return None


def _sanitize_for_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return json.loads(value.to_json(orient="records", date_format="iso"))
    if isinstance(value, pd.Series):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, pd.Index):
        return [_sanitize_for_json(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, set):
        return [_sanitize_for_json(v) for v in sorted(value, key=lambda item: repr(item))]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        try:
            return _sanitize_for_json(value.item())
        except Exception:
            pass
    return str(value)


def _load_ticker(symbol: str) -> tuple[yf.Ticker | None, list[str]]:
    notes: list[str] = []
    try:
        ticker = yf.Ticker(symbol)
    except Exception as error:  # pragma: no cover - network errors
        notes.append(f"Error initializing ticker {symbol}: {error}")
        return None, notes
    try:
        isin = ticker.isin
    except Exception as error:  # pragma: no cover - remote service quirks
        notes.append(f"Unable to validate ticker {symbol}: {error}")
        return ticker, notes
    if isin is None:
        notes.append(f"Ticker {symbol} not found on Yahoo Finance.")
        return None, notes
    return ticker, notes


def _collect_info(ticker: yf.Ticker) -> tuple[dict[str, Any] | None, str | None]:
    try:
        info = ticker.info
    except Exception as error:  # pragma: no cover - remote service errors
        return None, f"Error retrieving company information: {error}"
    if not info:
        return None, "No company information returned by Yahoo Finance."
    return info, None


def _collect_history(
    ticker: yf.Ticker, period: str = SUMMARY_PERIOD, interval: str = SUMMARY_INTERVAL
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        history_df = ticker.history(period=period, interval=interval)
    except Exception as error:  # pragma: no cover - remote service errors
        return [], f"Error retrieving price history: {error}"
    if history_df.empty:
        return [], None
    reset_df = history_df.reset_index()
    if "Date" not in reset_df.columns:
        reset_df = reset_df.rename(columns={reset_df.columns[0]: "Date"})
    records = json.loads(reset_df.to_json(orient="records", date_format="iso"))
    return records, None


def _financial_statement_to_records(statement: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column in statement.columns:
        if isinstance(column, (pd.Timestamp, datetime)):
            date_value = column.isoformat()
        else:
            date_value = str(column)
        entry: dict[str, Any] = {"date": date_value}
        for index, value in statement[column].items():
            entry[str(index)] = _sanitize_for_json(value)
        records.append(entry)
    return records


def _collect_financial_statements(ticker: yf.Ticker) -> tuple[dict[str, Any], list[str]]:
    mapping = {
        FinancialType.income_stmt: "incomeStatement",
        FinancialType.quarterly_income_stmt: "quarterlyIncomeStatement",
        FinancialType.balance_sheet: "balanceSheet",
        FinancialType.quarterly_balance_sheet: "quarterlyBalanceSheet",
        FinancialType.cashflow: "cashflow",
        FinancialType.quarterly_cashflow: "quarterlyCashflow",
    }
    data: dict[str, Any] = {}
    errors: list[str] = []
    for statement_type, label in mapping.items():
        try:
            statement = getattr(ticker, statement_type.value)
        except Exception as error:  # pragma: no cover - remote service errors
            errors.append(f"{statement_type.value}: {error}")
            data[label] = []
            continue
        if statement is None or statement.empty:
            data[label] = []
            continue
        data[label] = _financial_statement_to_records(statement)
    return data, errors


def _dataframe_to_records(
    frame: pd.DataFrame | pd.Series | None,
    *,
    index_name: str | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, pd.Series):
        working = frame.to_frame()
    else:
        working = frame
    if working.empty:
        return []
    working = working.copy()
    if index_name:
        working.index.name = index_name
        working = working.reset_index()
    else:
        working = working.reset_index(drop=True)
    if max_rows is not None:
        working = working.head(max_rows)
    return json.loads(working.to_json(orient="records", date_format="iso"))


def _collect_holders(ticker: yf.Ticker) -> tuple[dict[str, Any], list[str]]:
    mapping = {
        HolderType.major_holders: ("majorHolders", "metric"),
        HolderType.institutional_holders: ("institutionalHolders", None),
        HolderType.mutualfund_holders: ("mutualFundHolders", None),
        HolderType.insider_transactions: ("insiderTransactions", None),
        HolderType.insider_purchases: ("insiderPurchases", None),
        HolderType.insider_roster_holders: ("insiderRosterHolders", None),
    }
    data: dict[str, Any] = {}
    errors: list[str] = []
    for holder_type, (label, index_name) in mapping.items():
        try:
            holder_df = getattr(ticker, holder_type.value)
        except Exception as error:  # pragma: no cover - remote service errors
            errors.append(f"{holder_type.value}: {error}")
            data[label] = []
            continue
        data[label] = _dataframe_to_records(holder_df, index_name=index_name)
    return data, errors


def _collect_actions(ticker: yf.Ticker) -> tuple[list[dict[str, Any]], str | None]:
    try:
        actions_df = ticker.actions
    except Exception as error:  # pragma: no cover - remote service errors
        return [], f"Error retrieving stock actions: {error}"
    return _dataframe_to_records(actions_df, index_name="Date"), None


def _collect_news(ticker: yf.Ticker, limit: int = NEWS_LIMIT) -> tuple[list[dict[str, Any]], str | None]:
    try:
        raw_news = ticker.news or []
    except Exception as error:  # pragma: no cover - remote service errors
        return [], f"Error retrieving news: {error}"
    items: list[dict[str, Any]] = []
    for article in raw_news[:limit]:
        content = article.get("content") or {}
        url = content.get("canonicalUrl", {}).get("url") or article.get("link")
        summary = content.get("summary") or content.get("description") or article.get("summary")
        published_ts = (
            content.get("providerPublishTime")
            or content.get("pubDate")
            or article.get("providerPublishTime")
            or article.get("pubDate")
        )
        items.append(
            {
                "title": content.get("title") or article.get("title"),
                "publisher": content.get("provider", {}).get("displayName") or article.get("publisher"),
                "summary": summary,
                "url": url,
                "publishedAt": _convert_timestamp(published_ts) if published_ts else None,
            }
        )
    return items, None


def _collect_recommendations(
    ticker: yf.Ticker, months_back: int = RECOMMENDATION_MONTHS
) -> tuple[dict[str, Any], list[str]]:
    data: dict[str, Any] = {}
    errors: list[str] = []
    try:
        recommendations_df = ticker.recommendations
    except Exception as error:  # pragma: no cover - remote service errors
        errors.append(f"{RecommendationType.recommendations.value}: {error}")
        recommendations_df = None
    if recommendations_df is not None and not recommendations_df.empty:
        data[RecommendationType.recommendations.value] = _dataframe_to_records(
            recommendations_df, index_name="Date"
        )
    else:
        data[RecommendationType.recommendations.value] = []
    try:
        upgrades_df = ticker.upgrades_downgrades
    except Exception as error:  # pragma: no cover - remote service errors
        errors.append(f"{RecommendationType.upgrades_downgrades.value}: {error}")
        upgrades_df = None
    if upgrades_df is not None and not upgrades_df.empty:
        working = upgrades_df.reset_index()
        if "GradeDate" in working.columns:
            working["GradeDate"] = pd.to_datetime(working["GradeDate"], errors="coerce", utc=True)
            cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.DateOffset(months=months_back)
            working = working[working["GradeDate"].notna()]
            working = working[working["GradeDate"] >= cutoff]
            working = working.sort_values("GradeDate", ascending=False)
            working = working.drop_duplicates(subset=["Firm"], keep="first")
        data[RecommendationType.upgrades_downgrades.value] = _dataframe_to_records(working)
    else:
        data[RecommendationType.upgrades_downgrades.value] = []
    return data, errors


def _format_price(value: Any, currency: str | None = None) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    formatted = f"{numeric:,.2f}"
    return f"{formatted} {currency}" if currency else formatted


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric * 100:.2f}%"


def _format_large_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    absolute = abs(numeric)
    if absolute >= 1_000_000_000_000:
        return f"{numeric / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{numeric / 1_000:.2f}K"
    if float(numeric).is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}"


def _make_symbol_url(symbol: str) -> str:
    return f"https://finance.yahoo.com/quote/{symbol.upper()}"


def _compose_summary_text(
    symbol: str,
    info: dict[str, Any] | None,
    history: list[dict[str, Any]],
    news: list[dict[str, Any]],
    recommendations: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> str:
    symbol_upper = symbol.upper()
    display_name = symbol_upper
    if info:
        display_name = str(info.get("longName") or info.get("shortName") or symbol_upper)
    lines = [f"{display_name} ({symbol_upper}) overview from Yahoo Finance."]

    exchange = info.get("fullExchangeName") if info else None
    currency = (info or {}).get("currency") or (info or {}).get("financialCurrency")
    if exchange or currency:
        lines.append(f"Exchange: {exchange or 'N/A'} | Currency: {currency or 'N/A'}.")

    price = None
    if info:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is not None:
        lines.append(f"Latest price: {_format_price(price, currency)}")

    if info:
        previous_close = info.get("previousClose")
        if previous_close is not None:
            lines.append(f"Previous close: {_format_price(previous_close, currency)}")
        market_cap = info.get("marketCap")
        if market_cap is not None:
            lines.append(f"Market cap: {_format_large_number(market_cap)}")
        pe_ratio = info.get("trailingPE")
        if pe_ratio is not None:
            lines.append(f"Trailing P/E: {_format_number(pe_ratio)}")
        dividend_yield = info.get("trailingAnnualDividendYield")
        if dividend_yield is not None:
            lines.append(f"Dividend yield: {_format_percent(dividend_yield)}")
        fifty_two_low = info.get("fiftyTwoWeekLow")
        fifty_two_high = info.get("fiftyTwoWeekHigh")
        if fifty_two_low is not None and fifty_two_high is not None:
            lines.append(
                "52-week range: "
                f"{_format_price(fifty_two_low, currency)} - {_format_price(fifty_two_high, currency)}"
            )

    if history:
        last_entry = history[-1]
        last_date = last_entry.get("Date") or last_entry.get("date")
        lines.append(
            f"Recent daily price history contains {len(history)} records ending {last_date}."
        )

    if news:
        lines.append("Latest headlines:")
        for article in news[:3]:
            title = article.get("title") or "Untitled"
            publisher = article.get("publisher") or "Unknown source"
            published = article.get("publishedAt") or "date unavailable"
            lines.append(f"- {title} ({publisher}, {published})")

    if recommendations:
        upgrades = recommendations.get(RecommendationType.upgrades_downgrades.value) or []
        if upgrades:
            first_update = upgrades[0]
            firm = first_update.get("Firm") or first_update.get("firm") or "Unknown firm"
            action = first_update.get("ToGrade") or first_update.get("Action") or "update"
            grade_date = first_update.get("GradeDate") or first_update.get("date")
            lines.append(f"Most recent analyst update: {firm} -> {action} on {grade_date}.")
        rec_history = recommendations.get(RecommendationType.recommendations.value) or []
        if rec_history:
            lines.append(f"Recommendation history includes {len(rec_history)} entries.")

    if info and info.get("longBusinessSummary"):
        lines.append("")
        lines.append(str(info["longBusinessSummary"]))

    if errors:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {message}" for message in errors)

    if warnings:
        lines.append("")
        lines.append("Additional notes:")
        lines.extend(f"- {message}" for message in warnings)

    return "\n".join(lines).strip()


def _make_response(
    resource_id: str, title: str, text: str, url: str | None, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": resource_id,
        "title": title,
        "text": text,
        "url": url,
        "metadata": _sanitize_for_json(metadata),
    }


def _error_response(resource_id: str, message: str) -> dict[str, Any]:
    return {
        "id": resource_id,
        "title": "Yahoo Finance request failed",
        "text": message,
        "url": None,
        "metadata": {"error": message},
    }


def _parse_resource_id(resource_id: str) -> ResourceDescriptor:
    parts = resource_id.split(":")
    if len(parts) < 3:
        raise ValueError("Resource id must follow 'ticker:<SYMBOL>:<section>' format.")
    category, symbol, section, *extra = parts
    category = category.strip()
    symbol = symbol.strip()
    section = section.strip()
    if not category:
        raise ValueError("Resource id is missing a category prefix (expected 'ticker').")
    if not symbol:
        raise ValueError("Resource id is missing the ticker symbol.")
    if not section:
        raise ValueError("Resource id is missing the resource section (for example 'summary').")
    return ResourceDescriptor(resource_id, category, symbol, section, tuple(extra))


def _handle_summary(descriptor: ResourceDescriptor) -> dict[str, Any]:
    if descriptor.category != "ticker":
        return _error_response(
            descriptor.raw, f"Unsupported resource category '{descriptor.category}'."
        )
    ticker, warnings = _load_ticker(descriptor.symbol)
    if ticker is None:
        message = warnings[0] if warnings else f"Unable to load ticker {descriptor.symbol}."
        return _error_response(descriptor.raw, message)

    info, info_error = _collect_info(ticker)
    history, history_error = _collect_history(ticker)
    financials, financial_errors = _collect_financial_statements(ticker)
    holders, holder_errors = _collect_holders(ticker)
    actions, actions_error = _collect_actions(ticker)
    news, news_error = _collect_news(ticker)
    recommendations, recommendation_errors = _collect_recommendations(ticker)

    errors = [
        message
        for message in [
            info_error,
            history_error,
            actions_error,
            news_error,
            *financial_errors,
            *holder_errors,
            *recommendation_errors,
        ]
        if message
    ]

    metadata: dict[str, Any] = {
        "symbol": descriptor.symbol.upper(),
        "section": descriptor.section,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "history": {
            "period": SUMMARY_PERIOD,
            "interval": SUMMARY_INTERVAL,
            "data": history,
        },
        "financialStatements": financials,
        "holders": holders,
        "actions": actions,
        "news": news,
        "recommendations": recommendations,
    }
    if info is not None:
        metadata["info"] = _sanitize_for_json(info)
    if errors:
        metadata["errors"] = errors
    if warnings:
        metadata["warnings"] = warnings

    text = _compose_summary_text(
        descriptor.symbol,
        info,
        history,
        news,
        recommendations,
        errors,
        warnings,
    )
    url = _make_symbol_url(descriptor.symbol)
    title = f"{descriptor.symbol.upper()} Yahoo Finance overview"
    return _make_response(descriptor.raw, title, text, url, metadata)


@yfinance_server.tool(
    name="search",
    description="""Search Yahoo Finance for companies, ticker symbols, and related entities.

Args:
    query: str
        Free-form text describing the company, ticker, or topic to search for.
""",
    annotations=ToolAnnotations(
        title="Search Yahoo Finance",
        readOnlyHint=True,
        openWorldHint=True,
    ),
)
async def search(
    query: Annotated[str, Field(min_length=1, description="Ticker symbol, company name, or topic to search for.")]
) -> str:
    try:
        search_client = yf.Search(query, max_results=DEFAULT_SEARCH_QUOTE_COUNT, news_count=0)
    except Exception as error:  # pragma: no cover - remote service errors
        return json.dumps({"query": query, "error": f"Error performing search: {error}"})

    results: list[dict[str, Any]] = []
    quotes: list[dict[str, Any]] = []
    for quote in search_client.quotes[:DEFAULT_SEARCH_QUOTE_COUNT]:
        symbol = (quote.get("symbol") or "").strip()
        if not symbol:
            continue
        name = quote.get("shortname") or quote.get("longname") or symbol
        url = _make_symbol_url(symbol)
        entry = {
            "type": quote.get("quoteType"),
            "symbol": symbol,
            "name": name,
            "exchange": quote.get("exchDisp"),
            "score": quote.get("score"),
            "currency": quote.get("currency"),
            "url": url,
        }
        quotes.append(entry)
        results.append(
            {
                "id": f"ticker:{symbol}:summary",
                "type": "quote",
                "title": f"{name} ({symbol}) overview",
                "url": url,
                "snippet": (
                    f"Comprehensive Yahoo Finance data for {symbol}, including prices, financials, holders, "
                    "news, and analyst activity."
                ),
                "symbol": symbol,
                "exchange": quote.get("exchDisp"),
                "score": quote.get("score"),
            }
        )

    payload = {
        "query": query,
        "results": results,
        "quotes": quotes,
    }
    return json.dumps(payload)


@yfinance_server.tool(
    name="fetch",
    description="""Fetch Yahoo Finance data for an ID returned by the search tool.

Args:
    id: str
        Identifier from the search results (for example `ticker:AAPL:summary`).
""",
    annotations=ToolAnnotations(
        title="Fetch Yahoo Finance resource",
        readOnlyHint=True,
        openWorldHint=True,
    ),
)
async def fetch(
    resource_id: Annotated[str, Field(min_length=1, description="ID returned from the search tool.")]
) -> str:
    try:
        descriptor = _parse_resource_id(resource_id)
    except ValueError as error:
        return json.dumps(_error_response(resource_id, str(error)))

    if descriptor.section == "summary":
        response = _handle_summary(descriptor)
    else:
        response = _error_response(
            descriptor.raw, f"Unsupported resource section '{descriptor.section}'."
        )
    return json.dumps(response)


if __name__ == "__main__":
    transport = os.getenv("YFINANCE_MCP_TRANSPORT", "streamable-http")
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "Unsupported transport set via YFINANCE_MCP_TRANSPORT. "
            "Valid options are: stdio, sse, streamable-http."
        )

    print(f"Starting Yahoo Finance MCP server using {transport} transport...")
    yfinance_server.run(transport=transport)
