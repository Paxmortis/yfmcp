import base64
import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any
from urllib.parse import parse_qs

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
# limited Accept values (e.g. ChatGPT Team connectors).
def _relaxed_check_accept_headers(self, request):
    accept_header = request.headers.get("accept")
    if not accept_header:
        return True, True

    accept_types = [
        media_type.strip() for media_type in accept_header.split(",") if media_type.strip()
    ]
    wildcard = any(media_type in {"*/*"} for media_type in accept_types)

    has_json = (
        any(media_type.startswith(CONTENT_TYPE_JSON) for media_type in accept_types) or wildcard
    )
    has_sse = (
        any(media_type.startswith(CONTENT_TYPE_SSE) for media_type in accept_types) or wildcard
    )

    return has_json, has_sse


StreamableHTTPServerTransport._check_accept_headers = _relaxed_check_accept_headers  # type: ignore[assignment]

yfinance_server = FastMCP(
    "yfinance",
    host="0.0.0.0",
    port=8090,
    sse_path="/sse",
    message_path="/messages/",
    streamable_http_path="/mcp",
    instructions="""# Yahoo Finance MCP Server

This server exposes two tools that comply with the ChatGPT connector requirements.

## Tools

- **search(query)** – Search Yahoo Finance for ticker symbols and related headlines. Results include identifiers that can be used with the `fetch` tool.
- **fetch(id)** – Retrieve structured Yahoo Finance data for the supplied identifier. Supported identifiers include:
  - `quote:<symbol>`
  - `history:<symbol>[?period=<period>&interval=<interval>]`
  - `financials:<symbol>:<financial_statement>`
  - `holders:<symbol>:<holder_type>`
  - `options:<symbol>:expirations`
  - `options:<symbol>:chain:<expiration>[?type=calls|puts]`
  - `recommendations:<symbol>:recommendations`
  - `recommendations:<symbol>:upgrades_downgrades[?months=<int>]`
  - `news:<encoded-payload>` (returned directly from `search` results)

### Financial statement types
- income_stmt
- quarterly_income_stmt
- balance_sheet
- quarterly_balance_sheet
- cashflow
- quarterly_cashflow

### Holder types
- major_holders
- institutional_holders
- mutualfund_holders
- insider_transactions
- insider_purchases
- insider_roster_holders

### Recommendation types
- recommendations
- upgrades_downgrades
""",
)

DEFAULT_QUOTE_RESULTS = 5
DEFAULT_NEWS_RESULTS = 5

FINANCIAL_URL_SUFFIX = {
    FinancialType.income_stmt: "financials",
    FinancialType.quarterly_income_stmt: "financials",
    FinancialType.balance_sheet: "balance-sheet",
    FinancialType.quarterly_balance_sheet: "balance-sheet",
    FinancialType.cashflow: "cash-flow",
    FinancialType.quarterly_cashflow: "cash-flow",
}

QUOTE_INFO_KEYS = [
    "regularMarketPrice",
    "regularMarketChange",
    "regularMarketChangePercent",
    "regularMarketDayHigh",
    "regularMarketDayLow",
    "regularMarketPreviousClose",
    "regularMarketOpen",
    "regularMarketVolume",
    "currency",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "fiftyTwoWeekRange",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "dividendYield",
    "beta",
]

COMPANY_INFO_KEYS = [
    "shortName",
    "longName",
    "longBusinessSummary",
    "sector",
    "industry",
    "website",
    "country",
    "city",
    "address1",
    "phone",
    "fullTimeEmployees",
]


def _convert_timestamp(value: int | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _clean_value(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, pd.DateOffset)):
        return str(value)
    if isinstance(value, dict):
        return {key: _clean_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - fallback conversion
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback conversion
            pass
    return value


def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_value(value) for key, value in metadata.items()}


def _encode_payload(payload: dict[str, Any]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload, default=str).encode("utf-8"))
    return encoded.decode("utf-8").rstrip("=")


def _decode_payload(encoded: str) -> dict[str, Any]:
    padding = "=" * (-len(encoded) % 4)
    data = base64.urlsafe_b64decode(f"{encoded}{padding}".encode("utf-8"))
    return json.loads(data.decode("utf-8"))


def _build_fetch_response(
    item_id: str,
    title: str,
    url: str | None,
    metadata: dict[str, Any],
    *,
    text: str | None = None,
) -> str:
    metadata_ready = _serialize_metadata(metadata)
    payload = {
        "id": item_id,
        "title": title,
        "url": url,
        "text": text if text is not None else json.dumps(metadata_ready, indent=2),
        "metadata": metadata_ready,
    }
    return json.dumps(payload)


def _error_response(item_id: str, message: str) -> str:
    return _build_fetch_response(
        item_id,
        "Error",
        None,
        {"error": message},
        text=message,
    )


def _load_ticker(symbol: str) -> tuple[yf.Ticker | None, str | None]:
    try:
        ticker = yf.Ticker(symbol)
    except Exception as error:  # pragma: no cover - remote service errors
        return None, f"Error retrieving ticker {symbol}: {error}"

    try:
        if ticker.isin is None:
            return None, f"Company ticker {symbol} not found."
    except Exception as error:  # pragma: no cover - remote service errors
        return None, f"Error verifying ticker {symbol}: {error}"

    return ticker, None


def _quote_available_fetch_ids(symbol: str) -> list[str]:
    base = symbol.strip()
    return [
        f"quote:{base}",
        f"history:{base}?period=1mo&interval=1d",
        f"history:{base}?period=1y&interval=1wk",
        f"financials:{base}:{FinancialType.income_stmt.value}",
        f"financials:{base}:{FinancialType.quarterly_income_stmt.value}",
        f"financials:{base}:{FinancialType.balance_sheet.value}",
        f"financials:{base}:{FinancialType.quarterly_balance_sheet.value}",
        f"financials:{base}:{FinancialType.cashflow.value}",
        f"financials:{base}:{FinancialType.quarterly_cashflow.value}",
        f"holders:{base}:{HolderType.major_holders.value}",
        f"holders:{base}:{HolderType.institutional_holders.value}",
        f"holders:{base}:{HolderType.mutualfund_holders.value}",
        f"holders:{base}:{HolderType.insider_transactions.value}",
        f"holders:{base}:{HolderType.insider_purchases.value}",
        f"holders:{base}:{HolderType.insider_roster_holders.value}",
        f"recommendations:{base}:{RecommendationType.recommendations.value}",
        f"recommendations:{base}:{RecommendationType.upgrades_downgrades.value}?months=12",
        f"options:{base}:expirations",
    ]


def _dataframe_to_records(
    data: pd.DataFrame | None,
    *,
    index_name: str | None = None,
    reset_index: bool = True,
) -> list[dict[str, Any]]:
    if data is None or data.empty:
        return []

    frame = data
    if reset_index:
        if index_name is not None:
            frame = frame.reset_index(names=index_name)
        else:
            frame = frame.reset_index()

    return json.loads(frame.to_json(orient="records", date_format="iso"))


@yfinance_server.tool(
    name="search",
    description="""Search Yahoo Finance for companies, ticker symbols, and related news.

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
    query: Annotated[
        str,
        Field(
            min_length=1,
            description="Ticker symbol, company name, or topic to search for.",
        ),
    ],
) -> str:
    """Search Yahoo Finance for tickers and related news."""

    trimmed_query = query.strip()
    if not trimmed_query:
        return json.dumps({"query": query, "results": [], "error": "Empty query provided."})

    try:
        search_client = yf.Search(
            trimmed_query, max_results=DEFAULT_QUOTE_RESULTS, news_count=DEFAULT_NEWS_RESULTS
        )
    except Exception as error:  # pragma: no cover - remote service errors
        return json.dumps(
            {
                "query": trimmed_query,
                "results": [],
                "error": f"Error performing search: {error}",
            }
        )

    results: list[dict[str, Any]] = []
    quotes = list(search_client.quotes or [])
    for quote in quotes[:DEFAULT_QUOTE_RESULTS]:
        symbol = quote.get("symbol")
        if not symbol:
            continue
        name = quote.get("shortname") or quote.get("longname") or symbol
        exchange = quote.get("exchDisp")
        currency = quote.get("currency")
        snippet_parts = [f"Symbol: {symbol}"]
        if exchange:
            snippet_parts.append(f"Exchange: {exchange}")
        if currency:
            snippet_parts.append(f"Currency: {currency}")
        metadata = _serialize_metadata(
            {
                "type": "quote",
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "currency": currency,
                "score": quote.get("score"),
                "availableFetchIds": _quote_available_fetch_ids(symbol),
            }
        )
        results.append(
            {
                "id": f"quote:{symbol}",
                "title": name,
                "url": f"https://finance.yahoo.com/quote/{symbol}",
                "snippet": " | ".join(snippet_parts),
                "metadata": metadata,
            }
        )

    news_items = list(search_client.news or [])
    for article in news_items[:DEFAULT_NEWS_RESULTS]:
        title = article.get("title")
        url = article.get("link")
        if not title or not url:
            continue
        summary = article.get("summary")
        publisher = article.get("publisher")
        published_at = _convert_timestamp(article.get("providerPublishTime"))
        payload = {
            "type": "news",
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "url": url,
            "publishedAt": published_at,
            "relatedTickers": article.get("relatedTickers"),
        }
        encoded_id = _encode_payload(payload)
        snippet = summary or (publisher or url)
        results.append(
            {
                "id": f"news:{encoded_id}",
                "title": title,
                "url": url,
                "snippet": snippet,
                "metadata": _serialize_metadata(payload),
            }
        )

    return json.dumps({"query": trimmed_query, "results": results})


@yfinance_server.tool(
    name="fetch",
    description="""Retrieve structured Yahoo Finance data for an identifier returned by `search`.

Args:
    id: str
        Identifier value from a previous `search` response or one of the documented helper identifiers.
""",
    annotations=ToolAnnotations(
        title="Fetch Yahoo Finance item", readOnlyHint=True, openWorldHint=True
    ),
)
async def fetch(
    item_id: Annotated[
        str,
        Field(
            min_length=1,
            description="Identifier returned by the search tool (e.g. quote:AAPL).",
        ),
    ],
) -> str:
    normalized_id = item_id.strip()
    if not normalized_id:
        return _error_response(item_id, "Empty identifier provided.")

    prefix, sep, remainder = normalized_id.partition(":")
    if not sep or not remainder:
        return _error_response(normalized_id, "Identifier must be of the form '<type>:<value>'.")

    key = prefix.lower()
    if key == "quote":
        return _fetch_quote(normalized_id, remainder)
    if key == "history":
        return _fetch_history(normalized_id, remainder)
    if key == "financials":
        return _fetch_financials(normalized_id, remainder)
    if key == "holders":
        return _fetch_holders(normalized_id, remainder)
    if key == "options":
        return _fetch_options(normalized_id, remainder)
    if key == "recommendations":
        return _fetch_recommendations(normalized_id, remainder)
    if key == "news":
        return _fetch_news(normalized_id, remainder)

    return _error_response(normalized_id, f"Unsupported identifier prefix '{prefix}'.")


def _fetch_quote(item_id: str, remainder: str) -> str:
    symbol = remainder.strip()
    if not symbol:
        return _error_response(item_id, "Missing symbol in quote identifier.")

    ticker, error = _load_ticker(symbol)
    if error:
        return _error_response(item_id, error)
    assert ticker is not None

    try:
        info = ticker.get_info()
    except Exception:  # pragma: no cover - remote service errors
        info = {}

    history_records: list[dict[str, Any]] = []
    try:
        history_records = _dataframe_to_records(
            ticker.history(period="1mo", interval="1d"),
            index_name="Date",
        )
    except Exception:  # pragma: no cover - remote service errors
        history_records = []

    long_name = info.get("longName") or info.get("shortName") or symbol.upper()
    exchange = info.get("fullExchangeName") or info.get("exchange")
    currency = info.get("currency")
    price = info.get("regularMarketPrice")
    change = info.get("regularMarketChange")
    change_percent = info.get("regularMarketChangePercent")
    summary_lines = [f"{long_name} ({symbol.upper()})"]
    if exchange:
        summary_lines.append(f"Exchange: {exchange}")
    if price is not None:
        summary_lines.append(f"Price: {price}{f' {currency}' if currency else ''}")
    if change is not None and change_percent is not None:
        summary_lines.append(f"Change: {change} ({change_percent}%)")
    elif change is not None:
        summary_lines.append(f"Change: {change}")
    elif change_percent is not None:
        summary_lines.append(f"Change: {change_percent}%")
    range_52 = info.get("fiftyTwoWeekRange")
    if range_52:
        summary_lines.append(f"52-week range: {range_52}")
    summary_lines.append(f"Fetched at {datetime.now(timezone.utc).isoformat()}")

    metadata = {
        "symbol": symbol.upper(),
        "quote": {key: info.get(key) for key in QUOTE_INFO_KEYS if key in info},
        "company": {key: info.get(key) for key in COMPANY_INFO_KEYS if key in info},
        "history": history_records,
        "availableFetchIds": _quote_available_fetch_ids(symbol),
    }

    return _build_fetch_response(
        item_id,
        f"{long_name} overview",
        f"https://finance.yahoo.com/quote/{symbol}",
        metadata,
        text="\n".join(summary_lines),
    )


def _fetch_history(item_id: str, remainder: str) -> str:
    symbol, _, query = remainder.partition("?")
    symbol = symbol.strip()
    if not symbol:
        return _error_response(item_id, "Missing symbol in history identifier.")

    params = parse_qs(query)
    period = params.get("period", ["1mo"])[0]
    interval = params.get("interval", ["1d"])[0]

    ticker, error = _load_ticker(symbol)
    if error:
        return _error_response(item_id, error)
    assert ticker is not None

    try:
        history_df = ticker.history(period=period, interval=interval)
    except Exception as error:  # pragma: no cover - remote service errors
        return _error_response(item_id, f"Error retrieving historical data: {error}")

    records = _dataframe_to_records(history_df, index_name="Date")
    if not records:
        metadata = {
            "ticker": symbol.upper(),
            "period": period,
            "interval": interval,
            "prices": [],
        }
        return _build_fetch_response(
            item_id,
            f"{symbol.upper()} historical prices",
            f"https://finance.yahoo.com/quote/{symbol}/history",
            metadata,
            text=(
                f"No historical data available for {symbol.upper()} with period {period} and interval {interval}."
            ),
        )

    latest = records[-1]
    close_value = latest.get("Close")
    summary_lines = [
        f"{symbol.upper()} historical prices ({period}, {interval})",
        f"Most recent close on {latest.get('Date')}: {close_value}",
        f"Records returned: {len(records)}",
    ]
    metadata = {
        "ticker": symbol.upper(),
        "period": period,
        "interval": interval,
        "prices": records,
    }
    return _build_fetch_response(
        item_id,
        f"{symbol.upper()} historical prices",
        f"https://finance.yahoo.com/quote/{symbol}/history",
        metadata,
        text="\n".join(summary_lines),
    )


def _fetch_financials(item_id: str, remainder: str) -> str:
    symbol, _, fin_type_raw = remainder.partition(":")
    symbol = symbol.strip()
    fin_type_raw = fin_type_raw.strip()
    if not symbol or not fin_type_raw:
        return _error_response(
            item_id, "Financial statement identifier must include symbol and type."
        )

    try:
        fin_type = FinancialType(fin_type_raw)
    except ValueError:
        return _error_response(item_id, f"Invalid financial statement type '{fin_type_raw}'.")

    ticker, error = _load_ticker(symbol)
    if error:
        return _error_response(item_id, error)
    assert ticker is not None

    try:
        statement = getattr(ticker, fin_type.value)
    except Exception as error:  # pragma: no cover - remote service errors
        return _error_response(item_id, f"Error retrieving financial statement: {error}")

    if statement is None or statement.empty:
        return _build_fetch_response(
            item_id,
            f"{symbol.upper()} {fin_type.value}",
            f"https://finance.yahoo.com/quote/{symbol}/{FINANCIAL_URL_SUFFIX[fin_type]}",
            {
                "ticker": symbol.upper(),
                "financialType": fin_type.value,
                "statement": [],
            },
            text=f"No {fin_type.value} data available for {symbol.upper()}.",
        )

    records: list[dict[str, Any]] = []
    for column in statement.columns:
        if isinstance(column, pd.Timestamp):
            date_str = column.strftime("%Y-%m-%d")
        else:
            date_str = str(column)
        column_series = statement[column]
        entry = {"date": date_str}
        for metric, value in column_series.items():
            entry[str(metric)] = None if pd.isna(value) else _clean_value(value)
        records.append(entry)

    metadata = {
        "ticker": symbol.upper(),
        "financialType": fin_type.value,
        "statement": records,
    }

    return _build_fetch_response(
        item_id,
        f"{symbol.upper()} {fin_type.value}",
        f"https://finance.yahoo.com/quote/{symbol}/{FINANCIAL_URL_SUFFIX[fin_type]}",
        metadata,
        text=f"{symbol.upper()} {fin_type.value.replace('_', ' ')} covering {len(records)} periods.",
    )


def _fetch_holders(item_id: str, remainder: str) -> str:
    symbol, _, holder_raw = remainder.partition(":")
    symbol = symbol.strip()
    holder_raw = holder_raw.strip()
    if not symbol or not holder_raw:
        return _error_response(item_id, "Holder identifier must include symbol and holder type.")

    try:
        holder_type = HolderType(holder_raw)
    except ValueError:
        return _error_response(item_id, f"Invalid holder type '{holder_raw}'.")

    ticker, error = _load_ticker(symbol)
    if error:
        return _error_response(item_id, error)
    assert ticker is not None

    try:
        if holder_type == HolderType.major_holders:
            holder_df = ticker.major_holders
            if holder_df is not None:
                holder_df = holder_df.reset_index(names="metric").rename(columns={0: "value"})
        else:
            holder_df = getattr(ticker, holder_type.value)
            if holder_df is not None:
                holder_df = holder_df.reset_index(drop=True)
    except Exception as error:  # pragma: no cover - remote service errors
        return _error_response(item_id, f"Error retrieving holder data: {error}")

    records = _dataframe_to_records(holder_df, reset_index=False) if holder_df is not None else []
    metadata = {
        "ticker": symbol.upper(),
        "holderType": holder_type.value,
        "records": records,
    }

    if not records:
        text = f"No {holder_type.value.replace('_', ' ')} data available for {symbol.upper()}."
    else:
        text = f"{symbol.upper()} {holder_type.value.replace('_', ' ')} records returned: {len(records)}."

    return _build_fetch_response(
        item_id,
        f"{symbol.upper()} {holder_type.value}",
        f"https://finance.yahoo.com/quote/{symbol}/holders",
        metadata,
        text=text,
    )


def _fetch_options(item_id: str, remainder: str) -> str:
    symbol, _, rest = remainder.partition(":")
    symbol = symbol.strip()
    rest = rest.strip()
    if not symbol:
        return _error_response(item_id, "Options identifier must include a symbol.")

    ticker, error = _load_ticker(symbol)
    if error:
        return _error_response(item_id, error)
    assert ticker is not None

    if not rest or rest == "expirations":
        try:
            expirations = list(getattr(ticker, "options", []))
        except Exception as error:  # pragma: no cover - remote service errors
            return _error_response(item_id, f"Error retrieving option expirations: {error}")
        metadata = {
            "ticker": symbol.upper(),
            "expirationDates": expirations,
        }
        if not expirations:
            text = f"No option expirations available for {symbol.upper()}."
        else:
            text = f"{symbol.upper()} has {len(expirations)} option expiration dates. Next expiration: {expirations[0]}"
        return _build_fetch_response(
            item_id,
            f"{symbol.upper()} option expirations",
            f"https://finance.yahoo.com/quote/{symbol}/options",
            metadata,
            text=text,
        )

    if not rest.startswith("chain"):
        return _error_response(item_id, f"Unsupported options identifier segment '{rest}'.")

    _, _, chain_segment = rest.partition(":")
    expiration, _, query = chain_segment.partition("?")
    expiration = expiration.strip()
    if not expiration:
        return _error_response(item_id, "Missing expiration date for option chain identifier.")

    params = parse_qs(query)
    option_type = params.get("type", ["calls"])[0].lower()
    if option_type not in {"calls", "puts"}:
        return _error_response(item_id, "Option chain type must be 'calls' or 'puts'.")

    available_expirations = list(getattr(ticker, "options", []))
    if expiration not in available_expirations:
        return _error_response(
            item_id, f"No options available for {symbol.upper()} on {expiration}."
        )

    try:
        chain = ticker.option_chain(expiration)
    except Exception as error:  # pragma: no cover - remote service errors
        return _error_response(item_id, f"Error retrieving option chain: {error}")

    chain_df = getattr(chain, option_type, None)
    contracts = _dataframe_to_records(chain_df, reset_index=False) if chain_df is not None else []
    metadata = {
        "ticker": symbol.upper(),
        "expiration": expiration,
        "optionType": option_type,
        "contracts": contracts,
    }

    if not contracts:
        text = f"No {option_type} option contracts for {symbol.upper()} expiring on {expiration}."
    else:
        text = f"{symbol.upper()} {option_type} options for {expiration}. Contracts returned: {len(contracts)}."

    return _build_fetch_response(
        item_id,
        f"{symbol.upper()} {option_type} options ({expiration})",
        f"https://finance.yahoo.com/quote/{symbol}/options?date={expiration}",
        metadata,
        text=text,
    )


def _fetch_recommendations(item_id: str, remainder: str) -> str:
    symbol, _, rest = remainder.partition(":")
    symbol = symbol.strip()
    rest = rest.strip()
    if not symbol or not rest:
        return _error_response(item_id, "Recommendation identifier must include symbol and type.")

    rec_type_raw, _, query = rest.partition("?")
    rec_type_raw = rec_type_raw.strip()
    try:
        rec_type = RecommendationType(rec_type_raw)
    except ValueError:
        return _error_response(item_id, f"Invalid recommendation type '{rec_type_raw}'.")

    params = parse_qs(query)
    months_value = params.get("months", ["12"])[0]
    try:
        months_back = max(1, int(months_value))
    except (TypeError, ValueError):
        return _error_response(item_id, f"Invalid months parameter '{months_value}'.")

    ticker, error = _load_ticker(symbol)
    if error:
        return _error_response(item_id, error)
    assert ticker is not None

    if rec_type == RecommendationType.recommendations:
        try:
            recommendations_df = ticker.recommendations
        except Exception as error:  # pragma: no cover - remote service errors
            return _error_response(item_id, f"Error retrieving recommendations: {error}")

        records = _dataframe_to_records(recommendations_df, index_name="Date")
        metadata = {
            "ticker": symbol.upper(),
            "recommendationType": rec_type.value,
            "entries": records,
        }
        if not records:
            text = f"No analyst recommendations available for {symbol.upper()}."
        else:
            text = f"{symbol.upper()} analyst recommendations retrieved: {len(records)}."
        return _build_fetch_response(
            item_id,
            f"{symbol.upper()} recommendations",
            f"https://finance.yahoo.com/quote/{symbol}/analysis",
            metadata,
            text=text,
        )

    try:
        upgrades_df = ticker.upgrades_downgrades.reset_index()
    except Exception as error:  # pragma: no cover - remote service errors
        return _error_response(item_id, f"Error retrieving upgrades/downgrades: {error}")

    if upgrades_df is None or upgrades_df.empty:
        metadata = {
            "ticker": symbol.upper(),
            "recommendationType": rec_type.value,
            "entries": [],
        }
        return _build_fetch_response(
            item_id,
            f"{symbol.upper()} upgrades/downgrades",
            f"https://finance.yahoo.com/quote/{symbol}/analysis",
            metadata,
            text=f"No upgrades/downgrades available for {symbol.upper()}.",
        )

    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
    filtered = upgrades_df[upgrades_df["GradeDate"] >= cutoff]
    filtered = filtered.sort_values("GradeDate", ascending=False)
    latest_by_firm = filtered.drop_duplicates(subset=["Firm"])
    records = _dataframe_to_records(latest_by_firm, reset_index=False)
    metadata = {
        "ticker": symbol.upper(),
        "recommendationType": rec_type.value,
        "monthsBack": months_back,
        "entries": records,
    }
    if not records:
        text = f"No upgrades/downgrades for {symbol.upper()} in the past {months_back} months."
    else:
        text = f"{symbol.upper()} upgrades/downgrades in the last {months_back} months: {len(records)} unique firms."
    return _build_fetch_response(
        item_id,
        f"{symbol.upper()} upgrades/downgrades",
        f"https://finance.yahoo.com/quote/{symbol}/analysis",
        metadata,
        text=text,
    )


def _fetch_news(item_id: str, remainder: str) -> str:
    encoded = remainder.strip()
    if not encoded:
        return _error_response(item_id, "Missing payload in news identifier.")

    try:
        payload = _decode_payload(encoded)
    except Exception as error:
        return _error_response(item_id, f"Invalid news identifier payload: {error}")

    title = payload.get("title") or "News article"
    summary = payload.get("summary") or "No summary available."
    publisher = payload.get("publisher")
    published_at = payload.get("publishedAt")
    url = payload.get("url")

    summary_lines = [title]
    if publisher:
        summary_lines.append(f"Publisher: {publisher}")
    if published_at:
        summary_lines.append(f"Published: {published_at}")
    summary_lines.append("")
    summary_lines.append(summary)

    metadata = {
        "title": title,
        "summary": summary,
        "publisher": publisher,
        "publishedAt": published_at,
        "url": url,
        "relatedTickers": payload.get("relatedTickers"),
    }

    return _build_fetch_response(
        item_id,
        title,
        url,
        metadata,
        text="\n".join(line for line in summary_lines if line is not None),
    )


if __name__ == "__main__":
    transport = os.getenv("YFINANCE_MCP_TRANSPORT", "streamable-http")
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "Unsupported transport set via YFINANCE_MCP_TRANSPORT. "
            "Valid options are: stdio, sse, streamable-http."
        )

    print(f"Starting Yahoo Finance MCP server using {transport} transport...")
    yfinance_server.run(transport=transport)
