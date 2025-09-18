[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/alex2yang97-yahoo-finance-mcp-badge.png)](https://mseep.ai/app/alex2yang97-yahoo-finance-mcp)

# Yahoo Finance MCP Server

<div align="right">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a>
</div>

This is a Model Context Protocol (MCP) server that provides comprehensive financial data from Yahoo Finance. It allows you to retrieve detailed information about stocks, including historical prices, company information, financial statements, options data, and market news.

[![smithery badge](https://smithery.ai/badge/@Alex2Yang97/yahoo-finance-mcp)](https://smithery.ai/server/@Alex2Yang97/yahoo-finance-mcp)

## Demo

![MCP Demo](assets/demo.gif)

## MCP Tools

To comply with the ChatGPT connector restrictions documented in `restrictions.md`,
the server now exposes only the required `search` and `fetch` tools. Everything
that previously lived behind individual endpoints is bundled into the metadata
returned by `fetch`.

| Tool | Description |
|------|-------------|
| `search` | Search Yahoo Finance for ticker symbols or company names and return results whose IDs follow the `ticker:<SYMBOL>:summary` pattern. |
| `fetch` | Retrieve the resource identified by a search result. The response contains a natural-language summary plus structured metadata covering price history, financial statements, holders, corporate actions, news headlines, and analyst recommendations. |

`fetch` responses expose a `metadata` object with the following sections so that
agents can reproduce the functionality of the legacy tools:

- `info`: The raw Yahoo Finance company information dictionary.
- `history`: Recent OHLCV price history (default one month of daily bars).
- `financialStatements`: Annual and quarterly income statement, balance sheet,
  and cash flow statement data.
- `holders`: Major, institutional, mutual fund, and insider ownership data.
- `actions`: Dividend and split history.
- `news`: Latest Yahoo Finance headlines for the symbol.
- `recommendations`: Analyst recommendation history plus filtered upgrades and
  downgrades from the last twelve months.

## Real-World Use Cases

With this MCP server, you can use Claude to:

### Stock Analysis

- **Price Analysis**: "Show me the historical stock prices for AAPL over the last 6 months with daily intervals."
- **Financial Health**: "Get the quarterly balance sheet for Microsoft."
- **Performance Metrics**: "What are the key financial metrics for Tesla from the stock info?"
- **Trend Analysis**: "Compare the quarterly income statements of Amazon and Google."
- **Cash Flow Analysis**: "Show me the annual cash flow statement for NVIDIA."

### Market Research

- **News Analysis**: "Get the latest news articles about Meta Platforms."
- **Institutional Activity**: "Show me the institutional holders of Apple stock."
- **Insider Trading**: "What are the recent insider transactions for Tesla?"
- **Options Analysis**: "Get the options chain for SPY with expiration date 2024-06-21 for calls."
- **Analyst Coverage**: "What are the analyst recommendations for Amazon over the last 3 months?"

### Investment Research

- "Create a comprehensive analysis of Microsoft's financial health using their latest quarterly financial statements."
- "Compare the dividend history and stock splits of Coca-Cola and PepsiCo."
- "Analyze the institutional ownership changes in Tesla over the past year."
- "Generate a report on the options market activity for Apple stock with expiration in 30 days."
- "Summarize the latest analyst upgrades and downgrades in the tech sector over the last 6 months."

## Requirements

- Python 3.11 or higher
- Dependencies as listed in `pyproject.toml`, including:
  - mcp
  - yfinance
  - pandas
  - pydantic
  - and other packages for data processing

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Alex2Yang97/yahoo-finance-mcp.git
   cd yahoo-finance-mcp
   ```

2. Create and activate a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

## Usage

### Development Mode

You can test the server with MCP Inspector by running:

```bash
uv run server.py
```

This will start the server and allow you to test the available tools.

### Expose via ngrok (Streamable HTTP)

The server now runs using the MCP Streamable HTTP transport on `0.0.0.0:8090` at the `/mcp` endpoint. This works well with ngrok. To expose it publicly:

1. Start the server:
   ```bash
   uv run server.py
   ```

2. In another terminal, start ngrok:
   ```bash
   ngrok http 8090
   ```

3. Use the generated HTTPS URL **with the `/mcp` path appended** (for example, `https://<your-ngrok-id>.ngrok-free.app/mcp`) when configuring your MCP client.

> [!TIP]
> If you need the legacy SSE transport for an older client, set the environment variable `YFINANCE_MCP_TRANSPORT=sse` before starting the server. Otherwise, the Streamable HTTP transport is recommended and required for ChatGPT Team/Enterprise connectors.

### Integration with Claude for Desktop

To integrate this server with Claude for Desktop:

1. Install Claude for Desktop to your local machine.
2. Install VS Code to your local machine. Then run the following command to open the `claude_desktop_config.json` file:
   - MacOS: `code ~/Library/Application\ Support/Claude/claude_desktop_config.json`
   - Windows: `code $env:AppData\Claude\claude_desktop_config.json`

3. Edit the Claude for Desktop config file, located at:
   - macOS: 
     ```json
     {
       "mcpServers": {
         "yfinance": {
           "command": "uv",
           "args": [
             "--directory",
             "/ABSOLUTE/PATH/TO/PARENT/FOLDER/yahoo-finance-mcp",
             "run",
             "server.py"
           ]
         }
       }
     }
     ```
   - Windows:
     ```json
     {
       "mcpServers": {
         "yfinance": {
           "command": "uv",
           "args": [
             "--directory",
             "C:\\ABSOLUTE\\PATH\\TO\\PARENT\\FOLDER\\yahoo-finance-mcp",
             "run",
             "server.py"
           ]
         }
       }
     }
     ```

   - **Note**: You may need to put the full path to the uv executable in the command field. You can get this by running `which uv` on MacOS/Linux or `where uv` on Windows.

4. Restart Claude for Desktop

## License

MIT
