# Yahoo Finance MCP 服务器

<div align="right">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a>
</div>

这是一个基于模型上下文协议（MCP）的服务器，提供来自 Yahoo Finance 的全面金融数据。它允许您获取股票的详细信息，包括历史价格、公司信息、财务报表、期权数据和市场新闻。

[![smithery badge](https://smithery.ai/badge/@Alex2Yang97/yahoo-finance-mcp)](https://smithery.ai/server/@Alex2Yang97/yahoo-finance-mcp)

## 演示

![MCP 演示](assets/demo.gif)

## MCP 工具

为了遵守 `restrictions.md` 中记录的 ChatGPT 连接器规范，服务器现在只暴露必需的
`search` 和 `fetch` 两个工具，其余功能通过 `fetch` 返回的 `metadata` 提供。

| 工具 | 描述 |
|------|-------------|
| `search` | 在 Yahoo Finance 中搜索股票代码或公司名称，返回 `ticker:<股票代码>:summary` 形式的 ID。 |
| `fetch` | 根据搜索结果的 ID 获取详细数据，包含自然语言摘要和结构化元数据（价格历史、财务报表、持有人、公司行动、新闻以及分析师推荐等）。 |

`fetch` 返回的 `metadata` 包含以下部分，用于覆盖旧工具的功能：

- `info`: Yahoo Finance 的原始公司信息字典。
- `history`: 最近一个月的日线 OHLCV 价格历史。
- `financialStatements`: 年度与季度的利润表、资产负债表和现金流量表。
- `holders`: 主要、机构、共同基金和内幕持有人数据。
- `actions`: 分红与拆股历史。
- `news`: 该股票的最新 Yahoo Finance 新闻。
- `recommendations`: 分析师推荐历史以及最近 12 个月的评级上调和下调。

## 实际应用场景

使用此 MCP 服务器，您可以利用 Claude 进行：

### 股票分析

- **价格分析**："显示苹果公司过去 6 个月的每日历史股价。"
- **财务健康**："获取微软的季度资产负债表。"
- **业绩指标**："从股票信息中获取特斯拉的关键财务指标。"
- **趋势分析**："比较亚马逊和谷歌的季度利润表。"
- **现金流分析**："显示英伟达的年度现金流量表。"

### 市场研究

- **新闻分析**："获取关于 Meta Platforms 的最新新闻文章。"
- **机构活动**："显示苹果股票的机构股东。"
- **内幕交易**："特斯拉最近的内幕交易有哪些？"
- **期权分析**："获取 SPY 在 2024-06-21 到期的看涨期权链。"
- **分析师覆盖**："亚马逊过去 3 个月的分析师推荐是什么？"

### 投资研究

- "使用微软最新的季度财务报表创建其财务健康状况的全面分析。"
- "比较可口可乐和百事可乐的分红历史和股票拆分。"
- "分析特斯拉过去一年的机构所有权变化。"
- "生成一份关于苹果股票 30 天到期的期权市场活动报告。"
- "总结过去 6 个月科技行业的最新分析师评级变更。"

## 系统要求

- Python 3.11 或更高版本
- `pyproject.toml` 中列出的依赖项，包括：
  - mcp
  - yfinance
  - pandas
  - pydantic
  - 以及其他数据处理包

## 安装

1. 克隆此仓库：
   ```bash
   git clone https://github.com/Alex2Yang97/yahoo-finance-mcp.git
   cd yahoo-finance-mcp
   ```

2. 创建并激活虚拟环境，安装依赖：
   ```bash
   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

## 使用方法

### 开发模式

您可以通过运行以下命令使用 MCP Inspector 测试服务器：

```bash
uv run server.py
```

这将启动服务器并允许您测试可用工具。

### 与 Claude Desktop 集成

要将此服务器与 Claude Desktop 集成：

1. 在本地机器上安装 Claude Desktop。
2. 在本地机器上安装 VS Code。然后运行以下命令打开 `claude_desktop_config.json` 文件：
   - MacOS：`code ~/Library/Application\ Support/Claude/claude_desktop_config.json`
   - Windows：`code $env:AppData\Claude\claude_desktop_config.json`

3. 编辑 Claude Desktop 配置文件，位于：
   - macOS：
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
   - Windows：
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

   - **注意**：您可能需要在命令字段中填入 uv 可执行文件的完整路径。您可以通过在 MacOS/Linux 上运行 `which uv` 或在 Windows 上运行 `where uv` 来获取此路径。

4. 重启 Claude Desktop

## 许可证

MIT 