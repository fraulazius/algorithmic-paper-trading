# Algorithmic Paper Trading System

Python-based paper-trading system for US equities using hourly market data, technical signal generation, rolling parameter optimization, backtesting, and automated execution through the Alpaca paper-trading API.

> **Paper trading only.** This project is for educational and research purposes and does not enable live trading.

## Overview

The system monitors a diversified universe of US equities and, for each trading cycle:

1. downloads recent hourly market data with `yfinance`;
2. optimizes strategy parameters over a rolling 60-day training window;
3. generates signals from horizontal and oblique support/resistance structures;
4. checks market status, positions, open orders, and available buying power;
5. submits paper orders through Alpaca;
6. logs trades, optimized parameters, and portfolio snapshots.

## Main Features

- Hourly market data across a broad US-equity universe
- Horizontal support/resistance signals
- Oblique support/resistance trendline signals
- Grid-search optimization of line and risk tolerances
- Historical strategy backtesting
- Automated Alpaca paper execution
- Maximum allocation per stock
- Market-open and duplicate-order safeguards
- CSV logging for trades, parameters, and account runs

## Strategy

### Horizontal support and resistance

The algorithm detects local extrema and compares current turning points with previously observed support and resistance levels. Signals are generated when the relevant price levels fall within a configurable tolerance.

### Oblique support and resistance

Recent local minima are used to construct projected support trendlines, while recent local maxima define resistance trendlines. The lines are projected to the current confirmed bar and may generate signals when price is sufficiently close to them.

Oblique signals take priority when both signal types are available.

## Parameter Optimization

The strategy performs a grid search over:

- `line_tolerance`
- `risk_tolerance`

Each parameter combination is backtested across the training universe. The combination with the highest aggregate historical training profit is selected for the next paper-trading cycle.

The training window uses the previous 60 days of available data.

## Execution and Controls

Buy signals are submitted as market orders in Alpaca's paper environment, subject to buying power and a configurable maximum allocation per stock.

Sell signals are submitted as limit orders. The current implementation blocks a sell when the estimated current price is at or below the Alpaca average entry price.

Operational safeguards include:

- explicit paper-trading mode;
- no order submission when the market is closed;
- duplicate open-order prevention;
- no additional buy when a position already exists;
- capped allocation per stock;
- API credentials loaded from environment variables.

## Project Structure

```text
algorithmic-paper-trading/
├── paper_trading_system.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your Alpaca paper credentials to `.env`:

```text
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Then run:

```bash
python paper_trading_system.py
```

## Generated Output

The system may generate:

- `alpaca_paper_trades.csv`
- `alpaca_paper_parameters.csv`
- `alpaca_paper_runs.csv`

These runtime files are excluded from version control.

## Technology

Python, NumPy, Pandas, yfinance, Alpaca-py, python-dotenv.

## Limitations

This is an experimental project. Parameter selection may overfit historical data, and the internal backtest does not fully model transaction costs, slippage, liquidity, or market impact. Yahoo Finance market data may also differ from broker execution data.

The rule that avoids selling below the average entry price is an experimental design choice, not a general risk-management principle, and may keep losing positions open for extended periods.

Past backtest or paper-trading performance does not imply future results.

## Disclaimer

For educational and research purposes only. This repository does not constitute investment advice.
