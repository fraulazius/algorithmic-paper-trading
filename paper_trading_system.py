import os
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
)


# ============================================================
# ALPACA PAPER TRADER
# ============================================================
#
# PAPER TRADING ONLY
#
# Strategy:
#
#   1. Horizontal support/resistance
#   2. Oblique support/resistance trendlines
#   3. Parameter optimization
#   4. 60-day training window
#
# SELL PROTECTION:
#
#   A position may NEVER be sold below its
#   Alpaca average entry price.
#
# SELL ORDERS ARE LIMIT ORDERS.
#
# MARKET-CLOSED PROTECTION:
#
#   No stock orders are submitted when Alpaca
#   reports that the market is closed.
#
# ============================================================


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY:
    raise RuntimeError(
        "ALPACA_API_KEY not found in .env"
    )

if not ALPACA_SECRET_KEY:
    raise RuntimeError(
        "ALPACA_SECRET_KEY not found in .env"
    )


# ============================================================
# ALPACA CLIENT
# ============================================================

trading_client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=True,
)


# ============================================================
# TICKERS
# ============================================================

TICKERS = [

    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
    "CRM", "AMD", "ADBE", "CSCO", "INTC",
    "QCOM", "MU", "IBM", "NOW", "INTU",
    "PANW", "CRWD", "PLTR", "SNOW", "SHOP",

    # Internet / Consumer Technology
    "AMZN", "META", "GOOGL", "GOOG", "NFLX",
    "TSLA", "UBER", "ABNB", "BKNG", "DASH",

    # Financial
    "JPM", "BAC", "WFC", "C", "GS",
    "MS", "BLK", "SCHW", "AXP", "COF",
    "V", "MA", "PYPL", "ICE", "CME",

    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK",
    "PFE", "TMO", "ABT", "AMGN", "GILD",
    "ISRG", "SYK", "DHR", "BSX", "MDT",

    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT",
    "TJX", "NKE", "LULU", "MCD", "SBUX",
    "KO", "PEP", "PM", "MO", "MDLZ",

    # Industrials / Aerospace
    "GE", "CAT", "HON", "RTX", "BA",
    "LMT", "NOC", "GD", "UPS", "FDX",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",

    # Communication
    "T", "VZ", "TMUS", "CMCSA", "DIS",

    # Utilities / Real Estate
    "NEE", "DUK", "SO", "AEP", "AMT",

    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX",
]


# ============================================================
# SETTINGS
# ============================================================

INTERVAL = "1h"

DOWNLOAD_PERIOD = "60d"

TRAINING_DAYS = 60

LOOP_SECONDS = 300

MAX_ALLOCATION_PER_STOCK = 1000.0


# ============================================================
# STRATEGY SETTINGS
# ============================================================

LINE_TOLERANCES = np.array([
    0.002, 0.004, 0.006, 0.008, 0.010, 0.012,
    0.015, 0.020, 0.025, 0.030, 0.040, 0.050,
])

RISK_TOLERANCES = np.array([
    0.002, 0.005, 0.010, 0.015, 0.020, 0.025,
    0.030, 0.040, 0.050, 0.075, 0.100,
])


# ============================================================
# OBLIQUE LINE SETTINGS
# ============================================================

OBLIQUE_LOOKBACK = 30
OBLIQUE_MIN_POINTS = 2
OBLIQUE_TOLERANCE = 0.015


# ============================================================
# FILES
# ============================================================

TRADES_FILE = "alpaca_paper_trades.csv"
PARAMETERS_FILE = "alpaca_paper_parameters.csv"
RUN_FILE = "alpaca_paper_runs.csv"


# ============================================================
# GET WINDOW
# ============================================================

def get_window(price, start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    normalized = price.index.normalize()
    mask = (
        (normalized >= start_date)
        &
        (normalized <= end_date)
    )
    return price.loc[mask]


# ============================================================
# MARKET CLOCK
# ============================================================

def market_is_open():
    try:
        clock = trading_client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        print(f"MARKET CLOCK ERROR: {e}")
        return False


def print_market_status():
    try:
        clock = trading_client.get_clock()
        print()
        print(f"Alpaca market open: {clock.is_open}")
        print(f"Market timestamp: {clock.timestamp}")
        print(f"Next open: {clock.next_open}")
        print(f"Next close: {clock.next_close}")
        print()
    except Exception as e:
        print(f"Could not read market clock: {e}")


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():
    print()
    print("=" * 90)
    print("DOWNLOADING MARKET DATA")
    print("=" * 90)
    print(f"Tickers : {len(TICKERS)}")
    print(f"Period  : {DOWNLOAD_PERIOD}")
    print(f"Interval: {INTERVAL}")

    data = {}

    for number, ticker in enumerate(TICKERS, start=1):
        print(f"[{number:3d}/{len(TICKERS)}] {ticker:<6}", end=" ")

        try:
            df = yf.download(
                ticker,
                period=DOWNLOAD_PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )

            if df.empty:
                print("NO DATA")
                continue

            close = df["Close"]

            if isinstance(close, pd.DataFrame):
                if ticker in close.columns:
                    close = close[ticker]
                else:
                    close = close.iloc[:, 0]

            close = close.dropna().astype(float)

            if len(close) < 100:
                print(f"NOT ENOUGH DATA ({len(close)})")
                continue

            if hasattr(close.index, "tz") and close.index.tz is not None:
                close.index = close.index.tz_localize(None)

            data[ticker] = close
            print(f"OK - {len(close)} bars")

        except Exception as e:
            print(f"ERROR - {e}")

    print()
    print(f"Downloaded {len(data)} / {len(TICKERS)}")

    if not data:
        raise RuntimeError("No market data downloaded.")

    return data


# ============================================================
# LOCAL EXTREMA
# ============================================================

def find_extrema(price):
    maxima = []
    minima = []

    for i in range(1, len(price) - 1):
        p0 = float(price.iloc[i - 1])
        p1 = float(price.iloc[i])
        p2 = float(price.iloc[i + 1])

        if p1 > p0 and p1 > p2:
            maxima.append(i)

        if p1 < p0 and p1 < p2:
            minima.append(i)

    return maxima, minima


# ============================================================
# HORIZONTAL STRATEGY
# ============================================================

def horizontal_signal(price, line_tolerance, risk_tolerance):
    if len(price) < 5:
        return None

    previous_price = float(price.iloc[-3])
    current_price = float(price.iloc[-2])
    next_price = float(price.iloc[-1])

    is_max = current_price > previous_price and current_price > next_price
    is_min = current_price < previous_price and current_price < next_price

    maxima, minima = find_extrema(price)

    if is_max and minima:
        previous_min_index = minima[-1]
        previous_min_price = float(price.iloc[previous_min_index])

        distance = abs(current_price - previous_min_price) / previous_min_price

        if distance <= line_tolerance:
            return {
                "signal": "SELL",
                "strategy": "HORIZONTAL",
                "signal_price": current_price,
                "execution_price": next_price,
            }

    if is_min and maxima:
        previous_max_index = maxima[-1]
        previous_max_price = float(price.iloc[previous_max_index])

        distance = abs(current_price - previous_max_price) / previous_max_price

        if distance > line_tolerance:
            return None

        entry_price = next_price

        if entry_price <= 0:
            return None

        risk = (entry_price - previous_max_price) / entry_price
        risk = max(0.0, risk)

        if risk > risk_tolerance:
            return None

        return {
            "signal": "BUY",
            "strategy": "HORIZONTAL",
            "signal_price": current_price,
            "execution_price": entry_price,
            "risk": risk,
        }

    return None


# ============================================================
# LINE PROJECTION
# ============================================================

def project_line(
    point1_index,
    point1_price,
    point2_index,
    point2_price,
    target_index,
):
    if point2_index == point1_index:
        return None

    slope = (
        point2_price - point1_price
    ) / (
        point2_index - point1_index
    )

    projected = (
        point2_price
        +
        slope
        *
        (
            target_index
            -
            point2_index
        )
    )

    return projected


# ============================================================
# OBLIQUE STRATEGY
# ============================================================

def oblique_signal(price):
    if len(price) < OBLIQUE_LOOKBACK:
        return None

    recent = price.iloc[-OBLIQUE_LOOKBACK:].reset_index(drop=True)
    maxima, minima = find_extrema(recent)

    if (
        len(maxima) < OBLIQUE_MIN_POINTS
        and
        len(minima) < OBLIQUE_MIN_POINTS
    ):
        return None

    current_index = len(recent) - 2
    current_price = float(recent.iloc[current_index])
    next_price = float(recent.iloc[-1])

    if len(minima) >= 2:
        i1 = minima[-2]
        i2 = minima[-1]
        p1 = float(recent.iloc[i1])
        p2 = float(recent.iloc[i2])

        support = project_line(
            i1,
            p1,
            i2,
            p2,
            current_index,
        )

        if support is not None and support > 0:
            distance_to_support = abs(current_price - support) / support

            if (
                distance_to_support <= OBLIQUE_TOLERANCE
                and
                p2 >= p1
            ):
                return {
                    "signal": "BUY",
                    "strategy": "OBLIQUE_SUPPORT",
                    "signal_price": current_price,
                    "execution_price": next_price,
                    "line_price": support,
                }

    if len(maxima) >= 2:
        i1 = maxima[-2]
        i2 = maxima[-1]
        p1 = float(recent.iloc[i1])
        p2 = float(recent.iloc[i2])

        resistance = project_line(
            i1,
            p1,
            i2,
            p2,
            current_index,
        )

        if resistance is not None and resistance > 0:
            distance_to_resistance = abs(current_price - resistance) / resistance

            if (
                distance_to_resistance <= OBLIQUE_TOLERANCE
                and
                p2 <= p1
            ):
                return {
                    "signal": "SELL",
                    "strategy": "OBLIQUE_RESISTANCE",
                    "signal_price": current_price,
                    "execution_price": next_price,
                    "line_price": resistance,
                }

    return None


# ============================================================
# COMBINED SIGNAL
# ============================================================

def generate_signal(price, line_tolerance, risk_tolerance):
    horizontal = horizontal_signal(
        price,
        line_tolerance,
        risk_tolerance,
    )

    oblique = oblique_signal(price)

    if oblique is not None:
        return oblique

    if horizontal is not None:
        return horizontal

    return None


# ============================================================
# STRATEGY BACKTEST
# ============================================================

def run_strategy(
    price,
    line_tolerance,
    risk_tolerance,
    initial_capital=1000.0,
):
    capital = float(initial_capital)
    shares = 0.0
    buy_price = 0.0
    state = "FLAT"
    trades = 0

    for t in range(2, len(price)):
        historical = price.iloc[:t + 1]

        signal = generate_signal(
            historical,
            line_tolerance,
            risk_tolerance,
        )

        if signal is None:
            continue

        execution_price = float(signal["execution_price"])

        if execution_price <= 0:
            continue

        if (
            signal["signal"] == "BUY"
            and
            state == "FLAT"
        ):
            if capital <= 0:
                continue

            buy_price = execution_price
            shares = capital / buy_price
            capital = 0.0
            state = "LONG"
            trades += 1

        elif (
            signal["signal"] == "SELL"
            and
            state == "LONG"
        ):
            if execution_price <= buy_price:
                continue

            capital = shares * execution_price
            shares = 0.0
            state = "FLAT"
            trades += 1

    if state == "LONG":
        final_price = float(price.iloc[-1])
        capital = shares * final_price

    profit = capital - initial_capital

    return {
        "final": capital,
        "profit": profit,
        "trades": trades,
    }


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_parameters(training_data):
    results = []

    for line_tol in LINE_TOLERANCES:
        for risk_tol in RISK_TOLERANCES:
            total_profit = 0.0
            total_final = 0.0
            total_trades = 0

            for ticker, price in training_data.items():
                if len(price) < 20:
                    continue

                result = run_strategy(
                    price,
                    line_tol,
                    risk_tol,
                    MAX_ALLOCATION_PER_STOCK,
                )

                total_profit += result["profit"]
                total_final += result["final"]
                total_trades += result["trades"]

            results.append({
                "line_tolerance": line_tol,
                "risk_tolerance": risk_tol,
                "total_profit": total_profit,
                "final_capital": total_final,
                "total_trades": total_trades,
            })

    results_df = pd.DataFrame(results)
    best_index = results_df["total_profit"].idxmax()
    best = results_df.loc[best_index]

    return best, results_df


# ============================================================
# ACCOUNT / POSITIONS / ORDERS
# ============================================================

def get_account():
    return trading_client.get_account()


def get_positions():
    positions = trading_client.get_all_positions()
    result = {}

    for position in positions:
        result[position.symbol] = position

    return result


def get_open_orders():
    try:
        return trading_client.get_orders(filter="open")
    except Exception:
        return []


def has_open_order(symbol):
    orders = get_open_orders()

    for order in orders:
        if str(order.symbol).upper() == symbol.upper():
            return True

    return False


def get_buying_power():
    account = get_account()
    return float(account.buying_power)


# ============================================================
# LOG ORDER
# ============================================================

def log_order(
    date,
    symbol,
    side,
    qty,
    order_id,
    line_tolerance,
    risk_tolerance,
    strategy,
    limit_price=None,
):
    row = pd.DataFrame([{
        "date": date,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "order_id": str(order_id),
        "line_tolerance": line_tolerance,
        "risk_tolerance": risk_tolerance,
        "strategy": strategy,
        "limit_price": limit_price,
    }])

    exists = os.path.exists(TRADES_FILE)

    row.to_csv(
        TRADES_FILE,
        mode="a",
        header=not exists,
        index=False,
    )


# ============================================================
# SUBMIT BUY
# ============================================================

def submit_buy(
    symbol,
    estimated_price,
    line_tolerance,
    risk_tolerance,
    strategy,
):
    if not market_is_open():
        print(f"  SKIP BUY {symbol}: market is closed.")
        return None

    if has_open_order(symbol):
        print(f"  SKIP BUY {symbol}: existing open order.")
        return None

    positions = get_positions()

    if symbol in positions:
        current_qty = float(positions[symbol].qty)

        if current_qty > 0:
            print(f"  SKIP BUY {symbol}: position already exists.")
            return None

    buying_power = get_buying_power()

    allocation = min(
        MAX_ALLOCATION_PER_STOCK,
        buying_power,
    )

    if allocation <= 1.0:
        print(f"  SKIP BUY {symbol}: insufficient buying power.")
        return None

    if estimated_price <= 0:
        return None

    qty = allocation / estimated_price
    qty = round(qty, 6)

    if qty <= 0:
        return None

    client_order_id = (
        "strategy-"
        + symbol.lower()
        + "-"
        + uuid.uuid4().hex[:12]
    )

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )

    print()
    print(
        f"  >>> PAPER BUY "
        f"{symbol} "
        f"qty={qty} "
        f"strategy={strategy}"
    )

    try:
        order = trading_client.submit_order(
            order_data=order_request
        )

        print(
            f"  ORDER SUBMITTED "
            f"{symbol} "
            f"{order.id}"
        )

        log_order(
            datetime.now().isoformat(),
            symbol,
            "BUY",
            qty,
            order.id,
            line_tolerance,
            risk_tolerance,
            strategy,
        )

        return order

    except Exception as e:
        print(f"  BUY ERROR {symbol}: {e}")
        return None


# ============================================================
# SUBMIT PROFIT-PROTECTED SELL
# ============================================================

def submit_sell(
    symbol,
    current_price,
    line_tolerance,
    risk_tolerance,
    strategy,
):
    if not market_is_open():
        print(f"  SKIP SELL {symbol}: market is closed.")
        return None

    if has_open_order(symbol):
        print(f"  SKIP SELL {symbol}: existing open order.")
        return None

    positions = get_positions()

    if symbol not in positions:
        print(f"  SKIP SELL {symbol}: no position.")
        return None

    position = positions[symbol]
    qty = float(position.qty)
    avg_entry = float(position.avg_entry_price)

    if qty <= 0:
        return None

    if current_price <= avg_entry:
        print()
        print(
            f"  BLOCK SELL {symbol}: "
            f"current=${current_price:.2f} "
            f"<="
            f"entry=${avg_entry:.2f}"
        )
        return None

    limit_price = max(
        current_price,
        avg_entry,
    )

    if limit_price >= 1.0:
        limit_price = round(limit_price, 2)
    else:
        limit_price = round(limit_price, 4)

    qty = round(qty, 6)

    client_order_id = (
        "strategy-"
        + symbol.lower()
        + "-"
        + uuid.uuid4().hex[:12]
    )

    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        client_order_id=client_order_id,
    )

    print()
    print(f"  >>> PAPER SELL {symbol}")
    print(f"      strategy     = {strategy}")
    print(f"      quantity     = {qty}")
    print(f"      avg entry    = ${avg_entry:.2f}")
    print(f"      current      = ${current_price:.2f}")
    print(f"      sell limit   = ${limit_price:.2f}")

    try:
        order = trading_client.submit_order(
            order_data=order_request
        )

        print(
            f"  SELL LIMIT SUBMITTED "
            f"{symbol} "
            f"{order.id}"
        )

        log_order(
            datetime.now().isoformat(),
            symbol,
            "SELL",
            qty,
            order.id,
            line_tolerance,
            risk_tolerance,
            strategy,
            limit_price,
        )

        return order

    except Exception as e:
        print(f"  SELL ERROR {symbol}: {e}")
        return None


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    ticker,
    price,
    test_date,
    line_tolerance,
    risk_tolerance,
):
    history_start = (
        pd.Timestamp(test_date)
        -
        pd.Timedelta(days=TRAINING_DAYS)
    )

    history = get_window(
        price,
        history_start,
        test_date,
    )

    if len(history) < 20:
        return

    signal = generate_signal(
        history,
        line_tolerance,
        risk_tolerance,
    )

    if signal is None:
        return

    execution_price = float(signal["execution_price"])

    print()
    print(
        f"{ticker}: "
        f"{signal['signal']} "
        f"signal "
        f"strategy={signal['strategy']} "
        f"@ ${execution_price:.2f}"
    )

    if signal["signal"] == "BUY":
        submit_buy(
            ticker,
            execution_price,
            line_tolerance,
            risk_tolerance,
            signal["strategy"],
        )

    elif signal["signal"] == "SELL":
        submit_sell(
            ticker,
            execution_price,
            line_tolerance,
            risk_tolerance,
            signal["strategy"],
        )


# ============================================================
# OPTIMIZE FOR DAY
# ============================================================

def optimize_for_day(data, test_date):
    training_start = (
        pd.Timestamp(test_date)
        -
        pd.Timedelta(days=TRAINING_DAYS)
    )

    training_end = (
        pd.Timestamp(test_date)
        -
        pd.Timedelta(days=1)
    )

    training_data = {}

    for ticker, price in data.items():
        window = get_window(
            price,
            training_start,
            training_end,
        )

        if len(window) >= 20:
            training_data[ticker] = window

    if not training_data:
        return None

    best, optimization_df = optimize_parameters(
        training_data
    )

    return best


# ============================================================
# SAVE PARAMETERS
# ============================================================

def save_parameters(date, best):
    row = pd.DataFrame([{
        "date": date,
        "line_tolerance": float(best["line_tolerance"]),
        "risk_tolerance": float(best["risk_tolerance"]),
        "training_profit": float(best["total_profit"]),
        "training_final": float(best["final_capital"]),
        "training_trades": int(best["total_trades"]),
    }])

    exists = os.path.exists(PARAMETERS_FILE)

    row.to_csv(
        PARAMETERS_FILE,
        mode="a",
        header=not exists,
        index=False,
    )


# ============================================================
# SAVE RUN
# ============================================================

def save_run(
    date,
    account,
    line_tolerance,
    risk_tolerance,
):
    row = pd.DataFrame([{
        "date": date,
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
        "line_tolerance": line_tolerance,
        "risk_tolerance": risk_tolerance,
    }])

    exists = os.path.exists(RUN_FILE)

    row.to_csv(
        RUN_FILE,
        mode="a",
        header=not exists,
        index=False,
    )


# ============================================================
# GET LATEST DATE
# ============================================================

def get_latest_date(data):
    dates = set()

    for price in data.values():
        d = (
            pd.DatetimeIndex(price.index)
            .normalize()
            .unique()
        )

        dates.update(d)

    if not dates:
        return None

    return pd.Timestamp(max(dates))


# ============================================================
# PROCESS DATE
# ============================================================

def process_date(data, test_date):
    print()
    print("=" * 90)
    print(f"PROCESSING {test_date.date()}")
    print("=" * 90)

    if not market_is_open():
        print("MARKET CLOSED.")
        print("NO ORDERS WILL BE SUBMITTED.")
        return

    account = get_account()

    print(
        f"Portfolio before: "
        f"${float(account.portfolio_value):,.2f}"
    )

    print(
        f"Buying power: "
        f"${float(account.buying_power):,.2f}"
    )

    best = optimize_for_day(
        data,
        test_date,
    )

    if best is None:
        print("No training data.")
        return

    best_line = float(best["line_tolerance"])
    best_risk = float(best["risk_tolerance"])

    print()
    print(
        f"OPTIMIZED: "
        f"LINE={best_line * 100:.2f}% "
        f"RISK={best_risk * 100:.2f}%"
    )

    save_parameters(
        test_date,
        best,
    )

    for ticker in TICKERS:
        if ticker not in data:
            continue

        try:
            process_symbol(
                ticker,
                data[ticker],
                test_date,
                best_line,
                best_risk,
            )

        except Exception as e:
            print(f"{ticker}: ERROR {e}")

    time.sleep(2)

    account_after = get_account()

    print()
    print("=" * 90)

    print(
        f"Portfolio after: "
        f"${float(account_after.portfolio_value):,.2f}"
    )

    print(
        f"Cash: "
        f"${float(account_after.cash):,.2f}"
    )

    print(
        f"Buying power: "
        f"${float(account_after.buying_power):,.2f}"
    )

    print("=" * 90)

    save_run(
        test_date,
        account_after,
        best_line,
        best_risk,
    )


# ============================================================
# RUN ONCE
# ============================================================

def run_once():
    print()
    print("=" * 90)
    print("ALPACA PAPER STRATEGY")
    print("=" * 90)

    print()
    print("MODE: PAPER TRADING ONLY")
    print("LIVE TRADING: DISABLED")
    print()

    print_market_status()

    account = get_account()

    print(f"Account status: {account.status}")
    print(f"Paper cash: ${float(account.cash):,.2f}")
    print(
        f"Portfolio value: "
        f"${float(account.portfolio_value):,.2f}"
    )
    print(
        f"Buying power: "
        f"${float(account.buying_power):,.2f}"
    )

    data = download_data()
    latest_date = get_latest_date(data)

    if latest_date is None:
        print("No market date.")
        return

    print()
    print(
        f"Latest Yahoo date: "
        f"{latest_date.date()}"
    )

    process_date(
        data,
        latest_date,
    )

    positions = get_positions()

    print()
    print("=" * 90)
    print("CURRENT PAPER POSITIONS")
    print("=" * 90)

    if not positions:
        print("No open positions.")

    else:
        for symbol, position in positions.items():
            print(
                f"{symbol:<6} "
                f"qty={float(position.qty):.6f} "
                f"avg=${float(position.avg_entry_price):.2f} "
                f"market=${float(position.market_value):,.2f}"
            )


# ============================================================
# CONTINUOUS LOOP
# ============================================================

def main():
    processed_date = None

    while True:
        try:
            print()
            print("=" * 90)
            print(
                f"CHECKING BOT "
                f"{datetime.now().isoformat()}"
            )
            print("=" * 90)

            print_market_status()

            data = download_data()
            latest_date = get_latest_date(data)

            if latest_date is None:
                print("No latest market date.")

            else:
                if (
                    processed_date is None
                    or
                    latest_date > processed_date
                ):
                    process_date(
                        data,
                        latest_date,
                    )

                    processed_date = latest_date

                else:
                    print(
                        f"No new market date. "
                        f"Latest: "
                        f"{latest_date}"
                    )

        except KeyboardInterrupt:
            print()
            print("Bot stopped by user.")
            break

        except Exception as e:
            print()
            print(f"MAIN LOOP ERROR: {e}")

        print()
        print(
            f"Next check in "
            f"{LOOP_SECONDS} seconds..."
        )

        time.sleep(LOOP_SECONDS)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
