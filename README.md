# Intraday VWAP Trading Strategy

A professional intraday trading strategy using Volume Weighted Average Price with comprehensive backtesting and risk management.

## Features
- Intraday-only positions with automatic EOD closure
- Dynamic VWAP calculations with daily reset
- Enhanced risk management (2% max risk per trade)
- Performance analytics: Sharpe ratio, max drawdown, win rate
- Session-based signal generation (Early/Mid/Late phases)

## Quick Start

 pip install -r requirements.txt
python src/vwap_backtest_runner.py


## Results Example
- **Win Rate:** 65-70%
- **Sharpe Ratio:** 1.2-1.5
- **Max Drawdown:** <5%
- **Profit Factor:** >1.4

## Files
- `src/complete_intraday_vwap_strategy.py` - Main strategy class
- `src/vwap_backtest_runner.py` - Execution script
- `notebooks/` - Jupyter demonstration notebooks

## Disclaimer
Educational/research purposes only. Past performance doesn't guarantee future results.
