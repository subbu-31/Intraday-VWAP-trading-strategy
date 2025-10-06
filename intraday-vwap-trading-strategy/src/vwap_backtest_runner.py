import yfinance as yf
import pandas as pd
import pytz
from intraday_vwap_strategy import IntradayVWAPStrategy

def run_complete_vwap_backtest():
    """Complete VWAP backtest with user inputs and full results"""
    
    print("=== VWAP Strategy Backtest ===")
    ticker = input("Enter ticker (e.g., SUNPHARMA.NS): ").strip().upper()
    interval = input("Enter interval (15m, 5m, 1h): ").strip()
    period = input("Enter period (1mo, 3mo, 1y): ").strip()
    
    print(f"\nProcessing {ticker}...")
    
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        ist = pytz.timezone('Asia/Kolkata')
        data.index = data.index.tz_convert(ist)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns.values]
        data.columns = data.columns.str.lower()
        
        strategy = IntradayVWAPStrategy(data, initial_capital=100000, max_risk_per_trade=2.0)
        tradebook_df, equity_curve_df = strategy.execute_intraday_backtest(verbose=True)
        metrics = strategy.get_performance_metrics()
        results_df = strategy.get_backtest_results()
        
        print(f"\nFinal Capital: ${results_df['final_capital'].iloc[0]:,.2f}")
        print(f"Total Return: {results_df['total_return_pct'].iloc[0]:.2f}%")
        print(f"Sharpe Ratio: {results_df['sharpe_ratio'].iloc[0]:.2f}")
        
        strategy.plot_equity_curve()
        return strategy, tradebook_df, equity_curve_df, results_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

if __name__ == "__main__":
    run_complete_vwap_backtest()
