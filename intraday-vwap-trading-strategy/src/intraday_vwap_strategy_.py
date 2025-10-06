# ===================================================================
# COMPLETE INTRADAY-ONLY VWAP AND ANCHORED VWAP TRADING STRATEGY
# Same-day entry and exit only, with automatic position management
# Clean console output, human-written style, structured results
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import warnings

warnings.filterwarnings('ignore')


# 1. INTRADAY SESSION MANAGEMENT CLASS

class IntradaySession:
    def __init__(self):
        # NSE trading hours (adjusted for common intraday datasets)
        self.market_open = time(9, 15)
        self.market_close = time(15, 25)  # aligns with common yfinance bar availability

        # Intraday windows
        self.early_session = (time(9, 15), time(11, 0))
        self.mid_session = (time(11, 0), time(14, 0))
        self.late_session = (time(14, 0), time(15, 25))

    def is_market_open(self, timestamp):
        current_time = timestamp.time()
        return self.market_open <= current_time <= self.market_close

    def get_session_phase(self, timestamp):
        current_time = timestamp.time()
        if self.early_session[0] <= current_time < self.early_session[1]:
            return 'EARLY'
        elif self.mid_session[0] <= current_time < self.mid_session[1]:
            return 'MID'
        elif self.late_session[0] <= current_time <= self.late_session[1]:
            return 'LATE'
        else:
            return 'CLOSED'

    def minutes_to_close(self, timestamp):
        current_time = timestamp.time()
        close_datetime = datetime.combine(timestamp.date(), self.market_close)
        current_datetime = datetime.combine(timestamp.date(), current_time)
        if current_datetime <= close_datetime:
            delta = close_datetime - current_datetime
            return int(delta.total_seconds() / 60)
        return 0


# 2. ENHANCED VWAP CALCULATIONS FOR INTRADAY

def calculate_intraday_vwap(df):
    """Calculate intraday VWAP that resets each trading day."""
    df_copy = df.copy()

    # Typical price and price*volume
    df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
    df_copy['pv'] = df_copy['typical_price'] * df_copy['volume']
    df_copy['trading_day'] = df_copy.index.date

    # Daily cumulative values (reset each day)
    df_copy['daily_cum_pv'] = df_copy.groupby('trading_day')['pv'].cumsum()
    df_copy['daily_cum_volume'] = df_copy.groupby('trading_day')['volume'].cumsum()
    df_copy['intraday_vwap'] = df_copy['daily_cum_pv'] / df_copy['daily_cum_volume']

    return df_copy


def calculate_session_anchored_vwap(df, session_start_time='09:15'):
    """Calculate VWAP anchored to a specific intraday time (e.g., market open)."""
    df_copy = df.copy()
    df_copy['time_str'] = df_copy.index.strftime('%H:%M')

    # Find anchor points (e.g., market open each day)
    anchor_mask = df_copy['time_str'] == session_start_time

    # Reset cumulative values at each anchor point
    df_copy['anchor_group'] = anchor_mask.cumsum()

    # Calculate VWAP from each anchor point
    df_copy['anchored_cum_pv'] = df_copy.groupby('anchor_group')['pv'].cumsum()
    df_copy['anchored_cum_volume'] = df_copy.groupby('anchor_group')['volume'].cumsum()
    df_copy['session_anchored_vwap'] = df_copy['anchored_cum_pv'] / df_copy['anchored_cum_volume']

    return df_copy


# 3. MAIN INTRADAY VWAP STRATEGY CLASS

class IntradayVWAPStrategy:
    def __init__(self, df, initial_capital=100000, max_risk_per_trade=2.0):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade  # percent
        self.session = IntradaySession()

        # State and results containers
        self.reset_trading_state()

        # Outputs
        self.tradebook_df = pd.DataFrame()      # filled after backtest
        self.equity_curve_df = pd.DataFrame()   # filled during backtest
        self.backtest_results_df = pd.DataFrame()  # filled after metrics

    def reset_trading_state(self):
        """Reset all trading variables."""
        self.capital = self.initial_capital
        self.daily_capital_start = self.initial_capital

        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.target_price = 0.0

        self.trades = []
        self.equity_curve = []
        self.current_trading_day = None

    def prepare_intraday_data(self):
        """Prepare data with intraday indicators."""
        # Calculate intraday VWAP (resets daily)
        self.df = calculate_intraday_vwap(self.df)

        # Calculate session anchored VWAP
        temp_df = calculate_session_anchored_vwap(self.df, '09:15')
        self.df['session_anchored_vwap'] = temp_df['session_anchored_vwap']

        # VWAP deviation bands
        self.df['vwap_upper_band'] = self.df['intraday_vwap'] * 1.002  # +0.2%
        self.df['vwap_lower_band'] = self.df['intraday_vwap'] * 0.998  # -0.2%

        # Price relative to VWAP
        self.df['price_vs_vwap_pct'] = (
            (self.df['close'] - self.df['intraday_vwap']) / self.df['intraday_vwap'] * 100
        )

        # Volume analysis
        self.df['volume_ma_20'] = self.df['volume'].rolling(20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_ma_20']

        # Session phase and time to close
        self.df['session_phase'] = self.df.index.map(self.session.get_session_phase)
        self.df['minutes_to_close'] = self.df.index.map(self.session.minutes_to_close)

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Position size based on risk management."""
        if stop_loss_price == 0:
            risk_per_share = entry_price * 0.01  # fallback to 1% price risk
        else:
            risk_per_share = abs(entry_price - stop_loss_price)

        max_risk_amount = self.capital * (self.max_risk_per_trade / 100.0)
        position_size = int(max_risk_amount / risk_per_share)

        # Capital constraint (90%) and a demo cap for bar-based data
        max_position_by_capital = int(self.capital * 0.9 / entry_price)
        return max(0, min(position_size, max_position_by_capital, 10))

    def generate_intraday_signals(self):
        """Generate intraday trading signals with enhanced logic."""
        self.df['signal'] = 0
        self.df['signal_strength'] = 0
        self.df['signal_reason'] = ''

        # Start after 20 bars for moving averages
        for i in range(20, len(self.df)):
            current_row = self.df.iloc[i]
            prev_row = self.df.iloc[i - 1]

            # Skip if market is closed or less than 30 minutes to close
            if current_row['session_phase'] == 'CLOSED' or current_row['minutes_to_close'] < 30:
                continue

            # Buy signal
            buy_conditions = [
                current_row['close'] > current_row['intraday_vwap'],
                prev_row['close'] <= prev_row['intraday_vwap'],
                current_row['close'] > current_row['session_anchored_vwap'],
                current_row['volume_ratio'] > 1.2,
                current_row['session_phase'] in ['EARLY', 'MID']
            ]

            # Sell signal
            sell_conditions = [
                current_row['close'] < current_row['intraday_vwap'],
                prev_row['close'] >= prev_row['intraday_vwap'],
                current_row['close'] < current_row['session_anchored_vwap'],
                current_row['volume_ratio'] > 1.1,
                current_row['session_phase'] in ['EARLY', 'MID']
            ]

            signal = 0
            signal_strength = 0
            signal_reason = ""

            if all(buy_conditions):
                signal = 1
                signal_strength = sum([
                    2 if current_row['volume_ratio'] > 1.5 else 1,
                    2 if current_row['session_phase'] == 'EARLY' else 1,
                    1 if current_row['price_vs_vwap_pct'] > 0.1 else 0
                ])
                signal_reason = (
                    f"BUY: VWAP cross, Volume {current_row['volume_ratio']:.1f}x, "
                    f"Session {current_row['session_phase']}"
                )

            elif all(sell_conditions):
                signal = -1
                signal_strength = sum([
                    2 if current_row['volume_ratio'] > 1.5 else 1,
                    2 if current_row['session_phase'] == 'EARLY' else 1,
                    1 if current_row['price_vs_vwap_pct'] < -0.1 else 0
                ])
                signal_reason = (
                    f"SELL: VWAP cross, Volume {current_row['volume_ratio']:.1f}x, "
                    f"Session {current_row['session_phase']}"
                )

            self.df.iloc[i, self.df.columns.get_loc('signal')] = signal
            self.df.iloc[i, self.df.columns.get_loc('signal_strength')] = signal_strength
            self.df.iloc[i, self.df.columns.get_loc('signal_reason')] = signal_reason

    def execute_intraday_backtest(self, verbose=True):
        """Execute intraday backtest with automatic EOD closure."""
        if verbose:
            print("Preparing indicators and signals...")

        self.prepare_intraday_data()
        self.generate_intraday_signals()

        for i in range(len(self.df)):
            self._process_intraday_bar(i)

        # Close any remaining position at end
        if self.position != 0:
            self._close_position(len(self.df) - 1, "End of backtest")

        # Build outputs
        self.tradebook_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame(
            columns=[
                'entry_time', 'exit_time', 'trading_day', 'type',
                'entry_price', 'exit_price', 'quantity', 'pnl', 'return_pct',
                'duration_minutes', 'entry_session', 'exit_session', 'exit_reason'
            ]
        )
        self.equity_curve_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame(
            columns=['timestamp', 'price', 'position', 'capital', 'unrealized_pnl', 'equity', 'trading_day', 'session_phase']
        )

        if verbose:
            print("Backtest complete.")
            print(f"Total trades: {len(self.tradebook_df)}")
            print(f"Final capital: {self.capital:,.2f}")
            if len(self.tradebook_df) > 0:
                total_return_pct = (self.tradebook_df['pnl'].sum() / self.initial_capital) * 100.0
                print(f"Total return: {total_return_pct:.2f}%")

        return self.tradebook_df, self.equity_curve_df

    def _process_intraday_bar(self, i):
        """Process each bar with intraday-specific logic."""
        current_row = self.df.iloc[i]
        timestamp = self.df.index[i]
        signal = current_row['signal']

        # New trading day handling
        trading_day = timestamp.date()
        if self.current_trading_day != trading_day:
            if self.position != 0:
                self._close_position(i, "New trading day - auto close")
            self.current_trading_day = trading_day
            self.daily_capital_start = self.capital

        # Force close positions 15 minutes before market close
        if current_row['minutes_to_close'] <= 15 and self.position != 0:
            self._close_position(i, "Approaching market close")
            self._update_equity_curve(i)
            return

        # Process signals
        if signal == 1 and self.position <= 0:
            if self.position < 0:
                self._close_position(i, "Buy signal - closing short")
            if self.position == 0:
                self._open_intraday_position(i, 'LONG')

        elif signal == -1 and self.position >= 0:
            if self.position > 0:
                self._close_position(i, "Sell signal - closing long")
            if self.position == 0:
                self._open_intraday_position(i, 'SHORT')

        # Check stops/targets
        if self.position != 0:
            self._check_stop_loss_target(i)

        # Update equity curve every bar
        self._update_equity_curve(i)

    def _open_intraday_position(self, i, position_type):
        """Open new intraday position with risk management."""
        current_row = self.df.iloc[i]
        timestamp = self.df.index[i]
        entry_price = current_row['close']

        # Stop loss based on VWAP distance
        if position_type == 'LONG':
            self.stop_loss = max(
                entry_price * 0.998,           # 0.2% stop
                current_row['vwap_lower_band']  # VWAP lower band
            )
            self.target_price = entry_price * 1.006  # 0.6% target (3:1)
        else:
            self.stop_loss = min(
                entry_price * 1.002,            # 0.2% stop
                current_row['vwap_upper_band']  # VWAP upper band
            )
            self.target_price = entry_price * 0.994  # 0.6% target

        position_size = self.calculate_position_size(entry_price, self.stop_loss)
        if position_size > 0:
            self.position = position_size if position_type == 'LONG' else -position_size
            self.entry_price = entry_price
            self.entry_time = timestamp

    def _check_stop_loss_target(self, i):
        """Check if stop loss or target is hit."""
        current_row = self.df.iloc[i]
        current_price = current_row['close']

        if self.position > 0:
            if current_price <= self.stop_loss:
                self._close_position(i, f"Stop loss hit @ {current_price:.2f}")
            elif current_price >= self.target_price:
                self._close_position(i, f"Target hit @ {current_price:.2f}")

        elif self.position < 0:
            if current_price >= self.stop_loss:
                self._close_position(i, f"Stop loss hit @ {current_price:.2f}")
            elif current_price <= self.target_price:
                self._close_position(i, f"Target hit @ {current_price:.2f}")

    def _close_position(self, i, reason):
        """Close current position and record trade."""
        if self.position == 0:
            return

        current_row = self.df.iloc[i]
        timestamp = self.df.index[i]
        exit_price = current_row['close']

        # P&L
        if self.position > 0:
            pnl = (exit_price - self.entry_price) * self.position
            trade_type = 'LONG'
        else:
            pnl = (self.entry_price - exit_price) * abs(self.position)
            trade_type = 'SHORT'

        self.capital += pnl
        trade_duration = timestamp - self.entry_time
        return_pct = (pnl / (self.entry_price * abs(self.position))) * 100.0

        entry_session = (
            self.df.loc[self.entry_time, 'session_phase']
            if self.entry_time in self.df.index else 'UNKNOWN'
        )

        trade_record = {
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'trading_day': timestamp.date(),
            'type': trade_type,
            'entry_price': float(self.entry_price),
            'exit_price': float(exit_price),
            'quantity': int(abs(self.position)),
            'pnl': float(pnl),
            'return_pct': float(return_pct),
            'duration_minutes': int(trade_duration.total_seconds() / 60),
            'entry_session': entry_session,
            'exit_session': current_row['session_phase'],
            'exit_reason': reason
        }
        self.trades.append(trade_record)

        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.target_price = 0.0

    def _update_equity_curve(self, i):
        """Update equity curve with current mark-to-market equity."""
        current_row = self.df.iloc[i]
        timestamp = self.df.index[i]
        current_price = current_row['close']

        # Unrealized P&L
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
        elif self.position < 0:
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
        else:
            unrealized_pnl = 0.0

        current_equity = float(self.capital + unrealized_pnl)

        self.equity_curve.append({
            'timestamp': timestamp,
            'price': float(current_price),
            'position': int(self.position),
            'capital': float(self.capital),
            'unrealized_pnl': float(unrealized_pnl),
            'equity': current_equity,
            'trading_day': timestamp.date(),
            'session_phase': current_row['session_phase']
        })

    # ---------- PERFORMANCE AND REPORTING ----------

    def get_performance_metrics(self, trading_minutes_per_year=252*390):
        """Calculate performance metrics including Sharpe and Max Drawdown."""
        # Build tradebook DF if needed
        tradebook = pd.DataFrame(self.trades) if self.trades else pd.DataFrame(
            columns=['entry_time','exit_time','trading_day','type','entry_price',
                     'exit_price','quantity','pnl','return_pct','duration_minutes',
                     'entry_session','exit_session','exit_reason']
        )

        # Equity curve DF
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame(
            columns=['timestamp','price','position','capital','unrealized_pnl','equity',
                     'trading_day','session_phase']
        )

        # If no data, return minimal
        if equity_df.empty:
            return {'error': 'No equity data. Run backtest first.'}

        # Compute returns from equity curve
        equity_series = equity_df.set_index('timestamp')['equity'].astype(float)
        equity_returns = equity_series.pct_change().dropna()

        # Intraday bar period estimation for annualization
        # Use index frequency if available else infer by median timediff
        if len(equity_df) > 1:
            times = equity_df['timestamp'].sort_values().to_list()
            diffs = np.diff(pd.to_datetime(times).astype('int64')) / 1e9  # seconds
            median_sec = np.median(diffs) if len(diffs) > 0 else 900.0  # default 15m
            bars_per_day = int((6.5*3600) / median_sec) if median_sec > 0 else 26
        else:
            bars_per_day = 26  # default for 15m bars in a ~6.5h session

        bars_per_year = bars_per_day * 252

        # Sharpe ratio (risk-free ~0)
        mean_ret = equity_returns.mean()
        std_ret = equity_returns.std(ddof=1)
        sharpe = (mean_ret / std_ret) * np.sqrt(bars_per_year) if std_ret > 0 else np.nan

        # Max Drawdown
        roll_max = equity_series.cummax()
        drawdown = (equity_series / roll_max) - 1.0
        max_dd = drawdown.min()

        # Basic metrics
        total_pnl = float(tradebook['pnl'].sum()) if not tradebook.empty else 0.0
        final_capital = float(self.initial_capital + total_pnl)
        total_return_pct = (total_pnl / self.initial_capital) * 100.0 if self.initial_capital > 0 else np.nan

        total_trades = int(len(tradebook)) if not tradebook.empty else 0
        wins = int((tradebook['pnl'] > 0).sum()) if not tradebook.empty else 0
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (
            abs(tradebook.loc[tradebook['pnl'] > 0, 'pnl'].sum()) /
            abs(tradebook.loc[tradebook['pnl'] < 0, 'pnl'].sum())
            if (not tradebook.empty) and (tradebook['pnl'] < 0).any() else np.inf
        )

        metrics = {
            'initial_capital': float(self.initial_capital),
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'total_return_pct': float(total_return_pct),
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate_pct': float(win_rate),
            'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else np.inf,
            'sharpe_ratio': float(sharpe) if pd.notnull(sharpe) else np.nan,
            'max_drawdown_pct': float(max_dd * 100.0) if pd.notnull(max_dd) else np.nan
        }

        # Store backtest_results as a single-row DataFrame
        self.backtest_results_df = pd.DataFrame([metrics])

        return metrics

    def get_tradebook(self):
        """Return the tradebook DataFrame."""
        if self.tradebook_df is None or self.tradebook_df.empty:
            return pd.DataFrame(columns=[
                'entry_time','exit_time','trading_day','type','entry_price','exit_price',
                'quantity','pnl','return_pct','duration_minutes','entry_session',
                'exit_session','exit_reason'
            ])
        return self.tradebook_df.copy()

    def get_equity_curve(self):
        """Return the equity curve DataFrame."""
        if self.equity_curve_df is None or self.equity_curve_df.empty:
            return pd.DataFrame(columns=[
                'timestamp','price','position','capital','unrealized_pnl','equity',
                'trading_day','session_phase'
            ])
        return self.equity_curve_df.copy()

    def get_backtest_results(self):
        """Return the backtest results DataFrame (summary row)."""
        return self.backtest_results_df.copy() if self.backtest_results_df is not None else pd.DataFrame()

    def plot_equity_curve(self, figsize=(12, 5), title='Equity Curve'):
        """Plot the equity curve based on recorded equity."""
        if self.equity_curve_df is None or self.equity_curve_df.empty:
            print("No equity curve to plot. Run the backtest first.")
            return

        ec = self.equity_curve_df.copy()
        ec = ec.set_index('timestamp')
        plt.figure(figsize=figsize)
        plt.plot(ec.index, ec['equity'], label='Equity', color='navy')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Equity (currency units)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# 4. EXAMPLE USAGE (optional)
def run_intraday_strategy_example():
    """
    Example usage outline:
      1) Ensure df has columns: open, high, low, close, volume with DatetimeIndex in IST.
      2) Initialize strategy and run backtest:
         strategy = IntradayVWAPStrategy(df, initial_capital=100000)
         tradebook, equity = strategy.execute_intraday_backtest()
         metrics = strategy.get_performance_metrics()
         results_df = strategy.get_backtest_results()
         strategy.plot_equity_curve()
    """
    print("Intraday VWAP Strategy module loaded.")
    print("Provide a DataFrame with columns: open, high, low, close, volume and a DatetimeIndex.")


if __name__ == "__main__":
    run_intraday_strategy_example()

