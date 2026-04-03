import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

from src.inference import load_best_model, predict_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_patchtst_backtest(df: pd.DataFrame, 
                          initial_capital: float = 10000.0,
                          fee_pct: float = 0.0008,      
                          slippage_pct: float = 0.0004,
                          min_confidence: float = 60.0):
    """
    Walk-forward backtest with improved PatchTST signals.
    """
    model = load_best_model()
    
    equity = [initial_capital]
    position = 0
    entry_price = 0.0
    trades = []
    equity_curve = [initial_capital]

    logger.info(f"Starting backtest on {len(df):,} candles | Initial capital: ${initial_capital:,.0f}")

    # Use last 3000 candles (~10 days)
    test_df = df.tail(3000).copy()

    for i in range(512, len(test_df) - 20):
        current_slice = test_df.iloc[:i+1]
        current_price = current_slice['close'].iloc[-1]

        signal_dict = predict_signal(current_slice, model)
        signal = signal_dict['signal']
        confidence = signal_dict['confidence']

        # Exit logic
        if position != 0:
            pnl_pct = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
            pnl_pct = pnl_pct * position

            if (position == 1 and signal in ["SHORT", "STRONG_SHORT", "FLAT"]) or \
               (position == -1 and signal in ["LONG", "STRONG_LONG", "FLAT"]):
                final_equity = equity[-1] * (1 + pnl_pct - fee_pct)
                equity.append(final_equity)
                trades.append({
                    "type": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct * 100,
                    "confidence": confidence
                })
                position = 0

        # Entry logic
        if position == 0 and confidence >= min_confidence:
            if signal in ["LONG", "STRONG_LONG"]:
                position = 1
                entry_price = current_price * (1 + slippage_pct)
                equity.append(equity[-1] * (1 - fee_pct))
            elif signal in ["SHORT", "STRONG_SHORT"]:
                position = -1
                entry_price = current_price * (1 - slippage_pct)
                equity.append(equity[-1] * (1 - fee_pct))

        equity_curve.append(equity[-1])

    # Final metrics
    equity_series = pd.Series(equity_curve, index=test_df.index[-len(equity_curve):])
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
    num_trades = len(trades)
    win_rate = len([t for t in trades if t.get("pnl_pct", 0) > 0]) / num_trades * 100 if num_trades > 0 else 0
    max_dd = ((equity_series / equity_series.cummax()) - 1).min() * 100

    print("\n" + "="*70)
    print("PATCHTST BACKTEST RESULTS (Improved Signals)")
    print("="*70)
    print(f"Period                : {test_df.index[0].date()} → {test_df.index[-1].date()}")
    print(f"Final Equity          : ${equity[-1]:,.2f}")
    print(f"Total Return          : {total_return:+.2f}%")
    print(f"Number of Trades      : {num_trades}")
    print(f"Win Rate              : {win_rate:.1f}%")
    print(f"Max Drawdown          : {max_dd:.2f}%")
    print("="*70)

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_series.index, equity_series, label="Equity Curve", color="#00cc66", linewidth=2)
    plt.title("PatchTST Backtest - Equity Curve (Improved Signals)")
    plt.ylabel("Account Value ($)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtest_equity_curve_improved.png", dpi=200)
    plt.show()

    return equity_series, trades


if __name__ == "__main__":
    df = pd.read_parquet("data/raw/binance_btcusdt_5m.parquet")
    equity_curve, trades = run_patchtst_backtest(df, min_confidence=60.0)