# src/data_fetch/unified_fetcher.py
import ccxt
import pandas as pd
from datetime import datetime
from typing import Optional, Union, Literal
import time

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class UnifiedDataFetcher:
    """Unified OHLCV fetcher supporting:
    - CCXT: Any crypto exchange (Binance, Bybit, etc.) → deep history
    - Alpaca: US stocks + popular cryptos (BTC/USD, ETH/USD, etc.)
    Returns clean pandas DataFrame ready for training PatchTST, LightGBM, etc.
    """

    def __init__(
        self,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
    ):
        # CCXT exchanges cache
        self.ccxt_exchanges: dict = {}

        # Alpaca clients
        self.crypto_client = CryptoHistoricalDataClient()  # No keys needed for crypto
        self.stock_client = None
        if alpaca_api_key and alpaca_secret_key:
            self.stock_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)

    def _get_ccxt_exchange(self, exchange_name: str = "binance") -> ccxt.Exchange:
        if exchange_name not in self.ccxt_exchanges:
            self.ccxt_exchanges[exchange_name] = getattr(ccxt, exchange_name)({
                "enableRateLimit": True,
            })
        return self.ccxt_exchanges[exchange_name]

    def fetch_data(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: int = 5000,
        source: Literal["auto", "ccxt", "alpaca"] = "auto",
        ccxt_exchange: str = "binance",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for any asset.
        
        Examples:
            fetch_data("BTC/USDT", "5m")                    # CCXT Binance (best for deep crypto history)
            fetch_data("BTC/USD", "5m", source="alpaca")    # Alpaca crypto (no keys needed)
            fetch_data("AAPL", "15m", source="alpaca")      # Stock (requires API keys)
        """
        if source == "auto":
            # Prefer CCXT for USDT pairs (deeper history), Alpaca otherwise
            source = "ccxt" if "USDT" in symbol.upper() else "alpaca"

        if source == "ccxt":
            return self._fetch_ccxt(symbol, timeframe, start_date, end_date, limit, ccxt_exchange)
        else:
            return self._fetch_alpaca(symbol, timeframe, start_date, end_date)

    def _fetch_ccxt(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        limit: int,
        exchange_name: str,
    ) -> pd.DataFrame:
        exchange = self._get_ccxt_exchange(exchange_name)
        since = int(pd.Timestamp(start_date).timestamp() * 1000) if start_date else None

        ohlcv = []
        while True:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if len(data) == 0:
                break
            ohlcv.extend(data)
            since = data[-1][0] + 1
            if end_date and pd.Timestamp(data[-1][0], unit="ms") >= pd.Timestamp(end_date):
                break
            time.sleep(0.25)  # Be nice to rate limits

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def _fetch_alpaca(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
    ) -> pd.DataFrame:
        tf_map = {
            "1m": TimeFrame(1, TimeFrameUnit.Minute),
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Minute)

        # Alpaca uses BTC/USD format (not BTC/USDT)
        alpaca_symbol = symbol.replace("USDT", "USD").upper()

        # Decide crypto vs stock
        is_crypto = any(c in alpaca_symbol for c in ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOGE"])

        if is_crypto:
            req = CryptoBarsRequest(
                symbol_or_symbols=[alpaca_symbol],
                timeframe=tf,
                start=start_date,
                end=end_date,
            )
            bars = self.crypto_client.get_crypto_bars(req)
        else:
            if not self.stock_client:
                raise ValueError(
                    "Alpaca API keys are required for stock data. "
                    "Pass alpaca_api_key and alpaca_secret_key to UnifiedDataFetcher()"
                )
            req = StockBarsRequest(
                symbol_or_symbols=[alpaca_symbol],
                timeframe=tf,
                start=start_date,
                end=end_date,
                feed="sip",
            )
            bars = self.stock_client.get_stock_bars(req)

        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)  # Remove symbol level from multi-index

        return df[["open", "high", "low", "close", "volume"]].copy()