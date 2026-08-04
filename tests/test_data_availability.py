import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_fear_greed.cache import (
    AFGI_CACHE_SCHEMA_VERSION,
    load_cache,
    update_fear_greed_cache,
)
from market_fear_greed.data_sources import load_market_fear_greed_source_data
from market_fear_greed.enhanced_index import (
    ENHANCED_CACHE_SCHEMA_VERSION,
    SENTIMENT_WEIGHTS,
    build_afgi_enhanced_timeseries,
    load_enhanced_cache,
    update_enhanced_afgi_cache,
)
from market_fear_greed.pg_adapter import clear_cache, pg_load_trade_calendar


class DataAvailabilityTest(unittest.TestCase):
    @staticmethod
    def _daily(trade_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [trade_date],
                "date": [pd.to_datetime(trade_date)],
                "close": [10.0],
                "amount": [1000.0],
            }
        )

    def test_postgres_trade_calendar_requires_downloaded_stock_daily(self):
        clear_cache()
        query_result = pd.DataFrame(
            {"trade_date": ["20260803"], "date": [pd.Timestamp("2026-08-03")]}
        )
        with patch("market_fear_greed.pg_adapter._query_df", return_value=query_result) as query:
            result = pg_load_trade_calendar("PG_LOCAL", "20260803", "20260804")

        sql, params = query.call_args.args
        self.assertIn("EXISTS (SELECT 1 FROM stock_daily", sql)
        self.assertEqual(params, ("20260803", "20260804"))
        self.assertEqual(result["trade_date"].tolist(), ["20260803"])

    def test_original_source_omits_calendar_date_without_stock_daily(self):
        empty = pd.DataFrame()
        daily_20260803 = self._daily("20260803")
        patches = {
            "load_trade_dates": ["20260803", "20260804"],
            "load_all_stock_daily": lambda _token, trade_date: daily_20260803 if trade_date == "20260803" else empty,
            "load_index_daily": empty,
            "load_stock_basic_safe": empty,
            "load_ivix_series": empty,
            "load_fund_daily": empty,
            "load_50etf_option_basic": empty,
            "load_50etf_option_daily": empty,
            "load_margin_series": empty,
            "load_futures_basis_proxy": empty,
        }
        with ExitStack() as stack:
            for name, value in patches.items():
                target = f"market_fear_greed.data_sources.{name}"
                if callable(value):
                    stack.enter_context(patch(target, side_effect=value))
                else:
                    stack.enter_context(patch(target, return_value=value))
            source = load_market_fear_greed_source_data("token", "20260803", "20260804")

        self.assertEqual(source["trade_dates"], ["20260803"])
        self.assertEqual(set(source["daily"]["trade_date"]), {"20260803"})

    def _build_enhanced_with_daily_dates(self, available_dates):
        def load_daily(_token, trade_date):
            return self._daily(trade_date) if trade_date in available_dates else pd.DataFrame()

        def build_rows(_daily_map, _trade_dates, target_dates, *_args):
            return pd.DataFrame(
                {
                    "trade_date": target_dates,
                    "date": pd.to_datetime(target_dates),
                }
            )

        with patch("market_fear_greed.enhanced_index.load_original_afgi_cache", return_value=pd.DataFrame()), patch(
            "market_fear_greed.enhanced_index.load_trade_dates",
            return_value=["20260803", "20260804"],
        ), patch("market_fear_greed.enhanced_index._local_stock_daily_dates", return_value=[]), patch(
            "market_fear_greed.enhanced_index.load_stock_basic_safe", return_value=pd.DataFrame()
        ), patch("market_fear_greed.enhanced_index.load_stock_daily_cached", side_effect=load_daily), patch(
            "market_fear_greed.enhanced_index._build_index_history", return_value=pd.DataFrame()
        ), patch(
            "market_fear_greed.enhanced_index._supplement_original_core_indicators",
            return_value=pd.DataFrame(),
        ), patch(
            "market_fear_greed.enhanced_index.load_industry_history_cached", return_value=pd.DataFrame()
        ), patch(
            "market_fear_greed.enhanced_index._build_stock_raw_rows_fast", side_effect=build_rows
        ), patch("market_fear_greed.enhanced_index._attach_original_afgi", side_effect=lambda _token, frame, *_args, **_kwargs: frame), patch(
            "market_fear_greed.enhanced_index._finalize_enhanced_scores", side_effect=lambda frame: frame
        ):
            return build_afgi_enhanced_timeseries("token", "20260803", "20260804")

    def test_enhanced_builder_omits_date_without_stock_daily(self):
        result = self._build_enhanced_with_daily_dates({"20260803"})
        self.assertEqual(result["trade_date"].tolist(), ["20260803"])

    def test_enhanced_builder_keeps_date_with_stock_daily(self):
        result = self._build_enhanced_with_daily_dates({"20260803", "20260804"})
        self.assertEqual(result["trade_date"].tolist(), ["20260803", "20260804"])

    def test_enhanced_cache_removes_unavailable_invalid_tail(self):
        rows = {
            "date": pd.to_datetime(["2026-08-03", "2026-08-04"]),
            "trade_date": ["20260803", "20260804"],
            "afgi_enhanced": [35.0, 25.0],
            "enhanced_schema_version": [ENHANCED_CACHE_SCHEMA_VERSION] * 2,
        }
        for component in SENTIMENT_WEIGHTS:
            rows[f"{component}_status"] = ["ok", "missing"]
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "enhanced.csv"
            frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
            with patch("market_fear_greed.enhanced_index._local_stock_daily_dates", return_value=[]), patch(
                "market_fear_greed.enhanced_index.load_trade_dates", return_value=["20260803"]
            ), patch("market_fear_greed.enhanced_index._finalize_enhanced_scores", side_effect=lambda value: value):
                result = update_enhanced_afgi_cache(
                    "token", "20260803", "20260804", cache_path=cache_path
                )
            saved = load_enhanced_cache(cache_path)

        self.assertEqual(result["trade_date"].tolist(), ["20260803"])
        self.assertEqual(saved["trade_date"].tolist(), ["20260803"])

    def test_original_cache_removes_unavailable_invalid_tail(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-03", "2026-08-04"]),
                "trade_date": ["20260803", "20260804"],
                "total_amount": [1000.0, None],
                "volume_source": ["Tushare全A日线成交额", None],
                "volume_status": ["ok", "stale"],
                "afgi": [35.0, 25.0],
                "afgi_schema_version": [AFGI_CACHE_SCHEMA_VERSION] * 2,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "original.csv"
            frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
            with patch("market_fear_greed.cache.load_trade_dates", return_value=["20260803"]):
                result = update_fear_greed_cache(
                    "token", "20260803", "20260804", cache_path=cache_path
                )
            saved = load_cache(cache_path)

        self.assertEqual(result["trade_date"].tolist(), ["20260803"])
        self.assertEqual(saved["trade_date"].tolist(), ["20260803"])


if __name__ == "__main__":
    unittest.main()
