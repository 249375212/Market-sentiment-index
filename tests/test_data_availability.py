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
from market_fear_greed.charts import make_history_chart
from market_fear_greed.data_sources import load_market_fear_greed_source_data
from market_fear_greed.fear_greed_index import _attach_index_closes
from market_fear_greed.enhanced_index import (
    ENHANCED_CACHE_SCHEMA_VERSION,
    SENTIMENT_WEIGHTS,
    build_afgi_enhanced_timeseries,
    load_enhanced_cache,
    update_enhanced_afgi_cache,
)
from market_fear_greed.pg_adapter import (
    clear_cache,
    pg_load_index_daily,
    pg_load_stock_basic_safe,
    pg_load_trade_calendar,
)


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

    def test_index_close_attachment_fills_missing_values_without_overwriting_existing(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-30", "2026-07-31"]),
                "trade_date": ["20260730", "20260731"],
                "sh_close": [3800.0, None],
            }
        )
        loaded = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-30", "2026-07-31"]),
                "close": [3801.0, 3832.26],
            }
        )

        with patch(
            "market_fear_greed.fear_greed_index.INDEX_CODES",
            {"sh": "000001.SH"},
        ), patch(
            "market_fear_greed.fear_greed_index.load_index_daily",
            return_value=loaded,
        ):
            result = _attach_index_closes(
                "token", frame, "20260730", "20260731"
            )

        self.assertEqual(result["sh_close"].tolist(), [3800.0, 3832.26])
        self.assertNotIn("sh_close_x", result.columns)
        self.assertNotIn("sh_close_y", result.columns)

    def test_postgres_index_loader_extends_missing_tail_from_local_stock_returns(self):
        cases = {
            "000001.SH": (3832.26, 0.01),
            "399006.SZ": (3343.96, -0.02),
        }
        for ts_code, (real_close, proxy_return) in cases.items():
            with self.subTest(ts_code=ts_code):
                clear_cache()
                real_rows = pd.DataFrame(
                    {
                        "trade_date": ["20260731"],
                        "date": [pd.Timestamp("2026-07-31")],
                        "open_price": [real_close - 10.0],
                        "high_price": [real_close + 10.0],
                        "low_price": [real_close - 20.0],
                        "close_price": [real_close],
                        "pre_close": [real_close - 5.0],
                        "pct_change": [0.13],
                        "volume": [1000.0],
                        "amount": [2000.0],
                    }
                )
                proxy_rows = pd.DataFrame(
                    {
                        "trade_date": ["20260803"],
                        "date": [pd.Timestamp("2026-08-03")],
                        "return_ratio": [proxy_return],
                    }
                )

                def query(sql, _params=None):
                    if "FROM index_daily" in sql:
                        return real_rows.copy()
                    if "stock_valuation_daily" in sql:
                        return proxy_rows.copy()
                    self.fail(f"unexpected SQL: {sql}")

                with patch("market_fear_greed.pg_adapter._query_df", side_effect=query):
                    result = pg_load_index_daily(
                        "PG_LOCAL", ts_code, "20260731", "20260803"
                    )

                self.assertEqual(result["trade_date"].tolist(), ["20260731", "20260803"])
                self.assertAlmostEqual(result.loc[0, "close"], real_close)
                self.assertAlmostEqual(
                    result.loc[1, "close"], real_close * (1.0 + proxy_return)
                )
                self.assertAlmostEqual(result.loc[1, "pre_close"], real_close)
                self.assertAlmostEqual(result.loc[1, "pct_chg"], proxy_return * 100.0)

    def test_postgres_index_loader_keeps_real_tail_without_proxying(self):
        clear_cache()
        real_rows = pd.DataFrame(
            {
                "trade_date": ["20260731", "20260803"],
                "date": pd.to_datetime(["2026-07-31", "2026-08-03"]),
                "open_price": [3820.0, 3810.0],
                "high_price": [3840.0, 3830.0],
                "low_price": [3810.0, 3800.0],
                "close_price": [3832.26, 3826.0],
                "pre_close": [3804.69, 3832.26],
                "pct_change": [0.724632, -0.16335],
                "volume": [1000.0, 1100.0],
                "amount": [2000.0, 2100.0],
            }
        )

        def query(sql, _params=None):
            if "FROM index_daily" in sql:
                return real_rows.copy()
            self.fail("real index tail must not be replaced by a stock proxy")

        with patch("market_fear_greed.pg_adapter._query_df", side_effect=query):
            result = pg_load_index_daily(
                "PG_LOCAL", "000001.SH", "20260731", "20260803"
            )

        self.assertEqual(result["trade_date"].tolist(), ["20260731", "20260803"])
        self.assertEqual(result["close"].tolist(), [3832.26, 3826.0])

    def test_postgres_index_loader_can_extend_single_missing_incremental_date(self):
        clear_cache()
        anchor_close = 3832.26
        anchor_row = pd.DataFrame(
            {
                "trade_date": ["20260731"],
                "date": [pd.Timestamp("2026-07-31")],
                "open_price": [3820.0],
                "high_price": [3840.0],
                "low_price": [3810.0],
                "close_price": [anchor_close],
                "pre_close": [3804.69],
                "pct_change": [0.724632],
                "volume": [1000.0],
                "amount": [2000.0],
            }
        )
        proxy_row = pd.DataFrame(
            {
                "trade_date": ["20260803"],
                "date": [pd.Timestamp("2026-08-03")],
                "return_ratio": [-0.001],
            }
        )

        def query(sql, _params=None):
            if "FROM index_daily" in sql and "ORDER BY trade_date DESC" in sql:
                return anchor_row.copy()
            if "FROM index_daily" in sql:
                return pd.DataFrame()
            if "stock_valuation_daily" in sql:
                return proxy_row.copy()
            self.fail(f"unexpected SQL: {sql}")

        with patch("market_fear_greed.pg_adapter._query_df", side_effect=query):
            result = pg_load_index_daily(
                "PG_LOCAL", "000001.SH", "20260803", "20260803"
            )

        self.assertEqual(result["trade_date"].tolist(), ["20260803"])
        self.assertAlmostEqual(result.loc[0, "close"], anchor_close * 0.999)
        self.assertAlmostEqual(result.loc[0, "pre_close"], anchor_close)

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

    def test_postgres_stock_basic_reads_l1_industry_classification(self):
        clear_cache()
        query_result = pd.DataFrame(
            {
                "symbol": ["600000"],
                "security_id": ["XSHG:600000"],
                "name": ["浦发银行"],
                "exchange": ["XSHG"],
                "list_date": [pd.Timestamp("1999-11-10")],
                "industry": ["银行"],
            }
        )
        with patch(
            "market_fear_greed.pg_adapter._query_df", return_value=query_result
        ) as query:
            result = pg_load_stock_basic_safe("PG_LOCAL")

        sql = query.call_args.args[0]
        self.assertIn("industry_info.level = 'L1'", sql)
        self.assertNotIn("industry_info.level = 'primary'", sql)
        self.assertEqual(result.loc[0, "industry"], "银行")

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

    def test_enhanced_cache_repairs_missing_index_overlays_without_rebuilding_sentiment(self):
        trade_date = "20260731"
        row = {
            "date": pd.Timestamp("2026-07-31"),
            "trade_date": trade_date,
            "afgi_enhanced": 35.0,
            "enhanced_schema_version": ENHANCED_CACHE_SCHEMA_VERSION,
            "sh_close": None,
            "cyb_close": None,
        }
        for component in SENTIMENT_WEIGHTS:
            row[f"{component}_status"] = "ok"
        frame = pd.DataFrame([row])
        index_history = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-31")],
                "trade_date": [trade_date],
                "sh_close": [3832.26],
                "cyb_close": [3343.96],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "enhanced.csv"
            frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
            with patch(
                "market_fear_greed.enhanced_index._local_stock_daily_dates",
                return_value=[],
            ), patch(
                "market_fear_greed.enhanced_index.load_trade_dates",
                return_value=[trade_date],
            ), patch(
                "market_fear_greed.enhanced_index._build_index_history",
                return_value=index_history,
            ) as index_loader, patch(
                "market_fear_greed.enhanced_index.build_afgi_enhanced_timeseries"
            ) as sentiment_builder:
                result = update_enhanced_afgi_cache(
                    "token", trade_date, trade_date, cache_path=cache_path
                )
            saved = load_enhanced_cache(cache_path)

        sentiment_builder.assert_not_called()
        index_loader.assert_called_once()
        self.assertEqual(result.loc[0, "sh_close"], 3832.26)
        self.assertEqual(result.loc[0, "cyb_close"], 3343.96)
        self.assertEqual(saved.loc[0, "sh_close"], 3832.26)
        self.assertEqual(saved.loc[0, "cyb_close"], 3343.96)
        self.assertIn(
            "上证指数",
            [trace.name for trace in make_history_chart(result, "上证指数").data],
        )
        self.assertIn(
            "创业板指",
            [trace.name for trace in make_history_chart(result, "创业板指").data],
        )

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
            with patch(
                "market_fear_greed.enhanced_index._local_stock_daily_dates", return_value=[]
            ), patch(
                "market_fear_greed.enhanced_index.load_trade_dates", return_value=["20260803"]
            ), patch(
                "market_fear_greed.enhanced_index._finalize_enhanced_scores",
                side_effect=lambda value: value,
            ), patch(
                "market_fear_greed.enhanced_index._build_index_history",
                return_value=pd.DataFrame(),
            ):
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
