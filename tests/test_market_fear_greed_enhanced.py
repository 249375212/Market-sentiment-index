import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_fear_greed.enhanced_indicators import (
    calc_limit_up_down_ratio,
    calc_ma20_ratio,
    calc_new_low_60d_ratio,
    calc_open_board_rate,
    calc_stock_industry_up_ratio,
    calc_up_5d_ratio,
    calc_up_ratio,
    calc_yesterday_limit_up_return,
)
from market_fear_greed.enhanced_index import (
    SENTIMENT_WEIGHTS,
    _effective_cached_dates,
    _finalize_enhanced_scores,
    _industry_frame_for_date,
    _supplement_original_core_indicators,
    calc_afgi_enhanced,
    calc_breadth_sentiment,
    get_enhanced_cache_status,
    latest_enhanced_indicator_table,
    load_enhanced_cache,
    plan_enhanced_update_ranges,
    save_enhanced_cache,
)
from market_fear_greed.charts import make_component_chart, make_gauge, make_history_chart
from market_fear_greed.fear_greed_index import score_raw_fear_greed_indicators
from market_fear_greed.indicators import calc_price_strength_indicator


class EnhancedFearGreedTest(unittest.TestCase):
    def _daily(self, date, closes, pre_closes=None, highs=None):
        codes = ["000001.SZ", "000002.SZ", "600000.SH"]
        pre_closes = pre_closes or [value - 1 for value in closes]
        highs = highs or closes
        return pd.DataFrame(
            {
                "ts_code": codes,
                "trade_date": [pd.Timestamp(date).strftime("%Y%m%d")] * len(codes),
                "date": [pd.Timestamp(date)] * len(codes),
                "close": closes,
                "pre_close": pre_closes,
                "high": highs,
                "pct_chg": [(close / previous - 1) * 100 for close, previous in zip(closes, pre_closes)],
                "vol": [100, 100, 100],
                "amount": [1000, 1000, 1000],
            }
        )

    def _history(self, periods=65):
        frames = []
        for index, trade_date in enumerate(pd.bdate_range("2024-01-01", periods=periods)):
            frames.append(self._daily(trade_date, [10 + index * 0.1, 20 - index * 0.1, 30 + index * 0.05]))
        return pd.concat(frames, ignore_index=True)

    def test_has_nine_weighted_components(self):
        self.assertEqual(len(SENTIMENT_WEIGHTS), 9)
        self.assertAlmostEqual(sum(SENTIMENT_WEIGHTS.values()), 1.0)

    def test_breadth_and_ma_indicators(self):
        current = self._daily("2024-04-01", [12, 18, 33], pre_closes=[11, 19, 32])
        history = self._history()
        up = calc_up_ratio(current)
        self.assertEqual(up["status"], "ok")
        self.assertAlmostEqual(up["value"], 2 / 3)
        ma20 = calc_ma20_ratio(history, current)
        self.assertEqual(ma20["status"], "ok")
        self.assertTrue(0 <= ma20["value"] <= 1)

    def test_limit_and_open_board_indicators(self):
        current = self._daily("2024-04-01", [11, 18, 27], pre_closes=[10, 20, 30], highs=[11, 20, 30])
        ratio = calc_limit_up_down_ratio(current)
        open_board = calc_open_board_rate(current)
        self.assertEqual(ratio["status"], "ok")
        self.assertEqual(open_board["status"], "ok")
        self.assertGreaterEqual(ratio["value"], 0)
        self.assertGreaterEqual(open_board["value"], 0)

    def test_profitability_indicators(self):
        history = self._history()
        yesterday = self._daily("2024-04-01", [11, 20, 30], pre_closes=[10, 20, 30], highs=[11, 20, 30])
        current = self._daily("2024-04-02", [11.5, 19, 31], pre_closes=[11, 20, 30])
        self.assertEqual(calc_yesterday_limit_up_return(yesterday, current)["status"], "ok")
        self.assertIn(calc_up_5d_ratio(history, current)["status"], {"ok", "missing"})
        self.assertIn(calc_new_low_60d_ratio(history, current)["status"], {"ok", "missing"})

    def test_score_bounds_and_cache_roundtrip(self):
        row = pd.Series({key: 50 + index * 3 for index, key in enumerate(SENTIMENT_WEIGHTS)})
        row["up_ratio_score"] = 80
        row["ma20_ratio_score"] = 70
        row["ma60_ratio_score"] = 60
        self.assertTrue(0 <= calc_breadth_sentiment(row) <= 100)
        self.assertTrue(0 <= calc_afgi_enhanced(row) <= 100)
        self.assertEqual(len(latest_enhanced_indicator_table(row)), 10)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "enhanced.csv"
            frame = pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-01-01")],
                    "trade_date": ["20240101"],
                    "afgi_enhanced": [60],
                }
            )
            save_enhanced_cache(frame, path)
            loaded = load_enhanced_cache(path)
        self.assertFalse(loaded.empty)

    def test_cache_status_distinguishes_first_run_from_incremental_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_path = temp_path / "enhanced.csv"
            raw_dir = temp_path / "raw"

            empty_status = get_enhanced_cache_status(cache_path, raw_dir)
            self.assertTrue(empty_status["needs_initialization"])
            self.assertIsNone(empty_status["latest_trade_date"])

            frame = pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-12-31")],
                    "trade_date": ["20241231"],
                    "afgi_enhanced": [55.0],
                }
            )
            save_enhanced_cache(frame, cache_path)
            raw_dir.mkdir()
            for trade_date in pd.bdate_range("2024-01-01", periods=252):
                (raw_dir / f"stock_daily_{trade_date:%Y%m%d}.csv").touch()

            ready_status = get_enhanced_cache_status(cache_path, raw_dir)
            self.assertFalse(ready_status["needs_initialization"])
            self.assertEqual(ready_status["history_trade_days"], 252)
            self.assertEqual(ready_status["earliest_trade_date"], "20241231")
            self.assertEqual(ready_status["latest_trade_date"], "20241231")

    def test_update_plan_fills_before_cache_and_after_cache(self):
        cache_status = {
            "needs_initialization": False,
            "earliest_trade_date": "20240110",
            "latest_trade_date": "20240730",
        }
        ranges = plan_enhanced_update_ranges(cache_status, "20230101", "20240731", "20240731")
        self.assertEqual(ranges, [("20230101", "20240109"), ("20240730", "20240731")])

    def test_first_run_update_plan_uses_requested_display_range(self):
        cache_status = {"needs_initialization": True}
        ranges = plan_enhanced_update_ranges(cache_status, "20240701", "20240731")
        self.assertEqual(ranges, [("20240701", "20240731")])

    def test_schema_upgrade_rebuilds_requested_display_range(self):
        cache_status = {
            "needs_initialization": False,
            "needs_rebuild": True,
            "earliest_trade_date": "20200101",
            "latest_trade_date": "20240731",
        }
        ranges = plan_enhanced_update_ranges(cache_status, "20230731", "20240731", "20240731")
        self.assertEqual(ranges, [("20230731", "20240731")])

    def test_custom_historical_range_does_not_expand_first_run_to_today(self):
        cache_status = {"needs_initialization": True}
        ranges = plan_enhanced_update_ranges(cache_status, "20200101", "20201231", "20240731")
        self.assertEqual(ranges, [("20200101", "20201231")])

    def test_newer_preset_does_not_download_gap_after_old_custom_cache(self):
        cache_status = {
            "needs_initialization": False,
            "earliest_trade_date": "20200101",
            "latest_trade_date": "20201231",
        }
        ranges = plan_enhanced_update_ranges(cache_status, "20230731", "20240731", "20240731")
        self.assertEqual(ranges, [("20230731", "20240731")])

    def test_missing_core_score_carries_latest_valid_value(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "trade_date": ["20240102", "20240103"],
                "raw_market_volatility": [0.2, None],
                "raw_amount_ratio": [1.1, None],
                "raw_new_high_ratio": [0.1, None],
                "raw_risk_appetite": [0.03, None],
                "volatility_status": ["ok", "missing"],
                "volume_status": ["ok", "missing"],
                "price_strength_status": ["ok", "missing"],
                "risk_appetite_status": ["ok", "missing"],
            }
        )
        scored = score_raw_fear_greed_indicators(frame)
        latest = scored.iloc[-1]
        for prefix in ("volatility", "volume", "price_strength", "risk_appetite"):
            self.assertEqual(latest[f"{prefix}_status"], "stale")
            self.assertEqual(latest[f"{prefix}_score"], scored.iloc[-2][f"{prefix}_score"])
            self.assertEqual(latest[f"{prefix}_score_source_date"], "20240102")

    def test_core_score_directions_match_greed_semantics(self):
        periods = 252
        frame = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=periods),
                "trade_date": pd.bdate_range("2024-01-01", periods=periods).strftime("%Y%m%d"),
                "raw_market_volatility": range(periods),
                "raw_amount_ratio": range(periods),
                "raw_new_high_ratio": range(periods),
                "raw_risk_appetite": range(periods),
                "volatility_status": "ok",
                "volume_status": "ok",
                "price_strength_status": "ok",
                "risk_appetite_status": "ok",
            }
        )
        latest = score_raw_fear_greed_indicators(frame).iloc[-1]
        self.assertEqual(latest["volatility_score"], 0.0)
        self.assertEqual(latest["price_strength_score"], 100.0)
        self.assertEqual(latest["risk_appetite_score"], 100.0)

    def test_price_strength_matches_original_120_sample_entry_rule(self):
        dates = pd.bdate_range("2024-01-02", periods=120)
        closes = [100.0 + index * 0.1 for index in range(len(dates))]
        daily = pd.DataFrame(
            {
                "ts_code": "000001.SZ",
                "date": dates,
                "trade_date": dates.strftime("%Y%m%d"),
                "close": closes,
            }
        )
        stock_basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["A"],
                "list_date": ["20200101"],
            }
        )
        result = calc_price_strength_indicator(daily, stock_basic)
        self.assertEqual(result.iloc[-2]["effective_stock_count"], 0)
        latest = result.iloc[-1]
        self.assertEqual(latest["effective_stock_count"], 1)
        self.assertEqual(latest["new_high_count"], 1)
        self.assertEqual(latest["raw_new_high_ratio"], 1.0)

    def test_component_statuses_and_chart_layout(self):
        row = {
            "date": pd.Timestamp("2024-01-03"),
            "trade_date": "20240103",
            "volatility_score": 20,
            "volume_score": 30,
            "price_strength_score": 40,
            "risk_appetite_score": 50,
            "volatility_status": "stale",
            "volume_status": "ok",
            "price_strength_status": "ok",
            "risk_appetite_status": "ok",
        }
        for raw_col in (
            "up_ratio",
            "ma20_ratio",
            "ma60_ratio",
            "limit_up_down_ratio",
            "open_board_rate",
            "yesterday_limit_up_return",
            "up_5d_ratio",
            "new_low_60d_ratio",
            "industry_up_ratio",
            "small_large_relative_strength",
        ):
            row[raw_col] = 0.5
            row[f"{raw_col}_status"] = "ok"
        finalized = _finalize_enhanced_scores(pd.DataFrame([row])).iloc[0]
        self.assertEqual(finalized["volatility_sentiment_status"], "stale")
        self.assertEqual(finalized["breadth_sentiment_status"], "ok")

        components = make_component_chart(finalized)
        self.assertEqual(components.data[0].orientation, "h")
        self.assertIn("延用", list(components.data[0].y)[0])
        gauge = make_gauge(finalized)
        self.assertEqual(list(gauge.data[0].domain.x), [0.08, 0.92])

    def test_history_chart_merges_sentiment_average_and_market_index(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        frame = pd.DataFrame(
            {
                "date": dates,
                "afgi_enhanced": range(30, 60),
                "afgi_enhanced_ma20": range(35, 65),
                "hs300_close": range(3500, 3530),
            }
        )
        chart = make_history_chart(frame, "沪深300")
        self.assertEqual([trace.name for trace in chart.data], ["市场情绪指数", "20日均值", "沪深300"])
        self.assertEqual(chart.data[-1].yaxis, "y2")
        self.assertEqual(chart.layout.yaxis.range, (0, 100))

    def test_current_industry_snapshot_fills_history_gap(self):
        history = pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "pct_chg": [1.0],
            }
        )
        spot = pd.DataFrame({"板块名称": ["行业A", "行业B"], "涨跌幅": [1.2, -0.4]})
        with patch(
            "market_fear_greed.enhanced_index.load_industry_spot_cached",
            return_value=spot,
        ) as loader:
            result = _industry_frame_for_date(history, "20240103")
        loader.assert_called_once_with("20240103")
        self.assertEqual(len(result), 2)
        self.assertIn("涨跌幅", result.columns)

    def test_stock_industry_fallback_uses_current_returns(self):
        current = self._daily(
            "2024-04-01",
            [10.5, 18, 33],
            pre_closes=[10, 20, 30],
        )
        basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
                "name": ["A", "B", "C"],
                "industry": ["银行", "银行", "电子"],
            }
        )
        result = calc_stock_industry_up_ratio(current, basic)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source"], "Tushare stock_basic + daily")
        self.assertAlmostEqual(result["value"], 0.5)

    def test_stale_core_component_requires_refresh(self):
        row = {
            "date": pd.Timestamp("2024-04-01"),
            "trade_date": "20240401",
        }
        for component in SENTIMENT_WEIGHTS:
            row[f"{component}_status"] = "ok"
        frame = pd.DataFrame([row])
        self.assertEqual(_effective_cached_dates(frame, "token"), {"20240401"})
        frame.loc[0, "volume_sentiment_status"] = "stale"
        self.assertEqual(_effective_cached_dates(frame, "token"), set())

    def test_lightweight_core_supplement_produces_current_values(self):
        dates = pd.bdate_range("2023-01-02", periods=260)
        daily_map = {
            day.strftime("%Y%m%d"): self._daily(
                day,
                [10 + index * 0.02, 20 + index * 0.01, 30 + index * 0.03],
            )
            for index, day in enumerate(dates)
        }
        stock_basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
                "name": ["A", "B", "C"],
                "industry": ["银行", "银行", "电子"],
                "list_date": ["20200101", "20200101", "20200101"],
            }
        )
        index_dates = dates
        index_history = pd.DataFrame(
            {
                "date": index_dates,
                "trade_date": index_dates.strftime("%Y%m%d"),
                "hs300_close": [4000 + index * 5 for index in range(len(index_dates))],
                "zz1000_close": [6000 + index * 8 for index in range(len(index_dates))],
            }
        )
        bond = pd.DataFrame(
            {
                "date": index_dates,
                "trade_date": index_dates.strftime("%Y%m%d"),
                "close": [100 + index * 0.01 for index in range(len(index_dates))],
            }
        )
        targets = dates.strftime("%Y%m%d").tolist()
        with patch(
            "market_fear_greed.enhanced_index.load_fund_daily",
            return_value=bond,
        ) as fund_loader, patch("market_fear_greed.enhanced_index.save_original_afgi_cache"):
            result = _supplement_original_core_indicators(
                "token",
                pd.DataFrame(),
                daily_map,
                stock_basic,
                index_history,
                targets,
            )
        for call in fund_loader.call_args_list:
            self.assertIn(targets[0], {str(value) for value in call[0]})
        self.assertGreaterEqual(result["raw_market_volatility"].notna().sum(), 240)
        self.assertGreaterEqual(result["raw_risk_appetite"].notna().sum(), 240)
        latest = result.iloc[-1]
        for prefix in ("volatility", "volume", "price_strength", "risk_appetite"):
            self.assertEqual(latest[f"{prefix}_status"], "ok")
        for raw_col in (
            "raw_market_volatility",
            "raw_amount_ratio",
            "raw_new_high_ratio",
            "raw_risk_appetite",
        ):
            self.assertTrue(pd.notna(latest[raw_col]))


if __name__ == "__main__":
    unittest.main()
