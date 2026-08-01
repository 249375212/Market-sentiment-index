# A股市场情绪指数

一个基于 Streamlit、Tushare 与 AkShare 的九分项 A 股市场情绪指数开源实现。本仓库目前只保留市场情绪指数，不包含个股情绪、ETF、行业排名、组合或回测模块。

## 九大分项

| 分项 | 权重 | 主要输入 |
| --- | ---: | --- |
| 波动率情绪 | 15% | 50ETF 20日年化历史波动率，缺失时使用沪深300，反向计分 |
| 成交情绪 | 15% | 全市场成交活跃度 |
| 股价强度情绪 | 10% | 全A 252日收盘新高股票占比 |
| 风险偏好情绪 | 10% | 沪深300与国债ETF的20日相对收益 |
| 市场广度情绪 | 15% | 上涨占比、MA20、MA60 |
| 涨跌停情绪 | 15% | 涨跌停比、炸板率、昨日涨停收益 |
| 赚钱效应 | 10% | 近5日上涨占比、创60日新低占比 |
| 板块扩散情绪 | 5% | 行业上涨占比 |
| 风格风险偏好 | 5% | 中证1000相对沪深300强弱 |

后五项由十个 A 股底层原始指标计算。各项统一转换为 0–100 分，数据缺失或历史样本不足时使用 50 分中性值。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:TUSHARE_TOKEN="你的 Token"
streamlit run app.py
```

也可以把 `.streamlit/secrets.toml.example` 复制为 `.streamlit/secrets.toml`，再填写 Token。该文件已被 Git 忽略。

启动后会自动打开浏览器，也可以手动访问 `http://localhost:8501`。终端需要在应用运行期间保持打开，按 `Ctrl+C` 停止服务。

## 数据更新

页面启动只读取本地缓存，不会自动联网。首次使用时，点击主界面的“获取历史数据并开始计算”，程序会下载所选区间以及滚动指标所需的前置历史行情、显示进度，并将结果保存到本地。前置行情只参与计算，不会扩大图表日期范围。首次处理耗时较长，请保持终端和浏览器页面开启。

主界面的“近1年、近3年、近5年、自定义”同时控制图表和数据更新范围。选择范围后点击“补齐并更新数据”，程序会同时检查本地最早和最新交易日：所选日期更早时向前补齐，存在新增交易日时向后补齐。

运行缓存位于 `data/market_fear_greed/`，默认不会提交到 Git。首次从空缓存构建需要下载滚动历史窗口，后续更新速度会明显加快。

本项目使用 Tushare 的 ETF 日线行情接口，因此 Token **至少需要 5000 积分**。积分不足时，波动率和风险偏好等分项可能无法取得完整数据。

## 测试

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 项目结构

```text
.
├── app.py
├── market_fear_greed/
│   ├── cache.py
│   ├── charts.py
│   ├── data_sources.py
│   ├── enhanced_explanation.py
│   ├── enhanced_index.py
│   ├── enhanced_indicators.py
│   ├── fear_greed_index.py
│   ├── indicators.py
│   ├── scoring.py
│   └── tushare_client.py
├── tests/
└── data/market_fear_greed/
```

## 声明

本项目仅用于量化研究、数据分析与技术交流，不构成任何投资建议。

## License

[MIT](LICENSE)
