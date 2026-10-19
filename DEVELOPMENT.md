# 開發說明

給要改程式的人看。使用方式見 [README.md](README.md)，每一版改了什麼見 [CHANGELOG.md](CHANGELOG.md)。

## 改完程式先跑測試

```bash
python -m tests                # 有進度條；同時開幾個行程跑（--jobs 1 一個一個跑）
python -m tests continue judge # 只跑名稱含這幾段文字的測試
```

- 共 114 個測試，不需要 LM Studio；四核心的電腦約 25 秒。
- 每個行程用不同的埠開假伺服器；上次各測試的耗時記在 `tests/.timings.json`（不進版本控制），慢的先派。
- `python -m unittest` 照樣可以用（沒有進度條）。
- GitHub Actions（`.github/workflows/tests.yml`）：每次 PR 與推到 main 時，在 Linux（Python 3.9、3.13）與 Windows（3.13）各跑一次。
- 全部用假資料或假伺服器；回報時要分清哪些功能實際在 LM Studio 上跑過、哪些只用假資料測過。

## 檔案結構

| 檔案 | 負責什麼 |
| --- | --- |
| `lijuncha.py` | 入口 |
| `config/config.json` | 版本號、屬性與群體、題庫、生成／評審／嵌入設定、對照模組、互斥規則（以企劃書為準） |
| `config/judge_prompts/` | 評審提示詞（檔名就是版本，改規則時換新檔名） |
| `config/names/` | 姓名資料檔與清單 `manifest.json`（原住民族名授權確認前不進版本控制） |
| `config/private/` | 不進版本控制：保留題 `reserved_questions.json`（`read_config` 有就合併進題庫）、作者機器的載入設定紀錄 `load_presets.json`、公開快照檢查用的 `denylist.txt` |
| `config/script/` | Unihan 繁簡字表 |
| `config/i18n/terms.json` | 中英對照詞表 |
| `ljc/cli.py` | 指令與選項 |
| `ljc/runner.py` | `run`、`status`、`check`：派發對話、進度、存檔、續跑、跨機器分工 |
| `ljc/maintain.py` | `merge`、`calibrate`、`continue` |
| `ljc/judge.py` | `judge` |
| `ljc/analyze.py`、`ljc/screen.py`、`ljc/joint.py`、`ljc/stats.py`、`ljc/verdict.py` | 分析、統計、判定 |
| `ljc/scorecard.py`、`ljc/report.py` | 報告的版面 |
| `ljc/metrics.py`、`ljc/tokens.py` | 指標、詞彙層向量與 token 檢查 |
| `ljc/personas.py`、`ljc/names.py` | `gen`：人物與姓名 |
| `ljc/confirm.py`、`ljc/probe.py`、`ljc/interpret.py`、`ljc/annotate.py` | 配對確認、操弄檢核、詮釋層、人工標註 |
| `ljc/styles.py`、`ljc/script.py` | 對照模組的寫法轉換、簡繁固定資料 |
| `ljc/layout.py` | `out/` 的資料夾安排 |
| `ljc/doctor.py` | `doctor`：換電腦或更新後的健檢（不改檔） |
| `ljc/envinfo.py`、`ljc/client.py`、`ljc/cloud.py`、`ljc/tuner.py` | 執行環境紀錄、與伺服器溝通、雲端、自動並行 |
| `ljc/i18n.py`、`ljc/i18n_en.py` | 介面語言（英文句型在 `i18n_en.py`） |
| `ljc/common.py` | 共用工具（含提示排版） |
| `ljc/pack.py`、`ljc/mock.py` | 打包與換電腦的解包驗證（`pack`、`unpack`）、假資料 |
| `tools/` | 姓名資料轉檔與清單（`names_manifest.py`）、驗證人物可重現（`verify_personas.py`）、token 機率實測（`logprobs_check.py`）、Unihan 字表、WGND 篩選、載入設定的人工紀錄、移出多跑的段、檢定力（`power.py`） |
| `tests/` | 自動測試、假伺服器、測試用設定檔與姓名資料（內容全是虛構的） |

相依方向是單向的：`cli → personas / runner / judge / maintain / analyze / annotate → screen → layout、tuner、client、metrics、stats、joint、verdict、report → common`。
`tuner.py`、`stats.py`、`joint.py` 與 `verdict.py` 不讀檔也不連網，可以單獨測試。

## 提示的排版規則

程式印出的每一段提示都照這幾條（`ljc/common.py` 的 `kv_block`、`bullets`、`stop`、`warn`、`Progress`、`next_steps`）：

- 一行只說一件事；一行最多一個「；」。
- 狀態：`── 標題`＋對齊的「項目：內容」。
- 停下來不做：`✗ 停在哪裡`，下面每行一個原因或做法。
- 要注意、照樣繼續：`⚠ 什麼事`，下面每行一個細節。
- 長時間的批次工作：進度條，約每 5% 一行。
- 結束時：`── 下一步`，每項一個指令。

## 名詞對照

程式的提示、報告與文件用同一套說法：

| 名稱 | 意思 | 在哪裡設定 |
|---|---|---|
| 段 | 一段對話：某位人物、某一題、第幾次重複；含 2 或 3 輪 | — |
| 則 | 一則回覆（一段兩輪就有兩則） | — |
| 揭露版／隱性版 | 人物直接說出身分／只給生活線索 | — |
| 身分位置 | 身分寫在第一句提問前（message）或放在系統提示（system） | `run --identity` |
| 每則上限 | 一則回覆最多幾個 token，含思考（max_tokens） | 設定檔 `generation.max_tokens`；評審是 `judge.max_tokens` |
| 上下文長度 | LM Studio 載入模型時的 Context Length，並行的各段平分 | LM Studio；正式值寫在 `generation.context_length` |
| 每段上下文 | 上下文長度 ÷ 並行數，每段至少 `generation.min_context_per_segment` | — |
| 並行數 | 同時進行幾段對話（或幾則評審） | `--workers`（預設 auto，自動測速） |
| 設定資料夾 | out 底下每組設定一個資料夾 | 由指令參數決定；`settings.json` 不要手動改 |

## 介面語言

- `--lang en`（或環境變數 `LJC_LANG=en`）：提示逐行比對 `ljc/i18n_en.py` 的句型換成英文；中文字串不用改。
- 送給模型的人物句、題目與回覆不翻；`analyze` 一律同時產生中英文報告，以中文版為準。
- 之後的新功能預設只做中文，作者要求時才補英文；還沒有英文的句子照樣印中文。
