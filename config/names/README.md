# 姓名資料檔

把這些檔放在這個資料夾，檔名不要改：

- `names_general.jsonl`：一般常見姓名（內政部《全國姓名統計分析》），格式見〈離均差 姓名資料檔規格〉
- `names_indigenous.jsonl`：原住民族名（原住民族委員會、原住民族語言研究發展基金會），格式同上
- `names_foreign.jsonl`：外籍身分（婚姻移民、產業移工、外國專業人員）的原文名，格式見〈離均差 原文姓名資料檔規格〉。
  - 外籍三群共用的 4 國（印尼、越南、菲律賓、泰國；作者 10-19 決定）：作者整理清單（`AUTHOR-LIST-2026`），
    每國每性別 15 個名字（越南、菲律賓、泰國另有 15 個姓氏），不代表常見程度。越南文保留聲調符號；泰國用護照上的拉丁字母拼法；
    印尼多半沒有家族姓氏，直接給完整姓名。建議再請母語者校對。
  - 法、英、美、加、澳：用 `python tools/make_names_foreign.py --raw <NAMES 資料夾>` 從作者下載的官方資料轉出（目前沒有輪到）。
- `names_foreign_test.jsonl`：測試用的外籍名字（Faker 語系詞表，不是人口統計，每筆 `test_only`）。
  用 `python tools/make_test_names.py` 產生；只有 `gen --draft` 會用，正式模式一律不用。

`raw/`：下載的原始資料與篩完的中間檔，整個資料夾不進版本控制。
- `raw/wgnd/wgnd_2_0_name-gender-code.csv`：WGND 2.0（Raffo 2021，doi:10.7910/DVN/MSEGSJ，CC0 1.0）。
  `python tools/filter_wgnd.py --move ~/Downloads/wgnd_2_0_name-gender-code.csv` 搬進來並印欄位與前 5 列；
  再執行 `python tools/filter_wgnd.py` 篩成 `raw/wgnd/wgnd_2_0_filtered.csv`（需要的 9 國、某一性別 wgt ≥ 0.9）。

`names_new_immigrant.jsonl`（新住民取用的中文姓名）從 v0.5.2 起不再使用：企劃書決定外籍身分一律用原文名。

**哪些進版本控制**（作者 10-18 決定：讓使用者能自己產生人物並驗證）：
- 放進儲存庫：`names_general.jsonl`、`names_foreign.jsonl`、`names_foreign_test.jsonl`，以及清單 `manifest.json`。
  放進去之前，先在 `manifest.json` 的 `license` 欄填好每個來源的授權與出處（政府開放資料通常要求標示出處）。
- 不放：`names_indigenous.jsonl`。族語發展會的人名資料庫標示版權所有，授權確認前不放進儲存庫；
  沒有這個檔的人產生不了正式人物（原住民族群體要用族名），只能用 `manifest.json` 的 SHA-256 確認自己拿到的是同一份。

**驗證**：
- `python tools/names_manifest.py`：依這個資料夾現在的檔重寫 `manifest.json`（每個檔的 SHA-256、筆數、各來源筆數；授權欄保留）。
- `python tools/names_manifest.py --check`：比對自己的檔和清單是不是同一份。
- `python tools/verify_personas.py --personas <人物版本>`：用人物檔記下的種子重新產生一次，逐筆比對。
  v0.10.1 起的人物檔才有完整紀錄；p1.2 是 v0.8.6 以前產生的，用現在的程式算出來會不同（少了屬性平衡互換）。

換行字元：`.gitattributes` 規定 `.jsonl` 一律用 `\n`，各平台取出的檔案指紋才會一樣。

外籍身分要輪流的原國籍寫在 `config/config.json` 的 `origin_countries`。
執行 `python lijuncha.py gen` 時會說還缺哪幾類姓名、缺哪一國。
