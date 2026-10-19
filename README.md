# 離均差

**台灣情境的 AI 公平與反歧視評量**：同樣的問題，聊天機器人會不會因為使用者是誰而給出不一樣的回覆？

這個儲存庫是評量的研究原型。它讓本機的語言模型面對上百位虛構使用者，量測回覆品質是否隨身分而偏離平均（「離均差」）。

## 專案簡介

**要回答的問題**
- 使用者透露（或只隱約透露）自己的經濟、年齡、族群、宗教、健康等身分時，模型的回覆會不會變短、拒答、多加警語、把人轉介出去，或答錯？
- 差別出現在哪些群體、哪些題型？是模型讀出了身分才造成的，還是別的原因？

**怎麼做**
1. **虛構人物**：12 個屬性、41 個群體。每位人物帶 4 個屬性，每個群體至少 10 人，共 104 位。每位人物有兩版：
   - 揭露版：直接說出身分；
   - 隱性版：只給生活線索。
2. **提問**：13 題日常問題，分成實用指引、查資訊、寫作、技術協助，另有高風險題。每段兩輪，每格重複 4 次。
3. **評審**：另一個評審模型判定每則回覆的類型、免責聲明、轉介是否恰當、數字與答案是否正確。評審看不到人物身分。
4. **分析**：每個指標做一個多元迴歸，同時估所有屬性的效果。以人物為單位做置換檢定與拔靴法，多重比較加以校正。每個「屬性 × 指標」給出判定與整體結論。
5. **確認與檢核**：
   - 篩檢標出的格子，再用只差一個屬性的配對人物確認；
   - 另外檢查模型是否真的讀出了身分（操弄檢核）；
   - 再用成對比較看回覆的語氣與刻板印象（詮釋層）。
6. **對照實驗**：身分放在提問前或系統提示、繁體或簡體、標點寫法、題目同義改寫，看結果會不會跟著變。

**目前進度（2026-10-17）**
- 受測模型：google/gemma-4-e4b（本機，LM Studio）。
- 第 1–3 次重複已完成（每次 2,704 段），第 4 次正在 M1 Pro 上跑。
- 評審模型：openai/gpt-oss-20b。

**原則**
- 不使用中國開發的模型、框架、嵌入與套件。
- 研究資料（人物、回覆、報告）不上傳。
- 13 題裡有 4 題是保留題，不公開（見下一節）；公開的程式只帶其餘 9 題，另有簡繁對照用的 S1、S2。
- 企劃書、規格與待決事項在作者的內部文件〈離均差 文件總覽〉。

## 保留題

為了避免題目被拿去訓練模型、讓評量失準，P2、P3、I3、W3 四題的全文、追問、同義改寫與標準答案不公開。

- **雜湊承諾**：保留題檔（`config/private/reserved_questions.json`）定稿版的 SHA-256 是
  `fa447e9f3ccf10c357b53f1f39eac4e6ffa277edd10f25850b705a137b738b63`
  （換行一律當成 `\n` 算）。日後公開或提供這個檔時，任何人都可以算一次，確認和這裡寫的相同、題目沒有事後修改。
- **申請取得**：研究或查核用途，請以書面向作者申請，並書面承諾：(1) 不用於訓練或微調任何模型；(2) 不公開、不轉給第三方。
- **沒有這個檔也能跑**：程式只用公開題，報告開頭會寫明用了哪些題（`題目：只有公開的 11 題…`）。
  已經有保留題回覆的資料，沒有這個檔時程式會停下，不會默默少評、少分析幾題。

## 下載與安裝

1. 安裝 Python（建議 3.13；3.9 也能跑，但詞彙層用的 `lmstudio` 套件要 3.10 以上）與 Git，以及 [LM Studio](https://lmstudio.ai/)（跑本機模型用）。Windows 安裝 Python 時要勾 Add python.exe to PATH。
2. 建一個上層資料夾，在裡面取得程式：
   ```bash
   git clone https://github.com/<帳號>/<儲存庫>.git ljch
   cd ljch
   python -m pip install -r requirements.txt
   python -m tests
   python lijuncha.py doctor
   ```
   `doctor` 檢查這台能不能跑：程式是不是最新版、套件、`out/` 寫不寫得進去、姓名資料、LM Studio 載了哪些模型。換電腦或更新程式後先跑一次。
   沒有 Git 的話，可以解開作者給的 zip，放到同樣的位置。
3. 姓名資料：一般與外籍的姓名檔隨程式提供；原住民族名（`names_indigenous.jsonl`）因授權還不能放進儲存庫，要另外向作者取得，放進 `config/names/`（見該資料夾的 README）。
4. 資料會自動寫到 `ljch` 旁邊的 `out/`，不用先建立。換電腦時把舊的 `out/` 整個搬到同樣的位置（`out/` 底下直接是各設定資料夾，不要多一層）。

**更新程式**：`git pull`，或解開新的 zip 換掉程式資料夾；`out/` 不要動。更新後再跑一次 `python -m tests` 與 `python lijuncha.py doctor`。

**Windows 寫不進檔案**（`Bad file descriptor`、git 的 `cannot open .git/...`）：可能是 Windows 安全性的「受控資料夾存取」擋住 Python 與 Git 寫進「文件」等資料夾。
- 到「Windows 安全性 → 病毒與威脅防護 → 勒索軟體防護」，把 `python.exe` 和 `git.exe` 加進允許的應用程式；
- 或把整個上層資料夾搬到不受保護的位置，例如 `C:\lijuncha\`。

## 操作指引

只要記四大步驟。每一步結束時，程式會印出「下一步」要執行的指令；不確定做到哪裡，就執行 `status`。

| 步驟 | 指令 | LM Studio 要載入 |
| --- | --- | --- |
| 1. 產生人物 | `python lijuncha.py gen`（第一次才需要） | 不用 |
| 2. 生成回覆 | `python lijuncha.py run --model <受測模型>` | 受測模型，例如 google/gemma-4-e4b |
| 3. 評審 | `python lijuncha.py judge` | 評審模型 openai/gpt-oss-20b（設定檔 `judge.model`） |
| 4. 分析 | `python lijuncha.py analyze` | 不用 |

- 開始前：LM Studio 的 Developer 分頁按 **Start Server**；載入模型的設定照設定檔 `generation.lmstudio_load`。
- 每一步都可以隨時 Ctrl+C，再執行同一行就接著做，做過的會略過。
- 第 2、3 步跑著的時候，程式會讓電腦不進入睡眠（螢幕照樣可以關，結束就恢復）；筆電闔上蓋子仍會睡。
- 第 2 步跑完時，被截斷的回覆會自動續寫，也會順便做操弄檢核的提問；第 3 步評完時會自動把推測歸類。
- `out/` 只有一組設定時不用 `--run`；有好幾組時，下一步的提示會寫好 `--run`。
- 報告在 `out/analysis/<設定資料夾>/<範圍>/report.md`，英文版是 `report_en.md`。
- 只想試流程：每一步都加 `--mock`（假回覆，不用模型）。
- 指令總表：`python lijuncha.py -h`；每個指令加 `-h` 看全部選項。

### 用幾台電腦分工

只能依重複次數分工，每次重複只來自一台機器；評審全部在同一台做。

```bash
python lijuncha.py run --model <受測模型> --rep-range 4 --machine m1pro   # 這台只跑第 4 次
python lijuncha.py merge <另一台的 out 資料夾>                            # 跑完後，在主資料那台合併
```

- 各機器要用同一個模型檔、同一組生成設定。模型檔或載入設定不同時，程式會停下。
- 忘了 `--rep-range`、又會跑到別台負責的重複時，程式會先問。
- 合併時，兩台都跑了同一次重複，就整次用跑滿的那一邊；分不出來時讓你選。
- 評審在別台評過的，這台會自動重評（規格：評審全部在同一台）；舊結果保留，報告列出兩台的不一致比例。
- LM Studio 的 LM Link 會把別台的模型接過來：這台執行 `run` 時，請求會送去別台、紀錄卻記成這台。`doctor` 與 `run` 找不到模型檔時會提醒。

### 換電腦

```bash
python lijuncha.py pack all            # 舊電腦：打成 out/packages/ 裡的一個 zip
python lijuncha.py unpack <zip 檔>      # 新電腦：驗證每個檔的 SHA-256，再解到這台的 out/
python lijuncha.py doctor              # 新電腦：檢查這台能不能跑
```

- `unpack` 發現這台已經有同名但內容不同的檔就停下，那是兩台各跑了一部分，要用 `merge`。
- 用隨身碟、區網或自己的雲端硬碟傳 zip 都可以；不要讓雲端同步正在寫入的 `out/`。

### 被截斷的回覆
- 回覆碰到每則上限（目前 3,700 token）被截斷時，續寫到每則上限加 10%（4,070）。續寫完才能評審。
- 第 2 步跑完時會自動做。要手動時，先 `continue --check` 試一則，再執行 `continue`。

### 其他功能（進階）

| 功能 | 指令 |
| --- | --- |
| 配對確認 | `confirm --run <篩檢的設定資料夾> --attr <屬性>`，再照提示跑確認用的人物 |
| 人工標註 | `annotate`：從全部回覆簡單隨機抽 200 則給人標，填好後再執行一次算一致程度 |
| 評審速度（全面重評前） | `judge --sample`：只評人工標註的 200 則，量這台的速度 |
| 第二評審 | `judge --second`：只評人工標註的樣本，結果另存、不進分析（模型是設定檔 `judge.second_model`） |
| token 機率實測 | `python tools/logprobs_check.py --model <受測模型>` |
| 詮釋層 | `interpret --run <確認的資料夾>`，`annotate --interp` 抽 10% 給人重評 |
| 身分位置比較 | `run --identity both` |
| 簡繁對照 | `script`，校對後 `run --module script` |
| 標點、同義改寫對照 | `run --module punctuation`、`run --module paraphrase` |
| 詞彙層 | `analyze --embed`（要先在 LM Studio 載入嵌入模型，另需 `lmstudio` 套件） |
| 打包產出 | `pack <步驟>` |
| 檢定力：看得出多小的差異、多跑重複或多加人物各有多少幫助 | `python tools/power.py --run <設定資料夾>`（加 `--judged` 用評審的指標） |

## 資料夾與資料安全

`out/` 在程式資料夾旁邊，不進版本控制：

```
上層資料夾/
├─ ljch/        ← 程式（這個儲存庫）
└─ out/         ← 人物、回覆、評審、報告、備份
```

- 金鑰只從環境變數讀；`out/`、`tuning.json`、`hashes.json`、`config/private/`（保留題）、`config/names/names_indigenous.jsonl`（授權確認前）不上傳。
- 公開版用 `python tools/public_snapshot.py` 產生：只帶白名單裡的檔，並檢查沒有保留題、原住民族名、研究資料、私人字串與金鑰。
- 提交前先 `git status`，確認沒有 `out`、`.docx`、`.pdf`、`.env`。
- 提交紀錄的 email 會公開：`git config user.email` 用 GitHub 給的 `…@users.noreply.github.com`。

## 更多文件

- [DEVELOPMENT.md](DEVELOPMENT.md)：改程式的人看（測試、檔案結構、提示排版規則、名詞對照）。
- [CHANGELOG.md](CHANGELOG.md)：每一版改了什麼。
- 完整操作說明：Claude Docs 的〈離均差 執行流程〉、〈離均差原型：執行指引〉。
