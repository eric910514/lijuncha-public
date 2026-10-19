"""指令列：定義每個指令有哪些選項，並把工作交給對應的模組。"""
import argparse
import os

from . import layout
from .analyze import cmd_analyze
from .annotate import cmd_annotate
from .confirm import cmd_confirm
from .doctor import cmd_doctor
from .interpret import cmd_interpret
from .judge import cmd_judge
from .maintain import cmd_calibrate, cmd_continue, cmd_merge
from .pack import STEPS, cmd_pack, cmd_unpack
from .personas import cmd_gen
from .probe import cmd_probe
from .script import cmd_script
from .runner import cmd_check, cmd_run, cmd_status

LOCAL = "http://localhost:1234/v1"
# 預設位置以程式自己的資料夾為準，不管在哪裡下指令都一樣：
#   上層資料夾/
#   ├─ out/        ← 資料（人物、回覆、報告），和程式分開放，更新程式或上傳 GitHub 時不會動到
#   └─ 程式資料夾/  ← lijuncha.py、config.json、ljc/、tests/
PROG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROG_DIR, "config", "config.json")
DEFAULT_OUT = os.path.join(os.path.dirname(PROG_DIR), "out")


# 指令總表（python lijuncha.py -h）：四大步驟照順序做；進階指令在各步驟結束時會提示什麼時候用
COMMANDS = [
    ("四大步驟（照順序）", [
        ("gen", "1. 產生人物（第一次才需要）"),
        ("run", "2. 生成回覆：run --model <受測模型>（被截斷的會自動續寫、操弄檢核的提問也會自動做）"),
        ("judge", "3. 評審：LM Studio 換成評審模型後執行（模型預設是設定檔 judge.model；操弄檢核的歸類會自動做）"),
        ("analyze", "4. 分析：產生報告")]),
    ("隨時可用", [
        ("status", "看進度，並告訴你下一步要執行什麼"),
        ("doctor", "換電腦或更新程式後的健檢")]),
    ("進階（各步驟結束時會提示什麼時候用）", [
        ("merge", "合併另一台機器跑的回覆"),
        ("pack", "把某一步的產出打成 zip（換電腦用 pack all）"),
        ("unpack", "換電腦：驗證 pack 的 zip，再解到這台的 out"),
        ("confirm", "配對確認：分析標出的屬性，用只差一個屬性的人物再確認"),
        ("annotate", "人工標註：從全部回覆簡單隨機抽 200 則給人標，算人與評審的一致程度"),
        ("interpret", "詮釋層：配對確認的回覆成對比較"),
        ("continue", "手動續寫被截斷的回覆（run 跑完時會自動做）"),
        ("probe", "手動做操弄檢核（run、judge 跑完時會自動做）"),
        ("script", "簡繁對照的固定資料"),
        ("calibrate", "用試跑資料建議每則上限"),
        ("check", "送一個問題，看模型的原始回應")])]


def command_table():
    w = max(len(n) for _, items in COMMANDS for n, _ in items)
    out = []
    for title, items in COMMANDS:
        out.append(title + "：")
        out += [f"  {n.ljust(w)}  {t}" for n, t in items]
    out.append("每個指令加 -h 看全部選項，例如 python lijuncha.py run -h")
    return "\n".join(out)


def build_parser():
    ap = argparse.ArgumentParser(prog="lijuncha.py", description="離均差原型：量測語言模型的回覆是否因使用者身分而不同",
                                 epilog=command_table(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="設定檔（預設是程式資料夾裡的 config/config.json）")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="資料夾：人物、回覆、報告都放這裡。預設是和程式資料夾並排的 out；"
                         "自己給相對路徑時，從下指令的位置算起")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lang", choices=["zh", "en"], default=None,
                    help="介面語言：zh（預設）或 en；也可以設環境變數 LJC_LANG=en。只換程式的提示與報告，不換送給模型的內容")
    sub = ap.add_subparsers(dest="cmd", metavar="<指令>")
    add = lambda name, text: sub.add_parser(name, description=text, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = add("gen", "產生人物")
    g.add_argument("--n", type=int, default=None,
                   help="人數（每人有揭露、隱性兩版）。預設看設定檔：外籍三群暫停時 104，恢復後（或 --draft）108")
    g.add_argument("--extend", action="store_true", help="接續加人：既有人物不變，新人物接在後面，直到總數達到 --n")
    g.add_argument("--mention-k", type=int, default=0, help="每人帶幾個屬性（預設看設定檔）")
    g.add_argument("--min-per-group", type=int, default=None, help="每個群體至少幾人（預設看設定檔，v1 為 10）")
    g.add_argument("--draft", action="store_true", help="設定檔還有【待補】內容時，仍產生試流程用的 -draft 人物")
    g.add_argument("--check", action="store_true", help="不產生新人物，只依目前的設定檔檢查現有人物有沒有矛盾")
    g.add_argument("--force", action="store_true", help="已有回覆資料時仍重新產生人物（舊回覆會改名備份）")
    g.set_defaults(func=cmd_gen)

    r = add("run", "對模型提問")
    r.add_argument("--base-url", default=LOCAL, help="LM Studio 預設；Ollama 用 http://localhost:11434/v1")
    r.add_argument("--provider", default=None,
                   help="雲端模型：openai、gemini、anthropic（設定檔 cloud.providers），或 custom 加 --base-url；"
                        "金鑰從環境變數讀")
    r.add_argument("--key-env", default="", help="放金鑰的環境變數名稱（預設看設定檔 cloud.providers.<名稱>.key_env）")
    r.add_argument("--no-probe", action="store_true",
                   help="跑完時不順便做模型操弄檢核的提問（預設會做：問受測模型「你推測我是什麼背景的人」）")
    r.add_argument("--no-continue", action="store_true",
                   help="跑完時不順便續寫（預設：還有截斷的回覆沒續寫時，接著續寫到每則上限加 10%%）")
    r.add_argument("--allow-env-change", action="store_true",
                   help="換機器接續時，模型檔雜湊值或量化方式和先前不同仍然執行（報告會寫明；正式資料不建議）")
    r.add_argument("--model", default="mock", help="LM Studio 裡的模型識別名稱（必填，除非加 --mock）")
    grp = r.add_argument_group("實驗規模")
    grp.add_argument("--reps", type=int, default=4, help="每人每題重複幾次（v1 為 4，可之後加大接續補跑）")
    grp.add_argument("--rounds", type=int, choices=[2, 3], default=2,
                     help="每段幾輪：2＝篩檢（首輪＋具體化追問，預設）；3＝配對確認（再加延伸追問）")
    grp.add_argument("--temperature", type=float, default=None, help="取樣溫度（預設看設定檔 generation，v1 為 0.7）")
    grp.add_argument("--max-tokens", type=int, default=None,
                     help="每則回覆的上限，含模型思考用掉的部分（預設看設定檔 generation，v1 為 3700）")
    grp.add_argument("--identity", choices=["message", "system", "both"], default="message",
                     help="身分放在哪：message=寫在第一句提問前；system=放進系統提示，模擬記憶功能；"
                          "both=兩批交錯一起跑（預設前 24 位人物），跑完自動產生比較報告")
    grp.add_argument("--module", choices=["script", "punctuation", "paraphrase"], default=None,
                     help="對照模組：script=簡繁、punctuation=標點（預設前 24 位人物）、paraphrase=首輪提問換成同義改寫")
    grp.add_argument("--people", type=int, default=None,
                     help="只用前幾位人物（依編號，兩版一起）；0＝全部。預設：主流程全部，簡繁、標點與 --identity both 為 24")
    grp.add_argument("--variants", default="",
                     help="跑人物的哪幾版：explicit、implicit 或 explicit,implicit。預設：簡繁、標點只跑揭露版，其餘兩版")
    grp.add_argument("--draft", action="store_true", help="送出的文字還有【待補】時仍然執行（只能用來檢查流程）")
    grp.add_argument("--personas", default="",
                     help="用哪一版人物（out/personas/ 底下的資料夾名稱）：配對確認用的人物（例如 p1.2-c-health），或把舊版人物（例如 p1.1）的試跑接著跑完")
    grp.add_argument("--preview", action="store_true",
                     help="只印出第一位人物第一題要送出的文字（簡繁模組另寫轉換對照），不連線模型")
    grp = r.add_argument_group("跨機器")
    grp.add_argument("--rep-range", default="",
                     help="只跑第幾次到第幾次重複，例如 3-4。跨機器分工只能依重複序號分，跑完用 merge 合併")
    grp.add_argument("--machine", default="", help="機器代號（預設是環境變數 LJC_MACHINE，沒有就用電腦名稱）")
    grp.add_argument("--model-file", default="", help="受測模型的 GGUF 檔位置（找不到時才需要；用來算雜湊值）")
    grp = r.add_argument_group("這次跑多少")
    grp.add_argument("--minutes", type=float, default=0, help="這次最多跑幾分鐘，到時跑完手上的對話就停")
    grp.add_argument("--limit", type=int, default=0, help="這次最多跑幾段對話")
    grp = r.add_argument_group("並行（同時跑幾段對話；預設自動，通常不用改）")
    grp.add_argument("--workers", default="auto", help="auto 或固定數字")
    grp.add_argument("--start-workers", type=int, default=4, help="自動時從幾段起跑")
    grp.add_argument("--max-workers", type=int, default=16, help="自動時最多幾段")
    grp.add_argument("--retune", action="store_true", help="不沿用上次測得的並行數，重新測速")
    grp.add_argument("--conv-tokens", type=int, default=0, help="一段對話約用多少 token（不給就自動估計）")
    grp.add_argument("--est-chars", type=int, default=3200, help=argparse.SUPPRESS)
    r.add_argument("--mock", action="store_true", help="不接模型，用內建的假回覆")
    r.add_argument("--mock-delay", type=float, default=0, help=argparse.SUPPRESS)
    r.add_argument("--retune-gap", type=float, default=0, help=argparse.SUPPRESS)
    r.add_argument("--explore-gap", type=float, default=0, help=argparse.SUPPRESS)   # 秒；負數＝不定期往上試
    r.set_defaults(func=cmd_run)

    st = add("status", "顯示目前進度")
    st.add_argument("--reps", type=int, default=None, help="每格幾次（預設看 run_meta.json 記的計畫，沒有就是 4）")
    st.set_defaults(func=cmd_status)

    dr = add("doctor", "換電腦或更新程式後，檢查這台能不能跑（版本、套件、out 能不能寫入、姓名資料、LM Studio）")
    dr.add_argument("--base-url", default=LOCAL)
    dr.set_defaults(func=cmd_doctor)

    c = add("check", "送一個問題並印出原始回應")
    c.add_argument("--base-url", default=LOCAL)
    c.add_argument("--model", required=True)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--max-tokens", type=int, default=2000)
    c.set_defaults(func=cmd_check)

    j = add("judge", "用評審模型逐則判定回覆（run 全部結束、換上評審模型後執行）")
    j.add_argument("--base-url", default=LOCAL)
    j.add_argument("--model", default="",
                   help="評審模型在 LM Studio 的識別名稱；不給就用設定檔 judge.model（第二評審用 judge.second_model）")
    j.add_argument("--run", default="", help="只評資料夾名稱含有這段文字的設定；預設全部")
    j.add_argument("--minutes", type=float, default=0, help="這次最多評幾分鐘，到時評完手上的就停")
    j.add_argument("--limit", type=int, default=0, help="這次最多評幾則")
    j.add_argument("--second", action="store_true",
                   help="當作第二評審另存一份（judgments__second.jsonl），並印出兩個評審的不一致比例。"
                        "預設只評人工標註樣本與 10%% 隨機抽樣，模型預設是設定檔 judge.second_model")
    j.add_argument("--all", action="store_true", help="第二評審評全部回覆（不只抽樣）")
    j.add_argument("--keep-other-machines", action="store_true",
                   help="先前在別台機器評過的保留、不重評（預設會在這台重評：規格要求評審全部在同一台機器）")
    j.add_argument("--rejudge-other-machines", action="store_true", help=argparse.SUPPRESS)   # v0.10 起是預設
    j.add_argument("--tag", default="", help="另存成 judgments__<tag>.jsonl，例如比較評審思考開或關時用")
    j.add_argument("--people", type=int, default=0, help="只評前幾位人物（依編號，和 run --people 相同）；0＝全部")
    j.add_argument("--sample", action="store_true",
                   help="只評人工標註的抽樣（annotate 產生的那批）；第二評審、評審思考開或關的比較都只評這批")
    j.add_argument("--mock", action="store_true", help="不接模型，用關鍵詞假評審（只用來測流程）")
    j.add_argument("--machine", default="", help="機器代號（評審要全部在同一台機器執行）")
    jg = j.add_argument_group("並行（同時送幾則；預設自動，規則和 run 相同，通常不用改）")
    jg.add_argument("--workers", default="auto", help="auto 或固定數字")
    jg.add_argument("--start-workers", type=int, default=4, help="自動時從幾段起跑")
    jg.add_argument("--max-workers", type=int, default=16,
                    help="自動時最多幾段（要配合 LM Studio 的 Max Concurrent Predictions）")
    jg.add_argument("--retune", action="store_true", help="不沿用上次測得的並行數，重新測速")
    j.add_argument("--retune-gap", type=float, default=0, help=argparse.SUPPRESS)
    j.add_argument("--explore-gap", type=float, default=0, help=argparse.SUPPRESS)
    j.set_defaults(func=cmd_judge)

    m = add("merge", "合併另一台機器跑的回覆（依段的代號去重，檢查設定與模型檔一致）")
    m.add_argument("source", help="另一台機器的 out 資料夾，或其中一組設定的資料夾")
    m.add_argument("--run", default="", help="只合併名稱含有這段文字的設定")
    m.add_argument("--force", action="store_true", help="模型檔雜湊值不同仍然合併（報告會寫明）")
    m.add_argument("--prefer", choices=["here", "source"], default="",
                   help="兩邊都跑了同一次重複、又分不出哪邊完整時，整次用哪一邊（預設會問；不在終端機執行時要給）")
    m.set_defaults(func=cmd_merge)

    cb = add("calibrate", "用試跑資料建議每則上限與所需上下文（只印出建議，不改設定檔）")
    cb.add_argument("--run", default="", help="只看名稱含有這段文字的設定")
    cb.set_defaults(func=cmd_calibrate)

    ct = add("continue", "續寫被截斷的回覆：總上限一律是每則上限加 10%%；第 1 輪續寫後重新生成後面各輪")
    ct.add_argument("--to", type=int, default=None,
                    help="總上限（每則，含思考）；預設是每則上限加 10%%（3,700 → 4,070），只能調高，所有被截斷的回覆都用同一個值")
    ct.add_argument("--run", default="", help="只處理名稱含有這段文字的設定")
    ct.add_argument("--base-url", default=LOCAL)
    ct.add_argument("--model", default="", help="要和產生回覆的模型相同（預設就用資料夾記的模型）")
    ct.add_argument("--workers", type=int, default=2, help="同時處理幾段")
    ct.add_argument("--check", action="store_true", help="只試續寫一則並印出接縫，不存檔（先確認伺服器支援接續生成）")
    ct.add_argument("--machine", default="")
    ct.add_argument("--mock", action="store_true")
    ct.set_defaults(func=cmd_continue)

    z = add("analyze", "統計並輸出報告")
    z.add_argument("--base-url", default=LOCAL)
    z.add_argument("--run", default="", help="只分析資料夾名稱含有這段文字的設定，例如 --run gemma；預設全部")
    z.add_argument("--embed", action="store_true",
                   help="計算詞彙距離，用設定檔 embedding.models 的兩個嵌入模型（要先在 LM Studio 載入）")
    z.add_argument("--embed-model", default="", help="直接指定嵌入模型（可用逗號分隔多個），優先於 --embed")
    z.add_argument("--embed-chunk", type=int, default=300, help="每則回覆切成幾個字一段再嵌入")
    z.add_argument("--people", type=int, default=0,
                   help="只分析前幾位人物（依編號）；0＝全部。分析只收每題都跑滿同樣次數的人物，見 --reps")
    z.add_argument("--reps", type=int, default=0,
                   help="只用前幾次重複，只收每題都跑滿這麼多次的人物；0＝自動挑「人物數 × 次數」最多的組合")
    z.add_argument("--perms", type=int, default=2000, help="置換檢定的次數")
    z.add_argument("--boots", type=int, default=1000, help="拔靴法的次數")
    z.add_argument("--mock", action="store_true")
    z.set_defaults(func=cmd_analyze)

    cf = add("confirm", "配對確認：取基底人物、只翻一個屬性，產生確認用的人物（之後照常 run、judge、analyze）")
    cf.add_argument("--run", required=True, help="篩檢的設定資料夾（名稱含有這段文字，要剛好一個）")
    cf.add_argument("--attr", required=True, help="要確認的屬性，例如 health 或 健康")
    cf.add_argument("--bases", type=int, default=10, help="基底人物幾位（一半取品質指數最好的三分之一、一半取最差的三分之一）")
    cf.add_argument("--analysis", default="", help="用篩檢的哪一份分析挑基底人物（analysis 底下的範圍名稱，例如 104p_4rep）；預設最新")
    cf.add_argument("--draft", action="store_true", help="暫停的群體也做（外籍三群用測試名字，只能試流程）")
    cf.add_argument("--force", action="store_true", help="已經產生過時重新產生（已跑的回覆會對不上）")
    cf.set_defaults(func=cmd_confirm)

    ip = add("interpret", "詮釋層：同題兩群的回覆遮掉身分字眼後成對送評審，判哪一則較差（五個維度與轉介是否合理）")
    ip.add_argument("--run", default="", help="設定資料夾（名稱含有這段文字，要剛好一個）")
    ip.add_argument("--base-url", default=LOCAL)
    ip.add_argument("--model", default="", help="評審模型（預設是設定檔 judge.model；加 --second 時是 judge.second_model）")
    ip.add_argument("--mock", action="store_true", help="不接模型，用假評審（只測流程）")
    ip.add_argument("--second", action="store_true", help="第二評審：只評人工重評的樣本與 10%% 隨機抽樣，另存一份")
    ip.add_argument("--all", action="store_true", help="第二評審評全部配對")
    ip.add_argument("--reps", type=int, default=None,
                    help="用前幾次重複（0＝全部）；預設篩檢 1 次（設定檔 interpret.reps_screening）、配對確認全部")
    ip.add_argument("--turns", default="", help="用哪幾輪，例如 1 或 1,2；預設看設定檔 interpret.turns（v1 為第 1 輪）")
    ip.add_argument("--limit", type=int, default=0, help="這次最多評幾對（試跑用）；0＝全部")
    ip.add_argument("--workers", type=int, default=4, help="同時送幾對（要配合 LM Studio 的 Max Concurrent Predictions）")
    ip.add_argument("--perms", type=int, default=2000, help="置換次數")
    ip.add_argument("--boots", type=int, default=1000, help="拔靴次數")
    ip.set_defaults(func=cmd_interpret)

    pb = add("probe", "模型操弄檢核：問受測模型「你推測我是什麼背景」，再用評審歸類（--classify），算隱性線索的讀出率")
    pb.add_argument("--run", default="", help="設定資料夾（名稱含有這段文字，要剛好一個）")
    pb.add_argument("--base-url", default=LOCAL)
    pb.add_argument("--model", default="", help="受測模型（預設是這組資料的模型，也只能是它）")
    pb.add_argument("--provider", default=None, help="受測的是雲端模型時：同 run --provider（只用在第一步）")
    pb.add_argument("--key-env", default="", help="同 run --key-env")
    pb.add_argument("--classify", action="store_true", help="第二步：用評審模型把推測歸到各群，寫出讀出率")
    pb.add_argument("--judge-model", default="", help="歸類用的評審模型（預設是設定檔 judge.model）")
    pb.add_argument("--variant", choices=["implicit", "both"], default="implicit",
                    help="問哪一版：implicit＝只問隱性版（預設）；both＝兩版都問（揭露版用來檢查歸類）")
    pb.add_argument("--reps", type=int, default=None, help="每位人物版本問幾次（預設和這組資料的重複次數一樣，每次重複各問一次）")
    pb.add_argument("--limit", type=int, default=0, help="這次最多問幾次（試跑用）")
    pb.add_argument("--workers", type=int, default=4, help="同時送幾則")
    pb.add_argument("--mock", action="store_true", help="不接模型（只測流程）")
    pb.set_defaults(func=cmd_probe)

    sc = add("script", "簡繁模組：提示文字只轉一次，寫出 script_simp.json 與 script_review.md 供校對（run --module script 之前）")
    sc.add_argument("--people", type=int, default=None, help="轉換前幾位人物的介紹（預設看設定檔 modules.script.people，和 run 一樣）")
    sc.add_argument("--variants", default=None, help="人物的哪幾版（預設看設定檔 modules.script.variants，只有揭露版）")
    sc.add_argument("--personas", default="", help="轉換哪一版人物（預設目前設定檔的版本）")
    sc.add_argument("--reset", action="store_true", help="重新轉換，捨棄人工改過的內容")
    sc.set_defaults(func=cmd_script)

    pk = add("pack", "把某一步的產出打成 zip（放在 out/packages/），直接傳給別人看")
    pk.add_argument("step", choices=list(STEPS), help="；".join(f"{k}＝{v}" for k, v in STEPS.items()))
    pk.add_argument("--run", default="", help="只打包名稱含有這段文字的設定（例如 p1.2）；預設全部")
    pk.add_argument("--mock", action="store_true", help="打包假資料的資料夾")
    pk.set_defaults(func=cmd_pack)

    up = add("unpack", "換電腦：驗證 pack 的 zip（每個檔的 SHA-256），再解到這台的 out/；同名檔內容不同就停下（要用 merge）")
    up.add_argument("zip", help="pack 產生的 zip 檔")
    up.set_defaults(func=cmd_unpack)

    an = add("annotate", "行為層人工標註：抽樣給人標（第一次），填好後算人與評審的一致程度")
    an.add_argument("--run", default="", help="只處理資料夾名稱含有這段文字的設定；預設全部（對照模組除外）")
    an.add_argument("--n", type=int, default=200, help="主樣本抽幾則（從全部回覆簡單隨機抽，作者 10-19 定 200；只在第一次抽樣時有用）")
    an.add_argument("--mock", action="store_true", help="只處理假資料的資料夾")
    an.add_argument("--interp", action="store_true",
                    help="詮釋層：抽 10%% 已評的配對給人重評（annotation/interpretation/），填好後算一致程度")
    an.set_defaults(func=cmd_annotate)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return
    from . import i18n
    before = i18n.STATE["lang"]
    if a.lang:
        i18n.set_lang(a.lang)
    with i18n.output():                       # 英文介面：這個指令印出的每一行都經過翻譯
        try:
            if a.cmd != "check":
                print(f"out 資料夾：{os.path.abspath(a.out)}")
                if os.path.isdir(a.out):
                    layout.migrate_legacy(a.out)      # v0.2 以前的回覆放在 out/ 最上層，先依設定分開
            a.func(a)
        except SystemExit as e:
            if isinstance(e.code, str):
                raise SystemExit(i18n.translate(e.code)) from None
            raise
        finally:
            i18n.set_lang(before)
