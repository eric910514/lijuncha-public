"""把分析結果寫成 results.csv 與 report.md。只負責排版，不做計算。"""
import csv

import numpy as np

from .metrics import metric_definitions

FLAG_P = 0.10    # 校正後 p 不超過這個值就列入「值得注意」（規格：q ≤ 0.10）


def _nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _fmt(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def quality_lines(q, thinking=None):
    """截斷率與思考內容的摘要行。"""
    L = [f"- 截斷率：{q['cut_rate']:.1%}（{q['n_cut']}/{q['n_rep']} 則回覆碰到每則上限被截斷；"
         f"含截斷回覆的對話 {q['seg_cut']}/{q['n_seg']} 段）"
         + ("。字數指標會被上限壓平，解讀時要保留" if q["n_cut"] else "")]
    if q.get("n_continued") or q.get("n_regenerated"):
        L.append(f"- 續寫：{q['n_continued']} 則被截斷的回覆從中斷處續寫（總上限 {q['cap_total']} token），"
                 f"{q['n_regenerated']} 則後續輪次用完整的回覆重新生成；兩者都有標記，各群截斷率見「只描述」一節")
    if q["think_share"] is not None:
        want = {True: "設定檔要求開啟", False: "設定檔要求關閉"}.get(thinking, "設定檔未指定")
        L.append(f"- 思考：{want}；伺服器有另外回傳思考內容的回覆 {q['think_share']:.0%}"
                 "（程式無法切換思考開關，要在 LM Studio 設定）")
    return L


def judge_line(info):
    """評審摘要一行。info 是 analyze.judged_for 的摘要。"""
    return (f"評審模型 {'、'.join(info['models']) or '—'}｜提示詞 {info['prompt']}｜已評 {info['n']}/{info['total']} 則"
            f"｜judge_error {info['errors']} 則｜有回傳思考內容 {info['think']} 則")


def write_paired(path, title, settings_line, intro, warns, results, noise, quals, n, perms, boots, lex, spread=None,
                 thinking=None, judge=None):
    """成對比較的報告：對照模組（簡繁、標點、無關擾動）與身分位置比較共用。"""
    L = [f"# {title}", ""]
    L += [f"> ⚠ {w}" for w in warns] + ([""] if warns else [])
    L += [f"- 設定：{settings_line}", f"- 成對比較的人物版本：{n} 筆"]
    if judge:
        L += [f"- {k} 的評審：{judge_line(v)}" for k, v in judge.items()]
    else:
        L.append("- 評審：尚未評審，行為指標用關鍵詞初篩")
    L += [
          f"- 置換次數 {perms}（隨機翻轉正負號）、拔靴次數 {boots}、多重比較以 Benjamini–Hochberg 校正", ""]
    L += intro + ["", "## 資料品質", ""]
    for k, (q, _) in quals.items():
        L += [f"**{k}**", ""] + quality_lines(q, thinking) + [""]
    L += ["## 結果", "",
          "每位人物版本先在每種條件下各算一個分數（題型內平均再以固定權重合併），再看同一人兩種條件的差。",
          "", "| 比較 | 指標 | 人數 | 參考平均 | 對照平均 | 平均差 | 95% 區間 | 校正後 p | 重複取樣的晃動 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        flag = "**" if r["校正後p"] < FLAG_P else ""
        L.append(f"| {r['比較']} | {r['指標']} | {r['人數']} | {_fmt(r['參考平均'])} | {_fmt(r['對照平均'])} | "
                 f"{flag}{_fmt(r['平均差'])}{flag} | [{_fmt(r['區間下界'])}, {_fmt(r['區間上界'])}] | "
                 f"{r['校正後p']:.3f} | {_fmt(noise.get(r['指標']))} |")
    if spread:
        L += ["", "## 各屬性的群體差異（描述）", "",
              "每個屬性在兩種條件下，偏離最大的群體與它的離均差。這裡只做描述，"
              "沒有控制其他屬性；控制其他屬性的同時估計見各批主流程的報告。", "",
              "| 屬性 | 指標 | message：偏離最大 | system：偏離最大 |", "| --- | --- | --- | --- |"]
        for r in spread:
            L.append(f"| {r['屬性']} | {r['指標']} | {r['message'][0]} {_fmt(r['message'][1])} | "
                     f"{r['system'][0]} {_fmt(r['system'][1])} |")
    L += ["", "## 怎麼讀", "",
          "- **平均差**：對照條件減參考條件，同一人比較後再平均。粗體表示校正後 p < 0.10。",
          "- **95% 區間**：重抽人物得到的範圍；不含 0 才表示方向穩定。",
          "- **重複取樣的晃動**：參考條件下，同一人重問一次分數大約會變動多少。平均差沒有明顯大於它時，多半只是晃動。",
          "", "## 指標定義", ""] + metric_definitions(lex, bool(judge))
    save_report(path, L)


def save_report(path, lines):
    """寫出 report.md。介面語言是英文（--lang en）時，標題、欄位與固定句子換成英文；數字與群體資料照舊。"""
    from .i18n import translate_doc
    with open(path, "w", encoding="utf-8") as f:
        f.write(translate_doc("\n".join(lines)) + "\n")


def write_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        if not results:          # 人數太少、每個指標都沒有估計時：留一個空檔，報告照樣寫
            return
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
