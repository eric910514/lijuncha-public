"""script 指令：簡繁模組的提示文字只轉換一次（〈簡繁詞表〉v0.3）。

  python lijuncha.py script                 轉換這一版人物要用到的提示文字，寫出 script_simp.json 與 script_review.md
  python lijuncha.py script --people 30     人數和 run --module script 一樣（預設看設定檔 modules.script.people）
  python lijuncha.py script --reset         重新轉換，捨棄人工改過的內容

轉換的文字：系統提示前綴、簡繁模組的題目（含同義改寫）與追問、選到的人物介紹（含姓名與人物句）。
固定資料放在人物資料夾（out/personas/<版本>/script_simp.json，和人物一樣不上傳）：每段有 orig（繁體原文）、
auto（程式轉的）、simp（實際送出的；作者校對時可以直接改）。run --module script 只讀 simp，不在執行時轉換。
再執行一次時，人工改過的段（simp 和 auto 不同）會保留；有新增或轉換結果變了，reviewed 會改回 false。"""
import os
import time

from . import __version__
from .common import read_config, bullets, followup_texts, load_json, save_json, stop
from .personas import load_personas
from .styles import REVIEW, TABLE, ScriptConverter, load_table, module_questions, script_cfg, table_path


def script_texts(cfg, personas):
    """簡繁模組會送出的每一段繁體文字：[(位置, 文字)]，同一段文字只列一次。"""
    out, seen = [], set()

    def add(where, t):
        if t and t not in seen:
            seen.add(t)
            out.append((where, t))
    add("系統提示前綴", cfg.get("identity_system_prefix", ""))
    for q in module_questions(cfg, "script"):
        add(f"{q['id']} 首輪", q["q"])
        add(f"{q['id']} 同義改寫", q.get("q_paraphrase") or "")
        for i, t in enumerate(followup_texts(q, 99)):
            add(f"{q['id']} 追問 {i + 1}", t)
    for p in personas:
        add(f"人物 {p['id']}（{'揭露' if p.get('variant') == 'explicit' else '隱性'}）", p["intro"])
    return out


def build_table(cfg, pdir, personas, reset=False):
    """轉換並寫出固定資料與審閱清單。回傳 (表, 新增段數, 轉換結果改變的段數, 保留的人工修改段數)。"""
    conv = ScriptConverter(cfg)
    old = None if reset else load_table(pdir)
    prev = {t["orig"]: t for t in (old or {}).get("texts", [])}
    texts, added, changed, manual = [], 0, 0, 0
    for where, t in script_texts(cfg, personas):
        auto = conv.convert(t)
        o = prev.pop(t, None)
        if o is None:
            added += 1
            simp = auto
        else:
            if o.get("auto") != auto:
                changed += 1
            edited = o.get("simp") != o.get("auto")
            manual += edited
            simp = o["simp"] if edited else auto
        texts.append({"where": where, "orig": t, "auto": auto, "simp": simp})
    # 這次沒用到、但以前轉過的段照樣留著（例如先前用較多人跑過），放在最後
    texts += list(prev.values())
    sc = script_cfg(cfg)
    reviewed = bool(old and old.get("reviewed")) and not added and not changed
    table = {"_說明": "簡繁模組的固定資料：orig＝繁體原文，auto＝程式轉的，simp＝實際送出的（校對時直接改這一欄）。"
                      "全部看過後把 reviewed 改成 true。由 python lijuncha.py script 產生。",
             "program_version": __version__, "terms_version": sc.get("terms_version"),
             "unihan": conv.t2s.describe(), "built": time.strftime("%Y-%m-%d %H:%M"),
             "reviewed": reviewed, "texts": texts}
    save_json(table_path(pdir), table)
    write_review(cfg, conv, table, os.path.join(pdir, REVIEW))
    return table, added, changed, manual


def write_review(cfg, conv, table, path):
    """審閱清單：逐段轉換前後對照；用到的清單詞；逐字轉換時有多個候選的字另外標出；人工改過的段。"""
    used_count, multi_where, rare_where = {}, {}, {}
    rows = []
    for t in table["texts"]:
        _, used, multi, rare = conv.explain(t["orig"])
        for u in used:
            used_count[u] = used_count.get(u, 0) + 1
        for ch, pick, cands in multi:
            multi_where.setdefault((ch, pick, "、".join(cands)), []).append(t["where"])
        for ch, to in rare:
            rare_where.setdefault((ch, to), []).append(t["where"])
        rows.append((t, multi + [(c, p, None) for c, p in rare]))
    edited = [t for t in table["texts"] if t["simp"] != t["auto"]]
    L = ["# 簡繁轉換審閱", "",
         f"- 詞表版本：{table.get('terms_version') or '未標示'}（〈離均差 簡繁詞表〉）",
         f"- 逐字對照：{table.get('unihan')}",
         f"- 固定資料：{TABLE}（{len(table['texts'])} 段；" + ("已校對" if table.get("reviewed") else "**尚未校對**") + "）",
         "- 處理順序：保留詞與替換詞先換成佔位符 → 其餘逐字轉簡體 → 保留詞只轉字形、替換詞換成大陸用語 → 引號改為“”。",
         "- 執行時直接讀固定資料的 simp 欄，不再轉換。", "",
         "## 怎麼校對", "",
         f"1. 看下面「逐段對照」。某一段要改：直接改 {TABLE} 裡那段的 simp（auto 是程式轉的，不要改）。",
         "2. 是詞表的問題（某個台灣用語要換、某個法律名稱要保留）：改設定檔 modules.script 的 keep／replace，"
         "再執行 python lijuncha.py script；人工改過的段會保留。",
         f"3. 全部看完：把 {TABLE} 的 reviewed 改成 true。", "",
         "## 逐字轉換有多個候選的字", "",
         "Unihan 給了兩個以上的簡體候選，程式取第一個和原字不同的。「著」另有規則：預設轉「着」，設定檔 modules.script.zhu_keep "
         "的念 zhù 的詞（顯著、著作……）保留「著」。意思還是不對時請改 simp。", ""]
    if multi_where:
        L += ["| 原字 | 用的字 | 全部候選 | 出現在 |", "| --- | --- | --- | --- |"]
        for (ch, pick, cands), where in sorted(multi_where.items(), key=lambda x: -len(x[1])):
            L.append(f"| {ch} | {pick} | {cands} | {'、'.join(where[:6])}{'……' if len(where) > 6 else ''}（{len(where)} 段） |")
    else:
        L.append("- 沒有。")
    L += ["", "## 轉成擴充區罕用字的字", "",
          "Unihan 對到的簡體字在 BMP 以外（很多字型顯示不出來，模型也可能讀不懂）。要改用常用字時，加進設定檔 "
          "modules.script.extra_chars（例如 閒→闲）再執行一次。", ""]
    if rare_where:
        L += ["| 原字 | 轉成 | 出現在 |", "| --- | --- | --- |"]
        L += [f"| {ch} | {to} | {'、'.join(w[:6])}{'……' if len(w) > 6 else ''}（{len(w)} 段） |"
              for (ch, to), w in sorted(rare_where.items(), key=lambda x: -len(x[1]))]
    else:
        L.append("- 沒有。")
    L += ["", "## 用到的清單詞", "", "| 原詞 | 結果 | 清單 | 段數 |", "| --- | --- | --- | --- |"]
    L += [f"| {a} | {b} | {k} | {n} |" for (a, b, k), n in sorted(used_count.items(), key=lambda x: (x[0][2], -x[1]))]
    if edited:
        L += ["", "## 人工改過的段", "", "| 位置 | 程式轉的 | 實際送出 |", "| --- | --- | --- |"]
        L += [f"| {t['where']} | {t['auto']} | {t['simp']} |" for t in edited]
    L += ["", "## 逐段對照", "", "| 位置 | 原文 | 轉換後（實際送出） | 多候選字 |", "| --- | --- | --- | --- |"]
    for t, multi in rows:
        L.append(f"| {t['where']} | {t['orig']} | {t['simp']} | {'、'.join(f'{c}→{p}' for c, p, _ in multi)} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def selected_personas(a, cfg, personas):
    from .runner import default_people, default_variants, select_people
    a.module = "script"
    if not hasattr(a, "identity"):
        a.identity = "message"
    people = select_people(personas, default_people(a, cfg))
    variants = default_variants(a, cfg)
    return [p for p in people if p.get("variant") in variants]


def cmd_script(a):
    cfg = read_config(a.config)
    personas, pdir = load_personas(a.out, cfg, a.personas or None)
    if not personas:
        stop(f"找不到人物：{pdir}", "先執行 gen")
    chosen = selected_personas(a, cfg, personas)
    table, added, changed, manual = build_table(cfg, pdir, chosen, reset=a.reset)
    n_p = len({p["id"] for p in chosen})
    print(bullets("── 簡繁固定資料", [
        ("固定資料", f"{table_path(pdir)}（{len(table['texts'])} 段；{n_p} 位人物）"), ("審閱清單", os.path.join(pdir, REVIEW)),
        ("新增", f"{added} 段"), ("轉換結果改變", f"{changed} 段" if changed else None),
        ("保留人工修改", f"{manual} 段" if manual else None),
        ("校對", "已校對" if table["reviewed"] else f"還沒校對：看過審閱清單後，把 {TABLE} 的 reviewed 改成 true")]))
