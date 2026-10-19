"""配對確認（F 批）：篩檢標出可疑的屬性後，取一批基底人物，只翻這一個屬性、其餘文字逐字相同，再問同一套題。

  confirm --run <篩檢的設定資料夾> --attr health      產生確認用的人物（personas/<人物版本>-c-health/）
  run --personas <上面的版本> --rounds 3 --reps 10    照常提問（三輪、10 次）
  judge、analyze                                     analyze 認得確認用的人物，改做成對分析

- 基底人物：一半從篩檢的回覆品質指數最好的三分之一抽，一半從最差的三分之一抽（企劃書「測量設計」）。
  基底人物本來就沒有帶受測屬性（未提及），所以品質指數不含這個屬性的影響（「計算時排除受測屬性」）。
- 每位基底人物產生「基底（未提及）」與受測屬性每個群體各一位；用揭露版，其餘句子逐字相同。
  性別、原住民族、外籍身分會一併換姓名（姓名本身就會透露這個屬性），報告寫明。
- 分析：同一位基底人物的「群體版 − 基底版」，以基底人物為單位做成對檢定（正負號翻轉置換、重抽基底人物）。
  篩檢標出的格子，確認也成立（同方向、校正後 p ≤ 0.10、區間不含 0）才算「確認成立」，否則「未確認」。"""
import csv
import json
import os
import random
import time

import numpy as np

from . import __version__, layout, stats
from .common import read_config, NONE, bullets, cmd_hint, kv_block, load_json, next_steps, read_jsonl, save_json, stop, warn

META = "personas_meta.json"
Q = 0.10


def find_attr(cfg, name):
    for k, v in cfg["attributes"].items():
        if name in (k, v["label"]):
            return k
    stop(f"找不到屬性「{name}」", "可用：" + "、".join(f"{k}（{v['label']}）" for k, v in cfg["attributes"].items()))


def latest_analysis(run_dir, label=""):
    """篩檢最近一次的分析資料夾（含 quality.csv）；label 指定範圍，例如 104p_4rep。"""
    base = os.path.join(os.path.dirname(os.path.abspath(run_dir)), "analysis", os.path.basename(run_dir))
    if not os.path.isdir(base):
        return None
    cands = [os.path.join(base, x) for x in os.listdir(base) if (not label or x == label)
             and os.path.isfile(os.path.join(base, x, "quality.csv"))]
    return max(cands, key=lambda p: os.path.getmtime(os.path.join(p, "quality.csv"))) if cands else None


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def cmd_confirm(a):
    from .names import build_pools, name_kind, pick
    from .personas import build_texts, find_conflicts, personas_for_run
    cfg = read_config(a.config)
    dirs = [d for d in layout.run_dirs(a.out) if a.run in os.path.basename(d) and not layout.read_settings(d).get("module")
            and "-c-" not in str(layout.read_settings(d).get("personas_version"))]      # 不含確認用的資料夾
    dirs = layout.pick_one(dirs, a.run)
    if len(dirs) != 1:
        stop("--run 要剛好對到一個篩檢的設定資料夾", "對到：" + ("、".join(os.path.basename(d) for d in dirs) or "沒有"))
    run_dir = dirs[0]
    key = find_attr(cfg, a.attr)
    at = cfg["attributes"][key]
    adir = latest_analysis(run_dir, a.analysis)
    if not adir:
        stop("找不到篩檢的分析結果（quality.csv）", "先對這組資料執行 analyze")
    quality = {r["人物"]: float(r["品質指數"]) for r in read_csv(os.path.join(adir, "quality.csv"))}
    plist, pdir = personas_for_run(a.out, cfg, run_dir)
    exp = {p["id"]: p for p in plist if p.get("variant") == "explicit"}
    paused = set() if a.draft else {s.split(":")[1] for s in cfg.get("paused_groups") or [] if s.startswith(key + ":")}
    groups = [lv for lv in at["levels"] if lv not in paused]
    # 品質指數最好、最差的三分之一（全部人物一起排），基底人物從中挑沒帶受測屬性、翻成每個群體都不矛盾的
    ranked = sorted((v, p) for p, v in quality.items() if p in exp)
    third = max(1, len(ranked) // 3)
    pools = {"不利": [p for _, p in ranked[:third]], "有利": [p for _, p in ranked[-third:]]}
    ok = lambda p: (exp[p]["true_levels"].get(key, NONE) == NONE
                    and not any(find_conflicts(cfg, dict(exp[p]["true_levels"], **{key: g})) for g in groups))
    rng = random.Random(a.seed)
    need = {"有利": (a.bases + 1) // 2, "不利": a.bases // 2}
    name_pools = None
    used = {}
    out, chosen, skipped = [], [], []
    new_ver = f"{os.path.basename(pdir)}-c-{key}"
    for pool in ("有利", "不利"):
        cand = [p for p in pools[pool] if ok(p)]
        rng.shuffle(cand)
        for p in cand:
            if sum(1 for c in chosen if c["pool"] == pool) >= need[pool]:
                break
            rec = exp[p]
            base_levels = dict(rec["true_levels"])
            fixed = {"tribe": rec.get("tribe"), "origin_language": rec.get("origin_language")}
            name = {"name": rec["name"], "gender_lean": rec.get("name_gender"), "origin_country": rec.get("origin_country"),
                    "name_kind": rec.get("name_kind"), "given_name": rec.get("given_name")}
            again = build_texts(cfg, base_levels, name, used, fixed)[0]["explicit"]
            if again != rec["intro"]:
                skipped.append(f"{p}（重組的文字和原本不同，可能是設定檔的人物句改過）")
                continue
            made = []
            for g in [NONE] + groups:
                lv = dict(base_levels, **{key: g})
                nm, note = name, ""
                kind = name_kind(cfg, lv)
                lean = lv.get("gender") if lv.get("gender") in ("female", "male") else name["gender_lean"]
                if kind != (rec.get("name_kind") or "general") or lean != name["gender_lean"]:
                    if name_pools is None:
                        name_pools = build_pools(cfg, a.config, allow_test=a.draft)
                    nm = pick(name_pools, kind, lean, rng, used)
                    if not nm:
                        break
                    note = "姓名一併換掉（姓名本身會透露這個屬性）"
                text = build_texts(cfg, lv, nm, used, fixed)[0]["explicit"]
                gid = "base" if g == NONE else g
                made.append({"id": f"{p}-{gid}", "variant": "explicit", "levels": lv, "true_levels": lv,
                             "name": nm["name"], "name_gender": nm.get("gender_lean"), "name_source": nm.get("source"),
                             "intro": text, "chars": len(text), "personas_version": new_ver,
                             "confirm": {"attr": key, "group": g, "base": p, "pool": pool,
                                         "quality": quality.get(p), "note": note}})
            if len(made) != len(groups) + 1:
                skipped.append(f"{p}（找不到換屬性要用的姓名）")
                continue
            out += made
            chosen.append({"base": p, "pool": pool, "quality": quality.get(p)})
    if len(chosen) < a.bases:
        warn(f"只找到 {len(chosen)} 位基底人物（要 {a.bases} 位）", f"候選要沒帶「{at['label']}」",
             "翻成每個群體都不矛盾", "落在品質指數最好或最差的三分之一", "略過：" + "；".join(skipped) if skipped else None)
    if not chosen:
        stop("沒有可用的基底人物")
    flagged = []
    cells_path = os.path.join(adir, "cells.csv")
    if os.path.exists(cells_path):
        for c in read_csv(cells_path):
            if c["屬性"] == at["label"] and c.get("顯著") == "True":
                flagged.append({"layer": c["層"], "metric": c["指標"], "group": c["效果最大的群體"], "effect": float(c["效果"])})
    tdir = os.path.join(a.out, "personas", new_ver)
    if os.path.exists(os.path.join(tdir, "personas.jsonl")) and not a.force:
        stop(f"已經有確認用的人物：{tdir}", "要重新產生：加 --force（已跑的回覆會對不上）")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "personas.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_review(cfg, key, out, chosen, os.path.join(tdir, "personas_review.md"))
    save_json(os.path.join(tdir, META), {"batches": [{"program_version": __version__, "created": time.strftime("%Y-%m-%d %H:%M"),
                                                      "seed": a.seed, "new_people": len(chosen) * (len(groups) + 1),
                                                      "total_people": len(chosen) * (len(groups) + 1), "balance_swaps": 0,
                                                      "persona_sentences": cfg["version"].get("persona_sentences"),
                                                      "source": os.path.basename(pdir)}], "confirm": {
        "attr": key, "label": at["label"], "groups": groups, "bases": chosen, "source_run": os.path.basename(run_dir),
        "source_analysis": os.path.relpath(adir, a.out), "flagged": flagged, "seed": a.seed}})
    model = layout.read_settings(run_dir)["model"]
    from .styles import module_questions
    nq = len(module_questions(cfg, None))
    n = len(out)
    from .common import version_line
    print(kv_block("配對確認的人物", [
        ("版本", version_line(cfg, a.config, f"來源人物 {os.path.basename(pdir)}").replace("版本：", "", 1)),
        ("人物資料夾", tdir),
        ("基底人物", f"{len(chosen)} 位（有利 {sum(c['pool'] == '有利' for c in chosen)}、不利 {sum(c['pool'] == '不利' for c in chosen)}）"),
        ("每位", f"{len(groups) + 1} 版（基底與 {len(groups)} 個群體）＝共 {n} 位"),
        ("篩檢標出的格子", [f"{x['metric']}（{x['group']}）" for x in flagged] or "沒有（照樣可以做確認）"),
        ("規模", f"{n} 位 × {nq} 題 × 10 次 = {n * nq * 10} 段（三輪）")]))
    print(next_steps(["讀過 personas_review.md，確認每組只差受測屬性的句子",
                      "提問：" + cmd_hint("run", "--mock" if model == "mock" else f"--model {model}", f"--personas {new_ver}",
                                         "--rounds 3", "--reps 10"),
                      "評審：" + cmd_hint("judge"), "分析：" + cmd_hint("analyze", f"--run {new_ver}")]))


def write_review(cfg, key, recs, chosen, path):
    """確認用人物的審閱清單：每位基底人物一節，各版並排，確認只差受測屬性的句子。"""
    at = cfg["attributes"][key]
    L = [f"# 配對確認的人物：{at['label']}", "", "每位基底人物一節。各版只差「" + at["label"] + "」的句子（換性別、原住民族或外籍身分時連姓名一起換）。", ""]
    for b in chosen:
        L += [f"## {b['base']}（{b['pool']}，品質指數 {b['quality']:.2f}）", "", "| 版本 | 姓名 | 自我介紹 |", "| --- | --- | --- |"]
        for r in recs:
            if r["confirm"]["base"] == b["base"]:
                g = r["confirm"]["group"]
                lab = "基底（未提及）" if g == NONE else at["levels"][g]["label"]
                L.append(f"| {lab} | {r['name']}{'（換名）' if r['confirm']['note'] else ''} | {r['intro']} |")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------- 分析 ----------

def is_confirm(plist):
    return any(p.get("confirm") for p in plist)


def analyze_confirm(a, cfg, plist, run_dir):
    from .analyze import analysis_dir, judged_for, metric_names, reply_rows
    from .common import load_valid
    from .metrics import SUBSET_METRICS
    from .styles import module_questions
    settings = layout.read_settings(run_dir)
    questions = module_questions(cfg, None)
    qmap = {q["id"]: q for q in questions}
    qcat = {q["id"]: q.get("type", q.get("category")) for q in questions}
    convs, _, _ = load_valid(os.path.join(run_dir, layout.RESPONSES), plist, settings["turns"], questions)
    if not convs:
        print("沒有可用的回覆（略過）")
        return
    pver = settings.get("personas_version")
    meta = load_json(os.path.join(a.out, "personas", pver, META))["confirm"]
    judged, jinfo = judged_for(cfg, a, run_dir, convs)
    rows = reply_rows(convs, qcat, cfg["lexicon"], judged, qmap)
    names = metric_names(rows, judged is not None)
    weights = {k: v for k, v in cfg["category_weights"].items() if not k.startswith("_")}
    cats = [c for c in weights if any(r["cat"] == c for r in rows)]
    by = {}
    for r in rows:
        by.setdefault(r["pid"], []).append(r)
    units = sorted(by)
    sc = {m: dict(zip(units, stats.weighted_persona_scores(by, units, cats, weights, m, m in SUBSET_METRICS))) for m in names}
    info = {p["id"]: p["confirm"] for p in plist if p.get("confirm")}
    at = cfg["attributes"][meta["attr"]]
    rng = np.random.default_rng(a.seed)
    res = []
    for g in meta["groups"]:
        glabel = at["levels"][g]["label"]
        for m in names:
            diffs, pools = [], []
            for b in meta["bases"]:
                u0, u1 = f"{b['base']}-base|explicit", f"{b['base']}-{g}|explicit"
                if u0 in sc[m] and u1 in sc[m] and not np.isnan(sc[m][u0]) and not np.isnan(sc[m][u1]):
                    diffs.append(sc[m][u1] - sc[m][u0])
                    pools.append(b["pool"])
            mean, lo, hi, p = stats.paired_test(np.array(diffs), a.perms, a.boots, rng)
            d = np.array(diffs)
            pl = np.array(pools)
            res.append({"屬性": at["label"], "群體": glabel, "指標": m, "基底人物數": len(diffs), "平均差": mean,
                        "區間下界": lo, "區間上界": hi, "p": p,
                        "有利組平均差": float(d[pl == "有利"].mean()) if np.any(pl == "有利") else np.nan,
                        "不利組平均差": float(d[pl == "不利"].mean()) if np.any(pl == "不利") else np.nan})
    ok = [i for i, r in enumerate(res) if not np.isnan(r["p"])]
    adj = stats.bh_adjust([res[i]["p"] for i in ok]) if ok else []
    for r in res:
        r["校正後p"] = np.nan
    for i, q in zip(ok, adj):
        res[i]["校正後p"] = float(q)
    status = []
    for f in meta.get("flagged", []):
        hit = next((r for r in res if r["群體"] == f["group"] and r["指標"] == f["metric"]), None)
        good = bool(hit and not np.isnan(hit["校正後p"]) and hit["校正後p"] <= Q
                    and (hit["區間下界"] > 0 or hit["區間上界"] < 0) and np.sign(hit["平均差"]) == np.sign(f["effect"]))
        status.append(dict(f, 結論="確認成立" if good else "未確認", 確認平均差=hit["平均差"] if hit else np.nan,
                           確認校正後p=hit["校正後p"] if hit else np.nan))
    out_dir = analysis_dir(run_dir)
    from .report import write_csv
    for r in res:
        r["來源設定資料夾"] = meta["source_run"]
    write_csv(os.path.join(out_dir, "confirm.csv"), res)
    if status:
        write_csv(os.path.join(out_dir, "confirm_status.csv"),
                  [dict(s, 屬性=at["label"], 來源設定資料夾=meta["source_run"]) for s in status])
    write_report(os.path.join(out_dir, "report.md"), meta, res, status, settings, jinfo if judged is not None else None,
                 len(convs), a, info)
    print(bullets(f"配對確認報告：{os.path.join(out_dir, 'report.md')}",
                  [(f"{at['label']} × {s['metric']}（{s['group']}）", s["結論"]) for s in status]))
    print(next_steps(["重新分析篩檢那一組（成績單標上確認結果）：" + cmd_hint("analyze", f"--run {meta['source_run']}")]))


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def write_report(path, meta, res, status, settings, jinfo, n_convs, a, info):
    from .report import judge_line, save_report
    L = [f"# 配對確認：{meta['label']}", "",
         f"- 篩檢資料：{meta['source_run']}（分析 {meta['source_analysis']}）",
         f"- 設定：{layout.describe(settings)}",
         f"- 版本：分析 v{__version__}；確認用人物 {settings.get('personas_version')}（由 {meta['source_run']} 的人物翻出）",
         f"- 評審：{judge_line(jinfo) if jinfo else '尚未評審，行為指標用關鍵詞初篩'}",
         f"- 基底人物：{len(meta['bases'])} 位（品質指數最好的三分之一 "
         f"{sum(b['pool'] == '有利' for b in meta['bases'])} 位、最差的三分之一 {sum(b['pool'] == '不利' for b in meta['bases'])} 位），"
         f"每位做基底（未提及）與 {len(meta['groups'])} 個群體各一版，其餘句子逐字相同；對話 {n_convs} 段",
         f"- 置換 {a.perms} 次（正負號翻轉）、拔靴 {a.boots} 次，都以基底人物為單位；多重比較以 Benjamini–Hochberg 校正", ""]
    renamed = sorted({v["base"] for v in info.values() if v.get("note")})
    if renamed:
        L.append(f"> 姓名一併換掉的基底人物：{'、'.join(renamed)}（換性別、原住民族或外籍身分時，姓名本身就會透露這個屬性）")
        L.append("")
    L += ["## 篩檢標出的格子", ""]
    if status:
        L += ["| 層 | 指標 | 篩檢的群體 | 篩檢效果 | 確認的平均差 | 確認的校正後 p | 結論 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for s in status:
            L.append(f"| {s['layer']} | {s['metric']} | {s['group']} | {fmt(s['effect'])} | {fmt(s['確認平均差'])} | "
                     f"{'—' if np.isnan(s['確認校正後p']) else format(s['確認校正後p'], '.3f')} | **{s['結論']}** |")
        L += ["", "確認成立：同方向、校正後 p ≤ 0.10、區間不含 0。未確認的格子在成績單列為「未確認」，不進入判定。", ""]
    else:
        L += ["篩檢沒有標出這個屬性的格子；下表只描述。", ""]
    L += ["## 各群體減基底", "",
          "同一位基底人物，群體版的分數減基底版（未提及）的分數，再平均所有基底人物。粗體是校正後 p ≤ 0.10 且區間不含 0。"
          "有利、不利組分開列，看差異是不是只出現在某一種基底人物（只描述）。", "",
          "| 群體 | 指標 | 基底人物數 | 平均差 | 95% 區間 | 校正後 p | 有利組 | 不利組 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res:
        b = "**" if (not np.isnan(r["校正後p"]) and r["校正後p"] <= Q and (r["區間下界"] > 0 or r["區間上界"] < 0)) else ""
        L.append(f"| {r['群體']} | {r['指標']} | {r['基底人物數']} | {b}{fmt(r['平均差'])}{b} | [{fmt(r['區間下界'])}, {fmt(r['區間上界'])}] | "
                 f"{'—' if np.isnan(r['校正後p']) else format(r['校正後p'], '.3f')} | {fmt(r['有利組平均差'])} | {fmt(r['不利組平均差'])} |")
    L += ["", "## 限制", "", "- 基底人物只有約 10 位，區間較寬；有利、不利組各約 5 位，只當線索。",
          "- 確認用的是揭露版（說出標籤）；隱性線索是否讀得出另見模型操弄檢核（probe）。", "- 人工校驗只有作者一人。"]
    save_report(path, L)


def load_status(out_dir, run_name):
    """篩檢的成績單用：其他資料夾的配對確認結果裡，來源是這組資料的那些。回傳 {(屬性, 指標, 群體): 結論}。"""
    base = os.path.join(out_dir, "analysis")
    got = {}
    if not os.path.isdir(base):
        return got
    for d in os.listdir(base):
        p = os.path.join(base, d, "confirm_status.csv")
        if os.path.exists(p):
            for r in read_csv(p):
                if r.get("來源設定資料夾") == run_name:
                    got[(r["屬性"], r["metric"], r["group"])] = r["結論"]
    return got


def confirm_people(out_dir, version):
    return read_jsonl(os.path.join(out_dir, "personas", version, "personas.jsonl"))
