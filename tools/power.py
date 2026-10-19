"""檢定力：用已經跑完的重複，估每個指標「看得出多小的群體差異」，再推算多跑幾次重複、多加人物各能縮小多少。

想法（白話）：
- 一位人物的分數會晃動，來源有兩個：
  ① 重問的晃動：同一位人物、同一題，再問一次回覆就不一樣（溫度 0.7）。多跑幾次重複可以平均掉。
  ② 人物之間的差異：扣掉所有屬性之後，人物和人物本來就不同（名字、寫法、生活線索）。多跑重複平均不掉，
     只能靠更多人物。
- 分析是每個指標一個多元迴歸（同時估所有屬性），所以群體效果的標準誤 ≈ √(①/重複次數 ＋ ②) × 設計的係數。
  兩版人物屬於同一人，用以人物為單位的穩健標準誤修正（設計效應）。
- 最小可偵測效果（MDE）：在 80% 檢定力、雙尾 α = 0.05 下，真實效果至少要多大才大概看得出來。
  「校正後」把 α 除以一層的檢定數（Benjamini–Hochberg 在只有一個真效果時就是這樣），比較保守。

只讀資料，不改任何檔。用法（在程式資料夾）：
    python tools/power.py --run p1.2                  關鍵詞初篩的指標（字數、拒答率……），每次重複都有
    python tools/power.py --run p1.2 --judged         評審的指標（要至少兩次重複都評完）
    python tools/power.py --run p1.2 --csv power.csv  另存每個群體的明細"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ljc import layout, probe, screen, stats  # noqa: E402
from ljc.analyze import attr_labels, choose_cohort, judged_for, metric_names, reply_rows, uid  # noqa: E402
from ljc.common import read_config, kv_block, load_json, load_valid, stop  # noqa: E402
from ljc.metrics import SUBSET_METRICS  # noqa: E402
from ljc.personas import personas_for_run  # noqa: E402
from ljc.styles import module_questions  # noqa: E402

PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = {0.80: 0.8416, 0.90: 1.2816}


def z_two_sided(alpha):
    """雙尾臨界值（不靠 scipy：用二分法反解常態分布）。"""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def unit(m):
    """指標的單位：比率換成百分點。"""
    if m.endswith("率") or "率（" in m:
        return 100.0, "點"
    if m == "字數":
        return 1.0, "字"
    return 1.0, ""


def cluster_se(X, y, groups):
    """以人物為單位的穩健標準誤（兩版屬於同一人）。回傳 (OLS 標準誤, 穩健標準誤)。"""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    s2 = float(e @ e) / max(n - p, 1)
    ols = np.sqrt(np.diag(XtX_inv) * s2)
    meat = np.zeros((p, p))
    for g in np.unique(groups):
        idx = groups == g
        sg = X[idx].T @ e[idx]
        meat += np.outer(sg, sg)
    G = len(np.unique(groups))
    adj = G / max(G - 1, 1) * (n - 1) / max(n - p, 1)
    rob = np.sqrt(np.clip(np.diag(XtX_inv @ meat @ XtX_inv) * adj, 0, None))
    return ols, rob, s2


def analyse(a):
    cfg = read_config(a.config)
    dirs = layout.pick_one([d for d in layout.run_dirs(a.out) if a.run in os.path.basename(d)], a.run)
    if len(dirs) != 1:
        stop("--run 要剛好對到一個設定資料夾", "對到：" + ("、".join(os.path.basename(d) for d in dirs) or "沒有"))
    run_dir = dirs[0]
    plist = personas_for_run(a.out, cfg, run_dir)[0]
    plist, _ = probe.recode(cfg, run_dir, plist)
    personas = {uid(p): p for p in plist}
    settings = layout.read_settings(run_dir)
    questions = module_questions(cfg, None)
    qmap = {q["id"]: q for q in questions}
    convs, _, _ = load_valid(os.path.join(run_dir, layout.RESPONSES), plist, settings["turns"], questions)
    convs, cohort = choose_cohort(convs, plist, questions, 0, a.reps)
    R0 = cohort["reps"]
    if R0 < 2:
        stop("至少要兩次重複都跑完，才分得出重問的晃動", f"現在：{cohort['people']} 位人物 × {R0} 次")
    qcat = {q["id"]: q.get("type", q.get("category")) for q in questions}
    weights = {k: v for k, v in cfg["category_weights"].items() if not k.startswith("_")}
    judged = None
    if a.judged:
        judged, info = judged_for(cfg, argparse.Namespace(config=a.config), run_dir, convs)
        if judged is None:
            stop("這組資料還沒有評審結果", "先執行 judge；或不加 --judged，用關鍵詞初篩的指標")
    rows = reply_rows(convs, qcat, cfg["lexicon"], judged, qmap)
    names = [m for m in metric_names(rows, judged is not None) if not m.startswith(("正確率（", "表態率（"))]
    pids = sorted({r["pid"] for r in rows})
    cats = [c for c in weights if any(r["cat"] == c for r in rows)]
    per_rep = []
    for r in range(R0):
        by = {p: [x for x in rows if x["pid"] == p and x["rep"] == r] for p in pids}
        per_rep.append({m: stats.weighted_persona_scores(by, pids, cats, weights, m, m in SUBSET_METRICS) for m in names})

    attrs = cfg["attributes"]
    keys = list(attrs)
    attr_names = [attrs[k]["label"] for k in keys]
    label_fn = lambda key, us: attr_labels(cfg, personas, us, key)
    n_tests = len(keys) * len(names)                      # 一層的檢定數：屬性 × 指標
    z_alpha, z_alpha_adj = z_two_sided(a.alpha), z_two_sided(a.alpha / n_tests)
    z_pow = Z[a.power]
    out = []
    for m in names:
        S = np.array([pr[m] for pr in per_rep])           # 重複 × 人物版本
        ok = ~np.isnan(S).any(axis=0)
        if ok.sum() < 20:
            out.append({"指標": m, "資料不足": int(ok.sum())})
            continue
        us = [p for p, k in zip(pids, ok) if k]
        S = S[:, ok]
        D, _ = screen.build_design(cfg, personas, us, keys, attr_names, label_fn)
        X = D.X
        mean = S.mean(axis=0)
        if float(np.var(S)) == 0:
            out.append({"指標": m, "全部相同": float(S.flat[0])})
            continue
        within = float(np.mean(S.var(axis=0, ddof=1)))    # 只跑一次重複時的重問晃動（變異）
        ols, rob, s2 = cluster_se(X, mean, D.pidx)
        between = max(s2 - within / R0, 0.0)              # 扣掉屬性後、人物之間的差異（變異）
        cols = [(name, l, D.cols[name][i]) for name in D.attrs for i, l in enumerate(D.levels[name])]
        deff = float(np.median([rob[j] / ols[j] for _, _, j in cols if ols[j] > 0])) if cols else 1.0
        V = np.diag(np.linalg.pinv(X.T @ X))
        scale, u = unit(m)
        noise_share = (within / R0) / (within / R0 + between) if within + between > 0 else float("nan")
        for R in a.reps_plan:
            for k in a.people_plan:
                var = between + within / R
                for name, level, j in cols:
                    se = math.sqrt(var * V[j] / k) * max(deff, 1.0)
                    out.append({"指標": m, "單位": u, "屬性": name, "群體": level, "人數": D.persona_counts(name).get(level),
                                "重複": R, "人物倍數": k, "標準誤": se * scale,
                                "MDE": (z_alpha + z_pow) * se * scale, "MDE（校正後）": (z_alpha_adj + z_pow) * se * scale,
                                "重問晃動占比": noise_share, "設計效應": deff, "現有重複": R0, "人數總計": len(set(D.pidx))})
    return out, cohort, n_tests, judged is not None


def summarize(out, a, cohort, n_tests, judged):
    R0 = cohort["reps"]
    print(kv_block("檢定力", [("資料", f"{cohort['people']} 位人物 × 前 {R0} 次重複"),
                             ("指標", "評審的判定" if judged else "關鍵詞初篩（還沒用評審的判定）"),
                             ("標準", f"80% 檢定力、雙尾 α = {a.alpha}；校正後再除以一層的 {n_tests} 個檢定"),
                             ("怎麼讀", ["MDE：真實的群體差異至少這麼大，才大概看得出來（越小越靈敏）",
                                       "重問晃動占比：人物分數的晃動裡，有多少來自重問；占比低，多跑重複就幫助不大"])]))
    metrics = list(dict.fromkeys(r["指標"] for r in out))
    for m in metrics:
        rs = [r for r in out if r["指標"] == m]
        if "資料不足" in rs[0]:
            print(kv_block(m, [("沒辦法估", f"每次重複都有值的人物版本只有 {rs[0]['資料不足']} 個（至少要 20）"),
                               ("怎麼辦", "評審要評完至少兩次重複" if judged else "等更多重複跑完")]))
            continue
        if "全部相同" in rs[0]:
            scale, u = unit(m)
            print(kv_block(m, [("沒辦法估", f"每位人物的分數都是 {rs[0]['全部相同'] * scale:g} {u}（沒有任何差異可以量）")]))
            continue
        base = [r for r in rs if r["重複"] == R0 and r["人物倍數"] == 1]
        if not base:
            continue
        u = base[0]["單位"]
        med = lambda xs, key: float(np.median([x[key] for x in xs]))
        line = lambda R, k: [r for r in rs if r["重複"] == R and r["人物倍數"] == k]
        b = med(base, "MDE（校正後）")
        f = (lambda v: f"{v:.2f}") if b < 1 else (lambda v: f"{v:.1f}")
        rows = [("重問晃動占比", f"{base[0]['重問晃動占比']:.0%}（現在 {R0} 次重複）"),
                ("設計效應", f"{base[0]['設計效應']:.2f}（兩版屬於同一人，標準誤放大的倍數）"),
                ("MDE 中位數", f"{f(med(base, 'MDE'))} {u}；校正後 {f(b)} {u}"),
                ("最不靈敏的群體", f"{f(max(x['MDE（校正後）'] for x in base))} {u}（校正後；通常是人數最少的）")]
        plans = []
        combos = [(R, 1) for R in a.reps_plan if R != R0] + [(R0, k) for k in a.people_plan if k != 1]
        for R, k in combos:
            v = med(line(R, k), "MDE（校正後）")
            if not math.isnan(v):
                what = f"重複改成 {R} 次" if k == 1 else f"人物 × {k:g}（{R0} 次重複）"
                plans.append(f"{what}：{f(v)} {u}（{(v / b - 1):+.0%}）")
        rows.append(("如果", plans))
        print(kv_block(m, rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(PROG), "out"))
    ap.add_argument("--config", default=os.path.join(PROG, "config", "config.json"))
    ap.add_argument("--run", required=True, help="設定資料夾名稱裡的一段（例如 p1.2）")
    ap.add_argument("--reps", type=int, default=0, help="用前幾次重複估（預設：人物都跑滿的最多次數）")
    ap.add_argument("--judged", action="store_true", help="用評審的判定（預設用關鍵詞初篩，每次重複都有）")
    ap.add_argument("--reps-plan", type=int, nargs="+", default=[2, 3, 4, 6, 8], help="要推算的重複次數")
    ap.add_argument("--people-plan", type=float, nargs="+", default=[1, 1.5, 2], help="人物數的倍數（同樣的組成）")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, choices=sorted(Z), default=0.80)
    ap.add_argument("--csv", default="", help="另存每個群體的明細")
    a = ap.parse_args(argv)
    out, cohort, n_tests, judged = analyse(a)
    if not out:
        stop("沒有可以估的指標", "每個指標至少要 20 位人物版本每次重複都有值")
    summarize(out, a, cohort, n_tests, judged)
    if a.csv:
        with open(a.csv, "w", encoding="utf-8-sig", newline="") as f:
            rows = [r for r in out if "全部相同" not in r and "資料不足" not in r]
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"明細：{a.csv}")


if __name__ == "__main__":
    main()
