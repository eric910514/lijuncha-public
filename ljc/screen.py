"""篩檢分析（E 批）：把逐則回覆的指標整理成迴歸要的表，做同時估計、判定，以及「只描述」的幾項。
計算在 joint 與 verdict，這裡負責把人物、群體、題目對上。"""
import numpy as np

from . import joint, stats, verdict
from .common import NONE
from .metrics import SUBSET_METRICS

VERSION_COL, LENGTH_COL, NAME_COL = "隱性版", "揭露版多出的字數", "名字為男性"


def persona_id(personas, u):
    return personas[u].get("id", u.split("|")[0])


def build_design(cfg, personas, units, attr_keys, attr_names, label_fn):
    """迴歸的表：每列一個人物版本。label_fn(key, 人物版本清單) → 群體陣列（analyze.attr_labels）。"""
    ids = sorted({persona_id(personas, u) for u in units})
    pos = {p: i for i, p in enumerate(ids)}
    pidx = np.array([pos[persona_id(personas, u)] for u in units])
    vidx = np.array([0 if personas[u].get("variant") == "explicit" else 1 for u in units])
    labels, by_persona = {}, {}
    both = []                                   # 每位人物兩版的鍵（缺一版時用另一版代替，只影響置換用的表）
    for p in ids:
        e, i = f"{p}|explicit", f"{p}|implicit"
        both.append((e if e in personas else i, i if i in personas else e))
    for key, name in zip(attr_keys, attr_names):
        labels[name] = label_fn(key, units)
        tab = np.empty((len(ids), 2), object)
        tab[:, 0] = label_fn(key, [b[0] for b in both])
        tab[:, 1] = label_fn(key, [b[1] for b in both])
        by_persona[name] = tab
    ld = np.array([personas[u].get("length_diff", np.nan) for u in units], float)
    lean = [personas[u].get("name_gender") for u in units]
    covars = {VERSION_COL: vidx.astype(float),
              LENGTH_COL: np.where(vidx == 0, ld, 0.0) if not np.isnan(ld).all() else np.zeros(len(units))}
    if any(lean):
        covars[NAME_COL] = np.array([1.0 if x == "male" else (0.0 if x == "female" else np.nan) for x in lean])
    return joint.Design(pidx, vidx, labels, by_persona, covars), ids


def layer_scores(by_pid, pids, cats, weights, names, qkeep=None, drop=frozenset()):
    """一層的人物分數。qkeep：只用這些題目（高風險層）；drop：再排除這些題目（合理相關豁免）。"""
    rows = {p: [r for r in by_pid[p] if (qkeep is None or r["qid"] in qkeep) and r["qid"] not in drop] for p in pids}
    cs = [c for c in cats if any(r["cat"] == c for p in pids for r in rows[p])]
    if not cs:
        return {m: np.full(len(pids), np.nan) for m in names}
    return {m: stats.weighted_persona_scores(rows, pids, cs, weights, m, m in SUBSET_METRICS) for m in names}


MIN_DF = 10      # 迴歸的列數至少要比欄位多這麼多，否則不估（例如評審只評了一小部分人物）


def joint_layer(D, layer, scores, vecs, exempt, rng, perms, boots, raw_means=True, skipped=None, unit_sd=None):
    """一層的同時估計：每個屬性每個指標的群體效果、區間與以人物為單位的置換 p。

    scores：{指標: 每列分數}；vecs：{詞彙距離名稱: 每列向量}；
    exempt：{屬性: ({指標: 排除後的分數}, {詞彙距離名稱: 排除後的向量})}。
    unit_sd：{指標: 單則回覆、題內的標準差}，標準化效果的分母；沒給時退回人物分數的標準差（舊做法）。
    回傳 (results, tests, sds)：results 每個群體一列（含基準群），tests 是 [(屬性, 指標, p)]。"""
    vecs = {k: joint.reduce_vectors(v) for k, v in (vecs or {}).items()}
    ys, wants, plan = {}, {}, {}
    for m, y in scores.items():
        ys[("all", m)] = y
    for k, V in vecs.items():
        ys[("all", k)] = V
    for a in D.attrs:
        ex_s, ex_v = exempt.get(a, ({}, {}))
        cols, kinds, keys = [], [], []
        for m in list(scores) + list(vecs):
            if m in vecs:
                V = joint.reduce_vectors(ex_v[m]) if m in ex_v else vecs[m]
                yk = (a, m) if m in ex_v else ("all", m)
                if m in ex_v:
                    ys[yk] = V
                kinds.append((m, "vector", list(range(sum(c.shape[1] for c in cols), sum(c.shape[1] for c in cols) + V.shape[1]))))
                cols.append(V)
            else:
                y = ex_s.get(m, scores[m])
                if np.sum(~np.isnan(y)) < D.X.shape[1] + MIN_DF:
                    if skipped is not None and np.sum(~np.isnan(y)):
                        skipped.append(f"{m}（{layer}：只有 {int(np.sum(~np.isnan(y)))} 個人物版本有值）")
                    continue
                yk = (a, m) if m in ex_s else ("all", m)
                if m in ex_s:
                    ys[yk] = y
                kinds.append((m, "scalar", sum(c.shape[1] for c in cols)))
                cols.append(y[:, None])
            wants[(a, m)] = (yk, a)
            keys.append(m)
        if cols:
            plan[a] = (np.column_stack(cols), kinds)
    ci, _ = joint.bootstrap(D, ys, wants, boots, rng)
    results, tests, sds = [], [], {}
    for a in D.attrs:
        if a not in plan:
            continue
        Y, kinds = plan[a]
        T = joint.AttrTest(D, a, Y, kinds)
        obs = T.observed()
        ps = T.permutation_p(perms, rng)
        counts = D.persona_counts(a)
        lab = D.labels[a]
        for m, kind, j in kinds:
            eff = obs[m]
            lo, hi = ci[(a, m)]
            tests.append((a, m, ps[m]))
            if kind == "scalar":
                y = Y[:, j]
                ok = ~np.isnan(y)
                sd = (unit_sd or {}).get(m, float(np.std(y[ok], ddof=1)) if ok.sum() > 2 else np.nan)
                sds[(a, m)] = sd
                base = float(y[ok].mean() - T.block[ok].mean(axis=0) @ np.nan_to_num(eff))
                rates = {D.ref[a]: base, **{l: base + e for l, e in zip(D.levels[a], eff)}}
                flip = (lambda v: 1 - v) if m in verdict.ADVERSE else (lambda v: v)    # 越高越不好的比率用 1 − 比率
                irs = verdict.impact_ratios({l: flip(v) for l, v in rates.items() if l != NONE}) \
                    if m in verdict.RATE_METRICS else {}
            else:
                y, sd, irs, rates = None, np.nan, {}, {}
            for gi, l in enumerate([D.ref[a]] + D.levels[a]):
                ref = gi == 0
                mean = float(np.nanmean(y[(lab == l)])) if (y is not None and np.any((lab == l) & ~np.isnan(y))) else np.nan
                e = 0.0 if ref else float(eff[gi - 1])
                results.append({"層": layer, "屬性": a, "群體": l, "人數": counts.get(l, 0), "指標": m,
                                "群體平均": mean, "效果": e,
                                "標準化": (e / sd if sd and sd > 0 else np.nan) if kind == "scalar" else np.nan,
                                "區間下界": np.nan if ref else float(lo[gi - 1]),
                                "區間上界": np.nan if ref else float(hi[gi - 1]),
                                "調整後比率": float(rates[l]) if (m in verdict.RATE_METRICS and l in rates) else np.nan,
                                "影響比": irs.get(l, np.nan), "p": ps[m], "基準": ref})
    adj = stats.bh_adjust([t[2] for t in tests]) if tests else []
    padj = {(t[0], t[1]): float(q) for t, q in zip(tests, adj)}
    for r in results:
        r["校正後p"] = padj[(r["屬性"], r["指標"])]
    return results, tests, padj, sds


def per_attribute_layer(make_design, attrs, layer, scores, vecs, exempt, rng, perms, boots, skipped=None, unit_sd=None):
    """人數太少、無法把所有屬性放進同一個迴歸時的退路：一次只放一個屬性（加上揭露版本、字數差、名字性別），
    其餘做法相同（以人物為單位置換殘差與重抽）。屬性之間的外溢扣不掉，只能當流程檢查。
    make_design(屬性) → 只有那個屬性的 Design。多重比較在所有屬性一起校正。"""
    results, tests, sds = [], [], {}
    for a in attrs:
        Da = make_design(a)
        r, t, _, sd = joint_layer(Da, layer, scores, vecs, {a: exempt.get(a, ({}, {}))}, rng, perms, boots,
                                  skipped=skipped, unit_sd=unit_sd)
        results += r
        tests += t
        sds.update(sd)
    adj = stats.bh_adjust([t[2] for t in tests]) if tests else []
    padj = {(t[0], t[1]): float(q) for t, q in zip(tests, adj)}
    for r in results:
        r["校正後p"] = padj[(r["屬性"], r["指標"])]
    return results, tests, padj, sds


def interaction_cols(cfg, D, units, personas, attr_names_by_key):
    """設定檔 interactions.pairs 的每一組一欄（兩個群體都帶的人物版本為 1）。回傳 {名稱: (欄, 人物數)}。"""
    out = {}
    attrs = cfg["attributes"]

    def ind(spec):
        key, lv = spec.split(":")
        name = attr_names_by_key.get(key)
        if name not in D.labels:
            return None, None
        lab = D.labels[name]
        if lv == "*":
            return lab != NONE, f"{name}任一群"
        label = attrs[key]["levels"].get(lv, {}).get("label")
        return (lab == label) if label else None, label

    for a, b in (cfg.get("interactions") or {}).get("pairs", []):
        ia, na = ind(a)
        ib, nb = ind(b)
        if ia is None or ib is None:
            continue
        col = (ia & ib).astype(float)
        n = len({D.pidx[i] for i in np.where(col > 0)[0]})
        if n:
            out[f"{na} × {nb}"] = (col, n)
    return out


def interactions(D, inter, scores, boots, rng):
    """交互作用：在同時估所有屬性的迴歸裡加交互項。只當訊號，不判定、不進多重比較校正。"""
    if not inter:
        return []
    cols = {k: v[0] for k, v in inter.items()}
    ys = {m: y for m, y in scores.items() if np.sum(~np.isnan(y)) >= 3 and np.nanstd(y) > 0}
    _, xci = joint.bootstrap(D, {}, {}, boots, rng, extra_cols=cols, extra_ys=ys)
    out = []
    for m, y in ys.items():
        coef = joint.interaction_coefs(D, list(cols.values()), y)
        lo, hi = xci[m]
        for i, k in enumerate(cols):
            out.append({"交互作用": k, "人物數": inter[k][1], "指標": m, "效果": float(coef[i]),
                        "區間下界": float(lo[i]), "區間上界": float(hi[i])})
    return out


def quality_index(all_rows, units, cats, weights, judged):
    """回覆品質指數（以人物版本計）：完整回答率、字數（題內標準化）、正確率、不當轉介率（反向）各自標準化後平均。
    還沒評審時只能用字數與拒答率（反向，關鍵詞初篩）。回傳 (每列指數, 用到的成分)。"""
    by_q = {}
    for r in all_rows:
        by_q.setdefault((r["qid"], r["turn"]), []).append(r)
    for g in by_q.values():
        v = np.array([float(len(r["text"])) for r in g])
        sd = v.std(ddof=1) if len(v) > 2 else 0
        for r, x in zip(g, v):
            r["_字數題內"] = float((x - v.mean()) / sd) if sd > 0 else 0.0
    by = {}
    for r in all_rows:
        by.setdefault(r["pid"], []).append(r)
    by = {u: by.get(u, []) for u in units}
    parts = [("完整回答率", 1), ("_字數題內", 1), ("正確率", 1), ("不當轉介率", -1)] if judged \
        else [("_字數題內", 1), ("拒答率", -1)]
    used, zs = [], []
    for m, sign in parts:
        s = stats.weighted_persona_scores(by, units, cats, weights, m, m in SUBSET_METRICS)
        ok = ~np.isnan(s)
        if ok.sum() < 3 or np.std(s[ok]) == 0:
            continue
        zs.append(sign * (s - np.nanmean(s)) / np.nanstd(s, ddof=1))
        used.append(m.lstrip("_").replace("字數題內", "字數（題內標準化）") + ("（反向）" if sign < 0 else ""))
    if not zs:
        return np.full(len(units), np.nan), used
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(zs), axis=0), used


def persona_quality(D, ids, index):
    """每位人物的回覆品質指數（兩版平均）：{人物編號: 指數}。配對確認挑基底人物用。"""
    out = {}
    for i, p in enumerate(ids):
        v = index[D.pidx == i]
        if np.any(~np.isnan(v)):
            out[p] = float(np.nanmean(v))
    return out


def worst_personas(D, ids, index, share, personas, units, cfg):
    """品質指數最差的一部分人物（預設 5%）與各自帶的屬性。每位人物取兩版的平均。"""
    per = np.array([np.nanmean(index[D.pidx == i]) if np.any((D.pidx == i) & ~np.isnan(index)) else np.nan
                    for i in range(len(ids))])
    ok = np.where(~np.isnan(per))[0]
    k = max(1, int(round(len(ok) * share)))
    pick = ok[np.argsort(per[ok])[:k]]
    attrs = cfg["attributes"]
    out = []
    for i in pick:
        rec = personas.get(f"{ids[i]}|explicit") or personas.get(f"{ids[i]}|implicit")
        tags = [f"{attrs[key]['label']}：{attrs[key]['levels'][lv]['label']}"
                for key, lv in (rec.get("true_levels") or rec.get("levels") or {}).items()
                if lv != NONE and key in attrs and lv in attrs[key]["levels"]]
        out.append({"人物": ids[i], "品質指數": float(per[i]), "屬性": tags})
    return out, int(len(ok))


def disclosure(D, ids, scores, boots, rng):
    """揭露版減隱性版：同一人物兩版的分數直接相減，再依人物揭露版所屬的群體平均（只描述、不判定）。
    回傳 [{屬性, 群體, 人數, 指標, 平均差, 區間下界, 區間上界}]，另含每個指標全體的一列（屬性「全部」）。"""
    out = []
    e_row = {D.pidx[i]: i for i in range(D.n) if D.vidx[i] == 0}
    i_row = {D.pidx[i]: i for i in range(D.n) if D.vidx[i] == 1}
    pairs = sorted(set(e_row) & set(i_row))
    if len(pairs) < 3:
        return out
    pe = np.array([e_row[p] for p in pairs])
    pi = np.array([i_row[p] for p in pairs])
    for m, y in scores.items():
        d = y[pe] - y[pi]
        groups = [("全部", "全部", np.ones(len(pairs), bool))]
        for a in D.attrs:
            lab = D.by_persona[a][np.array(pairs), 0]
            groups += [(a, l, lab == l) for l in sorted(set(lab)) if l != NONE]
        for a, l, sel in groups:
            v = d[sel & ~np.isnan(d)]
            if len(v) < 3:
                continue
            b = v[rng.integers(0, len(v), (boots, len(v)))].mean(axis=1)
            out.append({"屬性": a, "群體": l, "人數": int(len(v)), "指標": m, "平均差": float(v.mean()),
                        "區間下界": float(np.percentile(b, 2.5)), "區間上界": float(np.percentile(b, 97.5))})
    return out


def examples(all_rows, unit_labels, attr, group, metric, sign, k=3, width=160):
    """待配對確認的格子附 2 到 3 則原文例子：該群體裡，指標往效果方向最極端的回覆，每位人物最多一則。"""
    cand = [r for r in all_rows if unit_labels.get(r["pid"]) == group]
    if not cand:
        return []
    if metric.startswith("詞彙距離"):
        cand = sorted(cand, key=lambda r: (r["qid"], r["pid"], r["rep"], r["turn"]))
        key = None
    else:
        cand = [r for r in cand if not (r.get(metric) is None or (isinstance(r.get(metric), float) and np.isnan(r[metric])))]
        key = lambda r: -sign * r[metric]
        cand = sorted(cand, key=lambda r: (key(r), r["qid"], r["pid"], r["rep"], r["turn"]))
    out, seen = [], set()
    for r in cand:
        if r["pid"] in seen:
            continue
        seen.add(r["pid"])
        t = " ".join(r["text"].split())
        out.append({"人物": r["pid"], "題號": r["qid"], "第幾輪": r["turn"] + 1, "重複": r["rep"] + 1,
                    "值": None if key is None else r[metric], "原文": t[:width] + ("…" if len(t) > width else "")})
        if len(out) >= k:
            break
    return out
