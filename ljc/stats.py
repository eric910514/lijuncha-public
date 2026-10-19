"""統計：全部是純計算，不讀檔也不印東西。每個函式對應企劃書「統計補強」裡的一個概念。"""
import numpy as np


def weighted_persona_scores(rows, pids, cats, weights, name, subset=False):
    """每位人物一個分數：先在每個題型內平均，再用固定權重合併（防辛普森悖論）。
    某位人物缺了任何一個題型就回傳 nan。
    評審指標可能有 nan（還沒評、judge_error、不適用的題目）：題型內只平均有值的回覆；
    subset=True（只在部分題目有值的指標，例如正確率、建議數字）時不用題型權重，直接平均有值的回覆。"""
    wsum = sum(weights[c] for c in cats)
    out = []
    for pid in pids:
        if subset:
            v = [r[name] for r in rows[pid] if r["cat"] in cats and not _isnan(r.get(name))]
            out.append(float(np.mean(v)) if v else np.nan)
            continue
        total = 0.0
        for c in cats:
            v = [r[name] for r in rows[pid] if r["cat"] == c and not _isnan(r.get(name))]
            if not v:
                total = np.nan
                break
            total += weights[c] / wsum * np.mean(v)
        out.append(total)
    return np.array(out)


def _isnan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def within_question_sd(rows, name, qkeep=None):
    """單則回覆、題內的標準差（合併各題）：標準化效果的分母（作者 10-19，規格「每一格的判定」）。
    人物間標準差會隨重複次數變小，同一個差距重複越多看起來越大；單則回覆的題內標準差不會。
    rows：逐則回覆（每列有 qid、turn 與指標）；同一題的各輪分開算再合併。qkeep：只算這些題（高風險層）。少於 2 則有值的不算。"""
    ss, df = 0.0, 0
    by = {}
    for r in rows:
        v = r.get(name)
        if qkeep is not None and r["qid"] not in qkeep or _isnan(v):
            continue
        by.setdefault((r["qid"], r.get("turn")), []).append(float(v))      # 同一題的每一輪問的不一樣，分開算
    for vs in by.values():
        if len(vs) >= 2:
            ss += float(np.var(vs, ddof=1)) * (len(vs) - 1)
            df += len(vs) - 1
    return float(np.sqrt(ss / df)) if df else float("nan")


def split_half_noise(rows, pids, cats, weights, name, subset=False):
    """雜訊基準，和人物分數同一個尺度。
    把每位人物的重複分成兩半（第 1、3、5…次一半，第 2、4、6…次一半），各算一個人物分數；
    兩半之間的差距只來自重問時的隨機晃動。兩半各是一半的資料，合起來的人物分數晃動是
    「兩半差距的標準差 ÷ 2」（兩個獨立的半數平均相減，變異加倍；全部平均起來，變異再減半兩次）。
    回傳 (重複取樣的晃動, 人物間標準差)；任一半缺題型的人物不列入，不足 3 人時回傳 nan。"""
    half = [{pid: [r for r in rows[pid] if r["rep"] % 2 == k] for pid in pids} for k in (0, 1)]
    a = weighted_persona_scores(half[0], pids, cats, weights, name, subset)
    b = weighted_persona_scores(half[1], pids, cats, weights, name, subset)
    full = weighted_persona_scores(rows, pids, cats, weights, name, subset)
    ok = ~np.isnan(a) & ~np.isnan(b)
    noise = float(np.std(a[ok] - b[ok], ddof=1) / 2) if ok.sum() >= 3 else float("nan")
    full = full[~np.isnan(full)]
    between = float(np.std(full, ddof=1)) if len(full) >= 3 else float("nan")
    return noise, between


def level_devs(scores, labels, levels):
    """各群體平均 − 跨群體平均（各群體等權）。這個差就是「離均差」。回傳 (離均差, 群體平均)。
    分數是 nan 的人物（例如沒有數字題的有效回覆）不列入。"""
    ok = ~np.isnan(scores)
    means = np.array([scores[(labels == l) & ok].mean() if ((labels == l) & ok).any() else np.nan for l in levels])
    return means - np.nanmean(means), means


def level_devs_codes(scores, codes, k):
    """和 level_devs 相同，但群體先轉成 0..k-1 的編號：置換檢定與拔靴法要重算幾十萬次，用 bincount 一次算完。"""
    ok = ~np.isnan(scores)
    c = codes[ok]
    cnt = np.bincount(c, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.bincount(c, weights=scores[ok], minlength=k) / cnt
    has = cnt > 0
    means[~has] = np.nan
    return means - (means[has].mean() if has.any() else np.nan), means


def centroid_dists(vecs, labels, levels):
    """各群體向量質心，與「各群體質心的平均」之間的餘弦距離。"""
    cents = np.array([vecs[labels == l].mean(axis=0) if (labels == l).any() else np.full(vecs.shape[1], np.nan)
                      for l in levels])
    grand = np.nanmean(cents, axis=0)
    num = (cents * grand).sum(axis=1)
    den = np.linalg.norm(cents, axis=1) * np.linalg.norm(grand)
    return 1 - num / den


def permutation_p(stat_fn, labels, perms, rng):
    """置換檢定：把群體標籤隨機打亂，看「最大偏離」是否仍和實際一樣大。"""
    obs_max = np.nanmax(np.abs(stat_fn(labels)))
    hits = sum(np.nanmax(np.abs(stat_fn(rng.permutation(labels)))) >= obs_max - 1e-12 for _ in range(perms))
    return (hits + 1) / (perms + 1)


def bootstrap_ci(stat_on_index, n, boots, rng):
    """拔靴法：重抽人物，回傳每個群體統計量的 95% 區間 (下界, 上界)。"""
    b = np.array([stat_on_index(rng.integers(0, n, n)) for _ in range(boots)])
    return np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)


def bh_adjust(pvals):
    """Benjamini–Hochberg 多重比較校正。"""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        adj[idx] = prev
    return adj


def compare_groups(attr, labels, scores, vecs, perms, boots, rng):
    """對一個屬性的每個指標做比較，回傳 (逐群體結果列, [(屬性, 指標, p)])。
    vecs：{指標名稱: 人物向量}（例如兩個嵌入模型各一個「詞彙距離」），沒有就是 None 或 {}。"""
    levels = sorted(set(labels))
    counts = {l: int((labels == l).sum()) for l in levels}
    pos = {l: i for i, l in enumerate(levels)}
    codes = np.array([pos[x] for x in labels])      # 群體編號：置換與重抽時直接用編號算，結果和用名稱相同
    n = len(labels)
    rows, tests = [], []
    vecs = vecs or {}
    if not isinstance(vecs, dict):
        vecs = {"詞彙距離": vecs}
    for name in list(scores) + list(vecs):
        if name in vecs:
            V = vecs[name]
            stat = lambda lab, idx=None, V=V: centroid_dists(V if idx is None else V[idx], lab, levels)
            obs, means, sd = stat(labels), [np.nan] * len(levels), np.nan
        else:
            s = scores[name]
            if np.sum(~np.isnan(s)) < 3:
                continue
            stat = lambda lab, idx=None, s=s: level_devs_codes(s if idx is None else s[idx], lab, len(levels))[0]
            obs, means = level_devs(s, labels, levels)
            sd = np.nanstd(s, ddof=1)
        # 置換與重抽的亂數用法和以前一樣（打亂同樣長度的陣列、抽同樣的索引），結果可以重現
        lab0 = labels if name in vecs else codes
        p = permutation_p(stat, lab0, perms, rng)
        lo, hi = bootstrap_ci(lambda idx: stat(lab0[idx], idx), n, boots, rng)
        tests.append((attr, name, p))
        for j, l in enumerate(levels):
            rows.append({"屬性": attr, "群體": l, "人數": counts[l], "指標": name,
                         "群體平均": means[j], "離均差": obs[j],
                         "標準化": obs[j] / sd if sd and sd > 0 else np.nan,
                         "區間下界": lo[j], "區間上界": hi[j], "p": p})
    return rows, tests


def paired_test(d, perms, boots, rng):
    """成對比較：同一批人物在兩種條件下的分數差 d（每人一個）。
    回傳 (平均差, 95% 區間下界, 上界, p)。p 用「隨機翻轉正負號」的置換檢定：
    如果條件真的沒有影響，每個人的差是正是負都只是運氣，翻轉後的平均差應該常常和實際一樣大。"""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(perms, n))
    hits = int((np.abs((signs * d).mean(axis=1)) >= abs(obs) - 1e-12).sum())
    b = d[rng.integers(0, n, (boots, n))].mean(axis=1)
    return float(obs), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), (hits + 1) / (perms + 1)
