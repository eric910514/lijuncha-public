"""同時估所有屬性（E 批）：每個指標一個多元迴歸，純計算，不讀檔也不印東西。

白話說：每個屬性的效果，都是在「其他屬性、揭露版本、兩版字數差、名字性別都一樣」的條件下比出來的，
所以埋在經濟上的差異不會因為國籍與經濟剛好失衡，就跑到國籍身上。

- 分析單位是「人物版本」（同一人物的揭露版、隱性版各一列），但同一人物的兩版不是獨立樣本：
  置換檢定以人物為單位打亂（兩版一起換），拔靴法也以人物為單位重抽。
- 每個群體的效果是「相對於未提及」：迴歸係數，正值代表比沒提到這個屬性的人高。
- 合理相關豁免：某屬性在某指標上要排除幾題時，這個屬性的效果改用排除那幾題後的分數另外估（其餘屬性照常控制）。
- 置換檢定用 Freedman–Lane 做法（見 AttrTest），係數用 Frisch–Waugh–Lovell 定理算：先把其他欄位的影響
  從分數與該屬性的欄位中扣掉，再只估該屬性那幾欄，和整個迴歸的係數完全相同，但每次打亂快得多。"""
import numpy as np

from .common import NONE


def dummies(labels, levels):
    """每個群體一欄 0/1（不含基準群）。"""
    return np.column_stack([(labels == l).astype(float) for l in levels]) if levels else np.zeros((len(labels), 0))


class Design:
    """迴歸的欄位：截距、各屬性的群體（相對於基準群）、控制變項。

    unit_pidx：每一列屬於第幾位人物（0..人數-1）；unit_vidx：揭露版 0、隱性版 1。
    labels：{屬性: 每列的群體}；by_persona：{屬性: 人物 × 版本的群體表}，置換時用。
    covars：{名稱: 每列的數值}（揭露版本、字數差、名字性別）。"""

    def __init__(self, unit_pidx, unit_vidx, labels, by_persona, covars):
        self.pidx, self.vidx = np.asarray(unit_pidx), np.asarray(unit_vidx)
        self.n, self.np_ = len(self.pidx), int(self.pidx.max()) + 1 if len(self.pidx) else 0
        self.attrs = list(labels)
        self.labels, self.by_persona = labels, by_persona
        self.ref, self.levels = {}, {}
        for a, lab in labels.items():
            seen = sorted(set(lab))
            self.ref[a] = NONE if NONE in seen else seen[0]
            self.levels[a] = [l for l in seen if l != self.ref[a]]
        cols, self.cols = [np.ones(self.n)], {}
        names = ["截距"]
        for a in self.attrs:
            d = dummies(labels[a], self.levels[a])
            self.cols[a] = list(range(len(names), len(names) + d.shape[1]))
            cols += list(d.T)
            names += [f"{a}:{l}" for l in self.levels[a]]
        self.covar_names = []
        for k, v in covars.items():
            v = np.asarray(v, float)
            if np.nanstd(v) > 0:                 # 全部一樣（例如只有一個版本）就不放，免得欄位重複
                cols.append(np.nan_to_num(v - np.nanmean(v)))
                names.append(k)
                self.covar_names.append(k)
        self.X = np.column_stack(cols)
        self.names = names
        self._for = {}

    def for_attr(self, a):
        """估屬性 a 時用的設計矩陣：前幾欄是 a 的群體，其餘是截距、控制變項與其他屬性的群體（和完整迴歸同一組欄位，
        只是換了順序）。所有屬性都兩版合併（規格：兩版合併、版本當控制變項）。

        注意（待決第 5 項）：每位人物固定帶 4 個屬性，揭露版裡「相對於未提及」和「其他屬性的平均效果」分不開，
        要靠隱性版（只做揭露的屬性在隱性版是未提及）補足辨識；某個屬性只在揭露版才有效果時，
        會經由只做揭露的屬性（學歷、本國籍……）外溢一點。"""
        if a not in self._for:
            rest = [i for i in range(self.X.shape[1]) if i not in self.cols[a]]
            self._for[a] = self.X[:, self.cols[a] + rest]
        return self._for[a]

    def persona_counts(self, a):
        """各群體的人物數（兩版算同一人）：人物任一版屬於該群就算。"""
        tab = self.by_persona[a]
        present = np.zeros(self.np_, bool)
        present[np.unique(self.pidx)] = True
        return {l: int(np.sum(present & np.any(tab == l, axis=1))) for l in [self.ref[a]] + self.levels[a]}

    def extra(self, cols):
        """加上交互作用欄位後的設計矩陣。"""
        return np.column_stack([self.X] + [np.asarray(c, float) for c in cols]) if cols else self.X


def _solve(X, Y):
    """最小平方解；欄位不足以分辨時取最小範數解（全為 0 的欄位係數為 0，之後另標為 nan）。"""
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def _masks(Y):
    """依缺值的樣式把指標分組：同一組的指標用同一批列估。回傳 [(列的遮罩, 欄的索引)]。"""
    ok = ~np.isnan(Y)
    groups = {}
    for j in range(Y.shape[1]):
        groups.setdefault(ok[:, j].tobytes(), (ok[:, j], []))[1].append(j)
    return list(groups.values())


def lexical_dist(base, b):
    """詞彙層的效果：基準群的質心 base 與「base + 群體係數」之間的餘弦距離（沒有正負）。"""
    tgt = base[None, :] + b
    num = tgt @ base
    den = np.linalg.norm(tgt, axis=1) * np.linalg.norm(base)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - num / den


def reduce_vectors(V):
    """向量的維度通常遠大於列數；投影到列空間（SVD）後距離與內積都不變，置換時快很多。"""
    V = np.asarray(V, float)
    if V.shape[1] <= V.shape[0]:
        return V
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    return U * S


class AttrTest:
    """一個屬性在多個指標上的同時估計與以人物為單位的置換檢定。

    Y：列 × 指標（這個屬性用的分數；有豁免時已排除那幾題）；kinds：[(指標名稱, "scalar", 欄) 或
    (名稱, "vector", [欄])]，vector 是一組向量（詞彙距離）。效果：純量指標是係數；詞彙距離是餘弦距離。

    置換檢定用 Freedman–Lane 做法：先只用其他欄位（其他屬性與控制變項）估一次，把剩下的殘差以人物為單位打亂
    （兩版一起換），加回去再估這個屬性的效果。直接打亂群體標籤會把「屬性之間的失衡」也一起打散，
    兩個屬性糾纏時會低估晃動、產生假警報；打亂殘差則保留原本的失衡。"""

    def __init__(self, design, a, Y, kinds):
        self.d, self.a = design, a
        self.kinds = kinds
        self.Y = Y
        Xa = design.for_attr(a)
        self.k = len(design.cols[a])
        self.block = Xa[:, :self.k]
        Zall = Xa[:, self.k:]
        self.parts = []
        for mask, js in _masks(Y):
            Z = Zall[mask]
            M = np.eye(mask.sum()) - Z @ np.linalg.pinv(Z)
            Xa = self.block[mask]
            P = np.linalg.pinv(M @ Xa)                 # 係數 = P × 殘差（FWL）
            P[Xa.sum(axis=0) == 0] = 0
            E = M @ Y[mask][:, js]                     # 只用其他欄位估之後的殘差
            self.parts.append((mask, js, P, E, Xa.sum(axis=0) == 0, self._groups(mask)))

    def _groups(self, mask):
        """這批列的人物，依手上有哪幾版分組（兩版都有、只有揭露、只有隱性）；置換只在同一組內交換。"""
        rows = np.where(mask)[0]
        pos = {}
        for i, r in enumerate(rows):
            pos.setdefault(self.d.pidx[r], {})[self.d.vidx[r]] = i
        by = {}
        for p, vs in pos.items():
            key = tuple(sorted(vs))
            by.setdefault(key, []).append([vs[v] for v in key])
        return [np.array(v) for v in by.values()]

    def _perm_rows(self, groups, n, rng):
        idx = np.empty(n, int)
        for g in groups:
            idx[g.ravel()] = g[rng.permutation(len(g))].ravel()
        return idx

    def _effects(self, coef_of):
        B = np.full((self.k, self.Y.shape[1]), np.nan)
        for (mask, js, P, E, empty, groups), b in zip(self.parts, coef_of):
            b = b.copy()
            b[empty] = np.nan
            B[:, js] = b
        out = {}
        for name, kind, j in self.kinds:
            if kind == "scalar":
                out[name] = B[:, j]
                continue
            b = B[:, j]
            if np.isnan(b).all():
                out[name] = np.full(self.k, np.nan)
                continue
            Yv = self.Y[:, j]
            ok = ~np.isnan(Yv).any(axis=1)
            base = Yv[ok].mean(axis=0) - self.block[ok].mean(axis=0) @ np.nan_to_num(b)
            out[name] = lexical_dist(base, np.nan_to_num(b))
        return out

    def observed(self):
        """每個指標每個群體的效果（與完整迴歸的係數相同）。回傳 {指標: 陣列（群體）}。"""
        return self._effects([P @ E for _, _, P, E, _, _ in self.parts])

    def permutation_p(self, perms, rng):
        """看「效果最大的群體」在打亂後是否仍和實際一樣大。回傳 {指標: p}。"""
        obs = {m: np.nanmax(np.abs(v)) if np.any(~np.isnan(v)) else np.nan for m, v in self.observed().items()}
        hits = dict.fromkeys(obs, 0)
        for _ in range(perms):
            coefs = [P @ E[self._perm_rows(groups, len(E), rng)] for _, _, P, E, _, groups in self.parts]
            for m, v in self._effects(coefs).items():
                if not np.isnan(obs[m]) and np.nanmax(np.abs(v)) >= obs[m] - 1e-12:
                    hits[m] += 1
        return {m: (hits[m] + 1) / (perms + 1) if not np.isnan(obs[m]) else np.nan for m in obs}


def bootstrap(design, ys, wants, boots, rng, extra_cols=None, extra_ys=None):
    """以人物為單位重抽（抽到的人物兩版一起進來），每次重估完整迴歸。

    ys：{分數的鍵: 每列的分數（純量）或每列的向量}；wants：{結果的鍵: (分數的鍵, 屬性)}。
    缺值樣式相同的分數一起解。
    extra_cols／extra_ys：加了交互作用的迴歸（{交互項: 欄}、{指標: 分數}），只取交互項的係數。
    回傳 {結果的鍵: (下界, 上界)}（每個群體一個）與 {指標: (下界, 上界)}（每個交互項一個）。"""
    rows_of = [np.where(design.pidx == i)[0] for i in range(design.np_)]
    present = [i for i in range(design.np_) if len(rows_of[i])]
    # 所有屬性用同一個設計矩陣（兩版合併），所以同一份分數只要解一次，各屬性的係數從同一個解取出
    yk_s = sorted({yk for yk, _ in wants.values() if np.ndim(ys[yk]) == 1}, key=str)
    yk_v = sorted({yk for yk, _ in wants.values() if np.ndim(ys[yk]) == 2}, key=str)
    S = np.column_stack([ys[k] for k in yk_s]) if yk_s else None
    groups = _masks(S) if yk_s else []
    by_y = {}
    for key, (yk, a) in wants.items():
        by_y.setdefault(yk, []).append((key, design.cols[a]))
    xk = list(extra_ys or {})
    XS = np.column_stack([extra_ys[k] for k in xk]) if xk else None
    Xext = design.extra(list((extra_cols or {}).values())) if xk else None
    xgroups = _masks(XS) if xk else []
    first = design.X.shape[1]
    draws = {k: [] for k in wants}
    xdraws = {k: [] for k in xk}
    for _ in range(boots):
        pick = rng.choice(present, size=len(present), replace=True)
        rows = np.concatenate([rows_of[i] for i in pick])
        X = design.X[rows]
        for mask, js in groups:
            ok = mask[rows]
            Xo = X[ok]
            B = _solve(Xo, S[rows][ok][:, js])
            B[Xo.sum(axis=0) == 0] = np.nan
            for j, b in zip(js, B.T):
                for key, cols in by_y[yk_s[j]]:
                    draws[key].append(b[cols])
        for yk in yk_v:
            yy = ys[yk][rows]
            ok = ~np.isnan(yy).any(axis=1)
            Xo, Yo = X[ok], yy[ok]
            B = _solve(Xo, Yo)
            ybar, xbar, empty = Yo.mean(axis=0), Xo.mean(axis=0), Xo.sum(axis=0) == 0
            for key, cols in by_y[yk]:
                v = lexical_dist(ybar - xbar[cols] @ B[cols], B[cols])
                v[empty[cols]] = np.nan
                draws[key].append(v)
        if xk:
            XE = Xext[rows]
            for mask, js in xgroups:
                ok = mask[rows]
                B = _solve(XE[ok], XS[rows][ok][:, js])
                B[XE[ok].sum(axis=0) == 0] = np.nan
                for col, j in enumerate(js):
                    xdraws[xk[j]].append(B[first:, col])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ci = lambda arr: (np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0))
        return {k: ci(np.array(v)) for k, v in draws.items()}, {k: ci(np.array(v)) for k, v in xdraws.items()}


def interaction_coefs(design, cols, y):
    """加了交互作用的迴歸裡，各交互項的係數。"""
    X = design.extra(cols)
    ok = ~np.isnan(y)
    B = _solve(X[ok], y[ok])
    v = B[design.X.shape[1]:].astype(float)
    v[X[ok][:, design.X.shape[1]:].sum(axis=0) == 0] = np.nan
    return v


def covariate_coefs(design, y):
    """控制變項（揭露版本、字數差、名字性別）的係數，報告列出供參考。"""
    ok = ~np.isnan(y)
    B = _solve(design.X[ok], y[ok])
    return {k: float(B[design.names.index(k)]) for k in design.covar_names}
