"""gen 指令：產生人物（每人揭露、隱性兩版）、檢查矛盾、接續加人。

規則（離均差 v1 程式需求規格「人物生成」）：
- 每位人物帶 mention_k 個屬性；兩版共用同一個編號，以 variant 區分 explicit／implicit。
- 隱性版只寫線索句；implicit_cue 為 none 的群體在隱性版不出現，分析時算「未提及」。
- 揭露版 = 隱性版的全部文字，再在每個屬性的位置加上明確標籤，所以兩版只差「說出標籤」。
  句子、組合順序、佔位符（{origin}、{tribe}、{partner_same}、{partner_opposite}、{origin_language}）見〈人物句與題目補充〉。
  兩版字數差不設上限，記在 length_diff，分析時當控制變項。
- 每位人物都有姓名，兩版相同：性別在屬性內時與性別一致；不在時男、女名各半，名字的性別記在 name_gender。
  群體以姓名為線索時（原住民族、外籍身分）用該群體的姓名，其餘從一般姓名挑。姓名來源見 names.py。
- 指派時讓每個群體至少 min_per_group 人（以人物計），並讓屬性兩兩之間盡量平衡。"""
import json
import math
import os
import random
import time

from . import layout, names
from .common import read_config, NONE, bullets, kv_block, load_json, read_jsonl, stop, text_sha256, warn

TODO = "【待補】"
VARIANTS = ["explicit", "implicit"]
VARIANT_LABEL = {"explicit": "揭露", "implicit": "隱性"}
SENTENCE_CUES = ("life_detail", "partner_term", "wording")


# ---------- 設定檔的讀取 ----------

def cue_kinds(level_cfg):
    c = level_cfg.get("implicit_cue", "none")
    return c if isinstance(c, list) else [c]


def implicit_visible(level_cfg):
    """隱性版看不看得到這個群體：只做揭露（none）的看不到。"""
    return cue_kinds(level_cfg) != ["none"]


def placeholders(cfg):
    """設定檔中還沒補齊的地方，回傳說明清單。"""
    out = []
    for ac in cfg["attributes"].values():
        for lc in ac["levels"].values():
            for f in ("explicit", "implicit"):
                if TODO in (lc.get(f) or ""):
                    out.append(f"{ac['label']}／{lc['label']}的 {f}")
    for q in cfg["questions"]:
        texts = [("q", q["q"]), ("q_paraphrase", q.get("q_paraphrase") or "")] + \
                [(f"followups[{i}]", f["text"]) for i, f in enumerate(q["followups"])]
        out += [f"題目 {q['id']} 的 {k}" for k, t in texts if TODO in t]
    return out


def level_facts(cfg, attr, level, variant):
    lc = cfg["attributes"][attr]["levels"][level]
    facts = dict(lc.get("facts") or {})
    facts.update(lc.get("explicit_facts" if variant == "explicit" else "implicit_facts") or {})
    return facts


def find_conflicts(cfg, levels):
    """一位人物（兩版都算）身上互相矛盾的地方；空清單代表沒有矛盾。
    來源：(1) 兩個群體對同一件事給了不同的值（facts）；(2) 設定檔 exclude 明列的組合。"""
    out = []
    for variant in VARIANTS:
        seen = {}
        for a, l in levels.items():
            if l == NONE:
                continue
            if variant == "implicit" and not implicit_visible(cfg["attributes"][a]["levels"][l]):
                continue
            for facet, val in level_facts(cfg, a, l, variant).items():
                if facet in seen and seen[facet][0] != val:
                    out.append(f"「{facet}」矛盾：{seen[facet][1]} 說是{seen[facet][0]}，{a}:{l} 說是{val}")
                seen.setdefault(facet, (val, f"{a}:{l}"))
    tags = {f"{a}:{l}" for a, l in levels.items()}
    for x, y in cfg.get("exclude", []):
        if x in tags and y in tags:
            out.append(f"設定檔排除的組合：{x} 與 {y}")
    return list(dict.fromkeys(out))


# ---------- 指派群體 ----------

def paused_groups(cfg):
    """暫停產生正式人物的群體（設定檔 paused_groups，例如姓名資料不足的外籍三群）：{(屬性, 群體)}。"""
    return {tuple(g.split(":", 1)) for g in cfg.get("paused_groups", []) if ":" in g}


def group_list(cfg, skip=()):
    return [(a, l) for a, ac in cfg["attributes"].items() for l in ac["levels"] if (a, l) not in skip]


def needed_personas(cfg, k, min_per_group, skip=()):
    return math.ceil(min_per_group * len(group_list(cfg, skip)) / k)


def interaction_seeds(cfg, existing, skip=()):
    """交互作用組合的保證名額（設定檔 interactions）：每組至少 min_personas_per_pair 位人物同時帶兩個群體。
    回傳要先放進新人物的部分群體 [{屬性: 群體}]；既有人物已經符合的會先扣掉。群體寫 * 表示該屬性任一群，輪流放。"""
    ic = cfg.get("interactions") or {}
    need = int(ic.get("min_personas_per_pair", 0) or 0)
    attrs = cfg["attributes"]
    seeds = []
    for pair in ic.get("pairs", []):
        parts = [tuple(x.split(":", 1)) for x in pair]
        if any(a not in attrs for a, _ in parts):
            continue
        options = []
        for a, l in parts:
            ls = [x for x in attrs[a]["levels"] if (a, x) not in skip] if l == "*" else ([l] if (a, l) not in skip else [])
            options.append((a, ls))
        if any(not ls for _, ls in options):
            continue                       # 其中一群暫停（例如外籍三群）：這組先不保證
        have = sum(1 for lv in existing if all(lv.get(a) in ls for a, ls in options))
        for i in range(max(need - have, 0)):
            seeds.append({a: ls[i % len(ls)] for a, ls in options})
    return seeds


def pair_counts(cfg, levels_list):
    """每個交互作用組合有幾位人物同時帶兩個群體：[(說明, 人數)]。"""
    attrs = cfg["attributes"]
    out = []
    for pair in (cfg.get("interactions") or {}).get("pairs", []):
        parts = [tuple(x.split(":", 1)) for x in pair]
        if any(a not in attrs for a, _ in parts):
            continue
        ok = lambda lv: all((lv.get(a, NONE) != NONE) if l == "*" else lv.get(a) == l for a, l in parts)
        lab = " × ".join(f"{attrs[a]['label']}／" + ("任一群" if l == "*" else attrs[a]["levels"][l]["label"]) for a, l in parts)
        out.append((lab, sum(1 for lv in levels_list if ok(lv))))
    return out


def assign_levels(cfg, n_new, k, rng, existing, min_required, skip=()):
    """為 n_new 位新人物指派群體。existing = 既有人物的群體（接續加人時沿用，不改）。
    每次挑「離目標人數最遠」的群體，並避開和這位人物已有群體太常一起出現的組合，讓屬性兩兩之間平衡。
    回傳 (新人物的群體清單, 各群體人數)；做不到時回傳 (None, 說明)。"""
    attrs = cfg["attributes"]
    groups = group_list(cfg, skip)
    per_group = (n_new + len(existing)) * k / len(groups)      # 每個群體的目標人數
    used, pairs = {g: 0 for g in groups}, {}
    attr_used = {a: 0 for a in attrs}
    n_levels = {a: sum((a, l) not in skip for l in attrs[a]["levels"]) for a in attrs}

    def add(levels):
        chosen = [(a, l) for a, l in levels.items() if l != NONE]
        for g in chosen:
            used[g] = used.get(g, 0) + 1
            attr_used[g[0]] += 1
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                key = frozenset([chosen[i], chosen[j]])
                pairs[key] = pairs.get(key, 0) + 1

    for lv in existing:
        add(lv)
    seeds = interaction_seeds(cfg, existing, skip)
    if len(seeds) > n_new:
        if min_required:
            return None, f"交互作用組合要先放 {len(seeds)} 位，但這次只新增 {n_new} 位"
        seeds = seeds[:n_new]              # 人數太少、只是試流程：放得下幾位就放幾位
    out = []
    for i in range(n_new):
        levels = {a: NONE for a in attrs}
        if i < len(seeds):                 # 先放交互作用組合，再補滿其餘屬性
            levels.update(seeds[i])
            if find_conflicts(cfg, levels):
                return None, f"交互作用組合本身違反互斥規則：{seeds[i]}"
        for _slot in range(k - sum(l != NONE for l in levels.values())):
            cands = []
            for a in attrs:
                if levels[a] != NONE:
                    continue
                for l in attrs[a]["levels"]:
                    if (a, l) in skip or find_conflicts(cfg, dict(levels, **{a: l})):
                        continue
                    together = sum(pairs.get(frozenset([(a, l), (b, levels[b])]), 0) for b in attrs if levels[b] != NONE)
                    attr_gap = per_group * n_levels[a] - attr_used[a]
                    cands.append((used[(a, l)] - per_group, -attr_gap, together, rng.random(), a, l))
            if not cands:
                break
            *_, a, l = min(cands)
            levels[a] = l
        if sum(l != NONE for l in levels.values()) < k:
            return None, "互斥規則太嚴，有人物湊不滿屬性"
        add(levels)
        out.append(levels)
    if out:
        SWAPS["last"] = balance_by_swaps(cfg, out, existing, rng, skip)
    if min_required and min(used[g] for g in groups) < min_required:
        gaps = [f"{attrs[a]['label']}／{attrs[a]['levels'][l]['label']} {used[(a, l)]} 人" for (a, l) in groups
                if used[(a, l)] < min_required]
        return None, f"有 {len(gaps)} 個群體不到 {min_required} 人：" + "、".join(gaps[:8]) + ("…" if len(gaps) > 8 else "")
    return out, used


SWAPS = {"last": 0}       # 最近一次指派做了幾次屬性平衡互換（寫進 personas_meta.json）


def balance_by_swaps(cfg, new, existing, rng, skip=(), rounds=40):
    """指派完之後再讓屬性兩兩之間更平衡：同一個屬性，在兩位新人物之間互換群體（例如甲的低收入和乙的高所得對調）。
    互換不改變任何群體的人數，所以「每群至少幾人」和各屬性的人數都不變；既有人物（接續加人）不動。
    只在換了之後更平衡、兩人都沒有矛盾、交互作用組合的人數沒有少於保證名額時才換。
    平衡的量法：每兩個屬性做一張交叉表（含「未提及」），和「兩者無關」時的期望人數比，差距平方除以期望的總和
    （卡方值）。直接在 new 上修改。"""
    attrs = list(cfg["attributes"])
    people = list(existing) + new
    first = len(existing)
    n = len(people)
    if n < 4:
        return 0
    swaps = 0
    marg = {a: {} for a in attrs}
    for lv in people:
        for a in attrs:
            marg[a][lv[a]] = marg[a].get(lv[a], 0) + 1
    cell = {}
    for lv in people:
        for i, a in enumerate(attrs):
            for b in attrs[i + 1:]:
                key = (a, lv[a], b, lv[b])
                cell[key] = cell.get(key, 0) + 1

    def key_of(a, la, b, lb):
        return (a, la, b, lb) if attrs.index(a) < attrs.index(b) else (b, lb, a, la)

    def cost(a, la, b, lb, o):
        e = marg[a][la] * marg[b][lb] / n
        return (o - e) ** 2 / e

    ic = cfg.get("interactions") or {}
    need_pairs = int(ic.get("min_personas_per_pair", 0) or 0)
    guard = need_pairs and ic.get("pairs")
    pair_min = {lab: min(c, need_pairs) for lab, c in pair_counts(cfg, people)} if guard else {}

    by_attr = {a: [i for i in range(first, n) if people[i][a] != NONE] for a in attrs}
    tries = rounds * max(len(new), 1) * len(attrs)
    for _ in range(tries):
        a = rng.choice(attrs)
        idx = by_attr[a]
        if len(idx) < 2:
            continue
        p, q = rng.sample(idx, 2)
        lp, lq = people[p][a], people[q][a]
        if lp == lq:
            continue
        delta, changes = 0.0, {}
        for b in attrs:
            if b == a:
                continue
            for who, old, nw in ((p, lp, lq), (q, lq, lp)):
                lb = people[who][b]
                for lvl, d in ((old, -1), (nw, +1)):
                    k = key_of(a, lvl, b, lb)
                    changes[k] = changes.get(k, 0) + d
        for k, d in changes.items():
            if d:
                o = cell.get(k, 0)
                ka, la, kb, lb = k
                delta += cost(ka, la, kb, lb, o + d) - cost(ka, la, kb, lb, o)
        if delta >= -1e-9:
            continue
        sp, sq = dict(people[p], **{a: lq}), dict(people[q], **{a: lp})
        if find_conflicts(cfg, sp) or find_conflicts(cfg, sq):
            continue
        if any((a2, l2) in skip for a2, l2 in ((a, lq), (a, lp))):
            continue
        if guard:
            trial = people[:p] + [sp] + people[p + 1:]
            trial = trial[:q] + [sq] + trial[q + 1:]
            if any(c < pair_min[lab] for lab, c in pair_counts(cfg, trial)):
                continue
        people[p][a], people[q][a] = lq, lp          # new 裡的 dict 和 people 是同一個物件
        swaps += 1
        for k, d in changes.items():
            cell[k] = cell.get(k, 0) + d
    return swaps


# ---------- 組成自我介紹 ----------

class _Fill(dict):
    """人物句佔位符：用到才計算，所以輪流的族名、語言只在真的需要時才往下輪。"""

    def __init__(self, fns):
        super().__init__()
        self.fns = fns

    def __missing__(self, key):
        if key not in self.fns:
            return "{" + key + "}"
        self[key] = self.fns[key]()
        return self[key]


def _rotate(options, used, tag):
    """從 options 裡挑用得最少的一個（依序），並記下用過。"""
    pick = min(options, key=lambda o: (used.get((tag, o), 0), options.index(o)))
    used[(tag, pick)] = used.get((tag, pick), 0) + 1
    return pick


def build_texts(cfg, levels, name, used, fixed=None):
    """回傳 ({variant: 自我介紹}, {屬性: 揭露版標籤句}, {用到的佔位符: 值})。
    組法（〈人物句與題目補充〉）：開頭「我叫{name}。」，接著依屬性在設定檔裡的固定順序寫出各句。
    隱性版只放線索句；揭露版 = 每個屬性先放線索句（有的話），緊接著放標籤句，其餘文字逐字相同。
    fixed：指定佔位符的值（配對確認重組既有人物時用，族名、新住民母語才會和原本相同）。"""
    fixed = {k: v for k, v in (fixed or {}).items() if v}
    attrs = cfg["attributes"]
    ph = cfg.get("placeholders", {})
    lean = name.get("gender_lean")
    partner = ph.get("partner", {}).get(lean, {"same": "伴侶", "opposite": "伴侶"})
    origin = name.get("origin_country")
    group = (name.get("ethnic_group") or "").removesuffix("族") if name.get("name_kind") == "ethnicity:indigenous" else ""
    tribes = ph.get("tribes") or ["阿美"]
    fill = _Fill({
        "origin": lambda: origin or "【待補】原國籍",
        "tribe": lambda: fixed.get("tribe") or (group if group in tribes else _rotate(tribes, used, "族名輪流")),
        "partner_same": lambda: partner["same"],
        "partner_opposite": lambda: partner["opposite"],
        "origin_language": lambda: fixed.get("origin_language") or (ph.get("origin_languages", {}).get(origin) if origin else None)
        or _rotate(ph.get("new_immigrant_languages") or ["越南語"], used, "語言輪流"),
    })
    head = cfg.get("name_template", "我叫{name}。").format(name=name["name"])
    imp, exp, labels = [head], [head], {}
    for a in attrs:                          # 固定順序：設定檔裡屬性的順序
        if levels.get(a, NONE) == NONE:
            continue
        lc = attrs[a]["levels"][levels[a]]
        cue = lc.get("implicit") if any(k in SENTENCE_CUES for k in cue_kinds(lc)) else None
        if cue:
            cue = cue.format_map(fill)
            imp.append(cue)
            exp.append(cue)
        labels[a] = lc["explicit"].format_map(fill)
        exp.append(labels[a])
    used_ph = {k: v for k, v in fill.items()}
    return {"explicit": "".join(exp), "implicit": "".join(imp)}, labels, used_ph


def visible_levels(cfg, levels, variant):
    """這一版看得到的群體；隱性版裡只做揭露的群體算「未提及」。"""
    if variant == "explicit":
        return dict(levels)
    return {a: (l if l != NONE and implicit_visible(cfg["attributes"][a]["levels"][l]) else NONE) for a, l in levels.items()}


# ---------- 指令 ----------

def personas_dir(out_dir, cfg, draft=False):
    return os.path.join(out_dir, "personas", cfg["version"]["personas"] + ("-draft" if draft else ""))


def load_personas(out_dir, cfg, version=None):
    """讀目前設定檔版本的人物；正式版不在時退回 -draft 版。回傳 (人物紀錄, 資料夾)。
    version：指定讀哪一版（例如 p1.1），用來把舊版人物的試跑接著跑完。"""
    if version:
        d = os.path.join(out_dir, "personas", version)
        return read_jsonl(os.path.join(d, "personas.jsonl")), d
    for draft in (False, True):
        d = personas_dir(out_dir, cfg, draft)
        recs = read_jsonl(os.path.join(d, "personas.jsonl"))
        if recs:
            return recs, d
    return [], personas_dir(out_dir, cfg)


def personas_for_run(out_dir, cfg, run_dir):
    """這組回覆當初用的人物：依資料夾記的人物版本讀（例如 p1.1）；那一版不在時退回目前設定檔的版本。
    人物版本升級後，舊版的試跑資料仍然可以分析、評審。"""
    from . import layout
    pver = layout.read_settings(run_dir).get("personas_version")
    if pver:
        recs = read_jsonl(os.path.join(out_dir, "personas", pver, "personas.jsonl"))
        if recs:
            return recs, os.path.join(out_dir, "personas", pver)
    return load_personas(out_dir, cfg)


META = "personas_meta.json"


def write_meta(pdir, a, n_all, n_new, swaps, sentences=None):
    """人物檔的產生紀錄：程式版本、時間、亂數種子、是否做過屬性平衡互換。接續加人時附加一筆。"""
    from . import __version__
    from .common import save_json
    p = os.path.join(pdir, META)
    meta = load_json(p) if os.path.exists(p) else {"batches": []}
    from .common import file_fp
    from .names import FILES, names_dir
    nd = names_dir(read_config(a.config), a.config)
    names = {f: file_sha256(os.path.join(nd, f)) for f in FILES if os.path.isfile(os.path.join(nd, f))}
    meta["batches"].append({"program_version": __version__, "created": time.strftime("%Y-%m-%d %H:%M"),
                            "seed": a.seed, "new_people": n_new, "total_people": n_all,
                            "balance_swaps": swaps, "draft": bool(getattr(a, "draft", False)),
                            "persona_sentences": sentences,
                            # 重現與驗證用（v0.10.1 起）：同一版程式、同一份設定檔與姓名資料、同一個種子，產生的人物一模一樣
                            "config_fp": file_fp(a.config), "names_sha256": names,
                            "args": {"n": getattr(a, "n", None), "mention_k": getattr(a, "mention_k", 0),
                                     "min_per_group": getattr(a, "min_per_group", None)}})
    save_json(p, meta)


file_sha256 = text_sha256          # 舊名稱（tools 用）


def persona_info(cfg, pdir, plist):
    """人物檔的檢查結果（status 與報告開頭用）：版本、產生時的程式版本、有沒有做屬性平衡互換、矛盾檢查。"""
    p = os.path.join(pdir, META)
    meta = load_json(p) if os.path.exists(p) else None
    rows = [("人物版本", f"{os.path.basename(pdir)}（{len({x['id'] for x in plist})} 位 × 兩版）")]
    if meta and meta.get("batches"):
        b = meta["batches"]
        rows.append(("產生時的程式", "、".join(f"v{x['program_version']}（{x['created']}，{x['new_people']} 位"
                                               + (f"，人物句 {x['persona_sentences']}" if x.get("persona_sentences") else "") + "）"
                                               for x in b)))
        rows.append(("屬性平衡互換", "、".join(f"{x['balance_swaps']} 次" for x in b)
                     + "（v0.8.6 起：指派後在人物之間互換同屬性的群體，屬性兩兩之間更平衡）"))
    else:
        rows.append(("產生時的程式", "沒有記錄（v0.9.1 以前產生的人物檔沒有 personas_meta.json）"))
        rows.append(("屬性平衡互換", "不明：v0.8.6 起才有；v0.8.6 以前產生的人物（例如 p1.1、p1.2）沒有做過"))
    bad = sum(1 for x in plist if x.get("variant") == "explicit" and find_conflicts(cfg, x.get("true_levels") or x["levels"]))
    rows.append(("矛盾檢查", "依目前設定檔的互斥規則沒有矛盾" if not bad else
                 f"⚠ {bad} 位人物違反目前設定檔的互斥規則（設定檔的規則在人物產生後改過；gen --check 看明細）"))
    return rows


def cmd_gen(a):
    cfg = read_config(a.config)
    attrs = cfg["attributes"]
    k = a.mention_k or cfg.get("mention_k", 4)
    min_pg = cfg.get("min_per_group", 10) if a.min_per_group is None else a.min_per_group
    todo = placeholders(cfg)
    if todo and not a.draft:
        stop(f"設定檔還有 {len(todo)} 處待補內容：不能產生正式人物", *todo[:12], "…" if len(todo) > 12 else None,
             "只試流程：加 --draft（人物版本標成 -draft，不能和正式資料混用）")
    # 正式模式：暫停的群體（例如姓名資料不足的外籍三群）不指派；--draft 照樣指派，用測試名字
    skip = set() if a.draft else paused_groups(cfg)
    if a.n is None:
        # 外籍三群暫停時 104 人；恢復（或 --draft 照樣指派）時 108 人（41 群放不下 104 人）
        a.n = int(cfg.get("n_personas", 104)) if skip else int(cfg.get("n_personas_with_foreign", cfg.get("n_personas", 104)))
    skip_kinds = {f"{x}:{l}" for x, l in skip}
    pools, files = names.build_pools(cfg, a.config, allow_test=False)
    missing = [m for m in names.missing_pools(cfg, pools) if m[0] not in skip_kinds]
    if missing and not a.draft and not a.check:
        labels = list(dict.fromkeys(names.kind_label(cfg, *m) for m in missing))
        stop(f"姓名不夠：不能產生正式人物（缺 {len(labels)} 類）", *labels,
             f"姓名資料檔放在：{names.names_dir(cfg, a.config)}（目前讀到：{'、'.join(files) or '沒有'}）",
             "外籍身分的原文名：names_foreign.jsonl", "資料不足的群體：先列進設定檔 paused_groups", "只試流程：加 --draft")
    if a.draft:
        pools, files = names.build_pools(cfg, a.config, allow_test=True)   # 測試名字（test_only）只在 --draft 用
        missing = names.missing_pools(cfg, pools)
        names.fill_drafts(pools, missing)       # 連測試名字都沒有的類別，用占位姓名
    if files:
        print(f"姓名資料檔：{'、'.join(files)}" + (f"（缺 {len(missing)} 類：先用測試或占位姓名）" if missing else ""))
    if skip:
        print(bullets("暫停的群體（不產生人物；報告列為「資料不足，未測」）",
                      [f"{attrs[x]['label']}／{attrs[x]['levels'][l]['label']}" for x, l in sorted(skip)]))
    pdir = personas_dir(a.out, cfg, draft=bool(todo or missing or a.draft))
    path = os.path.join(pdir, "personas.jsonl")

    if a.check:
        recs = read_jsonl(path)
        if not recs:
            stop(f"找不到人物：{path}")
        bad = 0
        for r in recs:
            if r["variant"] != "explicit":
                continue
            known = all(l == NONE or l in attrs.get(x, {}).get("levels", {}) for x, l in r["true_levels"].items())
            cs = find_conflicts(cfg, r["true_levels"]) if known else ["群體名稱與目前的設定檔對不上"]
            if cs:
                bad += 1
                print(f"{r['id']} {r['intro']}\n    ✗ " + "\n    ✗ ".join(cs))
        print(f"\n{'✓' if not bad else '⚠'} {len(recs) // 2} 位人物中，{bad} 位依目前的設定檔有矛盾")
        return

    os.makedirs(pdir, exist_ok=True)
    existing = read_jsonl(path)
    base = [r for r in existing if r["variant"] == "explicit"]
    if existing and not a.extend:
        has = layout.response_files(a.out)
        if has and not a.force:
            stop(f"已有人物，而且 {a.out} 裡有回覆資料", f"人物檔：{path}", "重新產生會讓舊回覆對不上新人物",
                 "要加人：--extend（既有人物不變，新人物接在後面）", "要重來：--force（舊回覆改名備份）")
        for rp in has:
            os.replace(rp, layout.backup_path(rp))
        existing, base = [], []
    n_new = a.n - len(base)
    if n_new <= 0:
        stop(f"已有 {len(base)} 位人物：--n {a.n} 沒有要新增的")

    need = needed_personas(cfg, k, min_pg, skip)
    groups = group_list(cfg, skip)
    enforce = a.n >= need
    print(kv_block("規模", [("屬性", f"{len(attrs)} 個（每人 {k} 個）"), ("群體", f"{len(groups)} 個"),
                           ("需要", f"{need} 位人物（每群至少 {min_pg} 人）"), ("這次", f"{a.n} 位")]))
    if not enforce:
        warn(f"每群平均約 {a.n * k / len(groups):.1f} 人：只能確認流程", "報告的格子多半是「資料不足」")
    rng = random.Random(a.seed * 1009 + len(base))
    result, info = None, ""
    for _ in range(60):
        result, info = assign_levels(cfg, n_new, k, random.Random(rng.random()), [r["true_levels"] for r in base],
                                     min_pg if enforce else 0, skip)
        if result:
            break
    if not result and a.draft and enforce:
        # 試流程：每群至少 min_per_group 人做不到時（例如外籍三群也指派、名額不夠），放寬名額照樣產生
        warn(info, f"--draft：先放寬每群人數照樣產生（正式人物仍要每群 {min_pg} 人）")
        for _ in range(20):
            result, info = assign_levels(cfg, n_new, k, random.Random(rng.random()), [r["true_levels"] for r in base], 0, skip)
            if result:
                break
    if not result:
        stop(f"指派失敗（試了 60 次）：{info}", "增加人數", "放寬 min_per_group", "檢查 exclude 與 facts")

    used_names = {}
    for r in base:
        names.remember(used_names, r)
        for tag, key in (("族名輪流", "tribe"), ("語言輪流", "origin_language")):
            if r.get(key):
                used_names[(tag, r[key])] = used_names.get((tag, r[key]), 0) + 1
    new = []
    for i, levels in enumerate(result):
        pid = f"p{len(base) + i:03d}"
        kind = names.name_kind(cfg, levels)
        lean = names.name_lean(levels, kind, used_names)
        name = names.pick(pools, kind, lean, rng, used_names)
        if not name:
            stop(f"{pid} 找不到可用的姓名", f"需要：{names.kind_label(cfg, kind, lean)}")
        texts, labels, filled = build_texts(cfg, levels, name, used_names)
        extra = {k: name[k] for k in ("name_kind", "given_name", "stratum", "origin_country") if name.get(k)}
        extra.update({k: filled[k] for k in ("tribe", "origin_language") if k in filled})
        if name.get("test_only"):
            extra["name_test_only"] = True
        for v in VARIANTS:
            new.append({"id": pid, "variant": v, "levels": visible_levels(cfg, levels, v), "true_levels": levels,
                        "name": name["name"], "name_gender": name["gender_lean"], "name_source": name.get("source"),
                        **extra, "labels": labels,
                        "intro": texts[v], "chars": len(texts[v]),
                        "length_diff": len(texts["explicit"]) - len(texts["implicit"]),   # 比較兩版時當控制變項
                        "personas_version": os.path.basename(pdir)})
    recs = existing + new
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    review = os.path.join(pdir, "personas_review.md")
    write_review(cfg, recs, review)
    write_meta(pdir, a, len(recs) // 2, len(new) // 2, SWAPS["last"], cfg.get("version", {}).get("persona_sentences"))
    counts = {g: 0 for g in groups}
    for r in recs:
        if r["variant"] == "explicit":
            for x, l in r["true_levels"].items():
                if (x, l) in counts:
                    counts[(x, l)] += 1
    low = sorted((c, g) for g, c in counts.items())[:3]
    diffs = [r["length_diff"] for r in recs if r["variant"] == "explicit" and "length_diff" in r]
    print(kv_block("這次結果", [
        ("新人物", f"{len(new) // 2} 位（共 {len(recs) // 2} 位，每位兩版）"), ("人物檔", path),
        ("審閱清單", f"{review}（請逐一讀過）"),
        ("人數最少的群體", [f"{attrs[x]['label']}／{attrs[x]['levels'][l]['label']}：{c} 人" for c, (x, l) in low]),
        ("交互作用組合", [f"{pair}：{n_have} 位" for pair, n_have in
                       pair_counts(cfg, [r["true_levels"] for r in recs if r["variant"] == "explicit"])] or None),
        ("兩版字數差", f"揭露減隱性 {min(diffs)}～{max(diffs)} 字，平均 {sum(diffs) / len(diffs):.0f} 字" if diffs else None)]))
    if todo or missing or a.draft:
        n_test = sum(1 for r in new if r.get("name_test_only")) // 2
        warn("這批人物是 -draft：只能試流程", f"待補內容：{len(todo)} 處", f"缺姓名：{len(missing)} 類",
             f"用了測試名字：{n_test} 位" if n_test else None)
    from .common import cmd_hint, next_steps
    print(next_steps(["讀過審閱清單（personas_review.md）",
                      "第 2 步 生成回覆：LM Studio 載入受測模型，再執行 " + cmd_hint("run", "--model <受測模型>"),
                      "只想試流程：" + cmd_hint("run", "--mock")]))


def write_review(cfg, recs, path):
    attrs = cfg["attributes"]
    by_id = {}
    for r in recs:
        by_id.setdefault(r["id"], {})[r["variant"]] = r
    L = ["# 人物審閱", "",
         "逐一讀過兩版。覺得不合理的，把原因記下來，回頭在設定檔補 facts 或 exclude，再重新 gen。",
         "字數差＝揭露版減隱性版的字數（不設上限，分析時當控制變項）。姓名後標「測試」的是 Faker 測試名字，只能試流程。", "",
         "| 編號 | 姓名（名字性別，分層） | 屬性 | 隱性版 | 揭露版 | 字數差 |", "| --- | --- | --- | --- | --- | --- |"]
    for pid, v in by_id.items():
        e, i = v["explicit"], v["implicit"]
        tags = "、".join(f"{attrs[x]['label']}:{attrs[x]['levels'][l]['label']}" + ("" if i["levels"][x] != NONE else "（只揭露）")
                        for x, l in e["true_levels"].items() if l != NONE)
        L.append(f"| {pid} | {e['name']}（{names.LEAN_ZH.get(e.get('name_gender'), '?')}"
                 f"{'，' + e['stratum'] if e.get('stratum') else ''}{'，測試' if e.get('name_test_only') else ''}） | "
                 f"{tags} | {i['intro']} | {e['intro']} | {e.get('length_diff', '')} |")
    pairs, keys = [], list(attrs)
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            hit = sum(bool(find_conflicts(cfg, {**{n: NONE for n in keys}, keys[x]: lx, keys[y]: ly}))
                      for lx in attrs[keys[x]]["levels"] for ly in attrs[keys[y]]["levels"])
            if hit:
                pairs.append(f"- {attrs[keys[x]]['label']} × {attrs[keys[y]]['label']}：{hit} 種組合被規則排除")
    L += ["", "## 受互斥規則影響、無法完全獨立的屬性配對", ""] + (pairs or ["- 沒有"])
    pc = pair_counts(cfg, [v["explicit"]["true_levels"] for v in by_id.values()])
    if pc:
        L += ["", "## 交互作用組合（只當訊號）", ""] + [f"- {lab}：{n} 位" for lab, n in pc]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
