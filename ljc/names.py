"""姓名：讀取 config/names/ 的姓名資料檔，組成各類姓名池，替人物挑名字。

哪一類姓名由人物的群體決定：群體的 implicit_cue 含 name 時，用那個群體的姓名池；都沒有時用一般姓名。國籍優先於族群。

性別（企劃書 v1）：v1 不用中性名。性別在屬性內時，名字的性別與它一致；性別不在屬性內時，男、女名各半
（在同一類姓名裡輪流，用得少的那個性別先），名字的性別記在人物紀錄的 name_gender，分析時當控制變項。

姓名池（資料檔有就用資料檔，沒有就用設定檔 names 直接列出的完整姓名）：
- 一般（names_general.jsonl）：姓氏前 100 大（等機率）＋名字（男、女各前 100 大與各出生年代前 10 大）。
  使用人數少於 names_min_count（預設 100）的名字不用；同時出現在男、女名單的名字不用（性別不明確）。
- 原住民族（names_indigenous.jsonl）：personal_name，只寫族名。依（族別, 名字）去重複；兩個來源性別標示對不上、
  或是男女共名的不用。16 族輪流。
- 外籍身分（婚姻移民、移工、外籍專業人士；names_foreign.jsonl）：原文名，名字配合原國籍並記錄性別。
  原國籍清單寫在設定檔 origin_countries，各國輪流、數量相同。資料檔可以給完整姓名（full_name），
  或姓氏與名字分開（surname、given_name），再由 naming_rule 指定順序（family_first／given_first／given_only）與分隔。
  格式見〈原文姓名資料檔規格〉。缺任何一國、任何一個性別的名字，gen 都會列出來並拒絕產生正式人物。"""
import os

from .common import NONE, read_jsonl

FILES = ("names_general.jsonl", "names_indigenous.jsonl", "names_foreign.jsonl", "names_foreign_test.jsonl")
LEANS = ("female", "male")          # v1 不用中性名
LEAN_ZH = {"female": "女性", "male": "男性"}
TODO = "【待補】"


def names_dir(cfg, config_path):
    """姓名資料檔的資料夾：設定檔的 names_dir（相對於設定檔所在的資料夾），預設 config/names/。"""
    d = cfg.get("names_dir", "names")
    return d if os.path.isabs(d) else os.path.join(os.path.dirname(os.path.abspath(config_path)), d)


def read_records(cfg, config_path):
    """回傳 {檔名: 紀錄清單}；檔案不在就不列。"""
    d = names_dir(cfg, config_path)
    return {f: read_jsonl(os.path.join(d, f)) for f in FILES if os.path.isfile(os.path.join(d, f))}


def _src(r):
    s = r.get("source") or {}
    if isinstance(s, str):
        return s
    return " ".join(x for x in (s.get("source_id"), s.get("locator")) if x)


def _lean(r):
    """紀錄的性別。姓氏等沒有性別的紀錄回傳 None；也接受 sex／gender 欄位。男女共名（neutral）回傳 neutral。"""
    v = r.get("gender_lean") or r.get("sex") or r.get("gender")
    v = {"男": "male", "女": "female", "M": "male", "F": "female"}.get(v, v)
    return v if v in ("female", "male", "neutral") else None


def _common_enough(r, min_count):
    """使用人數低於門檻的名字不用；來源沒有人數的紀錄照常使用。"""
    c = r.get("count")
    return not isinstance(c, (int, float)) or c >= min_count


def _entry(name, lean, source, **meta):
    return {"name": name, "gender_lean": lean, "source": source, **meta}


# ---------- 從資料檔組成姓名池 ----------
# 姓名池的格式：{"why": 說明, "strata": [{"label": 分層名稱, "given": [名字], "surnames": [姓氏] 或 None,
#                                        "order": 姓名順序, "sep": 分隔}]}
# 挑名字時先挑用得最少的一層，再挑那層裡用得最少的名字；有姓氏清單時，再隨機配一個姓。

def _general_pools(recs, min_count):
    surnames = [_entry(r["name"], None, _src(r)) for r in recs if r.get("record_type") == "surname" and r.get("name")]
    by_name = {}
    for r in recs:
        if r.get("record_type") in ("given_name", "given_name_by_birth_period") and r.get("name") and _lean(r) \
                and _common_enough(r, min_count):
            e = by_name.setdefault(r["name"], {"leans": set(), "src": _src(r)})
            e["leans"].add(_lean(r))
    pools = {}
    if not surnames:
        return pools
    for lean in LEANS:
        given = [n for n, e in by_name.items() if e["leans"] == {lean}]    # 男女名單都有的名字不用
        if given:
            pools[("general", lean)] = {"why": "一般姓名", "strata": [{
                "label": "一般", "surnames": surnames, "order": "family_first", "sep": "",
                "given": [_entry(n, lean, by_name[n]["src"]) for n in sorted(given)]}]}
    return pools


def _indigenous_pools(recs):
    merged = {}
    for r in recs:
        if r.get("record_type") != "personal_name" or not r.get("name"):
            continue
        key = (r.get("ethnic_group") or "?", r["name"])
        e = merged.setdefault(key, {"leans": set(), "src": []})
        e["leans"].add(_lean(r))
        e["src"].append(_src(r))
    pools = {}
    for lean in LEANS:
        strata = {}
        for (grp, name), e in sorted(merged.items()):
            if e["leans"] == {lean}:            # 性別標示不一致、男女共名的名字不用
                strata.setdefault(grp, []).append(_entry(name, lean, "；".join(e["src"]), ethnic_group=grp))
        if strata:
            pools[("ethnicity:indigenous", lean)] = {"why": "原住民族名", "strata": [
                {"label": g, "surnames": None, "given": v} for g, v in sorted(strata.items())]}
    return pools


def origin_countries(cfg, level):
    """這個外籍身分要輪流的原國籍；設定檔沒寫時回傳 None（由資料決定），還是【待補】時回傳 []。"""
    v = (cfg.get("origin_countries") or {}).get(level)
    if v is None:
        return None
    if isinstance(v, str) or any(TODO in c for c in v):
        return []
    return list(v)


def _single_year(r, single):
    """只有單年資料的來源（例如蘇格蘭、澳洲只有 2025 年新生兒名字）：紀錄自己標了 single_year，或來源代號在設定檔清單裡。"""
    sid = (r.get("source") or {}).get("source_id") if isinstance(r.get("source"), dict) else None
    return bool(r.get("single_year")) or (sid in single)


def _rank_value(r, single):
    """依排名取前 top_n 名的紀錄用什麼排序：WGND 的 nobs；單年資料的使用人數。其餘回傳 None（套使用人數門檻）。"""
    if isinstance(r.get("nobs"), (int, float)):
        return r["nobs"]
    if _single_year(r, single) and isinstance(r.get("count"), (int, float)):
        return r["count"]
    return None


def _top_by_nobs(rs, top_n, single=()):
    """WGND 這類以出現次數（nobs）排序的資料，以及只有單年資料的來源：每國每性別只取前 top_n 名（不套使用人數門檻）。"""
    ranked = [r for r in rs if _rank_value(r, single) is not None]
    if not ranked:
        return rs
    keep = set()
    for key in {(r.get("record_type"), _lean(r)) for r in ranked}:
        grp = sorted((r for r in ranked if (r.get("record_type"), _lean(r)) == key), key=lambda r: -_rank_value(r, single))
        keep |= {id(r) for r in grp[:top_n]}
    return [r for r in rs if _rank_value(r, single) is None or id(r) in keep]


def _foreign_strata(recs, country, lean, min_count, top_n=100, single=()):
    """某一國、某一性別的原文名分層；沒有資料回傳 None。"""
    rs = [r for r in recs if r.get("origin_country") == country and r.get("name")
          and (_rank_value(r, single) is not None or _common_enough(r, min_count))]
    rs = _top_by_nobs(rs, top_n, single)
    full = [r for r in rs if r.get("record_type") == "full_name" and _lean(r) == lean]
    if full:
        return {"label": country, "surnames": None,
                "given": [_entry(r["name"], lean, _src(r), origin_country=country,
                                 **({"test_only": True} if r.get("test_only") else {})) for r in full]}
    rule = next((r for r in recs if r.get("record_type") == "naming_rule" and r.get("origin_country") == country), {})
    order = rule.get("order", "given_first")
    given = [r for r in rs if r.get("record_type") == "given_name" and _lean(r) == lean]
    sur = [r for r in rs if r.get("record_type") == "surname" and _lean(r) in (None, lean)]
    if not given or (order != "given_only" and not sur):
        return None
    return {"label": country, "order": order, "sep": rule.get("separator", " "),
            "surnames": None if order == "given_only" else [_entry(r["name"], None, _src(r)) for r in sur],
            "given": [_entry(r["name"], lean, _src(r), origin_country=country,
                             **({"test_only": True} if r.get("test_only") else {})) for r in given]}


def _inline_pools(names):
    """設定檔 names 直接列出的完整姓名（測試用設定檔、或資料檔沒有的類別）。有 origin_country 就依國家分層。"""
    pools = {}
    for n in names or []:
        if n.get("gender_lean") not in LEANS:
            continue
        if n.get("nationality"):
            kind = f"nationality:{n['nationality']}"
        elif n.get("ethnicity"):
            kind = f"ethnicity:{n['ethnicity']}"
        else:
            kind = "general"
        p = pools.setdefault((kind, n["gender_lean"]), {"why": "設定檔 names", "strata": []})
        label = n.get("origin_country") or "設定檔"
        st = next((s for s in p["strata"] if s["label"] == label), None)
        if not st:
            st = {"label": label, "surnames": None, "given": []}
            p["strata"].append(st)
        st["given"].append(dict(n))
    return pools


def build_pools(cfg, config_path, allow_test=False):
    """回傳 ({(類別, 性別): 姓名池}, 讀到的資料檔清單)。資料檔的姓名池優先，設定檔 names 補資料檔沒有的。
    allow_test=True（--draft）時，正式資料沒有的國家改用 names_foreign_test.jsonl 的測試名字（test_only）；
    正式模式一律不用測試名字。"""
    files = read_records(cfg, config_path)
    if not allow_test:
        files.pop("names_foreign_test.jsonl", None)
    min_count = int(cfg.get("names_min_count", 100))
    top_n = int(cfg.get("names_top_n", 100))
    single = set(cfg.get("names_single_year_sources", []))
    pools = _inline_pools(cfg.get("names"))
    pools.update(_general_pools(files.get("names_general.jsonl", []), min_count))
    pools.update(_indigenous_pools(files.get("names_indigenous.jsonl", [])))
    foreign = [r for r in files.get("names_foreign.jsonl", []) if not r.get("test_only")]
    test = [dict(r, test_only=True) for r in files.get("names_foreign_test.jsonl", [])]
    for level in (cfg.get("origin_countries") or {}):
        if level.startswith("_"):
            continue
        countries = origin_countries(cfg, level) or []
        for lean in LEANS:
            kind = f"nationality:{level}"
            inline = {s["label"]: s for s in pools.get((kind, lean), {}).get("strata", [])}
            strata = []
            for c in countries:
                st = _foreign_strata(foreign, c, lean, min_count, top_n, single) or inline.get(c) \
                    or (_foreign_strata(test, c, lean, min_count, top_n, single) if test else None)
                if st:
                    strata.append(st)
            if strata:
                pools[(kind, lean)] = {"why": "原文名", "strata": strata}
            else:
                pools.pop((kind, lean), None)
    return pools, sorted(files)


# ---------- 這位人物要哪一類姓名 ----------

def name_kind(cfg, levels):
    """依群體決定姓名類別：國籍或族群的 implicit_cue 含 name 時用該群體的姓名，否則用一般姓名。"""
    for attr in ("nationality", "ethnicity"):
        lv = levels.get(attr, NONE)
        if lv == NONE:
            continue
        lc = cfg["attributes"].get(attr, {}).get("levels", {}).get(lv, {})
        cue = lc.get("implicit_cue", "none")
        if "name" in (cue if isinstance(cue, list) else [cue]):
            return f"{attr}:{lv}"
    return "general"


def name_lean(levels, kind, used):
    """名字的性別：性別在屬性內就照它；不在時，同一類姓名裡男女輪流（用得少的先），各半。"""
    g = levels.get("gender", NONE)
    if g in LEANS:
        return g
    lean = min(LEANS, key=lambda l: (used.get(("性別", kind, l), 0), LEANS.index(l)))
    used[("性別", kind, lean)] = used.get(("性別", kind, lean), 0) + 1
    return lean


def needed_kinds(cfg):
    """所有可能用到的姓名類別：[(類別, 性別, 原國籍或 None)]。"""
    out = []
    kinds = ["general"]
    for attr in ("nationality", "ethnicity"):
        for lv in cfg["attributes"].get(attr, {}).get("levels", {}):
            k = name_kind(cfg, {attr: lv})
            if k != "general":
                kinds.append(k)
    for k in kinds:
        level = k.split(":", 1)[1] if ":" in k else None
        countries = origin_countries(cfg, level) if k.startswith("nationality:") else None
        for lean in LEANS:
            if countries is None:
                out.append((k, lean, None))
            elif not countries:
                out.append((k, lean, TODO))          # 原國籍清單還沒定
            else:
                out += [(k, lean, c) for c in countries]
    return out


def kind_label(cfg, kind, lean, country=None):
    if kind == "general":
        head = "一般姓名"
    else:
        attr, lv = kind.split(":", 1)
        head = cfg["attributes"].get(attr, {}).get("levels", {}).get(lv, {}).get("label", lv) + "的姓名"
    if country == TODO:
        return f"{head}：原國籍清單（設定檔 origin_countries）還是【待補】"
    return f"{head}（{(country + '，') if country else ''}{LEAN_ZH[lean]}）"


def missing_pools(cfg, pools):
    """缺的姓名類別：[(類別, 性別, 原國籍或 None)]。"""
    out = []
    for k, lean, c in needed_kinds(cfg):
        strata = pools.get((k, lean), {}).get("strata", [])
        if c is None:
            ok = any(s["given"] for s in strata)
        else:
            ok = c != TODO and any(s["label"] == c and s["given"] for s in strata)
        if not ok:
            out.append((k, lean, c))
    return out


def fill_drafts(pools, missing):
    """試流程用：缺的類別補上占位姓名。外籍身分若有些國家已有名字（正式或測試），只輪流那些國家，不補占位。"""
    for kind, lean, c in missing:
        p = pools.setdefault((kind, lean), {"why": "占位", "strata": []})
        if c not in (None, TODO) and any(s["given"] and not s.get("placeholder") for s in p["strata"]):
            continue
        label = "占位" if c in (None, TODO) else c
        if not any(s["label"] == label for s in p["strata"]):
            meta = {"origin_country": c} if c not in (None, TODO) else {}
            p["strata"].append({"label": label, "surnames": None, "placeholder": True, "given": [
                _entry(f"{TODO}{kind.split(':')[-1]}{lean}{label if label != '占位' else ''}{i}", lean, "占位", **meta)
                for i in range(8)]})


def pick(pools, kind, lean, rng, used):
    """挑一個名字。used 記錄用過的次數（名字、分層、性別），讓名字盡量不重複、各分層與男女輪流用到。"""
    pool = pools.get((kind, lean))
    if not pool:
        return None
    strata = [s for s in pool["strata"] if s["given"]]
    fewest = min(used.get(("層", kind, s["label"]), 0) for s in strata)
    st = rng.choice([s for s in strata if used.get(("層", kind, s["label"]), 0) == fewest])
    fewest = min(used.get(g["name"], 0) for g in st["given"])
    g = rng.choice([g for g in st["given"] if used.get(g["name"], 0) == fewest])
    for key in (("層", kind, st["label"]), g["name"]):
        used[key] = used.get(key, 0) + 1
    tag = {"name_kind": kind, "given_name": g["name"], "stratum": st["label"]}
    if not st.get("surnames"):
        return dict(g, **tag)
    s = rng.choice(st["surnames"])
    sep = st.get("sep", "")
    full = (g["name"] + sep + s["name"]) if st.get("order") == "given_first" else (s["name"] + sep + g["name"])
    meta = {k: v for k, v in {**s, **g}.items() if k not in ("name", "gender_lean", "source")}
    return _entry(full, lean, f"姓：{s['source']}｜名：{g['source']}", **meta, **tag)


def remember(used, rec):
    """接續加人時，把既有人物用過的名字、分層與性別記進 used。"""
    g = rec.get("given_name") or rec["name"]
    used[g] = used.get(g, 0) + 1
    kind = rec.get("name_kind")
    if kind and rec.get("stratum"):
        k = ("層", kind, rec["stratum"])
        used[k] = used.get(k, 0) + 1
    lean = rec.get("name_gender") or rec.get("name_gender_lean")
    if kind and lean in LEANS and rec.get("true_levels", {}).get("gender", NONE) not in LEANS:   # 只算性別不在屬性內的人物
        k = ("性別", kind, lean)
        used[k] = used.get(k, 0) + 1
