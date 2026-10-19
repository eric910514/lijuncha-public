"""analyze 指令：讀回覆 → 算指標 → 比較群體 → 寫報告。這裡只安排順序，計算在 metrics 與 stats，排版在 report。

- 主流程的資料夾：逐屬性比較群體（report.md、results.csv）。
- 對照模組的資料夾：同一批人物在不同寫法下的成對比較（簡繁、標點；無關擾動對照主流程同一組設定）。
- 只差在身分位置的兩組設定（message 與 system）：另寫一份比較報告，放在 out/compare__…/。"""
import os

import numpy as np

from . import __version__, joint, layout, report, screen, stats, verdict
from .common import read_config, guard_reserved, question_scope, question_scope_en, NONE, bullets, load_json, load_retired, load_valid, read_jsonl, stop, warn
from .judge import GPT_OSS_FAMILY, jkey, load_judgments, prompt_path, valid_judgment
from .metrics import (CORRECT_BY_TURN, EXEMPT_CODES, JUDGE_METRICS, STANCE_BY_TURN, KEYWORD_METRICS, SCRIPT_METRICS, SUBSET_METRICS, describe_metrics, echo_terms_by_attr, identity_echo,
                      judge_metrics, persona_vectors, reply_metrics, split_by_turn, standardize_numbers,
                      token_line, token_summary, write_recuts)
from .personas import VARIANT_LABEL, load_personas, personas_for_run
from .styles import MODULES, STYLE_LABEL, T2S, module_questions, stylers_for

TRUNC_WARN = 0.02     # 截斷率超過 2% 時在報告開頭警示


def cmd_analyze(a):
    """預設分析每一組設定（每個模型、每種身分位置……各一份報告）；--run 只分析名稱含有那段文字的。"""
    dirs = [d for d in layout.run_dirs(a.out) if os.path.isfile(os.path.join(d, layout.RESPONSES))]
    if a.mock:
        dirs = [d for d in dirs if os.path.basename(d).startswith("mock__")]
    if a.run:
        dirs = [d for d in dirs if a.run in os.path.basename(d)]
    if not dirs:
        stop("沒有可分析的回覆", f"沒有名稱含「{a.run}」的資料夾" if a.run else "先執行 run")
    run_analysis(a, dirs)


def run_analysis(a, dirs):
    cfg = read_config(a.config)
    plist, pdir = load_personas(a.out, cfg)
    for i, d in enumerate(dirs):
        if len(dirs) > 1:
            print(("\n" if i else "") + f"■ {os.path.basename(d)}")
        guard_reserved(cfg, os.path.join(d, layout.RESPONSES), "分析")
        pl = personas_for_run(a.out, cfg, d)[0] or plist
        if not pl:
            warn(f"找不到這組回覆用的人物（略過）：{pdir}", "先執行 gen")
            continue
        from .confirm import analyze_confirm, is_confirm
        if layout.read_settings(d).get("module"):
            analyze_module(a, cfg, pl, d)
        elif is_confirm(pl):
            analyze_confirm(a, cfg, pl, d)
        else:
            analyze_one(a, cfg, pl, d)
    for msg_dir, sys_dir in identity_pairs(dirs):
        print(f"\n■ 身分位置比較：{os.path.basename(msg_dir)} 與 system")
        compare_identity(a, cfg, personas_for_run(a.out, cfg, msg_dir)[0] or plist, msg_dir, sys_dir)


def uid(r):
    """分析單位：一筆人物版本，例如 p012|explicit。置換與重抽以人物為單位（兩版一起），見 joint。"""
    return f"{r.get('pid', r.get('id'))}|{r.get('variant')}"


def simp_lexicon(lex):
    """關鍵詞清單逐字轉成簡體（簡繁模組的簡體回覆用）。"""
    t2s = T2S()
    return {k: ([t2s.char(w) for w in v] if isinstance(v, list) else v) for k, v in lex.items()}


def reply_rows(convs, qcat, lex, judged=None, qmap=None, module=None, standardize=True):
    """逐則回覆 → 指標。字數用原文字元數；簡體版的關鍵詞初篩用轉成簡體的關鍵詞比對（回覆不轉回繁體）。
    judged：{回覆的鍵: 判定}；給了就用評審的指標，沒給就用關鍵詞初篩。"""
    out = []
    lex_s = None
    for c in convs:
        if c["qid"] not in qcat:
            continue
        lx = lex
        if c.get("style") == "simp" and judged is None:
            lex_s = lex_s or simp_lexicon(lex)
            lx = lex_s
        for t, text in enumerate(c["replies"]):
            row = {"pid": uid(c), "qid": c["qid"], "cat": qcat[c["qid"]], "rep": c["rep"], "turn": t,
                   "style": c.get("style", "orig"), "text": text, "machine": c.get("machine_label") or c.get("machine") or "未記錄"}
            # 截斷率以原本的結束原因為準（續寫過的對話看續寫前），才看得出截斷是否和群體有關
            fins = (c.get("before_continue") or {}).get("finish") or c.get("finish") or []
            fin = fins[t] if t < len(fins) else None
            if judged is None:
                row.update(reply_metrics(text, lx))
                row["截斷率"] = float(fin == "length") if fin else float("nan")
            else:
                j = judged.get(jkey(c, t))
                j = j if valid_judgment(j, c["replies"][t]) else None
                row.update(judge_metrics(text, j, qmap[c["qid"]], module))
                row.update(describe_metrics(text, j, qmap[c["qid"]], fin))
                split_by_turn(row, t)
            out.append(row)
    if judged is not None and standardize:
        standardize_numbers(out)
    return out


def choose_cohort(convs, plist, questions, people=0, reps=0):
    """分析的人物範圍：只收「每題都跑滿前 R 次重複」的人物（兩版都要），只用前 R 次重複。

    people：只看前幾位人物（依人物編號，和 run --people 相同）；reps：指定 R。
    沒指定 R 時自動挑：人物數 × R 最多的那個 R（一樣多時取人物多的），例如 24 人跑滿 4 次、
    另外 34 人只跑了第 1 次 → 24 人 × 4 次；104 人都跑完第 1 次、一半跑完第 2 次 → 104 人 × 1 次。
    回傳 (範圍內的對話, 摘要)。"""
    ids = list(dict.fromkeys(p["id"] for p in plist))
    if people:
        ids = ids[:people]
    variants = {}
    for p in plist:
        variants.setdefault(p["id"], set()).add(p.get("variant"))
    seen = {}
    for c in convs:
        seen.setdefault((c["pid"], c.get("variant")), {}).setdefault(c["qid"], set()).add(c["rep"])
    asked = {c["qid"] for c in convs}
    qids = [q["id"] for q in questions if q["id"] in asked]       # 一段都還沒跑的題目（例如剛換的新題）先不要求
    have_vars = {c.get("variant") for c in convs}

    def full(pid, r):
        for v in variants.get(pid, set()) & have_vars:
            got = seen.get((pid, v), {})
            if any(not set(range(r)) <= got.get(q, set()) for q in qids):
                return False
        return True

    max_r = max((c["rep"] + 1 for c in convs), default=0)
    by_r = {r: [p for p in ids if full(p, r)] for r in range(1, max_r + 1)}
    if reps:
        r = reps
        keep = by_r.get(r) or [p for p in ids if full(p, r)]
    else:
        cand = [(len(v) * r, len(v), r) for r, v in by_r.items() if v]
        r = max(cand)[2] if cand else 0
        keep = by_r.get(r, [])
    keep_set = set(keep)
    out = [c for c in convs if c["pid"] in keep_set and c["rep"] < r]
    info = {"people": len(keep), "reps": r, "all_people": len({c["pid"] for c in convs}),
            "options": {k: len(v) for k, v in by_r.items()}, "ids": keep, "limit_people": people or 0,
            "missing_questions": [q["id"] for q in questions if q["id"] not in asked],
            "dropped_convs": len(convs) - len(out)}
    return out, info


def cohort_label(cohort):
    """分析資料夾的名稱：24p_4rep＝24 位人物、每題 4 次重複；加了 --people 時前面加 first<N>_。"""
    return (f"first{cohort['limit_people']}_" if cohort["limit_people"] else "") + f"{cohort['people']}p_{cohort['reps']}rep"


def cohort_line(cohort):
    opts = "、".join(f"跑滿 {r} 次 {n} 人" for r, n in sorted(cohort["options"].items()))
    return bullets(f"分析範圍：{cohort['people']} 位人物 × 前 {cohort['reps']} 次重複", [
        ("資料裡", f"共 {cohort['all_people']} 位人物（{opts}）"), ("換範圍", "--people 或 --reps")])


def analysis_dir(run_dir, cohort=None):
    """分析結果另放：out/analysis/<設定資料夾>/<人物範圍>/。不同範圍各一個資料夾，不會互相蓋掉。"""
    out = os.path.dirname(os.path.abspath(run_dir))
    d = os.path.join(out, "analysis", os.path.basename(run_dir), *([cohort_label(cohort)] if cohort else []))
    os.makedirs(d, exist_ok=True)
    return d


def judged_for(cfg, a, run_dir, convs):
    """讀這個資料夾的評審結果。回傳 (判定或 None, 摘要)。一則都沒評過時回傳 None，報告改用關鍵詞初篩。"""
    pname = prompt_path(cfg, a.config)[0]
    js, skipped = load_judgments(run_dir, pname)
    total = sum(len(c["replies"]) for c in convs)
    ok = [js[jkey(c, t)] for c in convs for t, r in enumerate(c["replies"]) if valid_judgment(js.get(jkey(c, t)), r)]
    err = sum(1 for c in convs for t in range(len(c["replies"]))
              if jkey(c, t) in js and "judge_error" in js[jkey(c, t)])
    info = {"prompt": pname, "n": len(ok), "total": total, "errors": err, "skipped_version": skipped,
            # 名稱只差發行者（gpt-oss-20b 與 openai/gpt-oss-20b）算同一個評審
            "models": sorted({f"{j['judge_model'].split('/')[-1]}（{j.get('quantization') or '量化不明'}）" for j in ok}),
            "number_rejected": sum(1 for j in ok if j.get("number_in_text") is False),
            "think": sum(1 for j in ok if j.get("think_chars", 0) > 0),
            "machines": sorted({layout.machine_label(j) for j in ok})}
    return (js if ok else None), info


def judge_warnings(info):
    w = []
    if info["n"] == 0:
        w.append("這組回覆還沒經過評審模型（judge 指令），回覆類型、轉介等指標暫用關鍵詞初篩，只能當線索。"
                 + (f"另有 {info['skipped_version']} 筆判定的提示詞版本和設定檔不同，沒有採用。" if info["skipped_version"] else ""))
    elif info["n"] < info["total"]:
        w.append(f"評審只完成 {info['n']}/{info['total']} 則回覆（judge_error {info['errors']} 則）；"
                 "沒評到的回覆不計入評審指標，結果會再變動。")
    if len(info["models"]) > 1:
        w.append("這組回覆混用了不同的評審模型：" + "、".join(info["models"]) + "。規格要求整個研究用同一個評審。")
    if len(info.get("machines", [])) > 1:
        w.append("評審在不只一台機器上執行：" + "、".join(info["machines"]) + "。規格要求評審全部在同一台機器、同一個評審模型檔。")
    return w


def metric_names(rows, judged, module=None):
    base = [m for m in JUDGE_METRICS if m != "正確率"] + CORRECT_BY_TURN       # 正確率分輪報告
    names = (base + (SCRIPT_METRICS if module == "script" else [])) if judged else KEYWORD_METRICS
    # 沒有任何數值的指標（例如這批題目沒有數字題）不列
    return [m for m in names if any(not _nan(r.get(m)) for r in rows)]


def _nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def embed_models(a, cfg):
    """要算哪些嵌入模型：--embed-model 直接指定（可用逗號分隔多個）；--embed 用設定檔 embedding.models。"""
    from .common import check_origin
    if getattr(a, "embed_model", ""):
        ms = [m.strip() for m in a.embed_model.split(",") if m.strip()]
    elif getattr(a, "embed", False):
        ms = list((cfg.get("embedding") or {}).get("models", []))
    else:
        ms = ["mock"] if a.mock else []
    check_origin(cfg, *ms, what="嵌入模型")
    return ms


def lex_name(model, models):
    """詞彙距離的指標名稱；只有一個嵌入模型時就叫「詞彙距離」。"""
    if len(models) <= 1:
        return "詞彙距離"
    short = model.split("/")[-1]
    short = short.replace("-GGUF", "").replace("multilingual-", "")
    return f"詞彙距離（{short}）"


def quality(convs, settings):
    """截斷率與思考內容：每則回覆的結束原因、伺服器另外回傳的思考字數。回傳 (摘要, 開頭警示)。"""
    fins = [f for c in convs for f in c.get("finish", [])]
    n_rep, n_cut = len(fins), sum(f == "length" for f in fins)
    seg_cut = sum(any(f == "length" for f in c.get("finish", [])) for c in convs)
    rate = n_cut / max(n_rep, 1)
    thinks = [x for c in convs for x in c.get("think_chars", [])]
    n_cont = sum(len(c.get("continued", [])) for c in convs)
    n_regen = sum(len(c.get("regenerated", [])) for c in convs)
    q = {"n_rep": n_rep, "n_cut": n_cut, "cut_rate": rate, "seg_cut": seg_cut, "n_seg": len(convs),
         "n_continued": n_cont, "n_regenerated": n_regen,
         "cap_total": max((c.get("max_tokens_total") or 0 for c in convs), default=0),
         "think_n": len(thinks), "think_share": (sum(x > 0 for x in thinks) / len(thinks)) if thinks else None}
    warns = []
    if rate > TRUNC_WARN:
        warns.append(f"截斷率 {rate:.1%}（{n_cut}/{n_rep} 則回覆碰到每則上限被截斷），超過 {TRUNC_WARN:.0%}："
                     "字數等指標會被上限壓平，請加大每則上限後補跑，或解讀時保留。")
    if settings.get("thinking") and thinks and not any(thinks) and settings.get("model") != "mock":
        warns.append("設定檔要求思考開啟，但伺服器沒有回傳任何思考內容。可能是模型沒有開啟思考，"
                     "或伺服器把思考混在回覆裡（會被算進字數）；請到 LM Studio 確認。")
    return q, warns


def analyze_one(a, cfg, plist, run_dir):
    rng = np.random.default_rng(a.seed)
    from . import probe
    probe_rows = probe.summary(cfg, run_dir, plist)
    plist, recoded = probe.recode(cfg, run_dir, plist)       # 讀不出來的隱性線索改當「未提及」
    personas = {uid(p): p for p in plist}
    settings = layout.read_settings(run_dir)
    questions = module_questions(cfg, None)
    qmap = {q["id"]: q for q in questions}
    convs, n_empty, n_stale = load_valid(os.path.join(run_dir, layout.RESPONSES), plist, settings["turns"], questions)
    envs = layout.read_meta(run_dir).get("envs") or {}
    for c in convs:
        c["machine_label"] = layout.machine_label(c, envs)     # 同一台電腦換了顯示卡也分得開
    if n_empty or n_stale:
        warn("略過一些紀錄", f"含空回覆：{n_empty} 段" if n_empty else None,
             f"人物或題目和設定檔對不上：{n_stale} 段" if n_stale else None)
    if not convs:
        print("沒有可用的回覆（略過）")
        return
    # 0. 分析的人物範圍：每位人物、每題都有同樣次數的重複，才不會把跑完與沒跑完的人物混在一起
    convs, cohort = choose_cohort(convs, plist, questions, getattr(a, "people", 0), getattr(a, "reps", 0))
    print(cohort_line(cohort))
    if not convs:
        print(bullets("✗ 略過：沒有任何人物每題都跑完同樣的次數", ["先讓第 1 次重複跑完（status 看進度）"]))
        return
    out_dir = analysis_dir(run_dir, cohort)
    qcat = {q["id"]: q.get("type", q.get("category")) for q in questions}
    weights = {k: v for k, v in cfg["category_weights"].items() if not k.startswith("_")}
    lex = cfg["lexicon"]

    # 1. 逐則回覆 → 指標（評審過就用評審的判定，否則用關鍵詞初篩）
    judged, jinfo = judged_for(cfg, a, run_dir, convs)
    all_rows = reply_rows(convs, qcat, lex, judged, qmap)
    names = metric_names(all_rows, judged is not None)
    pids = sorted({r["pid"] for r in all_rows})
    by_pid = {pid: [r for r in all_rows if r["pid"] == pid] for pid in pids}
    cats = [c for c in weights if any(r["cat"] == c for r in all_rows)]

    # 2. 每位人物一個分數；缺題型的人物先排除（以字數判斷，字數每則都有）
    pscore = lambda rows, pp, m: stats.weighted_persona_scores(rows, pp, cats, weights, m, m in SUBSET_METRICS)
    scores = {m: pscore(by_pid, pids, m) for m in names}
    keep = ~np.isnan(scores["字數"])
    n_all = len(pids)
    if not keep.any():
        print(bullets(f"✗ 略過：{n_all} 位人物都還沒問完所有題型", [
            "分數先在每個題型內平均再合併：每位人物每種題型至少要一段對話",
            "第 1 次重複全部跑完（進度 50%）就可以做第一次分析"]))
        return
    if not keep.all():
        warn(f"{(~keep).sum()} 位人物還沒問完所有題型（排除）", f"這次分析 {keep.sum()} 位")
    pids = [p for p, k in zip(pids, keep) if k]
    scores = {m: v[keep] for m, v in scores.items()}
    noise = {m: stats.split_half_noise({p: by_pid[p] for p in pids}, pids, cats, weights, m, m in SUBSET_METRICS)
             for m in names}

    # 3. 詞彙層（有指定嵌入模型才算；兩個模型各算一次）
    models = embed_models(a, cfg)
    vec_fn, tok = {}, {}
    for em in models:
        def vf(exclude=frozenset(), em=em):
            texts = {pid: [r["text"] for r in by_pid[pid] if r["qid"] not in exclude] for pid in pids}
            return persona_vectors(texts, pids, cfg, run_dir, a.base_url, em, a.embed_chunk, a.mock or em == "mock",
                                   personas, tok=tok)
        vec_fn[lex_name(em, models)] = vf
    vecs = {name: f() for name, f in vec_fn.items()}
    tok_sum = token_summary(tok)
    write_recuts(run_dir, tok)
    for m, s_ in tok_sum.items():
        print("詞彙層分段：" + token_line(m, s_))

    # 4. 同時估所有屬性（每個指標一個多元迴歸；以人物為單位置換與重抽）
    attrs = cfg["attributes"]
    keys = list(attrs)
    attr_names = [attrs[x]["label"] for x in keys]
    label_fn = lambda key, us: attr_labels(cfg, personas, us, key)
    D, ids = screen.build_design(cfg, personas, pids, keys, attr_names, label_fn)
    hr_q = {q["id"] for q in questions if q.get("high_risk")}
    exempt, exempt_hr, exempt_notes = {}, {}, []
    for key, attr in zip(keys, attr_names):
        ex = exempt_questions(questions, key)
        ex_s, ex_v, ex_h = {}, {}, {}
        for m, qids in ex.items():
            if m in scores:
                ex_s[m] = screen.layer_scores(by_pid, pids, cats, weights, [m], drop=frozenset(qids))[m]
                ex_h[m] = screen.layer_scores(by_pid, pids, cats, weights, [m], hr_q, frozenset(qids))[m]
                exempt_notes.append((attr, m, sorted(qids)))
            if m == "詞彙距離" and vec_fn:
                for name, f in vec_fn.items():
                    ex_v[name] = f(frozenset(qids))
                exempt_notes.append((attr, "詞彙距離", sorted(qids)))
        exempt[attr] = (ex_s, ex_v)
        exempt_hr[attr] = (ex_h, {})
    skipped = []
    joint_ok = D.n >= D.X.shape[1] + screen.MIN_DF        # 人物版本要比迴歸的欄位多，才能同時估所有屬性
    hr_scores = screen.layer_scores(by_pid, pids, cats, weights, names, hr_q)
    in_scope = [r for p in pids for r in by_pid[p]]
    usd = {m: stats.within_question_sd(in_scope, m) for m in names}          # 標準化的分母：單則回覆、題內的標準差
    usd_hr = {m: stats.within_question_sd(in_scope, m, hr_q) for m in names}
    if joint_ok:
        results, tests, padj, sds = screen.joint_layer(D, "全部題目", scores, vecs, exempt, rng, a.perms, a.boots,
                                                       skipped=skipped, unit_sd=usd)
        hr = screen.joint_layer(D, "高風險題", hr_scores, {}, exempt_hr, rng, a.perms, a.boots, skipped=skipped,
                                unit_sd=usd_hr) if hr_q else ([], [], {}, {})
    else:
        mode_note = (f"人物版本只有 {D.n} 筆，少於同時估所有屬性需要的 {D.X.shape[1] + screen.MIN_DF} 筆"
                     f"（迴歸有 {D.X.shape[1]} 欄），改用逐屬性比較：屬性之間的外溢扣不掉，結果只能當流程檢查。")
        print("⚠ " + mode_note)
        name_of = dict(zip(attr_names, keys))
        one = lambda attr: screen.build_design(cfg, personas, pids, [name_of[attr]], [attr], label_fn)[0]
        results, tests, padj, sds = screen.per_attribute_layer(one, attr_names, "全部題目", scores, vecs, exempt, rng,
                                                               a.perms, a.boots, skipped=skipped, unit_sd=usd)
        hr = screen.per_attribute_layer(one, attr_names, "高風險題", hr_scores, {}, exempt_hr, rng, a.perms, a.boots,
                                        skipped=skipped, unit_sd=usd_hr) if hr_q else ([], [], {}, {})
    thresholds = (cfg.get("verdict") or {}).get("thresholds")
    cells = verdict.cell_verdicts("全部題目", results, tests, padj, thresholds, sds)
    cells_hr = verdict.cell_verdicts("高風險題", hr[0], hr[1], hr[2], thresholds, hr[3])
    lex_split = lexical_disagreements(attr_names, list(vecs), padj) if len(vecs) > 1 else []

    # 5. 只描述：交互作用、品質指數最差的人物、揭露減隱性、各群比例
    inter = screen.interactions(D, screen.interaction_cols(cfg, D, pids, personas, dict(zip(keys, attr_names))),
                                scores, a.boots, rng) if joint_ok else []
    qidx, q_parts = screen.quality_index(all_rows, pids, cats, weights, judged is not None)
    share = (cfg.get("interactions") or {}).get("worst_share", 0.05)
    worst, n_ranked = screen.worst_personas(D, ids, qidx, share, personas, pids, cfg)
    disc = screen.disclosure(D, ids, scores, a.boots, rng)
    covs = {m: joint.covariate_coefs(D, scores[m]) for m in names} if joint_ok else {}
    describe = describe_groups(cfg, personas, pids, by_pid, attr_names + ["揭露方式"])
    machines = machine_summary(all_rows, convs)
    unit_labels = {a_: dict(zip(pids, D.labels[a_])) for a_ in D.attrs}
    flagged_cells = [c for c in cells + cells_hr if c["顯著"]]
    for c in flagged_cells:
        sign = 1 if (c["效果"] or 0) >= 0 else -1
        c["例子"] = screen.examples(all_rows, unit_labels[c["屬性"]], c["屬性"], c["效果最大的群體"], c["指標"], sign)
    if models:
        layout.update_meta(run_dir, {"embedding": {"models": models, "chunk_chars": a.embed_chunk,
                                                   "prefixes": {m: embed_prefix_of(cfg, m) for m in models},
                                                   "tokens": tok_sum}})

    # 6. 資料完整度摘要
    in_scope = set(pids)
    cells_n = {}
    for c in convs:
        if uid(c) in in_scope:
            cells_n[(uid(c), c["qid"])] = cells_n.get((uid(c), c["qid"]), 0) + 1
    counts = [cells_n.get((p, q["id"]), 0) for p in pids for q in questions]
    wk = {}
    for c in convs:
        wk[c.get("workers", "?")] = wk.get(c.get("workers", "?"), 0) + 1
    wsum = sum(weights[c] for c in cats)
    qual, warns = quality(convs, settings)
    warns += judge_warnings(jinfo)
    if not joint_ok:
        warns.insert(0, mode_note)
    if cohort.get("missing_questions"):
        warns.insert(0, f"{'、'.join(cohort['missing_questions'])} 還沒有任何回覆（例如換成新版的題目還沒補跑），"
                        "這份報告暫時不含這幾題；補跑完再分析一次。")
    if any(g in str(settings.get("model", "")).lower() for g in GPT_OSS_FAMILY):
        warns.insert(0, "受測模型和主評審同屬 gpt-oss 家族：同家族的評審可能偏好同家族的寫法，請看「限制」裡第二評審的不一致比例。")
    spill = explicit_only_groups(cfg)
    for c in cells + cells_hr:
        if c["顯著"] and c["效果最大的群體"] in spill.get(c["屬性"], set()):
            c["說明"] = "；".join(x for x in (c["說明"], "可能為揭露外溢（只做揭露的群體，見待決第 5 項）") if x)
    from .confirm import load_status
    confirmed = load_status(a.out, os.path.basename(run_dir))
    for c in cells + cells_hr:
        st = confirmed.get((c["屬性"], c["指標"], c["效果最大的群體"]))
        if st and c["顯著"]:
            c["配對確認"] = st
            c["說明"] = c["說明"].replace(verdict.PENDING, f"配對確認：{st}")
            if st == verdict.UNCONFIRMED and c["結論"] in ("不通過", "需改善"):
                c["說明"] = "；".join(x for x in (c["說明"], f"篩檢結論 {c['結論']}") if x)
                c["結論"] = verdict.UNCONFIRMED        # 不進入整體結論
    retired = retired_rows(run_dir, questions, settings["turns"], plist, set(pids), cfg, a)
    if any(TODO_MARK in p.get("intro", "") for p in plist):
        warns.insert(0, "人物介紹含有【待補】：這批回覆只能用來檢查流程，不是正式資料。")
    echo = echo_summary(cfg, personas, convs, in_scope)
    meta = layout.read_meta(run_dir)
    times = sorted(c.get("time") for c in convs if c.get("time"))
    paused = paused_groups(cfg, D)
    info = {"model": convs[0].get("model"), "settings": layout.describe(settings), "n": len(pids), "n_all": n_all,
            "n_people": len(ids), "n_convs": len(convs),
            "n_replies": len(all_rows), "weights": {c: weights[c] / wsum for c in cats},
            "perms": a.perms, "boots": a.boots, "cell_min": min(counts), "cell_max": max(counts),
            "quality": qual, "warnings": warns, "thinking": settings.get("thinking"),
            "workers_note": "、".join(f"{k} 段並行 {v} 段對話" for k, v in sorted(wk.items(), key=lambda kv: str(kv[0]))),
            "metrics": names + list(vecs), "judge": jinfo if judged is not None else None,
            "embed": [(lex_name(m, models), m, "" if m == "mock" else embed_prefix_of(cfg, m)) for m in models],
            "exempt": exempt_notes, "lex_split": lex_split, "echo": echo, "describe": describe, "machines": machines,
            "load_settings": machine_loads(meta),
            "cohort": cohort, "out_dir": out_dir, "retired": retired, "skipped": list(dict.fromkeys(skipped)), "persona_info": persona_file_info(a.out, cfg, run_dir, plist),
            "settings_check": layout.check_settings(run_dir),
            "conditions": test_conditions(settings, meta, jinfo if judged is not None else None, times, models, cfg,
                                          read_jsonl(os.path.join(run_dir, layout.RESPONSES)) if settings.get("provider") else None,
                                          tok=tok_sum),
            "paused": [g for g, present in paused if not present],
            "paused_present": [g for g, present in paused if present],
            "high_risk": sorted(hr_q), "interactions": inter, "worst": worst, "n_ranked": n_ranked,
            "worst_share": share, "quality_parts": q_parts, "disclosure": disc, "covariates": covs,
            "thresholds": thresholds, "limits": limitations(cfg, plist, run_dir, jinfo, lex_split, models)}
    from .interpret import appendix_for_screening
    info["interp"] = appendix_for_screening(a.out, run_dir, cfg)
    info["probe"] = {"rows": probe_rows, "recoded": recoded, "min": (cfg.get("probe") or {}).get("min_read_rate")}
    if recoded:
        warns.append("模型操弄檢核：這幾群的隱性線索受測模型讀不出來，隱性版在分析時改當「未提及」：" + "、".join(recoded))
    info["overall"], info["overall_en"] = verdict.overall(cells + cells_hr, thresholds)
    from .scorecard import Lang, load_terms, write_reports
    info["mismatch"] = verdict.plan_mismatches(settings, info, cfg, Lang("en", load_terms(a.config)).group_key)
    info["persona_rows"] = [(k, PERSONA_EN.get(k, k), v, persona_en(k, v)) for k, v in info["persona_info"]]

    # 7. 輸出
    csv_path, md_path = os.path.join(out_dir, "results.csv"), os.path.join(out_dir, "report.md")
    cells_path = os.path.join(out_dir, "cells.csv")
    report.write_csv(csv_path, results + hr[0])
    report.write_csv(cells_path, [{k: v for k, v in c.items() if k != "例子"} for c in cells + cells_hr])
    if disc:
        report.write_csv(os.path.join(out_dir, "disclosure.csv"), disc)
    if inter:
        report.write_csv(os.path.join(out_dir, "interactions.csv"), inter)
    qual_rows = [{"人物": p, "品質指數": v} for p, v in screen.persona_quality(D, ids, qidx).items()]
    if qual_rows:
        report.write_csv(os.path.join(out_dir, "quality.csv"), qual_rows)      # 配對確認挑基底人物用
    from .common import save_json
    save_json(os.path.join(out_dir, "cohort.json"), {k: v for k, v in cohort.items()})
    same, check = write_reports(out_dir, a.config, info, results, cells, cells_hr, hr[0], noise, attr_names, lex)
    print(bullets("── 報告", [("報告", md_path), ("英文版", f"report_en.md（數字比對{'一致' if same else '不一致，以中文版為準'}）"),
                                ("明細", csv_path), ("每格判定", cells_path)]))
    for w in warns:
        print(f"⚠ {w}")
    print(f"整體結論：{info['overall']}")
    flagged = sorted((c for c in cells if c["校正後p"] <= verdict.Q), key=lambda c: c["校正後p"])
    print(f"校正後 p ≤ {verdict.Q:.2f} 的組合：{len(flagged)} 個")
    for c in flagged:
        print(f"  {c['屬性']} × {c['指標']}  p_adj={c['校正後p']:.3f}" + ("（待配對確認）" if c["顯著"] else ""))
    hr_flag = [c for c in cells_hr if c["校正後p"] <= verdict.Q]
    if hr_flag:
        print(f"高風險題：校正後 p ≤ {verdict.Q:.2f} 的組合 {len(hr_flag)} 個")
        for c in hr_flag:
            print(f"  高風險 {c['屬性']} × {c['指標']}  q={c['校正後p']:.3f}")
    from .common import cmd_hint, next_steps
    steps = []
    if judged is None:
        steps.append("先評審，指標才會是評審的判定（現在是關鍵詞初篩）：" + cmd_hint("judge", layout.run_arg(a.out, run_dir)))
    else:
        key_of = dict(zip(attr_names, keys))
        for at in list(dict.fromkeys(c["屬性"] for c in cells + cells_hr if c["顯著"]))[:3]:
            steps.append(f"配對確認（{key_of.get(at, at)}）：" + cmd_hint("confirm", layout.run_arg(a.out, run_dir, True),
                                                                   f"--attr {key_of.get(at, at)}"))
        steps.append("人工標註（檢查評審準不準，200 則）：" + cmd_hint("annotate", layout.run_arg(a.out, run_dir)))
    print(next_steps(steps))


PERSONA_EN = {"人物版本": "Persona version", "產生時的程式": "Generated by", "屬性平衡互換": "Balance swaps",
              "矛盾檢查": "Conflict check"}


def persona_en(k, v):
    """人物檔資訊的英文（固定句型）。"""
    if k == "人物版本":
        return v.replace("位 × 兩版", " personas × 2 versions").replace("（", " (").replace("）", ")")
    if k == "產生時的程式":
        return "not recorded (persona files before v0.9.1 have no personas_meta.json)" if "沒有記錄" in v else v
    if k == "屬性平衡互換":
        return "unknown: added in v0.8.6; personas generated earlier (e.g. p1.1, p1.2) have none" if "不明" in v else \
            v.split("（")[0].replace("次", " swaps")
    if k == "矛盾檢查":
        return "no conflicts under the current exclusion rules" if "沒有矛盾" in v else "conflicts found (see report.md)"
    return v


def explicit_only_groups(cfg):
    """只做揭露的群體（隱性版沒有線索句）：{屬性: {群體}}。這些群體被標出時，可能是別的屬性的揭露效果外溢。"""
    out = {}
    for key, at in cfg["attributes"].items():
        for lv, spec in at["levels"].items():
            if not spec.get("implicit") or spec.get("implicit_cue") == "none":
                out.setdefault(at["label"], set()).add(spec["label"])
    return out


def retired_rows(run_dir, questions, turns, plist, in_scope, cfg, a):
    """換題前的舊版回覆（例如 q1.0 的 I1）：只描述，不列入分析。每個舊版一列：段數、平均字數，評審過的再列分輪正確率。"""
    recs = [r for r in load_retired(os.path.join(run_dir, layout.RESPONSES), questions, turns) if uid(r) in in_scope]
    if not recs:
        return []
    from .common import retired_hashes
    old = retired_hashes(questions, turns)
    js, _ = load_judgments(run_dir, prompt_path(cfg, a.config)[0])
    out = []
    for (qid, qh), ver in old.items():
        rs = [r for r in recs if r["qid"] == qid and r.get("qh") == qh]
        if not rs:
            continue
        row = {"題號": qid, "題庫版本": ver, "段數": len(rs), "人物版本": len({uid(r) for r in rs}),
               "平均字數": float(np.mean([len(x) for r in rs for x in r["replies"]]))}
        for t in range(turns):
            res = [js.get(jkey(r, t), {}).get("result", {}).get("correct") for r in rs
                   if valid_judgment(js.get(jkey(r, t)), r["replies"][t])]
            dec = [x for x in res if x in ("correct", "incorrect")]
            row[f"正確率（第 {t + 1} 輪）"] = dec.count("correct") / len(dec) if dec else np.nan
            row[f"已評（第 {t + 1} 輪）"] = len(res)
        out.append(row)
    return out


def persona_file_info(out, cfg, run_dir, plist):
    from .personas import persona_info
    pdir = personas_for_run(out, cfg, run_dir)[1]
    return persona_info(cfg, pdir, plist)


def paused_groups(cfg, D):
    """設定檔 paused_groups 的群體：[(「屬性：群體」, 這批資料裡有沒有)]。"""
    out = []
    for spec in cfg.get("paused_groups") or []:
        key, lv = spec.split(":")
        at = cfg["attributes"].get(key)
        if not at or lv not in at["levels"]:
            continue
        name, label = at["label"], at["levels"][lv]["label"]
        out.append((f"{name}：{label}", name in D.labels and label in set(D.labels[name])))
    return out


PROG_TERMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "i18n", "terms.json")


def test_conditions(settings, meta, jinfo, times, models, cfg, conv_recs=None, tok=None):
    """報告開頭的測試條件（規格：存取層級、模型版本與量化、測試方式、發布日期、測試日期、生成設定、系統提示詞、評審、嵌入模型）。
    回傳 [(項目, 英文項目, 內容, 英文內容)]。"""
    from .envinfo import clean_lms_field
    envs = list((meta.get("envs") or {}).values())
    quant = "、".join(sorted({e.get("quantization") or "不明" for e in envs})) or "未記錄"
    files = "、".join(sorted({e.get("model_file") or "不明" for e in envs})) or "未記錄"
    lms = "、".join(sorted({clean_lms_field(e.get("lmstudio")) or "不明" for e in envs})) or "未記錄"
    gpus = "、".join(sorted({e.get("gpu") or "不明" for e in envs})) or "未記錄"
    mock_ = settings.get("model") == "mock"
    sysp = ("無（身分放在系統提示裡模擬記憶功能，除此之外沒有系統提示詞）", "none (identity in the system prompt to simulate "
            "memory; nothing else)") if settings.get("identity") == "system" else \
        (("無", "none") if not settings.get("system_prompt") else ("有（見 settings.json）", "yes (see settings.json)"))
    hist = layout.caps(settings)
    cap_en = f"{settings['max_tokens']} tokens" + (f" (raised from {'→'.join(map(str, hist[:-1]))})" if len(hist) > 1 else "")
    gen_en = (f"{settings.get('model')} | identity {'in system prompt' if settings.get('identity') == 'system' else 'before the question'}"
              f" | temperature {settings['temperature']:g} | max tokens {cap_en} | {settings['turns']} turns"
              f" | questions/personas {settings.get('questions_version')}/{settings.get('personas_version')}")
    jl = report.judge_line(jinfo) if jinfo else "尚未評審，行為指標用關鍵詞初篩"
    jl_en = (f"{', '.join(jinfo['models'])} | prompt {jinfo['prompt']} | judged {jinfo['n']}/{jinfo['total']} | "
             f"judge_error {jinfo['errors']}") if jinfo else "not judged yet; keyword screening only"
    when = (f"{times[0][:10]} 到 {times[-1][:10]}", f"{times[0][:10]} to {times[-1][:10]}") if times else ("未記錄", "not recorded")
    ver = meta.get("program_version") or settings.get("version") or "未記錄"
    vs = meta.get("versions") or {}
    terms = load_json(PROG_TERMS_PATH) if os.path.exists(PROG_TERMS_PATH) else {}
    vline = (f"企劃書 {vs.get('plan') or settings.get('plan_version') or '未記錄'}｜題庫 {settings.get('questions_version')}"
             f"｜人物 {settings.get('personas_version')}｜人物句 {vs.get('persona_sentences') or '未記錄'}"
             f"｜中英詞表 {terms.get('_status', '未記錄')}｜題目 {question_scope(cfg)}")
    vline_en = (f"plan {vs.get('plan') or settings.get('plan_version') or 'n/a'} | questions {settings.get('questions_version')}"
                f" | personas {settings.get('personas_version')} | persona sentences {vs.get('persona_sentences') or 'n/a'}"
                f" | term list {terms.get('_status', 'n/a')} | questions: {question_scope_en(cfg)}")
    prov = settings.get("provider")
    rows_cloud = []
    if prov:
        from .cloud import usage_line
        line, u, c = usage_line(cfg, settings.get("model"), conv_recs or [])
        rows_cloud = [("呼叫與費用", "Calls and cost", line,
                       f"{u['calls']} calls | input {u['input']:,} tokens | output {u['output']:,} tokens"
                       + (f" | cost about {c['amount']:.2f} {c['currency']}" if c else " | cost: no unit price in config"))]
        if ((meta.get("cloud") or {}).get("compat") or {}).get("no_temperature"):
            rows_cloud.append(("溫度", "Temperature", f"服務不收溫度（推理模型），改用供應商預設值；設定的 {settings.get('temperature')} 沒有送出",
                               f"not accepted by the service (reasoning model); provider default used instead of {settings.get('temperature')}"))
    access = (("假資料", "mock data") if mock_ else
              (f"付費 API（{prov}，OpenAI 相容介面）", f"paid API ({prov}, OpenAI-compatible)") if prov else
              ("本機執行，開放權重（LM Studio 的 OpenAI 相容介面）", "local, open weights (LM Studio OpenAI-compatible API)"))
    return [
        ("存取層級", "Access", access[0], access[1]),
        ("受測模型", "Model", f"{settings.get('model')}｜量化 {quant}｜模型檔 {files}",
         f"{settings.get('model')} | quantization {quant} | file {files}"),
        ("執行環境", "Environment", f"顯示卡 {gpus}｜LM Studio {lms}", f"GPU {gpus} | LM Studio {lms}"),
        ("測試方式", "Method", meta.get("test_method") or ("假回覆（--mock）" if mock_ else "LM Studio 本機 API"),
         "mock replies" if mock_ else (f"paid API ({prov})" if prov else "LM Studio local API")),
        *rows_cloud,
        ("模型發布日期", "Model release date", meta.get("model_release") or "未記錄（程式讀不到，請手動補上）",
         meta.get("model_release") or "not recorded"),
        ("測試日期", "Test dates", when[0], when[1]),
        ("生成設定", "Generation", layout.describe(settings), gen_en),
        ("系統提示詞", "System prompt", sysp[0], sysp[1]),
        ("評審", "Judge", jl, jl_en),
        ("嵌入模型", "Embedding models", "、".join(models) if models else "未計算", ", ".join(models) if models else "not computed"),
        *([("嵌入分段", "Embedding segments", "；".join(token_line(m, s) for m, s in tok.items()),
            "; ".join(f"{m}: max {s['max_tokens']} tokens (limit {s['limit']}), re-split {s['recuts']} times"
                      for m, s in tok.items()))] if tok else []),
        ("程式版本", "Program version", f"分析 {__version__}；生成 {ver}", f"analysis {__version__}; generation {ver}"),
        ("其他版本", "Other versions", vline, vline_en)]


def limitations(cfg, plist, run_dir, jinfo, lex_split, models):
    """報告的「限制」：固定的幾項，加上這批資料能查到的（人工標註、第二評審、嵌入模型）。回傳 [(中文, 英文)]。"""
    L = [("人工校驗目前只有作者一人：能算人與評審模型的一致程度，量不到人與人之間的一致程度。",
          "Only one human annotator (the author): agreement with the judge can be measured, agreement between people cannot.")]
    from .annotate import agreement_summary
    ag = agreement_summary(run_dir)
    L.append((f"人與評審模型的一致程度：{ag}", f"Human–judge agreement: {ag} (see annotation/behavior)") if ag else
             ("人與評審模型的一致程度：還沒有人工標註（annotate 指令產生抽樣，填好後再跑一次）。",
              "Human–judge agreement: no annotation yet (run annotate, fill labels.csv, run it again)."))
    from .judge import agreement, load_judgments
    if jinfo and os.path.exists(os.path.join(run_dir, "judgments__second.jsonl")):
        a_, _ = load_judgments(run_dir, jinfo["prompt"])
        b_, _ = load_judgments(run_dir, jinfo["prompt"], "second")
        ag2 = agreement(a_, b_)
        txt = "、".join(f"{f} {x}/{n}（{x / n:.0%}）" for f, (x, n) in ag2.items())
        L.append(("兩個評審不一致的比例：" + (txt if ag2 else "沒有共同評過的回覆"),
                  "Second-judge disagreement: " + (", ".join(f"{f} {x}/{n}" for f, (x, n) in ag2.items()) if ag2 else "no overlap")))
    else:
        L.append(("兩個評審不一致的比例：還沒有第二評審的結果（第二評審評人工標註樣本與 10% 隨機抽樣：judge --second）。",
                  "Second-judge disagreement: no second judge yet (it judges the annotation sample plus a 10% random "
                  "sample: judge --second)."))
    from .judge import machine_agreement, machine_agreement_text
    mtxt = machine_agreement_text(machine_agreement(run_dir, jinfo["prompt"])) if jinfo else ""
    if mtxt:
        L.append(("換機器重評的不一致比例（待決第 19 項：舊結果保留，以最後評的那台為準）：" + mtxt,
                  "Re-judged on another machine (old results kept; the last machine counts): " + mtxt))
    if len(models) > 1:
        names = [x[0] for x in lex_split]
        L.append(("兩個嵌入模型結論不一致的格子：" + ("、".join(names) if names else "無"),
                  "Cells where the two embedding models disagree: " + (", ".join(names) if names else "none")))
    L.append(("系統提示詞：無。實際產品多半帶有系統提示詞，結果只代表裸模型。",
              "No system prompt: real products usually have one, so results describe the bare model."))
    L.append(("篩檢結果：被標出的格子都要等配對確認（F 批）才算定案。",
              "Screening only: flagged cells are final only after paired confirmation (batch F)."))
    L.append(("每位人物固定帶 4 個屬性：某屬性只在揭露版才有效果時，可能經由只做揭露的屬性外溢（待決第 5 項）。",
              "Each persona carries exactly 4 attributes; an effect present only in the explicit version can leak into "
              "explicit-only attributes (open item 5)."))
    if any(p.get("levels", {}).get("language") == "taiwanese" for p in plist):
        L.append(("台語的線索句用華台夾雜的寫法，同時帶有口語語氣；模型的反應可能混入對語氣的反應。",
                  "The Taiwanese cue is mixed Mandarin–Taiwanese with a colloquial tone; responses may react to the tone."))
    single = [s.lower() for s in cfg.get("names_single_year_sources") or []]
    if any(any(s in (p.get("name_source") or "").lower() for s in single) for p in plist):
        L.append(("部分外籍姓名（蘇格蘭、澳洲）只有單年的新生兒名字，用在成年人身上年代不合。",
                  "Some foreign names (Scotland, Australia) come from one year of newborn names, which do not match adults' ages."))
    if any((p.get("name_source") or "").find("Faker") >= 0 for p in plist):
        L.append(("外籍姓名用 Faker 測試詞表（非人口統計），只能用來檢查流程。",
                  "Foreign names come from Faker test word lists (not population data); pipeline check only."))
    return L


def embed_prefix_of(cfg, model):
    from .metrics import embed_prefix
    return embed_prefix(cfg, model)


DESCRIBE = [*STANCE_BY_TURN, "有給數字率", "截斷率"]


def describe_groups(cfg, personas, pids, by_pid, attr_names):
    """只描述、不判定：各群的表態率（標準答案題）、有給數字率（數字題）、截斷率。
    每位人物先平均自己的回覆，再平均群內人物。回傳 [{屬性, 群體, 人數, 各比例}]。"""
    present = [m for m in DESCRIBE if any(not _nan(r.get(m)) for p in pids for r in by_pid[p])]
    if not present:
        return []
    per = {}
    for m in present:
        per[m] = np.array([np.mean(v) if (v := [r[m] for r in by_pid[p] if not _nan(r.get(m))]) else np.nan
                           for p in pids])
    out = []
    attrs = cfg["attributes"]
    for key, attr in zip(list(attrs) + ["_variant"], attr_names):
        labels = attr_labels(cfg, personas, pids, key)
        for lv in sorted(set(labels)):
            idx = labels == lv
            row = {"屬性": attr, "群體": lv, "人數": int(idx.sum())}
            for m in present:
                v = per[m][idx]
                row[m] = float(np.nanmean(v)) if np.any(~np.isnan(v)) else np.nan
            out.append(row)
    return out


def machine_loads(meta):
    """各機器的 LM Studio 載入設定：{機器: (一行說明, 來源)}。程式自動記的（v0.9.6 起）優先，其次是人工比對的紀錄（load_manual）。"""
    from .envinfo import LOAD_CONTENT, _norm, load_text
    out = {}
    for e in (meta.get("envs") or {}).values():
        if e.get("load"):
            out[layout.machine_label({"machine": e.get("machine"), "gpu": e.get("gpu")})] = (
                load_text(e["load"]), "程式自動記錄（LM Studio /api/v1/models，讀不到的用設定檔聲明）")
    for m, rec in (meta.get("load_manual") or {}).items():
        content = {k: {"value": _norm(k, v), "from": "人工"} for k, v in rec.items() if k in LOAD_CONTENT}
        sp = {"parallel": rec["parallel"]} if rec.get("parallel") else {}
        out.setdefault(m, (load_text({"content": content, "speed": sp}), rec.get("_source") or "人工紀錄"))
    return out


def machine_summary(rows, convs):
    """各機器的截斷率與字數分布（跨機器分工時檢查機器之間有沒有系統性差異）。"""
    by = {}
    for r in rows:
        by.setdefault(r["machine"], []).append(r)
    segs = {}
    for c in convs:
        segs.setdefault(c.get("machine_label") or c.get("machine") or "未記錄", []).append(c)
    out = []
    for m, rs in sorted(by.items()):
        lens = np.array([len(r["text"]) for r in rs])
        cut = [r["截斷率"] for r in rs if not _nan(r.get("截斷率"))]
        reps = sorted({c["rep"] + 1 for c in segs.get(m, [])})
        out.append({"機器": m, "段數": len(segs.get(m, [])), "回覆數": len(rs), "重複序號": reps,
                    "截斷率": float(np.mean(cut)) if cut else np.nan,
                    "字數中位數": float(np.median(lens)), "字數第90百分位": float(np.percentile(lens, 90))})
    return out


def exempt_questions(questions, key):
    """這個屬性在哪些指標上要排除哪些題目：{指標名稱: {題號}}。指標名稱「詞彙距離」代表所有嵌入模型。"""
    out = {}
    for q in questions:
        if key in (q.get("relevant_attributes") or []):
            for code in q.get("relevant_exempt_metrics") or []:
                for m in EXEMPT_CODES.get(code, []):
                    out.setdefault(m, set()).add(q["id"])
    return out


def lexical_disagreements(attr_names, lex_metrics, padj):
    """兩個嵌入模型的結論不一致的屬性：一個校正後 p 低於標記門檻、另一個沒有。"""
    out = []
    for attr in attr_names:
        ps = {m: padj.get((attr, m)) for m in lex_metrics}
        if all(p is not None for p in ps.values()):
            flags = {m: p < report.FLAG_P for m, p in ps.items()}
            if len(set(flags.values())) > 1:
                out.append((attr, ps))
    return out


def echo_summary(cfg, personas, convs, in_scope):
    """身分複述（檢查項，不判定）：第 1 輪回覆是否複述使用者自己說過的身分字眼。
    回傳 {版本: (有複述的段數, 段數)}、{屬性: (有複述, 段數)} 與最常被複述的字眼。"""
    by_var, by_attr, terms = {}, {}, {}
    for c in convs:
        if uid(c) not in in_scope:
            continue
        p = personas[uid(c)]
        said = identity_echo(cfg, p, c["replies"][0])
        v = p.get("variant")
        n, k = by_var.get(v, (0, 0))
        by_var[v] = (n + bool(said), k + 1)
        for t in said:
            terms[t] = terms.get(t, 0) + 1
        for key, ts in echo_terms_by_attr(cfg, p).items():
            n, k = by_attr.get(key, (0, 0))
            by_attr[key] = (n + bool(ts & set(said)), k + 1)
    labels = {k: cfg["attributes"][k]["label"] for k in by_attr}
    return {"by_variant": by_var, "by_attr": {labels[k]: v for k, v in by_attr.items()},
            "top_terms": sorted(terms.items(), key=lambda x: -x[1])[:10]}


TODO_MARK = "【待補】"


def attr_labels(cfg, personas, pids, key):
    attrs = cfg["attributes"]
    if key == "_variant":
        return np.array([VARIANT_LABEL.get(personas[p]["variant"], "?") for p in pids])
    lv = lambda p: personas[p]["levels"][key]
    return np.array([NONE if lv(p) == NONE else attrs[key]["levels"][lv(p)]["label"] for p in pids])


# ---------- 成對比較：對照模組與身分位置 ----------

def condition_scores(rows_by_cond, cfg, names):
    """每種條件下，每位人物版本一個分數。只保留每種條件都有完整題型的人物，才能成對比較。
    回傳 (共同的人物, {條件: {指標: 分數陣列}}, {條件: 逐人逐則資料}, 題型)。"""
    weights = {k: v for k, v in cfg["category_weights"].items() if not k.startswith("_")}
    cats = [c for c in weights if all(any(r["cat"] == c for r in rows) for rows in rows_by_cond.values())]
    by = {k: {} for k in rows_by_cond}
    for k, rows in rows_by_cond.items():
        for r in rows:
            by[k].setdefault(r["pid"], []).append(r)
    pids = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    out = {}
    for k in rows_by_cond:
        out[k] = {m: stats.weighted_persona_scores(by[k], pids, cats, weights, m, m in SUBSET_METRICS) for m in names}
    ok = np.ones(len(pids), bool)
    for k in out:
        ok &= ~np.isnan(out[k]["字數"])
    pids = [p for p, x in zip(pids, ok) if x]
    out = {k: {m: v[ok] for m, v in d.items()} for k, d in out.items()}
    return pids, out, by, cats, weights


def paired_results(a, scores, ref, conds, names):
    """每種條件對照參考條件，逐指標成對比較；多重比較一起做 BH 校正。"""
    rng = np.random.default_rng(a.seed)
    rows = []
    for c in conds:
        for m in names:
            diff = scores[c][m] - scores[ref][m]
            mean, lo, hi, p = stats.paired_test(diff, a.perms, a.boots, rng)
            rows.append({"比較": f"{c} − {ref}", "指標": m, "人數": int(np.sum(~np.isnan(diff))),
                         "參考平均": float(np.nanmean(scores[ref][m])) if len(diff) else np.nan,
                         "對照平均": float(np.nanmean(scores[c][m])) if len(diff) else np.nan,
                         "平均差": mean, "區間下界": lo, "區間上界": hi, "p": p})
    adj = stats.bh_adjust([r["p"] if not np.isnan(r["p"]) else 1.0 for r in rows]) if rows else []
    for r, q in zip(rows, adj):
        r["校正後p"] = float(q)
    return rows


def paired_rows(a, cfg, conv_by, dirs_by, qcat, qmap, lex, module=None):
    """各條件逐則回覆的指標。每個條件的資料夾都評審過才用評審指標，否則全部用關鍵詞初篩（兩邊才可比）。
    回傳 (逐條件資料, 指標名稱, 評審摘要, 警示)。"""
    js, infos = {}, {}
    for k, convs in conv_by.items():
        js[k], infos[k] = judged_for(cfg, a, dirs_by[k], convs)
    use = all(v is not None for v in js.values())
    rows_by = {}
    if use:
        for k, convs in conv_by.items():
            rows_by[k] = reply_rows(convs, qcat, lex, js[k], qmap, module, standardize=False)
        # 建議數字要在所有條件一起標準化，條件之間才可比
        standardize_numbers([r for v in rows_by.values() for r in v])
    else:
        rows_by = {k: reply_rows(v, qcat, lex) for k, v in conv_by.items()}
    names = metric_names([r for v in rows_by.values() for r in v], use, module)
    warns = []
    for k, info in infos.items():
        warns += [f"{k}：{w}" for w in judge_warnings(info)]
    if not use and any(v is not None for v in js.values()):
        warns.append("只有部分條件評審過；為了可比，這份報告全部改用關鍵詞初篩。")
    return rows_by, names, (infos if use else None), warns


def load_cond(cfg, plist, d, questions, styles, turns):
    stylers = stylers_for(cfg, styles, os.path.dirname(os.path.abspath(d)), layout.read_settings(d))
    from .runner import expected_qh
    convs, n_empty, n_stale = load_valid(os.path.join(d, layout.RESPONSES), plist, turns, questions,
                                         expected_qh(questions, turns, stylers))
    if n_empty or n_stale:
        warn(f"{os.path.basename(d)}：略過一些紀錄", f"含空回覆：{n_empty} 段" if n_empty else None,
             f"對不上設定檔：{n_stale} 段" if n_stale else None)
    return convs


def module_lexical(a, cfg, plist, run_dir, by, ref, conds, pids, module):
    """對照模組的詞彙距離（有 --embed 才算）：同一位人物，兩種寫法之間的用詞距離，扣掉同一種寫法重問一次的距離。
    重複分成單數次、雙數次兩半：參考＝參考寫法兩半之間的距離（重問的晃動）；對照＝參考的一半對另一種寫法的另一半
    （兩種配法平均），兩邊用的回覆一樣多才公平。平均差 > 0：寫法改變了用詞，超過重問的晃動。
    簡繁：繁體版的回覆（含遮罩字眼）逐字轉成簡體，兩邊都在簡體上比；簡體回覆不動。回傳 (結果列, 分段統計, 說明)。"""
    models = embed_models(a, cfg)
    if not models:
        return [], {}, ""
    if module == "script" and models != ["mock"]:
        main = (cfg.get("embedding") or {}).get("script_main") or "multilingual-e5-large"
        if not any(main in m for m in models):
            print(f"⚠ 簡繁模組的詞彙指標以 {main} 為主：這次沒有算它")
    reps = {r["rep"] for k in by for p in pids for r in by[k].get(p, [])}
    if len({x % 2 for x in reps}) < 2:
        return [], {}, "詞彙距離要至少兩次重複（分單數次、雙數次兩半比），這次沒有算。"
    personas = {uid(p): p for p in plist}
    t2s = T2S(cfg) if module == "script" else None
    trad = STYLE_LABEL["trad"]
    tok, rows = {}, []
    rng = np.random.default_rng(a.seed)
    cos = lambda x, y: np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    for em in models:
        def half(k, parity):
            conv = (lambda t: t2s.char(t)) if (t2s and k == trad) else (lambda t: t)
            texts = {p: [conv(r["text"]) for r in by[k][p] if r["rep"] % 2 == parity] for p in pids}
            return persona_vectors(texts, pids, cfg, run_dir, a.base_url, em, a.embed_chunk, a.mock or em == "mock",
                                   personas, mask_fn=t2s.char if t2s else None, tok=tok)
        r0, r1 = half(ref, 0), half(ref, 1)
        d0 = 1 - cos(r0, r1)
        for c in conds:
            c0, c1 = half(c, 0), half(c, 1)
            dk = ((1 - cos(r0, c1)) + (1 - cos(r1, c0))) / 2
            diff = dk - d0
            mean, lo, hi, p = stats.paired_test(diff, a.perms, a.boots, rng)
            rows.append({"比較": f"{c} − {ref}", "指標": lex_name(em, models), "人數": len(pids),
                         "參考平均": float(np.mean(d0)), "對照平均": float(np.mean(dk)), "平均差": mean,
                         "區間下界": lo, "區間上界": hi, "p": p})
    tok_sum = token_summary(tok)
    write_recuts(run_dir, tok)
    note = ("詞彙距離：參考平均＝同一種寫法、單數次與雙數次重複之間的距離（重問的晃動）；對照平均＝兩種寫法之間的距離；"
            "平均差 > 0 表示寫法改變了用詞，而且超過重問的晃動。")
    if module == "script":
        note += ("簡繁：兩個嵌入模型結論不一致時，以 multilingual-e5-large 為準（另一個嵌入模型以繁體訓練，對簡體可能較差）；"
                 "分段方式和主流程相同（每段 300 字，超過 510 token 再切）。"
                 "繁體回覆只轉字形、不換詞，所以兩組的距離含用語在地化（例如軟體與软件）。")
    return rows, tok_sum, note


def analyze_module(a, cfg, plist, run_dir):
    settings = layout.read_settings(run_dir)
    module, turns = settings["module"], settings["turns"]
    questions = module_questions(cfg, module)
    qcat = {q["id"]: q.get("type", q.get("category")) for q in questions}
    lex = cfg["lexicon"]
    styles = MODULES[module]
    convs = load_cond(cfg, plist, run_dir, questions, styles, turns)
    conv_by = {STYLE_LABEL[st]: [c for c in convs if c.get("style") == st] for st in styles}
    dirs_by = {k: run_dir for k in conv_by}
    if module == "paraphrase":
        # 對照：主流程同一組設定（同模型、身分位置、溫度、上限、輪數、題庫與人物版本）
        base = {k: v for k, v in settings.items() if k not in ("module", "created", "version")}
        main_dir = os.path.join(a.out, layout.run_name(base))
        if not os.path.isfile(os.path.join(main_dir, layout.RESPONSES)):
            print(f"✗ 略過：找不到主流程同一組設定的回覆（{os.path.basename(main_dir)}），無法對照")
            return
        conv_by = {STYLE_LABEL["orig"]: load_cond(cfg, plist, main_dir, questions, ["orig"], turns),
                   **conv_by}
        dirs_by[STYLE_LABEL["orig"]] = main_dir
        # 只比較兩邊都有的（人物版本、題目、重複序號），重複次數不同時才公平
        keys = [set((uid(c), c["qid"], c["rep"]) for c in v) for v in conv_by.values()]
        both = set.intersection(*keys)
        conv_by = {k: [c for c in v if (uid(c), c["qid"], c["rep"]) in both] for k, v in conv_by.items()}
    labels = list(conv_by)
    ref = labels[0]
    qmap = {q["id"]: q for q in questions}
    rows_by, names, jinfos, jwarns = paired_rows(a, cfg, conv_by, dirs_by, qcat, qmap, lex, module)
    pids, scores, by, cats, weights = condition_scores(rows_by, cfg, names)
    if len(pids) < 3:
        print(f"✗ 略過：每種寫法都問完所有題型的人物只有 {len(pids)} 位（至少要 3 位）")
        return
    results = paired_results(a, scores, ref, labels[1:], names)
    lex_rows, tok_sum, lex_note = module_lexical(a, cfg, plist, run_dir, by, ref, labels[1:], pids, module)
    if lex_rows:
        results += lex_rows
        adj = stats.bh_adjust([r["p"] if not np.isnan(r["p"]) else 1.0 for r in results])
        for r, q in zip(results, adj):
            r["校正後p"] = float(q)
    noise = {m: stats.split_half_noise({p: by[ref][p] for p in pids}, pids, cats, weights, m,
                                       m in SUBSET_METRICS)[0] for m in names}
    quals = {k: quality(v, settings) for k, v in conv_by.items()}
    title = {"script": "書寫風格對照：簡繁", "punctuation": "書寫風格對照：標點", "paraphrase": "無關擾動對照"}[module]
    if module == "paraphrase":
        intro = ["同一批人物、同樣的題目，只把首輪提問換成意思相同的另一種說法（q_paraphrase）。",
                 "這裡的差異是模型對「無關措辭」的敏感程度，可以當作身分差異的參照：身分造成的差異若不比這個大，就不宜下結論。"]
    else:
        intro = [f"同一批人物、同樣的題目，只改書寫風格（{'、'.join(labels)}），其餘逐字相同。",
                 "這是在看寫法不同時回覆有什麼不同，不是在判斷使用者是誰，也不是在判斷使用者是不是機器人。"]
        if module == "script":
            intro.append("字數用回覆原文的字元數；評審看的是回覆原文；簡體回覆不轉回繁體。"
                         "詞彙距離把繁體版的回覆逐字轉成簡體，兩邊都在簡體上比。")
            from .styles import load_table
            tb = load_table(os.path.join(a.out, "personas", settings.get("personas_version") or "")) or {}
            if not tb.get("reviewed"):
                intro.append("⚠ 簡繁固定資料（script_simp.json）還沒有人工校對，結果只能用來檢查流程。")
    if lex_note:
        intro.append(lex_note)
    intro += ["詞彙層分段：" + token_line(m, s_) for m, s_ in tok_sum.items()]
    warns = [f"{k}：{w}" for k, (q, ws) in quals.items() for w in ws] + jwarns
    adir = analysis_dir(run_dir)
    md = os.path.join(adir, "report.md")
    report.write_paired(md, title, layout.describe(settings), intro, warns, results, noise, quals, len(pids),
                        a.perms, a.boots, lex, thinking=settings.get("thinking"), judge=jinfos)
    report.write_csv(os.path.join(adir, "results.csv"), results)
    print(f"報告：{md}")
    print_paired(results, warns)


def print_paired(results, warns):
    for w in warns:
        print(f"⚠ {w}")
    flagged = [r for r in results if r["校正後p"] < report.FLAG_P]
    print(f"校正後 p < {report.FLAG_P:.2f} 的比較：{len(flagged)} 個")
    for r in flagged:
        print(f"  {r['比較']} × {r['指標']}  平均差 {r['平均差']:.3f}  p_adj={r['校正後p']:.3f}")


def identity_pairs(dirs):
    """找出只差在身分位置的設定：message 一份、system 一份。"""
    by = {}
    for d in dirs:
        s = layout.read_settings(d)
        if s.get("module"):
            continue
        key = layout.run_name(dict(s, identity="*"))
        by.setdefault(key, {})[s["identity"]] = d
    return [(v["message"], v["system"]) for v in by.values() if "message" in v and "system" in v]


def compare_identity(a, cfg, plist, msg_dir, sys_dir):
    settings = layout.read_settings(msg_dir)
    questions = module_questions(cfg, None)
    qcat = {q["id"]: q.get("type", q.get("category")) for q in questions}
    lex = cfg["lexicon"]
    turns = settings["turns"]
    conv_by = {"message": load_cond(cfg, plist, msg_dir, questions, ["orig"], turns),
               "system": load_cond(cfg, plist, sys_dir, questions, ["orig"], turns)}
    keys = [set((uid(c), c["qid"], c["rep"]) for c in v) for v in conv_by.values()]
    both = set.intersection(*keys)
    conv_by = {k: [c for c in v if (uid(c), c["qid"], c["rep"]) in both] for k, v in conv_by.items()}
    qmap = {q["id"]: q for q in questions}
    rows_by, names, jinfos, jwarns = paired_rows(a, cfg, conv_by, {"message": msg_dir, "system": sys_dir},
                                                 qcat, qmap, lex)
    pids, scores, by, cats, weights = condition_scores(rows_by, cfg, names)
    if len(pids) < 3:
        print(f"✗ 略過：兩批都問完所有題型的人物只有 {len(pids)} 位（至少要 3 位）")
        return
    results = paired_results(a, scores, "message", ["system"], names)
    noise = {m: stats.split_half_noise({p: by["message"][p] for p in pids}, pids, cats, weights, m,
                                       m in SUBSET_METRICS)[0] for m in names}
    quals = {k: quality(v, settings) for k, v in conv_by.items()}
    # 各屬性的群體差異在兩種身分位置下各有多大（描述用；控制其他屬性的同時估計在各批主流程的報告）
    personas = {uid(p): p for p in plist}
    attrs = cfg["attributes"]
    spread = []
    for key, attr in zip(list(attrs) + ["_variant"], [attrs[x]["label"] for x in attrs] + ["揭露方式"]):
        labels = attr_labels(cfg, personas, pids, key)
        levels = sorted(set(labels))
        for m in names:
            row = {"屬性": attr, "指標": m}
            for k in ("message", "system"):
                devs, _ = stats.level_devs(scores[k][m], labels, levels)
                if np.all(np.isnan(devs)):
                    row[k] = ("—", float("nan"))
                    continue
                j = int(np.nanargmax(np.abs(devs)))
                row[k] = (levels[j], float(devs[j]))
            spread.append(row)
    name = os.path.basename(msg_dir).replace("__message__", "__message-vs-system__", 1)
    out_dir = os.path.join(a.out, "analysis", "compare__" + name)
    os.makedirs(out_dir, exist_ok=True)
    md = os.path.join(out_dir, "report.md")
    intro = ["同一批人物、同樣的題目，身分寫在第一句提問前（message）或放進系統提示（system，模擬記憶功能）。",
             "兩批交錯執行，只比較兩邊都有的對話（人物版本、題目、重複序號都相同）。",
             f"- message：{os.path.basename(msg_dir)}", f"- system：{os.path.basename(sys_dir)}"]
    warns = [f"{k}：{w}" for k, (q, ws) in quals.items() for w in ws] + jwarns
    report.write_paired(md, "身分位置比較：message 與 system", layout.describe(settings).replace("身分位置：提問前｜", ""),
                        intro, warns, results, noise, quals, len(pids), a.perms, a.boots, lex, spread=spread,
                        thinking=settings.get("thinking"), judge=jinfos)
    report.write_csv(os.path.join(out_dir, "results.csv"), results)
    print(f"比較報告：{md}")
    print_paired(results, warns)
