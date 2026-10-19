"""run／status／check 指令：把每段對話派給模型、顯示進度、存檔，可中斷續跑。"""
import argparse
import json
import os
import platform
import random
import sys
import threading
import time
import urllib.error
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from . import __version__, envinfo, layout, mock
from .personas import TODO, VARIANT_LABEL, load_personas
from .client import ServerError, chat, chat_raw, gpu_status, loaded_context, model_info
from .client import compat as client_compat
from .common import (read_config, guard_reserved, reserved_notes, HOW_CONTEXT, PauseClock, how_max_tokens, keep_awake, ask, bullets, cmd_hint, stop, warn, done_key, fmt_dur, fmt_secs, followup_texts, gpu_row,
                     kv_block, load_json, load_retired, load_valid, next_steps, note_block, open_append, phash, qhash, read_jsonl, remain_rows,
                     report_block, save_json, version_rows)
from .styles import MODULES, STYLE_LABEL, Styler, load_table, module_questions, stylers_for
from .tuner import TUNE_RULE, Tuner

SHORT_IDENTITY = {"message": "提問前", "system": "系統提示"}


def plabel(p, arm=None, many=(False, False)):
    """進度行裡的人物標示，例如 p012 揭露；同時跑兩種身分位置或好幾種寫法時一併標出。"""
    out = f"{p['id']} {VARIANT_LABEL.get(p.get('variant'), '')}".strip()
    if arm and many[0]:
        out += f" {SHORT_IDENTITY[arm['identity']]}"
    if arm and many[1]:
        out += f" {STYLE_LABEL[arm['style']]}"
    return out


def main_questions(cfg):
    """主流程的題目（modules 含 main）；簡繁、標點等對照模組的專用題不在這裡。"""
    return module_questions(cfg, None)


def select_people(personas, n):
    """只取前 n 位人物（依編號，兩版一起）；n 為 0 時全部。"""
    if not n:
        return personas
    keep = set(sorted({p["id"] for p in personas})[:n])
    return [p for p in personas if p["id"] in keep]


def default_people(a, cfg):
    """沒有指定 --people 時：對照模組看設定檔（v1 簡繁、標點、無關擾動都是 24 人），身分位置小批 24 人，其餘全部人物。"""
    if a.people is not None:
        return a.people
    if a.module:
        return int((cfg.get("modules") or {}).get(a.module, {}).get("people", 0) or 0)
    if a.identity == "both":
        return 24
    return 0


def default_variants(a, cfg):
    """跑人物的哪幾版：--variants 優先；簡繁、標點看設定檔（v1 只跑揭露版）；其餘兩版都跑。"""
    if getattr(a, "variants", None):
        return [v.strip() for v in a.variants.split(",") if v.strip()]
    if a.module:
        v = (cfg.get("modules") or {}).get(a.module, {}).get("variants")
        if v:
            return list(v)
    return ["explicit", "implicit"]


def question_text(q, style):
    return (q.get("q_paraphrase") or "") if style == "para" else q["q"]


def build_messages(cfg, settings, p, q, turns, styler, identity):
    """組出一段對話要送的系統提示與每一輪的提問。寫法（styler）套在人物介紹、提問與追問上。"""
    sys_parts = [settings["system_prompt"]] if settings["system_prompt"] else []
    qtext = question_text(q, styler.style)
    if styler.style == "simp":         # 簡繁：查固定資料，人物介紹、前綴與題目各自是一段
        if identity == "system":
            sys_parts.append(styler(cfg.get("identity_system_prefix", "")) + styler(p["intro"]))
            first = styler(qtext)
        else:
            first = styler(p["intro"]) + styler(qtext)
    elif identity == "system":
        sys_parts.append(styler(cfg.get("identity_system_prefix", "") + p["intro"]))
        first = styler(qtext)
    else:
        first = styler(p["intro"] + qtext)      # 合起來再轉，標點版的句界才會一致
    users = [first] + [styler(t) for t in followup_texts(q, turns)]
    return ("\n\n".join(sys_parts) if sys_parts else None), users


def expected_qh(questions, turns, stylers):
    """每題在每種寫法下的指紋。原文沿用舊的算法，舊資料不必重跑。"""
    out = {}
    for st, sty in stylers.items():
        for q in questions:
            if st == "orig":
                out[(q["id"], st)] = qhash(q, turns)
            else:
                out[(q["id"], st)] = sty.fingerprint([question_text(q, st)] + followup_texts(q, turns))
    return out


def estimate_rows(cfg, speed, n_run, n_left, turns, minutes=0):
    """執行前的時間預估：生成依這台機器的實測（沒有就用設定檔的估計值），評審依設定檔的估計範圍。"""
    gen = cfg.get("generation", {})
    if speed:
        per, src = speed, "這台機器以前跑這個模型的實測"
    else:
        per, src = float(gen.get("est_secs_per_segment", 19)), "設定檔的估計值；這台還沒有實測"
    lo, hi = (cfg.get("judge", {}).get("est_secs_per_call") or [1, 4])[:2]
    calls = n_run * turns
    rows = [("每段", f"約 {per:.0f} 秒（{src}）"),
            ("這次", f"{n_run} 段，約 {fmt_dur(per * n_run)}"),
            ("全部剩下", f"{n_left} 段，約 {fmt_dur(per * n_left)}" if n_left != n_run else None),
            ("評審", f"{calls} 次呼叫，約 {fmt_dur(lo * calls)}～{fmt_dur(hi * calls)}（設定檔的估計範圍）")]
    if minutes:
        rows.append(("時間上限", f"--minutes {minutes:g}，約可跑 {int(minutes * 60 / per)} 段"
                     + ("（不到兩段的時間：可能只跑完手上那幾段）" if minutes * 60 < per * 2 else "")))
    return rows


def past_speed(out_dir, model, machine, turns):
    """這台機器、這個模型、同樣輪數，以前實測每完成一段要幾秒（從各資料夾的 run_meta.json 加總）。"""
    secs = segs = 0
    for d in layout.run_dirs(out_dir):
        m = layout.read_meta(d)
        if m.get("model") != model or m.get("rounds") != turns:
            continue
        for ss in m.get("sessions", []):
            if ss.get("machine") == machine and ss.get("segments", 0) >= 5:
                secs += ss["secs"]
                segs += ss["segments"]
    return secs / segs if segs else None


def resolve_generation(a, cfg):
    """指令沒給的生成設定，用設定檔 generation 的值。"""
    gen = cfg.get("generation", {})
    if a.temperature is None:
        a.temperature = gen.get("temperature", 0.7)
    if a.max_tokens is None:
        a.max_tokens = gen.get("max_tokens", 3200)


def parse_rep_range(text, reps):
    """--rep-range 3-4 → 第 3、4 次重複（程式內部從 0 起算：[2, 3]）。只給一個數字就是那一次。"""
    if not text:
        return list(range(reps))
    m = __import__("re").fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", text)
    if not m:
        stop(f"--rep-range 格式不對：{text!r}", "格式：開始-結束，例如 3-4（第 3、4 次重複）")
    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    if lo < 1 or hi < lo:
        stop(f"--rep-range {text} 不對", "重複序號從 1 起算", "結束不能小於開始")
    if hi > reps:
        stop(f"--rep-range {text} 超過 --reps {reps}", "要多跑幾次：一起加大 --reps")
    return list(range(lo - 1, hi))


def reclaim_candidates(d, ident, good, plan, here):
    """每次重複只來自一台機器：這台要跑、還沒跑滿的重複裡，別台已經跑了的段。回傳 {重複: (那些段, 計畫段數)}。
    機器代號不同、顯示卡卻相同的不算別台（多半是同一台換了 --machine 的寫法）。"""
    planned = {}
    for t in plan:
        if t[3]["identity"] == ident:
            k = done_key(task_record_key(t))
            planned.setdefault(k[3], set()).add(k)
    envs = layout.read_meta(d["dir"]).get("envs") or {}
    here_gpu = envinfo.current_gpu()
    same_hw = lambda r: bool(here_gpu) and envinfo.gpu_model((envs.get(r.get("env")) or {}).get("gpu")) == here_gpu
    out = {}
    for rep, keys in sorted(planned.items()):
        have = {done_key(r) for r in good if r["rep"] == rep and done_key(r) in keys}
        if len(have) >= len(keys):
            continue                       # 這次重複已經跑滿（不管哪台跑的）：這台沒有要跑的，不動
        others = [r for r in good if r["rep"] == rep and done_key(r) in keys and r.get("machine") and r["machine"] != here
                  and not same_hw(r)]
        if others:
            out[rep] = (others, len(keys))
    return out


def decide_reclaim(cands, here):
    """別台跑了不到一半：自動改在這台重跑。一半以上：問要不要整次在這台重跑（預設保留，這台只補剩下的；
    不在終端機時保留）。回傳要移出的段。"""
    take = []
    for rep, (others, n) in sorted(cands.items()):
        ms = "、".join(sorted({r["machine"] for r in others}))
        if len(others) < n / 2:
            take += others
            continue
        head = f"第 {rep + 1} 次重複：{ms} 已經跑了 {len(others)}/{n} 段（一半以上），這台要接著跑"
        items = ["y：整次在這台重跑（那些段移到備份；這次重複就只來自這台，要多花時間）",
                 "其他鍵：保留，這台只補剩下的（這次重複會由兩台跑；報告「各機器」一節列出）"]
        if not sys.stdin or not sys.stdin.isatty():
            warn(head, "不在終端機執行，沒辦法問：保留，這台只補剩下的")
            continue
        warn(head, *items)
        ans = ask(f"  第 {rep + 1} 次要整次在這台（{here}）重跑嗎？")
        if ans is None:
            print("  ・讀不到輸入：保留，這台只補剩下的")
        elif ans == "y":
            take += others
    return take


def apply_reclaim(d, take, here):
    """把要改在這台重跑的段移到 out/backups/（reclaimed），run_meta.json 記下來。"""
    gone = {(done_key(r), r.get("machine"), r.get("time")) for r in take}
    keep_line = lambda r: (done_key(r), r.get("machine"), r.get("time")) not in gone
    allrecs = read_jsonl(d["path"])
    bak = layout.backup_path(d["path"], "reclaimed")
    with open(bak, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allrecs if not keep_line(r)))
    tmp = d["path"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allrecs if keep_line(r)))
    os.replace(tmp, d["path"])
    by = {}
    for r in take:
        by[(r["rep"] + 1, r["machine"])] = by.get((r["rep"] + 1, r["machine"]), 0) + 1
    meta = layout.read_meta(d["dir"])
    meta.setdefault("reclaimed", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": here,
                                            "taken": [{"rep": k[0], "from": k[1], "segments": n} for k, n in sorted(by.items())],
                                            "backup": os.path.basename(bak)})
    layout.update_meta(d["dir"], meta)
    warn("每次重複只來自一台：改在這台重跑",
         *[f"第 {rep} 次：{m} 的 {n} 段移到備份，這台重跑" for (rep, m), n in sorted(by.items())], f"備份：{bak}")


def reps_guard(dirs, machine, tasks):
    """這台上一次用 --rep-range 只跑一部分重複（例如 3090 跑第 1–3 次、第 4 次交給 M1 Pro），這次卻沒給 --rep-range：
    先問一次（10-17 的 3090 就多跑了第 4 次的 8 段）。回答要跑就照樣跑：合併時兩邊都跑了的那次重複，
    merge 會整次用比較完整的一邊。不在終端機執行（沒辦法問）時停下，請明確給 --rep-range。"""
    last = None                 # 這台最近一次真的跑了段的執行（被 tools/drop_segments.py 整批移出的不算）
    for d in dirs.values():
        for ss in layout.read_meta(d["dir"]).get("sessions", []):
            if ss.get("machine") == machine and ss.get("rep_range") and ss.get("segments") and not ss.get("dropped_all") \
                    and (last is None or ss.get("start", "") >= last.get("start", "")):
                last = ss
    # 只看用 --rep-range 縮小過範圍的那次（--reps 1 這種只是總次數少，不算分工）；頭尾都要看（例如 M1 Pro 只跑 4-4）
    lo, hi = (int(last["rep_range"][0]), int(last["rep_range"][1])) if last else (1, 0)
    narrowed = last and (lo > 1 or hi < int(last.get("reps") or 0)) and last.get("rep_range_given", True)
    if not narrowed:
        return
    extra = sorted({t[2] + 1 for t in tasks if not lo <= t[2] + 1 <= hi})
    if not extra:
        return
    span = f"{lo}" if lo == hi else f"{lo}-{hi}"
    head = f"這台（{machine}）上一次只跑第 {span} 次重複；這次沒給 --rep-range，會跑到第 {'、'.join(map(str, extra))} 次"
    every = sorted(set(extra) | set(range(lo, hi + 1)))
    items = ["跨機器分工：別台負責的重複不要在這台跑",
             "照樣跑也可以：合併時兩邊都跑了的那次重複，merge 整次用比較完整的一邊",
             f"照原本分工：加 --rep-range {span}；確定這台也要跑：加 --rep-range {every[0]}-{every[-1]}（就不會再問）"]
    if not sys.stdin or not sys.stdin.isatty():
        stop(head, *items, "不在終端機執行，沒辦法問：請明確給 --rep-range")
    warn(head, *items)
    ans = ask(f"  這台也要跑第 {'、'.join(map(str, extra))} 次嗎？輸入 y 照樣跑，其他鍵停下：")
    if ans is None:
        stop("沒有跑", "讀不到輸入：請明確給 --rep-range")
    if ans != "y":
        stop("沒有跑", f"照原本分工：加 --rep-range {span}")


def plan_tasks(personas, questions, reps, seed, arms=None):
    """固定的執行順序：先讓所有人物（兩版各算一筆）跑完第 1 次重複，再跑第 2 次；每次重複內打亂。
    同時跑好幾組（身分位置、寫法）時，各組交錯進行。這樣不管在哪裡中斷，已完成的部分各群體、各組都大致平衡。
    reps 可以是次數，或要跑的重複序號清單（跨機器分工的 --rep-range）。每次重複的打亂順序只看序號，分機器跑也一樣。"""
    arms = arms or [{"identity": "message", "style": "orig"}]
    tasks = []
    for rep in (range(reps) if isinstance(reps, int) else reps):
        block = [(p, q, rep, arm) for p in personas for q in questions for arm in arms]
        random.Random(seed * 7919 + rep).shuffle(block)
        tasks += block
    return tasks


def task_record_key(t):
    p, q, rep, arm = t
    return {"pid": p["id"], "variant": p.get("variant"), "qid": q["id"], "rep": rep, "style": arm["style"]}


def arm_label(arm, many=(True, True)):
    return "／".join(x for x in (SHORT_IDENTITY[arm["identity"]] if many[0] else "",
                                 STYLE_LABEL[arm["style"]] if many[1] else "") if x)


def compare_after_run(a, dirs):
    """--identity both 兩批都跑完：直接產生各自的報告與比較報告。"""
    from .analyze import run_analysis
    print("\n── 兩批都跑完：接著產生報告與身分位置比較")
    ns = argparse.Namespace(**vars(a))
    for k, v in {"embed_model": "", "embed": False, "embed_chunk": 300, "perms": 2000, "boots": 1000, "run": ""}.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    run_analysis(ns, [d["dir"] for d in dirs.values()])


def estimate_need(a, turns, seen_tokens):
    """估計一段對話最多用多少 token，回傳 (數字, 來源說明)。優先序：
    1. --conv-tokens 直接指定；2. 已完成對話中的最大值；3. 輪數 × --est-chars（中文約一個字一個 token）。
    後兩者再加 15% 與 200 的餘裕。"""
    if a.conv_tokens:
        return int(a.conv_tokens), "--conv-tokens 指定"
    if seen_tokens:
        return int(max(seen_tokens) * 1.15) + 200, f"已完成的 {len(seen_tokens)} 段中最大 {max(seen_tokens)}，加餘裕"
    return int(turns * a.est_chars * 1.15) + 200, f"{turns} 輪 × --est-chars {a.est_chars}，加餘裕"


def script_table(a, cfg, pdir, personas):
    """簡繁模組的固定資料。沒有時：--mock、--draft、--preview 自動轉一次（只能試流程），正式執行要先跑 script 指令。
    用到的文字有缺（例如人數比上次轉的多）也一樣處理。"""
    table = load_table(pdir)
    need = None
    if table is not None:
        have = {t["orig"] for t in table["texts"]}
        from .script import script_texts
        need = [t for _, t in script_texts(cfg, personas) if t not in have]
    if table is not None and not need:
        return table
    why = "還沒有簡繁固定資料" if table is None else f"簡繁固定資料少了 {len(need)} 段（人數或題目和上次轉的不同）"
    if a.mock or a.draft or a.preview:
        from .script import build_table
        table = build_table(cfg, pdir, personas)[0]
        warn(why, f"試流程：先自動轉一次（{len(table['texts'])} 段，尚未校對）")
        return table
    extra = "".join(f" {k} {v}" for k, v in (("--people", a.people), ("--variants", getattr(a, "variants", None)),
                                             ("--personas", getattr(a, "personas", None))) if v)
    stop(why, f"先轉換：python lijuncha.py script{extra}", f"再看過：{os.path.join(pdir, 'script_review.md')}",
         "提示文字只轉一次，校對後固定")


def cmd_run(a):
    cfg = read_config(a.config)
    layout.tidy_backups(a.out)
    old_ver = getattr(a, "personas", "") or ""
    all_personas, pdir = load_personas(a.out, cfg, old_ver or None)
    if not all_personas:
        stop(f"找不到人物：{pdir}", "確認 --personas 的版本名稱（out/personas/ 底下的資料夾名稱）" if old_ver
             else "先產生人物：python lijuncha.py gen")
    is_confirm_set = any(p.get("confirm") for p in all_personas)
    if old_ver and old_ver != cfg.get("version", {}).get("personas") and not is_confirm_set:
        warn(f"用舊版人物 {old_ver} 接著跑（設定檔是 {cfg['version']['personas']}）", "只能用來把試跑資料跑完，不能當正式資料")
    if not a.mock and not a.preview and a.model == "mock":
        stop("沒有指定模型", "--model <LM Studio 裡的模型識別名稱>", "只試流程：加 --mock 用假回覆")
    from .common import check_origin
    check_origin(cfg, "" if a.mock else a.model, what="受測模型")
    from . import cloud
    cloud_name = cloud.setup(a, cfg)            # 雲端模型：網址與金鑰（金鑰只從環境變數讀，只放在記憶體）
    budget_cap, budget_used = cloud.check_budget(a, cfg, a.model) if cloud_name else (None, 0.0)
    resolve_generation(a, cfg)
    gen_cfg = cfg.get("generation", {})
    rep_list = parse_rep_range(getattr(a, "rep_range", ""), a.reps)
    strict = not (a.mock or a.draft or a.preview)        # 正式執行：上下文與每則上限都要照設定檔
    if strict and gen_cfg.get("max_tokens") and int(a.max_tokens) != int(gen_cfg["max_tokens"]):
        stop(f"--max-tokens {a.max_tokens} 和設定檔 generation.max_tokens（{gen_cfg['max_tokens']}）不同",
             "每則上限整個研究一律相同", *how_max_tokens(a.config))
    n_people_arg = default_people(a, cfg)
    variants = default_variants(a, cfg)
    if not set(variants) <= {"explicit", "implicit"}:
        stop("--variants 不對", "只能是 explicit、implicit，或兩者用逗號隔開")
    personas = [p for p in select_people(all_personas, n_people_arg) if p.get("variant") in variants]
    questions = module_questions(cfg, a.module)
    if not questions:
        stop(f"設定檔沒有給「{a.module}」模組的題目", "在 questions 的 modules 欄位標記")
    identities = ["message", "system"] if a.identity == "both" else [a.identity]
    styles = MODULES[a.module] if a.module else ["orig"]
    table = script_table(a, cfg, pdir, personas) if "simp" in styles else None
    stylers = {st: Styler(cfg, st, table) for st in styles}

    # ---- 每一組（身分位置 × 寫法）一個 arm；身分位置不同就是不同的設定資料夾，寫法不同放同一個資料夾
    arms, dirs = [], {}
    for ident in identities:
        settings = layout.settings_from_args(a, cfg, os.path.basename(pdir), identity=ident)
        if a.mock:
            settings["model"] = "mock"
        if cloud_name:
            settings["provider"] = cloud_name
        d = layout.open_run(a.out, settings)
        dirs[ident] = {"dir": d, "path": os.path.join(d, layout.RESPONSES), "settings": settings}
        guard_reserved(cfg, dirs[ident]["path"], "生成")
        for st in styles:
            arms.append({"identity": ident, "style": st, "styler": stylers[st], **dirs[ident]})
    first_settings = arms[0]["settings"]
    turns = first_settings["turns"]
    sysh = layout.sys_hash(first_settings["system_prompt"])
    many = (len(identities) > 1, len(styles) > 1)

    # ---- 送出的文字裡不能有【待補】（除非明說只是試流程）
    todo_hits = set()
    for arm in arms:
        for p in personas:
            for q in questions:
                if TODO in p["intro"]:
                    todo_hits.add("人物介紹")
                sysmsg, users = build_messages(cfg, arm["settings"], dict(p, intro=""), q, turns, arm["styler"],
                                               arm["identity"])
                if any(TODO in t for t in [sysmsg or ""] + users):
                    todo_hits.add(f"題目 {q['id']}" + ("（同義改寫）" if arm["style"] == "para" else ""))
    if todo_hits and not a.draft and not a.preview:
        for d in dirs.values():
            layout.drop_if_empty(d["dir"])
        stop("要送給模型的文字還有【待補】", "位置：" + "、".join(sorted(todo_hits)), "正式執行：先在設定檔補齊",
             "只試流程：加 --draft（回覆只能檢查流程，不能當正式資料）")

    # ---- 簡繁模組：讀轉好的固定資料（script 指令產生），執行時不轉換
    if a.module == "script":
        print(f"簡繁固定資料：{os.path.join(pdir, 'script_simp.json')}（{len(table['texts'])} 段；"
              + ("已校對" if table.get("reviewed") else "尚未校對") + "）")
    if a.preview:
        p0 = personas[0]
        for arm in arms:
            sysmsg, users = build_messages(cfg, arm["settings"], p0, questions[0], turns, arm["styler"], arm["identity"])
            print(f"\n── {arm_label(arm)} · {plabel(p0)} · {questions[0]['id']}")
            if sysmsg:
                print(f"[系統提示] {sysmsg}")
            for i, u in enumerate(users):
                print(f"[第 {i + 1} 輪] {u}")
        for d in dirs.values():
            layout.drop_if_empty(d["dir"])
        print("\n（--preview：只顯示要送出的文字，沒有連線模型）")
        return

    # ---- 讀已完成的部分；壞掉或過期的紀錄移到備份檔，之後自動補跑
    expect = expected_qh(questions, turns, stylers)
    plan = plan_tasks(personas, questions, rep_list, a.seed, arms)
    here = envinfo.machine_code(a)
    done, good_all, reclaim = set(), [], {}
    for ident, d in dirs.items():
        good, empty, stale = load_valid(d["path"], all_personas, turns, questions, expect)
        if empty or stale:
            keep = good + load_retired(d["path"], questions, turns)     # 舊版題目的回覆照樣保留
            bak = layout.backup_path(d["path"])
            os.replace(d["path"], bak)
            with open(d["path"], "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            warn(f"清理 {empty + stale} 段：移出後重新補跑", f"含空回覆：{empty} 段" if empty else None,
                 f"人物或題目和設定檔對不上：{stale} 段" if stale else None, f"備份：{bak}")
        # 別台在這台要跑的重複裡跑了的段：先決定要不要改在這台重跑（檔案等所有檢查都通過才動）
        take = decide_reclaim(reclaim_candidates(d, ident, good, plan, here), here)
        if take:
            reclaim[ident] = take
        taken = {(done_key(r), r.get("machine"), r.get("time")) for r in take}
        done |= {(ident, *done_key(r)) for r in good if (done_key(r), r.get("machine"), r.get("time")) not in taken}
        good_all += good
    total = len(plan)
    tasks = [t for t in plan if (t[3]["identity"], *done_key(task_record_key(t))) not in done]
    n_done0 = total - len(tasks)
    if a.limit:
        tasks = tasks[:a.limit]
    n_run = len(tasks)

    lock = threading.RLock()

    def say(msg):
        with lock:
            print(msg, flush=True)

    nq = len(questions)
    n_people = len({p["id"] for p in personas})
    notes = []                    # 執行前的提醒，最後集中列在「提醒」一節
    if a.module == "script" and not table.get("reviewed"):
        notes.append("簡繁固定資料還沒校對：結果只能檢查流程（看過 script_review.md 後，把 script_simp.json 的 reviewed 改成 true）")
    if todo_hits:
        notes.append("--draft：送出的文字含【待補】，這批回覆只能檢查流程")
    who = f"前 {n_people} 位人物" if n_people_arg else f"{n_people} 位人物"
    combos = len(arms)
    rr = f"（這台機器只跑第 {rep_list[0] + 1}–{rep_list[-1] + 1} 次）" if len(rep_list) != a.reps else ""
    nv = len(personas) // max(n_people, 1)
    scale = (f"{who} × {'兩版' if nv == 2 else '揭露版' if variants == ['explicit'] else '隱性版' if variants == ['implicit'] else f'{nv} 版'}"
             f" × {nq} 題 × 重複 {len(rep_list)} 次{rr}"
             + (f" × {combos} 組（{'、'.join(arm_label(x, many) for x in arms)}）" if combos > 1 else "")
             + f" = {total} 段（{total * turns} 則回覆）")
    machine = envinfo.machine_code(a)
    if strict and not getattr(a, "rep_range", "") and tasks:
        reps_guard(dirs, machine, tasks)
    est = estimate_rows(cfg, past_speed(a.out, a.model, machine, turns), n_run, total - n_done0, turns,
                        a.minutes) if n_run and not a.mock else []

    # ---- 並行數：自動或固定
    auto = str(a.workers).lower() == "auto"
    fixed = 0 if auto else max(1, int(a.workers))
    max_workers = max(a.max_workers, a.start_workers, fixed)   # 起跑數或固定數比上限大，就以它為準
    tune_path = os.path.join(a.out, "tuning.json")
    # 同一台電腦換了顯示卡，測得的並行數不能沿用：鍵裡加顯示卡型號
    gpu = None if a.mock else envinfo.current_gpu()
    tune_key = f"{platform.node()}|{a.base_url}|{a.model}" + (f"|{gpu}" if gpu else "")
    tune_all = load_json(tune_path) if os.path.exists(tune_path) else {}
    remembered = tune_all.get(tune_key, {})
    seen_tokens = [r["tokens"] for r in good_all if r.get("tokens")]
    need, need_from = estimate_need(a, max(turns, cfg.get('generation', {}).get('context_turns', turns)), seen_tokens)
    ctx = None if a.mock or cloud_name else loaded_context(a.base_url, a.model)
    minfo = None if a.mock or cloud_name else model_info(a.base_url, a.model)
    min_seg = int(gen_cfg.get("min_context_per_segment", 12000))
    want_ctx = gen_cfg.get("context_length")
    if not a.mock and not cloud_name:            # 雲端模型不做 LM Studio 的上下文檢查
        problems = []
        if minfo and str(minfo.get("compatibility_type", "")).lower() == "mlx":
            problems.append("LM Studio 載入的是 MLX 版：各機器要用同一個 GGUF 檔，Mac 不得改用 MLX")
        if ctx is None:
            problems.append("問不到模型載入的上下文長度（要用 LM Studio，--model 要和識別名稱一致）")
        else:
            if ctx[0] < min_seg:
                problems.append(f"上下文長度只有 {ctx[0]} token：每段至少要 {min_seg}（設定檔 generation.min_context_per_segment）")
            if want_ctx and int(want_ctx) != ctx[0]:
                problems.append(f"上下文長度是 {ctx[0]}，設定檔 generation.context_length 是 {want_ctx}：用設定檔的值重新載入模型")
        if not want_ctx:
            notes.append("設定檔 generation.context_length 還沒填：正式執行前用 calibrate 推算後寫進去（各機器才能對齊）")
        if problems and strict:
            for d in dirs.values():
                layout.drop_if_empty(d["dir"])
            stop("不能正式執行", *problems, *HOW_CONTEXT, "只試流程：加 --draft")
        notes += [f"（--draft）{pr}" for pr in problems]
    seg_need = max(need, min_seg)
    cap, cap_why = max_workers, None
    if ctx:
        fit = max(1, ctx[0] // seg_need)
        if fit < cap:
            cap, cap_why = fit, f"上下文由並行的各段平分，每段要 {seg_need}"
            want = min(max_workers, a.start_workers)
            if fit < want:
                notes.append(f"想並行 {want} 段：LM Studio 的 Context Length 要 {want * seg_need} 以上"
                             + (f"（這個模型最多 {ctx[1]}）" if ctx[1] else ""))
    elif not a.mock and not cloud_name:
        cap_why = "問不到上下文長度；伺服器容量不足時自動降"
    env = (cloud.env_record(a, cloud_name) if cloud_name else envinfo.collect(a, a.model, minfo, a.out, mock=a.mock)) \
        if n_run else None
    if env and not a.mock and not cloud_name:
        from .client import load_config
        load = envinfo.load_settings(load_config(a.base_url, a.model), gen_cfg.get("lmstudio_load"))
        if load:
            env["load"] = load
            sp, uni = load.get("speed") or {}, (load["content"].get("unified_kv") or {}).get("value")
            if sp.get("parallel") and sp.get("context_length") and uni is not True:
                per = int(sp["context_length"]) // int(sp["parallel"])
                if per < min_seg:
                    notes.append(f"每段只分到 {per:,} token（Max Concurrent {sp['parallel']}、沒開 Unified KV Cache；每段至少要 {min_seg:,}）："
                                 "調低 Max Concurrent，或開 Unified KV Cache")
    eid = envinfo.env_id(env) if env else None
    if env and not a.mock and not cloud_name and env["model_sha256"] == envinfo.UNKNOWN:
        notes.append(f"{env['model_file_note']}：跨機器合併時無法比對模型檔；{envinfo.LINK_HINT}")
    elif env and not a.mock and not cloud_name:
        notes += [n for n in envinfo.fit_notes(a.model, (minfo or {}).get("quantization"), getattr(a, "model_file", None))
                  if "LM Link" not in n]
    qids = [q["id"] for q in questions]
    switch = machine_switch(a, dirs, env, good_all, machine, total, rep_list, strict, notes, qids) \
        if env and not cloud_name else None
    for ident, take in reclaim.items():        # 檢查都通過了：把改在這台重跑的段移到備份
        apply_reclaim(dirs[ident], take, here)
        gone = {id(r) for r in take}
        good_all = [r for r in good_all if id(r) not in gone]
    # 自動模式不從 1 段慢慢爬：直接從 --start-workers 起跑（容不下時自動減少），再往上試
    start = fixed or max(1, min(cap, a.start_workers))
    tuner = Tuner(start, cap, bool(fixed), say,
                  retune_gap=a.retune_gap or 20 * 60,
                  explore_gap=0 if a.explore_gap < 0 else (a.explore_gap or 15 * 60))
    same_ctx = remembered.get("context") == (ctx[0] if ctx else None)
    same_setup = auto and same_ctx and remembered.get("rule") == TUNE_RULE and not a.retune
    probe = remembered.get("probe") if same_setup else None      # 上次測速做到一半就停了
    usable = same_setup and not probe and remembered.get("level")
    # 上次是頂到上限才停的，而這次上限比較高：從上次的數字接著往上試
    was_capped = usable and remembered.get("capped") and cap > int(remembered.get("cap", 0))
    workers_text, done_lv = f"自動：從 {tuner.level} 段起跑", None
    if probe:
        tuner.resume(probe)
        if tuner.probing:
            done_lv = "、".join(f"{k} 段每秒 {v:.0f} token" for k, v in sorted(tuner.measured.items()))
            workers_text = (f"自動：接續上次沒測完的進度（正在測 {tuner.level} 段，"
                            f"已收 {len(tuner.samples)}/{tuner.needed()} 個樣本）")
    elif usable and not was_capped:
        tuner.level = max(1, min(int(remembered["level"]), cap))
        tuner.probing = False
        workers_text = f"自動：沿用這台機器上次測得的 {tuner.level} 段"
    elif fixed:
        tuner.level = min(fixed, cap)
        workers_text = f"固定 {tuner.level} 段" + (f"（上下文長度只夠 {cap} 段）" if fixed > cap else "")
    elif was_capped:
        tuner.level = max(1, min(int(remembered["level"]), cap))
        workers_text = f"自動：從 {tuner.level} 段起跑"
    workers_rows = [("並行", workers_text),
                    ("並行上限", None if fixed else f"{cap} 段" + (f"（{cap_why}）" if cap_why else "")),
                    ("已量完", done_lv or None),
                    ("再測", f"每 {tuner.explore_gap / 60:g} 分鐘試一次更多段" if tuner.explore_gap and not fixed else None)]

    # ---- 執行前的說明：分節、對齊，同一件事和報告、文件用同一個名稱
    blocks = []
    for ident, d in dirs.items():
        title = "設定" + (f"（{SHORT_IDENTITY[ident]}）" if len(dirs) > 1 else "")
        blocks.append(kv_block(title, [("設定資料夾", os.path.basename(d["dir"]))] + layout.settings_rows(d["settings"])))
    if a.module:
        blocks[-1] += "\n" + kv_block("", [("寫法", "、".join(STYLE_LABEL[st] for st in styles)),
                                            ("題目", "、".join(q["id"] for q in questions))])
    sent = ((layout.read_meta(next(iter(dirs.values()))["dir"]).get("versions") or {}).get("persona_sentences")
            if old_ver and old_ver != cfg["version"].get("personas") else None)
    blocks.append(kv_block("版本", version_rows(cfg, a.config, os.path.basename(pdir), pdir, sent)))
    if switch:
        blocks.append(kv_block("換機器接續", switch))
    awake = keep_awake(not a.mock and not cloud_name).__enter__()     # 生成要跑好幾個小時：不讓電腦睡著（結束時恢復）
    blocks.append(kv_block("規模與進度", [
        ("規模", scale),
        ("已完成", f"{n_done0} 段（{n_done0 / total:.1%}）"),
        ("這次要跑", f"{n_run} 段" + (f"，最多 {a.minutes:g} 分鐘" if a.minutes else "")),
        awake.row()]))
    if est:
        blocks.append(kv_block("預估時間", est))
    if cloud_name:
        blocks.append(kv_block("雲端模型", [
            ("服務", f"{cloud_name}（{a.base_url}）"), ("金鑰", "從環境變數讀，不寫進任何檔案"),
            ("預算", f"上限 {budget_cap:g} 美元，到了就停" if budget_cap is not None else "設定檔沒有 cloud.budget_usd：不設上限"),
            ("已花", f"約 {budget_used:.2f} 美元（所有雲端資料合計）" if budget_cap is not None else None),
            ("服務條款", f"已讀（{a.cloud['tos_checked']}）" if a.cloud.get("tos_checked") else "還沒記錄讀過（見提醒）"),
            *workers_rows, ("用量", "每一輪一次呼叫；結束時列出呼叫次數、token 與費用")]))
    elif not a.mock:
        blocks.append(kv_block("上下文與並行", [
            ("上下文長度", f"{ctx[0]} token（LM Studio 的 Context Length"
                          + (f"；這個模型最多 {ctx[1]}" if ctx[1] else "") + "）" if ctx else "問不到"),
            ("設定檔要求", f"context_length {want_ctx or '還沒填'}；每段至少 {min_seg} token"),
            ("每段約需", f"{need} token（{need_from}）"),
            *workers_rows]))
        if env:
            blocks.append(kv_block("環境", [
                ("機器", env["machine"]), ("GPU", env["gpu"]),
                ("LM Studio", env["lmstudio"]), ("執行環境", env["runtime"]),
                ("模型檔", "不明" if env["model_file"] == envinfo.UNKNOWN else
                 [env["model_file"], f"雜湊值 {env['model_sha256'][:12]}｜{env['quantization']}｜{env['format']}"]),
                ("載入設定", envinfo.load_lines(env.get("load")))]))
        blocks.append(note_block("怎麼改", how_max_tokens(a.config) + HOW_CONTEXT))
    else:
        blocks.append(kv_block("並行", workers_rows))
    blocks.append(note_block("提醒", notes))
    print("\n".join(x for x in blocks if x))
    print("隨時可按 Ctrl+C：已存的都保留，再執行同一行接著跑。")
    if not tasks:
        print("沒有要跑的對話。")
        if n_done0 == total:
            auto_continue(a, cfg, dirs, fixed or a.start_workers)
            auto_probe(a, cfg, dirs, fixed or a.start_workers)
            print(run_next_steps(a, dirs, True, 0))
        for d in dirs.values():
            layout.drop_if_empty(d["dir"])
        if a.identity == "both" and n_done0 == total:
            compare_after_run(a, dirs)
        return

    halt = threading.Event()     # 硬停：Ctrl+C 或無法繼續，進行中的對話放棄
    drain = threading.Event()    # 軟停：--minutes 到了，不再派新的，進行中的對話跑完才停

    def check_time():
        if a.minutes and not drain.is_set() and time.time() - t0 >= a.minutes * 60:
            drain.set()
            say(f"  ── 時間到（{a.minutes:g} 分鐘）：不再派新的；手上 {len(pending)} 段跑完就停")

    def check_budget(rec):
        if budget_cap is None:
            return
        st["spent"] += cloud.rec_cost(cfg, a.model, rec)
        if not drain.is_set() and budget_used + st["spent"] >= budget_cap:
            drain.set()
            say(f"  ── 雲端預算到了（約 {budget_used + st['spent']:.2f}／{budget_cap:g} 美元）：不再派新的；手上 {len(pending)} 段跑完就停")
    st = {"ok": 0, "fail": 0, "cut": 0, "next_pct": 5.0, "last_eta": time.time(),
          "recent": deque(maxlen=12), "hard_fail_run": 0, "ctx_fail_at_1": 0, "max_tokens_seen": 0, "cap_errs": 0,
          "by_dir": {ident: 0 for ident in dirs}, "spent": 0.0}
    t0 = time.time()
    clock = PauseClock()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    outs = {ident: open_append(d["path"]) for ident, d in dirs.items()}
    queue = deque(tasks)
    retries = {}
    rng = random.Random(a.seed)
    tune_saved = {"rec": None, "warned": False}

    def save_tuning():
        """把測速進度或定案結果寫進 tuning.json。每完成一段對話就呼叫一次，
        所以按 Ctrl+C、--minutes 到時、關掉視窗或斷電，都只損失還在進行中的那幾段。"""
        if a.mock or fixed:
            return            # 固定並行數不是測出來的，不寫進去，免得之後的自動模式把它當成測速結果
        snap = tuner.snapshot()
        if snap:
            rec = {"probe": snap}
        elif not tuner.probing and st["ok"]:
            rec = {"level": tuner.level, "cap": tuner.cap, "capped": tuner.level >= tuner.cap}
        else:
            return            # 還在暖機、什麼都沒量到：保留原本的紀錄
        rec.update(rule=TUNE_RULE, context=ctx[0] if ctx else None)
        if rec == tune_saved["rec"]:
            return
        try:
            tune_all[tune_key] = dict(rec, time=time.strftime("%Y-%m-%d %H:%M"))
            save_json(tune_path, tune_all)
            tune_saved["rec"] = rec
        except OSError as e:
            if not tune_saved["warned"]:
                tune_saved["warned"] = True
                say(f"  ⚠ 測速紀錄寫不進 {tune_path}（{e}）：回覆照常存檔，下次要重新測速")

    def eta_line():
        n = st["ok"]
        if not st["recent"]:
            return
        el = clock.active(t0)
        # 最近幾段的平均耗時 ÷ 並行數 = 目前每完成一段要等多久
        per = sum(st["recent"]) / len(st["recent"]) / max(tuner.level, 1)
        left = len(queue) + len(pending)
        all_left = total - n_done0 - n
        say(report_block(f"進度報告（{time.strftime('%H:%M')}）", [
            ("進度", f"{(n_done0 + n) / total:.1%}（{n_done0 + n}/{total} 段）"),
            ("已耗時", f"{fmt_dur(el)}（這次執行" + (f"；暫停的 {fmt_dur(clock.paused)} 不算" if clock.paused else "") + "）"),
            ("並行", f"{tuner.level} 段，約每 {fmt_secs(per)} 秒完成一段"),
            *remain_rows(per, left, all_left),
            gpu_row(gpu_status())]))

    def one(task, level_at_start):
        p, q, rep, arm = task
        t1 = time.time()
        sysmsg, users = build_messages(cfg, arm["settings"], p, q, turns, arm["styler"], arm["identity"])
        msgs = [{"role": "system", "content": sysmsg}] if sysmsg else []
        replies, finishes, thinks, gen, tokens, turn_secs = [], [], [], 0, 0, []
        turn_tok, prompt_tok = [], []
        for u in users:
            if halt.is_set():
                return ("abandon", task, None)
            t_turn = time.time()
            msgs.append({"role": "user", "content": u})
            try:
                if a.mock:
                    with lock:
                        r, fin, think = mock.mock_reply(p, rng, arm["style"], qid=q["id"]), "stop", 0
                        usage = {"completion_tokens": len(r), "prompt_tokens": sum(len(m["content"]) for m in msgs)}
                    if a.mock_delay:
                        time.sleep(a.mock_delay)
                else:
                    r, fin, usage, think = chat(a.base_url, a.model, msgs, a.temperature, a.max_tokens,
                                                a.seed * 100003 + rep * 1009 + len(replies))
            except Exception as e:
                return ("error", task, e)
            msgs.append({"role": "assistant", "content": r})
            replies.append(r)
            finishes.append(fin)
            thinks.append(think)
            turn_secs.append(round(time.time() - t_turn, 1))
            gen += usage.get("completion_tokens") or len(r)
            tokens = max(tokens, usage.get("total_tokens") or 0)
            turn_tok.append(usage.get("completion_tokens"))
            prompt_tok.append(usage.get("prompt_tokens"))
        secs = time.time() - t1
        s0 = arm["settings"]
        rec = {"pid": p["id"], "variant": p.get("variant"), "ph": phash(p), "qid": q["id"],
               "qh": expect[(q["id"], arm["style"])], "rep": rep, "style": arm["style"],
               "model": s0["model"], "sysh": sysh, "ver": __version__,
               "pver": s0["personas_version"], "qver": s0["questions_version"],
               "temperature": a.temperature, "max_tokens": a.max_tokens, "identity": arm["identity"],
               "secs": round(secs, 1), "turn_secs": turn_secs, "workers": level_at_start, "tokens": tokens,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "prompts": users, "system": sysmsg, "replies": replies, "finish": finishes, "think_chars": thinks,
               # 跨機器：機器代號、環境代號（完整內容在 run_meta.json 的 envs）、這段實際可用的上下文與用量
               "machine": machine, "env": eid, "ctx": (ctx[0] // max(level_at_start, 1)) if ctx else None,
               "ctx_total": ctx[0] if ctx else None, "turn_tokens": turn_tok, "prompt_tokens": prompt_tok}
        if a.module:
            rec["module"] = a.module
        return ("ok", task, (rec, secs, gen, level_at_start))

    def handle(result):
        kind, task, payload = result
        p, q, rep, arm = task
        rkey = (p["id"], p.get("variant"), q["id"], rep, arm["identity"], arm["style"])
        if kind == "abandon":
            return
        if kind == "error":
            e = payload
            capacity = isinstance(e, ServerError) and e.capacity
            n_try = retries.get(rkey, 0)
            if capacity:
                st["cap_errs"] += 1
                at1 = tuner.level <= 1
                tuner.on_capacity_error()
                if isinstance(e, ServerError) and e.context and at1:
                    st["ctx_fail_at_1"] += 1
                if n_try < 3 and st["ctx_fail_at_1"] < 3:
                    retries[rkey] = n_try + 1
                    queue.append(task)
                    say(f"  [重排] {plabel(p, arm, many)} {q['id']} 第 {rep + 1} 次：{e}（稍後重試）")
                    return
            st["fail"] += 1
            st["hard_fail_run"] += 1
            say(f"  [失敗] {plabel(p, arm, many)} {q['id']} 第 {rep + 1} 次：{e}")
            if st["ctx_fail_at_1"] >= 3:
                say(bullets("✗ 停止：只跑 1 段時上下文長度仍然不夠", [f"Context Length 至少要 {seg_need}", *HOW_CONTEXT]))
                halt.set()
            elif st["hard_fail_run"] >= 3 and st["ok"] == 0:
                say(bullets("✗ 停止：連續失敗，沒有任何一段成功", ["先確認模型回覆正常：" + cmd_hint("check", "--model", a.model)]))
                halt.set()
            return
        rec, secs, gen, lvl = payload
        outs[arm["identity"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
        outs[arm["identity"]].flush()
        st["ok"] += 1
        st["by_dir"][arm["identity"]] += 1
        st["hard_fail_run"] = 0
        spanned = clock.spans_pause(secs)       # 跨過電腦休眠的這段，耗時含睡著的時間：不拿來算速度
        if not spanned:
            st["recent"].append(secs)
        st["max_tokens_seen"] = max(st["max_tokens_seen"], rec["tokens"])
        ncut = sum(f == "length" for f in rec["finish"])
        st["cut"] += ncut
        k = n_done0 + st["ok"]
        say(f"  [{k:>{len(str(total))}}/{total} {k / total:6.1%}] {plabel(p, arm, many)} {q['id']:<4} 第 {rep + 1} 次  "
            f"{secs:5.0f} 秒  字數 {'/'.join(str(len(t)) for t in rec['replies'])}"
            + (f"  ⚠ {ncut} 則被截斷" if ncut else ""))
        if not spanned:
            tuner.on_ok(gen / max(secs, 1e-6), lvl)
        if ctx and rec["tokens"]:      # 實際用量比估計大，就重新算上下文容得下幾段
            real_need = max(int(st["max_tokens_seen"] * 1.15) + 200, min_seg)
            fit_now = max(1, ctx[0] // real_need)
            if real_need > seg_need:
                tuner.set_cap(fit_now, f"對話實際用到 {st['max_tokens_seen']} 個 token，上下文容不下更多段")
            elif st["ok"] >= 3 and not seen_tokens and not a.conv_tokens and st["cap_errs"] == 0:
                tuner.raise_cap(min(max_workers, fit_now), f"對話實際最多用 {st['max_tokens_seen']} 個 token")
        pct = k / total * 100
        now = time.time()
        if pct >= st["next_pct"] or now - st["last_eta"] >= 180 or st["ok"] == 1:
            while st["next_pct"] <= pct:
                st["next_pct"] += 5.0
            st["last_eta"] = now
            eta_line()
        check_time()
        if cloud_name:
            check_budget(rec)

    def record_session():
        """把這次的條件與實測耗時寫進每個設定資料夾的 run_meta.json（給下次預估時間、也給報告用）。電腦休眠的時間不算。"""
        el = clock.active(t0)
        for ident, d in dirs.items():
            n = st["by_dir"][ident]
            if not n:
                continue
            s0 = d["settings"]
            base = {"program_version": __version__, "model": s0["model"],
                    "test_method": "假回覆（--mock）" if a.mock else (f"付費 API（{cloud_name}，{a.base_url}）" if cloud_name
                                                                    else f"本機伺服器，OpenAI 相容 API（{a.base_url}）"),
                    **({"cloud": {"provider": cloud_name, "base_url": a.base_url, "tos_checked": a.cloud.get("tos_checked"),
                                  "compat": client_compat(a.base_url)}} if cloud_name else {}),
                    "generation": {"temperature": a.temperature, "max_tokens": a.max_tokens,
                                   "thinking": s0.get("thinking"), "context_turns": cfg.get("generation", {}).get("context_turns")},
                    "system_prompt": s0["system_prompt"] or "無", "identity": ident, "rounds": turns,
                    "versions": {"plan": s0["plan_version"], "questions": s0["questions_version"],
                                 "personas": s0["personas_version"],
                                 # 用舊版人物接著跑時，人物句版本沿用這組資料原本記的
                                 "persona_sentences": (layout.read_meta(d["dir"]).get("versions") or {}).get(
                                     "persona_sentences") if old_ver and old_ver != cfg["version"].get("personas")
                                 else cfg.get("version", {}).get("persona_sentences")},
                    "module": a.module or None, "styles": styles,
                    "planned": {"people": n_people, "reps": a.reps, "variants": variants,
                                "questions": [q["id"] for q in questions]},
                    "context": {"configured": want_ctx, "min_per_segment": min_seg}}
            if env:
                envs = layout.read_meta(d["dir"]).get("envs", {})
                envs[eid] = env
                base["envs"] = envs
            layout.add_session(d["dir"], base, {
                "machine": machine, "start": started, "end": time.strftime("%Y-%m-%d %H:%M:%S"),
                # 同時跑兩組時，耗時依段數分攤
                "segments": n, "secs": round(el * n / max(st["ok"], 1), 1), "failed": st["fail"],
                "workers_final": tuner.level, "reps": a.reps, "people": n_people, "draft": bool(todo_hits),
                "rep_range": [rep_list[0] + 1, rep_list[-1] + 1], "rep_range_given": bool(getattr(a, "rep_range", "")), "env": eid,
                "loaded_context": ctx[0] if ctx else None})

    interrupted = False
    pending = {}
    ex = ThreadPoolExecutor(max_workers=max_workers)
    try:
        while True:
            gap = clock.tick()
            if gap:
                st["recent"].clear()
                tuner.on_pause()
                say(f"  ◆ 程式停了 {fmt_dur(gap)}（電腦休眠或暫停）：這段不算進速度與預估時間")
            # 並行數往下調時，等多出來的舊對話跑完再派新的，量到的速度才乾淨；
            # 往上調不必等：新派的對話一開始就處在新的並行數下，最先完成的幾段另有暖機丟掉
            mixed = any(l > tuner.level for l in pending.values())
            while queue and not halt.is_set() and not drain.is_set() and len(pending) < tuner.level \
                    and not (tuner.probing and mixed):
                t = queue.popleft()
                pending[ex.submit(one, t, tuner.level)] = tuner.level
            if not pending:
                if not queue or halt.is_set() or drain.is_set():
                    break
                time.sleep(1.0)       # 等並行數切換
                check_time()
                continue
            finished, _ = wait(list(pending), timeout=1.0, return_when=FIRST_COMPLETED)
            for fu in finished:
                pending.pop(fu)
                handle(fu.result())
            if finished:
                save_tuning()
            check_time()
            if time.time() - st["last_eta"] >= 180 and st["ok"]:
                st["last_eta"] = time.time()
                eta_line()
    except KeyboardInterrupt:
        interrupted = True
        halt.set()
        say("\n── 收到中斷：已存的都保留；進行中的幾段放棄，下次重跑")
    finally:
        ex.shutdown(wait=not interrupted, cancel_futures=True)
        with lock:
            for f in outs.values():
                f.close()
        record_session()
        awake.__exit__(None, None, None)
        for d in dirs.values():
            layout.drop_if_empty(d["dir"])
    save_tuning()
    if tuner.probing and not fixed:
        msg = "、".join(f"{k} 段每秒 {v:.0f} token" for k, v in sorted(tuner.measured.items()))
        print("\n" + bullets("── 測速還沒做完", [(f"{tuner.level} 段", f"已收 {len(tuner.samples)}/{tuner.needed()} 個樣本"),
                                                ("已量完", msg or None),
                                                None if a.mock else "進度已存：下次執行同一行會接著測" if tuner.snapshot()
                                                else "停在暖機階段：沒有量到可以存的結果"]))
    k = n_done0 + st["ok"]
    el = clock.active(t0)
    per = sum(st["recent"]) / len(st["recent"]) / max(tuner.level, 1) if st["recent"] else None
    print("\n" + kv_block("這次結果", [
        ("這次完成", f"{st['ok']} 段" + (f"（失敗 {st['fail']} 段）" if st["fail"] else "")),
        ("用時", fmt_dur(el)),
        ("暫停", f"{fmt_dur(clock.paused)}（電腦休眠；不算進用時）" if clock.paused else None),
        ("最後的並行數", f"{tuner.level} 段"),
        ("累計", f"{k}/{total} 段（{k / total:.1%}）"),
        ("剩下", (f"{total - k} 段" + (f"，照目前的速度約 {fmt_dur(per * (total - k))}" if per else "")) if k < total else None),
        ("截斷", f"{st['cut']} 則回覆碰到每則上限（全部跑完時自動續寫）" if st["cut"] else None),
        ("回覆檔", [d["path"] for d in dirs.values()]),
        ("雲端用量（這組資料累計）", cloud.usage_line(cfg, a.model, [r for d in dirs.values() for r in read_jsonl(d["path"])])[0]
         if cloud_name and os.path.exists(next(iter(dirs.values()))["path"]) else None)]))
    if k == total and not interrupted:
        auto_continue(a, cfg, dirs, tuner.level)
        auto_probe(a, cfg, dirs, tuner.level)
    print(run_next_steps(a, dirs, k == total, st["cut"]))
    if interrupted:
        os._exit(0)
    if a.identity == "both" and k == total:
        compare_after_run(a, dirs)


def auto_continue(a, cfg, dirs, workers):
    """run 跑完時，受測模型還載入：還有沒續寫的截斷回覆，就順便續寫到每則上限加 10%
    （和不加 --to 的 continue 相同；續寫完才能評審）。--no-continue 可關。"""
    if getattr(a, "no_continue", False) or getattr(a, "preview", False) or getattr(a, "provider", None):
        return
    from .maintain import cmd_continue
    for d in dirs.values():
        if not os.path.isdir(d["dir"]):
            continue
        pend = layout.pending_continuation(d["dir"])
        if not pend:
            continue
        to = layout.continue_target(d["dir"])
        print("\n" + bullets("── 續寫被截斷的回覆（受測模型還載入，順便做）", [
            ("段數", f"{len(pend)} 段"), ("總上限", f"{to}（每則上限加 10%）"), ("不要", "run 加 --no-continue")]))
        ns = argparse.Namespace(out=a.out, config=a.config, run=os.path.basename(d["dir"]), to=to, base_url=a.base_url,
                                model=a.model, workers=max(1, min(int(workers), 4)), check=False,
                                machine=getattr(a, "machine", ""), mock=a.mock)
        try:
            cmd_continue(ns)
        except SystemExit as e:
            warn("續寫沒有做", str(e.code), "之後單獨執行：" + cmd_hint("continue", f"--run {os.path.basename(d['dir'])}"))


def auto_probe(a, cfg, dirs, workers):
    """run 跑完時，受測模型還載入：順便做模型操弄檢核的提問（只做主流程；對照模組、配對確認不做）。"""
    if getattr(a, "no_probe", False) or getattr(a, "module", None) or getattr(a, "preview", False):
        return
    from .personas import personas_for_run
    from . import probe
    for d in dirs.values():
        if not os.path.isdir(d["dir"]):
            continue
        plist = personas_for_run(a.out, cfg, d["dir"])[0]
        if not plist or any(p.get("confirm") for p in plist):
            continue
        ns = argparse.Namespace(out=a.out, config=a.config, mock=a.mock, model=a.model, base_url=a.base_url,
                                variant="implicit", reps=a.reps, limit=0, workers=max(1, int(workers)))
        try:
            probe.ask(ns, cfg, d["dir"], layout.read_settings(d["dir"]), auto=True)
        except SystemExit as e:
            warn("模型操弄檢核沒有做", str(e.code), "之後單獨執行：" + cmd_hint("probe", f"--run {os.path.basename(d['dir'])}"))


def machine_switch(a, dirs, env, good, machine, total, rep_list, strict, notes, new_questions=None):
    """這組資料先前在別台機器跑過時（例如 3090 跑一半、換到 3080 Ti 接著跑）：比對模型檔與量化方式，
    列出各次重複由哪台跑、這台會接著跑哪些。模型檔不同時正式執行停下（--allow-env-change 可略過，報告會寫明）。"""
    prev = {}
    for d in dirs.values():
        prev.update(layout.read_meta(d["dir"]).get("envs") or {})
    # 機器＝機器代號＋顯示卡型號：同一台電腦換了顯示卡（例如 3090 換成 3080 Ti）也算換機器
    here_gpu = envinfo.gpu_model(env.get("gpu"))
    here = f"{machine}（{envinfo.short_gpu(here_gpu)}）" if here_gpu else machine
    others = {}
    label = lambda r: (lambda g: f"{r.get('machine')}（{envinfo.short_gpu(g)}）" if g else r.get("machine"))(
        envinfo.gpu_model((prev.get(r.get("env")) or {}).get("gpu")))
    for r in good:
        g = envinfo.gpu_model((prev.get(r.get("env")) or {}).get("gpu"))
        if (r.get("machine") and r["machine"] != machine) or (g and here_gpu and g != here_gpu):
            others[label(r)] = others.get(label(r), 0) + 1
    if not others:
        return None
    probs, extra = envinfo.env_conflicts(prev, env)
    # 載入設定（Flash Attention、推測解碼、KV 快取量化、Unified KV）：和先前各機器自動記下的、以及人工比對的紀錄比
    meta0 = {}
    for d in dirs.values():
        meta0.update(layout.read_meta(d["dir"]).get("load_manual") or {})
    prev_loads = [(e.get("machine") or "未記錄", (e.get("load") or {}).get("content")) for e in prev.values() if e.get("load")]
    prev_loads += [(m, {k: {"value": envinfo._norm(k, v)} for k, v in rec.items() if k in envinfo.LOAD_CONTENT})
                   for m, rec in meta0.items()]
    load_probs = envinfo.load_conflicts(prev_loads, env.get("load"))
    probs += load_probs
    if probs and strict and not getattr(a, "allow_env_change", False):
        stop("換機器接續：這台的環境和先前不同", *probs, "各機器要用同一個模型檔與量化方式（規格「跨機器分工」）",
             "做法：在 LM Studio 載入同一個 GGUF 檔", "確定要照樣跑：加 --allow-env-change（報告會寫明）")
    notes += [f"（--allow-env-change）{p}" for p in probs] + extra
    from .maintain import rep_machine_table
    t = rep_machine_table(good, prev)
    per = total // max(len(rep_list), 1)
    rows = [("先前的機器", "、".join(f"{m}：{n} 段" for m, n in sorted(others.items()))), ("這台", here)]
    reps = []
    for rep in sorted(t):
        n = sum(t[rep].values())
        reps.append(f"第 {rep} 次 {n}/{per} 段（" + "、".join(f"{m} {k}" for m, k in sorted(t[rep].items())) + "）")
    rows.append(("各次重複", reps))
    partial = [rep for rep in t if 0 < sum(t[rep].values()) < per]
    # 只差整題沒跑的（例如換題後要補跑的新版 I1）分開說：那幾題整題在這台跑，機器只和題目綁在一起，不和群體綁在一起
    by_rep = {}
    for r in good:
        by_rep.setdefault(r["rep"] + 1, {}).setdefault(r["qid"], 0)
        by_rep[r["rep"] + 1][r["qid"]] += 1
    all_q = set().union(*by_rep.values()) if by_rep else set()
    per_q = per // max(len(all_q | set(new_questions or [])), 1) if per else 0
    whole = {}
    for rep in partial:
        miss = sorted(set(new_questions or []) - set(by_rep.get(rep, {})))
        if miss and sum(t[rep].values()) + len(miss) * per_q >= per:
            whole[rep] = miss
    if whole:
        rows.append(("換題補跑", ["第 " + "、".join(map(str, whole)) + " 次重複只差新版的 "
                                + "、".join(sorted({q for v in whole.values() for q in v})) + "：整題在這台跑",
                                "機器只和這題綁在一起，不和群體綁在一起"]))
        partial = [rep for rep in partial if rep not in whole]
    if partial:
        rows.append(("注意", [f"第 {'、'.join(map(str, partial))} 次重複只跑了一部分：剩下的在這台跑",
                             "同一次重複內順序打亂，機器不會和群體綁在一起",
                             "規格建議在一次重複跑完時換機器；報告「各機器」一節會列出差異"]))
    rows.append(("模型檔", "和先前相同" if not probs and env.get("model_sha256") not in envinfo.SKIP_SHA else
                 ("；".join(probs) if probs else "無法比對（見提醒）")))
    ctxs = sorted(r["ctx"] for r in good if r.get("ctx"))
    if load_probs:
        load_row = list(load_probs)
    elif not prev_loads:
        load_row = "先前沒有記錄（v0.9.6 以前不讀）"
    elif env.get("load"):
        load_row = "和先前相同"
    else:
        load_row = "這台讀不到，無法比對（設定檔 generation.lmstudio_load 可聲明）"
    rows.append(("載入設定", load_row))
    rows.append(("先前每段上下文", f"約 {ctxs[len(ctxs) // 2]:,} token" if ctxs else None))
    return rows


def run_next_steps(a, dirs, complete, cut):
    """run 結束時的下一步：沒跑完就接著跑；跑完就換評審，順便校準每則上限。"""
    if not complete:
        return next_steps(["接著跑：再執行同一行（已存的會略過）", "看進度：" + cmd_hint("status")])
    items = []
    cfg = read_config(a.config)
    live = [d["dir"] for d in dirs.values() if os.path.isdir(d["dir"])]
    for d in live:
        if layout.pending_continuation(d):
            items.append(f"先續寫到 {layout.continue_target(d)}（評審前要做完）：" + cmd_hint("continue", layout.run_arg(a.out, d)))
    left = []
    for d in live:
        recs = read_jsonl(os.path.join(d, layout.RESPONSES))
        _, here, ok_by, _ = judged_by_machine(a, cfg, d, recs)
        if ok_by.get(here, 0) < sum(len(r.get("replies") or []) for r in recs):
            left.append(d)
    if left:
        judge_model = (cfg.get("judge") or {}).get("model")
        items.append("第 3 步 評審：LM Studio 卸下受測模型、載入評審模型" + (f" {judge_model}" if judge_model else "") + "，再執行")
        items += [cmd_hint("judge", "--mock" if a.mock else "", layout.run_arg(a.out, d)) for d in left]
    else:
        items += ["第 4 步 分析（評審已經評完）：" + cmd_hint("analyze", "--mock" if a.mock else "", layout.run_arg(a.out, d)) for d in live]
    return next_steps(items)


def cmd_status(a):
    """不執行，只顯示目前進度。每組設定（模型、身分位置、對照模組……）各列一段。
    run_meta.json 記有計畫的人數與重複次數時，以它為準（例如身分位置小批只跑前 24 位）。"""
    layout.tidy_backups(a.out)
    cfg = read_config(a.config)
    from .personas import personas_for_run
    cur, pdir = load_personas(a.out, cfg)
    dirs = layout.run_dirs(a.out)
    print(kv_block("版本", version_rows(cfg, a.config, None, pdir if cur else None)))
    for x in reserved_notes(cfg):
        print(f"⚠ {x}")
    if cur:
        print(f"人物：{len({p['id'] for p in cur})} 位 × 兩版（{os.path.basename(pdir)}）")
    else:
        print(bullets(f"⚠ 目前設定檔版本的人物還沒產生（{os.path.basename(pdir)}）", ["各組資料用自己當初的人物版本顯示"]))
    if not dirs:
        print("還沒有任何回覆。")
        return
    for d in dirs:
        s = layout.read_settings(d)
        personas = personas_for_run(a.out, cfg, d)[0] or cur
        if not personas:
            print(f"\n■ {os.path.basename(d)}：找不到這組資料用的人物（略過）")
            continue
        meta = layout.read_meta(d)
        planned = meta.get("planned", {})
        module = s.get("module")
        questions = module_questions(cfg, module)
        styles = MODULES[module] if module else ["orig"]
        stylers = stylers_for(cfg, styles, a.out, s)
        reps = a.reps or planned.get("reps") or 4
        n_ids, n_plan = len({p["id"] for p in personas}), planned.get("people") or 0
        people = [p for p in select_people(personas, n_plan if 0 < n_plan < n_ids else 0)
                  if p.get("variant") in planned.get("variants", ["explicit", "implicit"])]
        keys = {(p["id"], p.get("variant")) for p in people}
        qcat = {q["id"]: q.get("type", q.get("category")) for q in questions}
        ncat = len(set(qcat.values()))
        total = len(people) * len(questions) * reps * len(styles)
        good, empty, stale = load_valid(os.path.join(d, layout.RESPONSES), personas, s["turns"], questions,
                                        expected_qh(questions, s["turns"], stylers))
        done = {done_key(r) for r in good if r["rep"] < reps and r["qid"] in qcat and done_key(r)[:2] in keys
                and r.get("style", "orig") in styles}
        rows = [("設定", layout.describe(s))]
        from .personas import persona_info
        rows += persona_info(cfg, personas_for_run(a.out, cfg, d)[1], personas)
        pend = layout.pending_continuation(d)
        if pend:
            rows.append(("待續寫", [f"{len(pend)} 段（續寫到 {layout.continue_target(d, s)}＝每則上限加 10%；評審前要做完）",
                                   cmd_hint("continue", f"--run {os.path.basename(d)}")]))
        nv = len({p.get("variant") for p in people})
        rows.append(("計畫", f"{len({p['id'] for p in people})} 位人物 × "
                     f"{'兩版' if nv == 2 else '揭露版' if people and people[0].get('variant') == 'explicit' else '隱性版'}"
                     f" × {len(questions)} 題 × 重複 {reps} 次"
                     + (f" × {len(styles)} 種寫法" if len(styles) > 1 else "") + f" = {total} 段"))
        rows.append(("已完成", f"{len(done)}/{total} 段（{len(done) / max(total, 1):.1%}）"))
        rows.append(("各次重複", [f"第 {rep + 1} 次：{sum(1 for x in done if x[3] == rep)}/{total // reps}"
                                  for rep in range(reps)]))
        ready = sum(1 for k in keys if len({qcat[x[2]] for x in done if x[:2] == k}) == ncat)
        rows.append(("可以分析", f"{ready}/{len(keys)} 個人物版本（每種題型至少一段）"
                     + ("→ 可以執行 analyze" if ready == len(keys) else "")))
        if empty or stale:
            rows.append(("待清理", [f"含空回覆 {empty} 段；人物或題目和設定檔對不上 {stale} 段", "下次 run 自動清理並補跑"]))
        secs = [r["secs"] / max(r.get("workers") or 1, 1) for r in good[-30:] if "secs" in r]
        if secs and len(done) < total:
            per = sum(secs) / len(secs)
            rows.append(("速度", f"最近約每 {fmt_secs(per)} 秒完成一段"))
            rows.append(("剩餘", f"約 {fmt_dur(per * (total - len(done)))}"))
        n_rep = sum(len(r.get("finish", [])) for r in good)
        cut = sum(f == "length" for r in good for f in r.get("finish", []))
        if cut:
            rows.append(("截斷", f"{cut} 則回覆碰到每則上限被截斷（{cut / max(n_rep, 1):.1%}）"))
        machines = {r.get("machine") for r in good} - {None}
        if machines:
            from .maintain import rep_machine_table
            t = rep_machine_table(good, meta.get("envs"))
            rows.append(("機器", [f"第 {rep} 次：" + "、".join(f"{m} {n} 段" for m, n in sorted(t[rep].items()))
                                  for rep in sorted(t)]))
        from .judge import jkey, load_judgments, prompt_path, valid_judgment
        pname = prompt_path(cfg, a.config)[0]
        js, here, ok_by, err = judged_by_machine(a, cfg, d, good)
        ok = sum(ok_by.values())
        if js:
            rows.append(("評審", [f"已評 {ok}/{n_rep} 則" + (f"，judge_error {err} 則" if err else "")]
                         + ([f"{m}：{n} 則" + ("（這台）" if m == here else "") for m, n in sorted(ok_by.items())]
                            if len(ok_by) > 1 or (ok_by and here not in ok_by) else [])))
        sec, _ = load_judgments(d, pname, "second")
        if sec:
            n_sec = sum(1 for r in good for t, x in enumerate(r["replies"]) if valid_judgment(sec.get(jkey(r, t)), x))
            rows.append(("第二評審", f"已評 {n_sec} 則（另存，分析只用主評審）"))
        print(f"\n■ {os.path.basename(d)}")
        print(kv_block("", rows))
        print(status_next(a, d, s, len(done) < total, pend, ok_by.get(here, 0), n_rep))
        sw = layout.settings_warning(d, read_jsonl(os.path.join(d, layout.RESPONSES)))
        if sw:
            print(sw)


def judged_by_machine(a, cfg, d, recs):
    """這組回覆評過的則數，依評審的機器分。回傳 (評審紀錄, 這台的名稱, {機器: 則數}, judge_error 則數)。
    規格要求評審全部在同一台，所以「評完」以這台評的為準。"""
    from .envinfo import current_gpu, machine_code
    from .judge import jkey, load_judgments, prompt_path, valid_judgment
    js, _ = load_judgments(d, prompt_path(cfg, a.config)[0])
    here = layout.machine_label({"machine": machine_code(a), "gpu": None if getattr(a, "mock", False) else current_gpu()})
    ok_by, err = {}, 0
    for r in recs:
        for t, x in enumerate(r.get("replies") or []):
            j = js.get(jkey(r, t))
            if valid_judgment(j, x):
                lab = layout.machine_label(j)
                ok_by[lab] = ok_by.get(lab, 0) + 1
            elif j and "judge_error" in j:
                err += 1
    return js, here, ok_by, err


def status_next(a, d, s, unfinished, pend, judged_here, n_rep):
    """status 每組設定最後的「下一步」：照四大步驟，從還沒做完的那一步開始。"""
    run = layout.run_arg(a.out, d)
    model = s.get("model") or "<受測模型>"
    items = []
    if unfinished:
        items.append("第 2 步 生成回覆（還沒跑完）：" + cmd_hint("run", f"--model {model}" if model != "mock" else "--mock", run))
        items.append("剩下的重複由別台跑時：跑完後在這台合併 " + cmd_hint("merge", "<別台的 out 資料夾>"))
    if pend:
        items.append(f"續寫 {len(pend)} 段（評審前要做完）：" + cmd_hint("continue", run))
    if judged_here < n_rep:
        items.append(("第 3 步 評審" + ("（生成還沒跑完也可以先評）" if unfinished else "") + "："
                      + cmd_hint("judge", "--mock" if model == "mock" else "", run)))
    if not unfinished and not pend and judged_here >= n_rep:
        done_dir = os.path.join(a.out, "analysis", os.path.basename(d))
        items.append("第 4 步 分析" + ("（已分析過，報告在 " + done_dir + "；資料有變動再執行一次）" if os.path.isdir(done_dir) else "")
                     + "：" + cmd_hint("analyze", "--mock" if model == "mock" else "", run))
        if os.path.isdir(done_dir):
            items.append("之後（進階）：配對確認 confirm（analyze 結束時列出要確認的屬性）、人工標註 annotate")
    return next_steps(items)


def cmd_check(a):
    """送出一個問題並印出伺服器的原始回應，用來確認連線與回覆格式。"""
    from .common import check_origin
    check_origin(read_config(a.config), a.model, what="模型")
    ctx = loaded_context(a.base_url, a.model)
    if ctx:
        ctx_row = f"{ctx[0]} token（LM Studio 的 Context Length；這個模型最多 {ctx[1]}）"
    else:
        ctx_row = "問不到（不是 LM Studio，或模型名稱不符）"
    from .client import load_config
    cfg = read_config(a.config)
    print(kv_block("模型", [("上下文長度", ctx_row), ("每則上限", f"{a.max_tokens} token（max_tokens，含思考）"),
                            ("載入設定", envinfo.load_lines(envinfo.load_settings(
                                load_config(a.base_url, a.model), (cfg.get("generation") or {}).get("lmstudio_load"))))]))
    try:
        d = chat_raw(a.base_url, a.model, [{"role": "user", "content": "請用一句話自我介紹。"}],
                     a.temperature, a.max_tokens, a.seed)
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        if isinstance(e, urllib.error.HTTPError):
            raise
        stop(f"連不上 {a.base_url}（{getattr(e, 'reason', e)}）", "LM Studio 的 Developer 分頁：開啟 Server",
             "確認 --base-url 的位址與 port")
    print(json.dumps(d, ensure_ascii=False, indent=2)[:3000])
    ch = d["choices"][0]
    msg = ch.get("message", {})
    text = (msg.get("content") or "").strip()
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    print("\n" + kv_block("判讀", [("回覆", f"{len(text)} 字"), ("思考", f"{len(think)} 字"),
                                    ("結束原因", ch.get("finish_reason"))]))
    if not text:
        warn("回覆是空的：思考用掉了整個每則上限（正式執行前要解決）", "加大每則上限，或關閉模型的思考功能", *how_max_tokens(a.config))
    elif ch.get("finish_reason") == "length":
        warn("回覆碰到每則上限被截斷", "建議加大每則上限", *how_max_tokens(a.config))
    else:
        print("✓ 看起來正常")
