"""跨機器與截斷的維護指令：merge（合併另一台機器的回覆）、calibrate（校準每則上限與上下文）、
continue（續寫被截斷的回覆）。

- merge：跨機器只能依重複序號分工（--rep-range）。合併時依段的代號（身分位置、人物、版本、題號、重複序號、寫法）
  去重，並檢查兩邊的設定與模型檔雜湊值一致。
- calibrate：用試跑資料建議每則上限（未截斷回覆 token 數第 99 百分位的 1.2 倍）並推算所需上下文；
  只印出建議，作者確認後自己寫進設定檔。
- continue：不得只重跑被截斷的對話。對所有被截斷的回覆，從中斷處用同一模型與設定接著生成，
  總上限一律提高到同一個新值；第 1 輪被續寫的段，第 2 輪用完整的第 1 輪重新生成。續寫與重生成都會標記。"""
import json
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from . import envinfo, layout
from .client import chat_raw
from .common import (read_config, HOW_CONTEXT, how_max_tokens, Progress, ask, bullets, cmd_hint, done_key, fmt_dur, kv_block, load_json,
                     note_block, read_jsonl, save_json, stop)

SETTINGS_IGNORE = ("created", "version", "max_tokens_history", "questions_version_history")


# ---------- merge ----------

def seg_key(r):
    return (r.get("identity") or "message", *done_key(r))


def cmd_merge(a):
    src = os.path.abspath(a.source)
    if os.path.isfile(os.path.join(src, layout.SETTINGS)):
        src_dirs = [src]
    else:
        src_dirs = [d for d in layout.run_dirs(src) if os.path.isfile(os.path.join(d, layout.RESPONSES))]
    if a.run:
        src_dirs = [d for d in src_dirs if a.run in os.path.basename(d)]
    if not src_dirs:
        stop(f"在 {src} 找不到要合併的回覆", "要指向另一台機器的 out 資料夾，或其中一組設定的資料夾")
    if os.path.abspath(a.out) == src or os.path.dirname(src_dirs[0]) == os.path.abspath(a.out):
        stop("來源和目前的 out 是同一個資料夾：不用合併")
    for sd in src_dirs:
        merge_one(a, sd, os.path.join(a.out, os.path.basename(sd)))


def _settings_core(s):
    return {k: v for k, v in s.items() if k not in SETTINGS_IGNORE}


def merge_one(a, sd, dd):
    name = os.path.basename(sd)
    print(f"\n■ {name}")
    ss = layout.read_settings(sd)
    if os.path.isfile(os.path.join(dd, layout.SETTINGS)):
        ds = layout.read_settings(dd)
        if _settings_core(ss) != _settings_core(ds):
            diff = sorted(k for k in set(_settings_core(ss)) | set(_settings_core(ds))
                          if _settings_core(ss).get(k) != _settings_core(ds).get(k))
            print(f"  ✗ 不合併：兩邊的設定不同（{'、'.join(diff)}）")
            return
    else:
        os.makedirs(dd, exist_ok=True)
        shutil.copy2(os.path.join(sd, layout.SETTINGS), os.path.join(dd, layout.SETTINGS))
    local = read_jsonl(os.path.join(dd, layout.RESPONSES))
    incoming = read_jsonl(os.path.join(sd, layout.RESPONSES))
    smeta, dmeta = layout.read_meta(sd), layout.read_meta(dd)
    envs = dict(dmeta.get("envs", {}))
    for k, v in smeta.get("envs", {}).items():
        envs.setdefault(k, v)

    # 1. 生成設定與模型檔要一致
    # 每則上限調高過的資料夾，同一組設定裡本來就有不同上限的段，所以上限改比 settings.json（見上面）
    gens = {(r.get("temperature"), r.get("sysh", ""), r.get("model")) for r in local + incoming}
    if len(gens) > 1:
        print(f"  ✗ 不合併：兩邊的生成設定不同（溫度、系統提示、模型）：{sorted(map(str, gens))}")
        return
    shas = {envs.get(r.get("env"), {}).get("model_sha256", envinfo.UNKNOWN) for r in local + incoming}
    known = shas - {envinfo.UNKNOWN, "mock"}
    if len(known) > 1 and not a.force:
        print(bullets("  ✗ 不合併：兩邊的模型檔雜湊值不同（不是同一個模型檔）", ["、".join(x[:12] for x in sorted(known)),
                                                                     "確定要合併：加 --force（報告會寫明）"], indent=4))
        return
    if envinfo.UNKNOWN in shas:
        n_unknown = sum(1 for r in local + incoming if envs.get(r.get("env"), {}).get("model_sha256", envinfo.UNKNOWN)
                        == envinfo.UNKNOWN)
        print(f"  ⚠ {n_unknown} 段沒有模型檔雜湊值（舊版程式或找不到模型檔）：無法確認是同一個模型檔")
    formats = {envs.get(r.get("env"), {}).get("format") for r in local + incoming} - {None, envinfo.UNKNOWN, "mock"}
    if len(formats) > 1:
        print(f"  ✗ 不合併：兩邊的模型格式不同（{'、'.join(sorted(formats))}；各機器要用同一個 GGUF 檔）")
        return

    # 2. 依段的代號去重；兩邊回覆不同時，先排除不是真衝突的，真衝突整次重複由一邊覆蓋（choose_reps）
    plan = merge_plan(local, incoming, planned_per_rep(dmeta, smeta))
    winners = choose_reps(a, plan)
    if winners is None:
        print("  ✗ 不合併（沒有改任何檔案）")
        return
    rows, replaced = apply_plan(local, incoming, plan, winners)
    path = os.path.join(dd, layout.RESPONSES)
    if rows != local:
        if os.path.isfile(path):
            shutil.copy2(path, layout.backup_path(path))
        if replaced:
            with open(layout.backup_path(path, "merge-replaced"), "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in replaced))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        os.replace(tmp, path)
    new = [r for r in rows if id(r) in {id(x) for x in incoming}]
    same, conflict = plan["same"], sum(len(v) for v in plan["conflicts"].values())
    meta = dict(dmeta)
    for k, v in smeta.items():
        meta.setdefault(k, v)
    meta["envs"] = envs
    manual = dict(smeta.get("load_manual") or {})
    manual.update(dmeta.get("load_manual") or {})
    if manual:
        meta["load_manual"] = manual
    meta["sessions"] = dmeta.get("sessions", []) + [dict(x, merged_from=name) for x in smeta.get("sessions", [])
                                                      if x not in dmeta.get("sessions", [])]
    meta.setdefault("merges", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": os.path.basename(
        os.path.dirname(sd)) + "/" + name, "added": len(new), "duplicates": same, "conflicts": conflict,
        "continued_taken": plan["updated"], "old_versions": plan["retired"],
        "reps": {str(r): w for r, w in winners.items()}, "replaced_local": len(replaced)})
    save_json(os.path.join(dd, layout.META), meta)
    print(kv_block("", [("新增", f"{len(new)} 段"), ("重複", f"{same} 段（內容相同）"),
                       ("取續寫過的", f"{plan['updated']} 段（同一段，一邊後來續寫過）" if plan["updated"] else None),
                       ("衝突", f"{conflict} 段（兩邊都跑了同一段、回覆不同）" if conflict else None),
                       ("整次覆蓋", [f"第 {r} 次：用{'來源' if w == 'source' else '這邊'}的"
                                    + (f"（這邊 {plan['count'][r][0]} 段移到備份）" if w == "source" else
                                       f"（來源 {plan['count'][r][1]} 段不採用）") for r, w in sorted(winners.items())] or None)]))
    if os.path.isfile(os.path.join(sd, "judgments.jsonl")):
        print("  ⚠ 來源資料夾的評審結果沒有合併（評審要全部在同一台機器、同一個評審模型檔執行）")
    print_rep_machine(rows)


def ckey(r):
    """合併比對用的鍵：段的代號加題目指紋（換題前的舊版回覆和新版不算同一段）。"""
    return (*seg_key(r), r.get("qh"))


def _depth(r):
    """同一段的處理程度：續寫過的總上限越高、續寫時間越晚，越完整。"""
    return int(r.get("max_tokens_total") or r.get("max_tokens") or 0), r.get("continue_time") or ""


def planned_per_rep(*metas):
    """每次重複計畫跑幾段（run_meta.json 的 planned：人數 × 版本 × 題數 × 寫法）；沒記就回傳 None。"""
    for m in metas:
        p = m.get("planned") or {}
        if p.get("people") and p.get("variants") and p.get("questions"):
            return int(p["people"]) * len(p["variants"]) * len(p["questions"]) * len(m.get("styles") or ["orig"])
    return None


def merge_plan(local, incoming, planned):
    """比對兩邊的回覆。回覆不同時，先排除不是真衝突的：
    - 同一段（同一台機器、同一個生成時間），一邊後來續寫過 → 取續寫過的（updated）。
    - 換題前的舊版題目（題目指紋不是最新的）→ 留這邊的（retired）。
    剩下的才是真衝突（兩台機器各跑了同一段），依重複序號分組，交給 choose_reps 整次決定用哪一邊。"""
    latest = {}
    for r in local + incoming:
        k = (r["qid"], r.get("style", "orig"))
        if r.get("time", "") >= latest.get(k, ("", None))[0]:
            latest[k] = (r.get("time", ""), r.get("qh"))
    current = lambda r: r.get("qh") == latest[(r["qid"], r.get("style", "orig"))][1]
    have = {ckey(r): r for r in local}
    plan = {"new": [], "updates": {}, "same": 0, "updated": 0, "retired": 0, "conflicts": {}, "count": {},
            "machines": {}, "planned": planned}
    for r in incoming:
        k = ckey(r)
        l = have.get(k)
        if l is None:
            plan["new"].append(r)
        elif l.get("replies") == r.get("replies") and (l.get("machine") == r.get("machine")
                                                       or not (l.get("machine") and r.get("machine"))):
            plan["same"] += 1                    # 內容相同、同一台（或沒記機器）：真的重複
        elif l.get("replies") == r.get("replies") and current(r):
            plan["conflicts"].setdefault(r["rep"] + 1, []).append(k)   # 內容相同但兩台各跑一次：仍要決定這次重複用哪台
        elif l.get("time") and l.get("time") == r.get("time") and l.get("machine") == r.get("machine"):
            if _depth(r) > _depth(l):
                plan["updates"][k] = r
                plan["updated"] += 1
        elif not current(r):
            plan["retired"] += 1
        else:
            plan["conflicts"].setdefault(r["rep"] + 1, []).append(k)
    for side, recs in ((0, local), (1, incoming)):
        for r in recs:
            if current(r):
                rep = r["rep"] + 1
                plan["count"].setdefault(rep, [0, 0])[side] += 1
                plan["machines"].setdefault(rep, [set(), set()])[side].add(r.get("machine") or "未記錄")
    return plan


def choose_reps(a, plan):
    """有真衝突的每一次重複：整次用比較完整的一邊（跑滿計畫段數的那邊）。
    兩邊都跑滿或都不完整時讓作者選（--prefer here／source 可以先指定）。回傳 {重複: "here"／"source"}，取消時 None。"""
    out = {}
    full = plan["planned"]
    for rep in sorted(plan["conflicts"]):
        lc, sc = plan["count"][rep]
        lm, sm = ("、".join(sorted(x)) for x in plan["machines"][rep])
        done_l, done_s = bool(full) and lc >= full, bool(full) and sc >= full
        if done_l != done_s:
            out[rep] = "here" if done_l else "source"
            continue
        suggest = "source" if sc > lc else "here"
        print(bullets(f"  ⚠ 第 {rep} 次重複兩邊都跑了，而且{'都跑滿' if done_l else '都不完整'}：整次要用哪一邊？", [
            ("1 這邊", f"{lc}{'/' + str(full) if full else ''} 段（{lm}）"),
            ("2 來源", f"{sc}{'/' + str(full) if full else ''} 段（{sm}）"),
            ("回覆不同", f"{len(plan['conflicts'][rep])} 段"),
            ("建議", f"{'2 來源' if suggest == 'source' else '1 這邊'}（段數多的；另一邊的這次重複全部不用，缺的段之後補跑）")], indent=4))
        pick = getattr(a, "prefer", "") or ""
        if not pick:
            if not sys.stdin or not sys.stdin.isatty():
                print("    ✗ 沒辦法問（不是在終端機執行）：加 --prefer here 或 --prefer source 再執行一次")
                return None
            ans = ask("    選 1 或 2（直接 Enter＝建議；q＝取消，不合併）：")
            if ans is None:
                print("    ✗ 讀不到輸入：加 --prefer here 或 --prefer source 再執行一次")
                return None
            if ans == "q":
                return None
            pick = {"1": "here", "2": "source", "": suggest}.get(ans)
            if pick is None:
                print("    看不懂的選項：不合併")
                return None
        out[rep] = pick
    return out


def apply_plan(local, incoming, plan, winners):
    """照計畫合併。回傳 (合併後的回覆, 被整次覆蓋而移出的這邊回覆)。"""
    take_src = {rep for rep, w in winners.items() if w == "source"}
    keep_here = {rep for rep, w in winners.items() if w == "here"}
    rows, replaced = [], []
    for r in local:
        if r["rep"] + 1 in take_src:
            replaced.append(r)
        else:
            rows.append(plan["updates"].get(ckey(r), r))
    rows += [r for r in incoming if r["rep"] + 1 in take_src]
    rows += [r for r in plan["new"] if r["rep"] + 1 not in take_src | keep_here]
    return rows, replaced


def rep_machine_table(recs, envs=None):
    t = {}
    for r in recs:
        m = layout.machine_label(r, envs) if envs else (r.get("machine") or "未記錄")
        t.setdefault(r["rep"] + 1, {}).setdefault(m, 0)
        t[r["rep"] + 1][m] += 1
    return t


def print_rep_machine(recs):
    t = rep_machine_table(recs)
    if not t:
        return
    print(bullets("  各次重複由哪台機器跑", [(f"第 {rep} 次", "、".join(f"{m} {n} 段" for m, n in sorted(t[rep].items())))
                                       for rep in sorted(t)], indent=4))
    mixed = [rep for rep in t if len(t[rep]) > 1]
    if mixed:
        print(bullets(f"  ⚠ 第 {'、'.join(map(str, mixed))} 次重複由不只一台機器跑",
                      ["切換機器盡量在一次重複跑完時進行", "報告會列出各機器的截斷率與字數分布"], indent=4))


# ---------- calibrate ----------

def char_ratio(recs):
    """沒有每輪 token 數的舊資料：用整段最後一輪的 token 總數 ÷ 字數，估計每個字約幾個 token（取中位數）。"""
    rs = []
    for r in recs:
        if not r.get("tokens") or not r.get("replies") or (r.get("finish") or [""])[-1] == "length":
            continue
        chars = len(r.get("system") or "") + sum(len(p) for p in r.get("prompts", [])) \
            + sum(len(x) for x in r["replies"]) + (r.get("think_chars") or [0])[-1]
        if chars:
            rs.append(r["tokens"] / chars)
    return float(np.median(rs)) if rs else 1.0


def reply_tokens(r, t, ratio):
    """第 t 輪回覆用掉的 token（含思考）。有每輪的實際數字就用，沒有就用字數估計。回傳 (數字, 是否實測)。"""
    tt = r.get("turn_tokens") or []
    if t < len(tt) and tt[t]:
        return int(tt[t]), True
    return int(ratio * (len(r["replies"][t]) + ((r.get("think_chars") or [0] * (t + 1))[t] or 0))), False


def last_prompt_tokens(r, ratio):
    pt = r.get("prompt_tokens") or []
    if pt and pt[-1]:
        return int(pt[-1]), True
    chars = len(r.get("system") or "") + sum(len(p) for p in r.get("prompts", [])) + sum(len(x) for x in r["replies"][:-1])
    return int(ratio * chars), False


def segment_floor(cap, prompt_tokens=800):
    """每段上下文的最低要求：三輪都用滿每則上限，加上提問（約 800 token）與一成餘裕，取到 256 的倍數。
    每則上限 3,700 時約 13,000（企劃書 10-16）。"""
    return int(math.ceil((3 * cap + prompt_tokens) * 1.1 / 256) * 256)


def calibrate(recs, min_seg, max_tokens, formal_cap=None):
    """回傳建議值與根據。formal_cap：設定檔的每則上限（全研究一律相同），用來推算每段上下文的最低要求。"""
    ratio = char_ratio(recs)
    toks, cut, measured, total = [], 0, 0, 0
    for r in recs:
        for t, f in enumerate(r.get("finish") or []):
            total += 1
            if f == "length":
                cut += 1
                continue
            v, real = reply_tokens(r, t, ratio)
            measured += real
            toks.append(v)
    prompts = [last_prompt_tokens(r, ratio)[0] for r in recs if r.get("replies")]
    if not toks:
        return None
    p99 = float(np.percentile(toks, 99))
    suggest = int(math.ceil(p99 * 1.2 / 100) * 100)
    p_prompt = float(np.percentile(prompts, 99)) if prompts else 0
    need2 = int(math.ceil((p_prompt + suggest) * 1.1 / 256) * 256)
    need3 = int(math.ceil((p_prompt + 2 * suggest + 100) * 1.1 / 256) * 256)
    return {"replies": total, "cut": cut, "cut_rate": cut / max(total, 1), "measured": measured,
            "ratio": ratio, "p99": p99, "suggest": suggest, "current": max_tokens, "p99_prompt": p_prompt,
            "need2": need2, "need3": need3, "floor": segment_floor(formal_cap) if formal_cap else 0,
            "per_segment": max(min_seg, need3, segment_floor(formal_cap) if formal_cap else 0), "min_seg": min_seg,
            "formal_cap": formal_cap}


def cmd_calibrate(a):
    cfg = read_config(a.config)
    gen = cfg.get("generation", {})
    min_seg = int(gen.get("min_context_per_segment", 12000))
    dirs = [d for d in layout.run_dirs(a.out) if os.path.isfile(os.path.join(d, layout.RESPONSES))
            and not layout.read_settings(d).get("module")]
    if a.run:
        dirs = [d for d in dirs if a.run in os.path.basename(d)]
    if not dirs:
        stop("沒有可以校準的試跑資料（主流程的回覆）" + (f"：名稱含「{a.run}」" if a.run else ""))
    for d in dirs:
        s = layout.read_settings(d)
        recs = read_jsonl(os.path.join(d, layout.RESPONSES))
        print(f"\n■ {os.path.basename(d)}")
        sw = layout.settings_warning(d, recs)
        if sw:
            print(sw)
        # 目前的每則上限以回覆紀錄為準（settings.json 可能被手動改過）
        used = sorted({int(r["max_tokens"]) for r in recs if r.get("max_tokens")})
        current = used[-1] if used else s["max_tokens"]
        c = calibrate(recs, min_seg, current, gen.get("max_tokens"))
        if not c:
            print("  ✗ 沒有未截斷的回覆：無法校準")
            continue
        src = "實測（每輪 token 數）" if c["measured"] == c["replies"] - c["cut"] else \
            f"部分或全部用字數估計（每字約 {c['ratio']:.2f} token；新版程式會記每輪的實際數字）"
        print(kv_block("根據", [
            ("回覆", f"{c['replies']} 則"), ("截斷", f"{c['cut']} 則（{c['cut_rate']:.1%}）"),
            ("token 數來源", src),
            ("每則用量", f"未截斷回覆（含思考）第 99 百分位 {c['p99']:.0f} token"),
            ("最後一輪提問", f"含前面的對話，第 99 百分位約 {c['p99_prompt']:.0f} token")], indent=4).replace("── ", "  ── ", 1))
        print(kv_block("建議", [
            ("每則上限", f"max_tokens = {c['suggest']}（第 99 百分位 × 1.2，取到百位；目前 {c['current']}）"),
            ("每段上下文", f"至少 {c['per_segment']} token"),
            ("推算", [f"實測：兩輪約 {c['need2']}、三輪約 {c['need3']}",
                     f"三輪都用滿每則上限 {c['formal_cap']}：約 {c['floor']}", f"設定檔下限：{c['min_seg']}"]),
            ("上下文長度", ["Context Length 是總量，由並行的各段平分",
                         "、".join(f"並行 {w} 段 {c['per_segment'] * w}" for w in (1, 2, 3, 4))])], indent=4).replace("── ", "  ── ", 1))
        if c["cut_rate"] > 0.01:
            print(bullets(f"    ⚠ 截斷率 {c['cut_rate']:.1%}：建議值可能偏低", ["被截斷的回覆真正多長不知道",
                                                                     "先用建議值試跑一小批，看截斷率是否降到 1% 以下"], indent=6))
        print(note_block("怎麼改（以上只是建議，作者確認後再改）", how_max_tokens(a.config) + HOW_CONTEXT, indent=4).replace("── ", "  ── ", 1))
        layout.update_meta(d, {"calibration": dict(c, time=time.strftime("%Y-%m-%d %H:%M"))})
    from .common import next_steps
    print(next_steps(["建議值交給企劃書對話確認，再寫進設定檔 generation.max_tokens、generation.context_length",
                      "調高每則上限：接續寫進同一個設定資料夾（被截斷的回覆要續寫）",
                      "調低每則上限：另開資料夾"]))


# ---------- continue ----------

def mock_continue(partial):
    return "（續）以上是完整的說明。", "stop", {"completion_tokens": 10}, 0


def continue_reply(a, model, msgs, partial, budget, temperature, seed):
    """從中斷處接著生成：把已經生成的部分當成最後一則 assistant 訊息送出（伺服器要支援接續生成）。"""
    if a.mock:
        return mock_continue(partial)
    d = chat_raw(a.base_url, model, msgs + [{"role": "assistant", "content": partial}], temperature, budget, seed)
    ch = d["choices"][0]
    msg = ch.get("message", {})
    return (msg.get("content") or ""), ch.get("finish_reason"), d.get("usage") or {}, \
        len(msg.get("reasoning_content") or msg.get("reasoning") or "")


def regenerate(a, model, msgs, budget, temperature, seed):
    if a.mock:
        return "（重生成）根據完整的第 1 輪回覆，補充如下。", "stop", {"completion_tokens": 12}, 0
    d = chat_raw(a.base_url, model, msgs, temperature, budget, seed)
    ch = d["choices"][0]
    msg = ch.get("message", {})
    return (msg.get("content") or "").strip(), ch.get("finish_reason"), d.get("usage") or {}, \
        len(msg.get("reasoning_content") or msg.get("reasoning") or "")


def fix_record(a, r, model, new_cap):
    """續寫一段對話裡被截斷的回覆；第 1 輪被續寫時，後面各輪用完整的回覆重新生成。回傳更新後的紀錄或錯誤說明。"""
    replies = list(r["replies"])
    finish = list(r["finish"])
    thinks = list(r.get("think_chars") or [0] * len(replies))
    tt = list(r.get("turn_tokens") or [None] * len(replies))
    t0 = finish.index("length")
    if tt[t0] is None:
        return None, "缺第 {} 輪的 token 數（舊版程式的資料）：算不出還剩多少額度".format(t0 + 1)
    seed = 7 + r["rep"] * 1009
    base = [{"role": "system", "content": r["system"]}] if r.get("system") else []
    continued, regenerated = [], []
    for t in range(t0, len(replies)):
        msgs = list(base)
        for i in range(t):
            msgs += [{"role": "user", "content": r["prompts"][i]}, {"role": "assistant", "content": replies[i]}]
        msgs.append({"role": "user", "content": r["prompts"][t]})
        if t > t0 and t0 in continued:
            text, fin, usage, think = regenerate(a, model, msgs, new_cap, r["temperature"], seed + t)
            replies[t], finish[t], thinks[t], tt[t] = text, fin, think, usage.get("completion_tokens")
            regenerated.append(t)
            if fin != "length" or tt[t] is None:
                continue
        if finish[t] != "length":
            continue
        budget = new_cap - int(tt[t])
        if budget <= 0:
            return None, f"第 {t + 1} 輪已用 {tt[t]} token：不少於總上限 {new_cap}"
        text, fin, usage, think = continue_reply(a, model, msgs, replies[t], budget, r["temperature"], seed + t)
        replies[t] = replies[t] + text
        finish[t] = fin
        thinks[t] = (thinks[t] or 0) + think
        tt[t] = int(tt[t]) + int(usage.get("completion_tokens") or 0)
        continued.append(t)
    out = dict(r)
    out.update({"replies": replies, "finish": finish, "think_chars": thinks, "turn_tokens": tt,
                "continued": sorted(set(r.get("continued", []) + continued)),
                "regenerated": sorted(set(r.get("regenerated", []) + regenerated)),
                "max_tokens_total": new_cap,
                "before_continue": r.get("before_continue") or {"replies": r["replies"], "finish": r["finish"],
                                                                 "think_chars": r.get("think_chars"),
                                                                 "turn_tokens": r.get("turn_tokens")},
                "continue_time": time.strftime("%Y-%m-%d %H:%M:%S"), "continue_machine": envinfo.machine_code(a)})
    out.pop("replies_trad", None)      # v0.9.6 起不再把簡體回覆轉回繁體（舊紀錄的這一欄續寫後就不準了）
    return out, None


def cmd_continue(a):
    dirs = [d for d in layout.run_dirs(a.out) if os.path.isfile(os.path.join(d, layout.RESPONSES))]
    if a.run:
        dirs = [d for d in dirs if a.run in os.path.basename(d)]
    if not a.mock:
        dirs = [d for d in dirs if not os.path.basename(d).startswith("mock__")]
    cloudy = [os.path.basename(d) for d in dirs if layout.read_settings(d).get("provider")]
    if cloudy:
        print(bullets("⚠ 雲端模型的資料夾還不支援續寫（略過）", cloudy))
        dirs = [d for d in dirs if os.path.basename(d) not in cloudy]
    if not dirs:
        stop("沒有可以續寫的回覆" + (f"：名稱含「{a.run}」" if a.run else ""))
    for d in dirs:
        s = layout.read_settings(d)
        meta = layout.read_meta(d)
        target = layout.continue_target(d, s)
        to = int(a.to or target)
        if to < target:
            print(bullets(f"■ {os.path.basename(d)}：--to {to} 不採用（略過）", [
                ("總上限", f"{target}（每則上限 {s['max_tokens']} 加 10%，或之前續寫用過的更高值）"),
                "總上限只能調高；所有被截斷的回覆一律相同"]))
            continue
        model = s["model"]
        if a.model and a.model != model and not a.mock:
            stop("續寫要用同一個模型", f"這組資料：{model}", f"--model：{a.model}")
        path = os.path.join(d, layout.RESPONSES)
        recs = read_jsonl(path)
        # 只續寫總上限低於這次總上限的段（已經續寫到這個值、仍被截斷的不再續寫）
        todo = [i for i, r in enumerate(recs) if "length" in (r.get("finish") or [])
                and int(r.get("max_tokens_total") or r.get("max_tokens") or 0) < to]
        print(f"\n■ {os.path.basename(d)}")
        print(kv_block("", [("被截斷", f"{len(todo)} 段"), ("總上限", f"{to}（每則，含思考）")]))
        if not todo:
            continue
        if a.check:
            r = recs[todo[0]]
            out, err = fix_record(a, r, model, to)
            if err:
                print(f"  ✗ 無法續寫：{err}")
                return
            t = out["continued"][0]
            old = r["replies"][t]
            print(f"  試續寫 {r['pid']} {r['variant']} {r['qid']} 第 {r['rep'] + 1} 次第 {t + 1} 輪（沒有存檔）")
            print(f"  ── 原本的結尾：…{old[-200:]}")
            print(f"  ── 接著生成的開頭：{out['replies'][t][len(old):][:300]}")
            print(bullets("  要確認", ["接著生成的內容是從中斷處接下去，不是重新開始回答",
                                     "確認後去掉 --check 正式續寫：" + cmd_hint("continue", f"--run {os.path.basename(d)}")], indent=4))
            return
        lock = threading.Lock()
        errors, done = {}, 0
        t0 = time.time()
        bar = Progress("續寫", len(todo))
        print("  隨時可按 Ctrl+C：做完的段會存檔，再執行同一行接著續寫")
        ex = ThreadPoolExecutor(max_workers=1 if a.mock else max(1, a.workers))
        futs = {ex.submit(fix_record, a, recs[i], model, to): i for i in todo}
        interrupted = False
        try:
            pending = set(futs)
            while pending:
                # 有逾時的等待：Windows 上 Ctrl+C 才收得到
                finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for f in finished:
                    i = futs[f]
                    try:
                        out, err = f.result()
                    except Exception as e:
                        out, err = None, f"{type(e).__name__}：{e}"
                    with lock:
                        if out:
                            recs[i] = out
                            done += 1
                        else:
                            errors[err] = errors.get(err, 0) + 1
                        bar.tick(bool(out))
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n── 收到中斷：做完的 {done} 段先存檔；進行中的放棄，再執行同一行接著續寫")
        ex.shutdown(wait=not interrupted, cancel_futures=True)
        if done:
            shutil.copy2(path, layout.backup_path(path, "before-continue"))
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
            cont = meta.get("continuation") or {"to": to, "sessions": []}
            if int(cont.get("to") or 0) != to:
                # 總上限調高（v0.9.7 起是每則上限加 10%）：記下先前的值；先前續寫過、仍被截斷的段這次一併續寫到新值
                cont["to_history"] = (cont.get("to_history") or []) + [{"to": cont.get("to"), "until": time.strftime("%Y-%m-%d %H:%M")}]
                cont["to"] = to
            cont["sessions"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "records": done, "to": to,
                                     "machine": envinfo.machine_code(a), "secs": round(time.time() - t0, 1)})
            layout.update_meta(d, {"continuation": cont})
        cut = [r for r in recs if "length" in (r.get("finish") or [])]
        left = sum(int(r.get("max_tokens_total") or r.get("max_tokens") or 0) < to for r in cut)
        print(kv_block("續寫結果", [("完成", f"{done} 段（{fmt_dur(time.time() - t0)}）"),
                           ("還沒續寫", f"{left} 段（再執行同一行）" if left else None),
                           ("到總上限仍截斷", f"{len(cut) - left} 段（不再續寫，報告列截斷率）" if len(cut) - left else None),
                           *[(f"✗ {n} 段", e) for e, n in errors.items()],
                           ("標記", "continued、regenerated；原本的回覆在 before_continue" if done else None),
                           ("評審", "這些回覆要重評（judge 會自動挑出來）" if done else None)]))
        if interrupted:
            os._exit(130)
