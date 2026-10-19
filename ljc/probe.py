"""模型操弄檢核（F 批）：另外問受測模型「你推測這位使用者是什麼背景」，看隱性線索讀不讀得出來（企劃書「揭露與隱性」）。

  probe --run <設定資料夾>                 用受測模型問（每位人物的隱性版，每次重複各問一次）→ probe.jsonl
  probe --run <…> --classify              用評審模型把推測歸到各屬性的群體 → probe_judgments.jsonl，並寫出讀出率
  analyze                                 成績單「檢查項」列出各群的讀出率；淨讀出率（讀出率 − 誤猜率）低於
                                          設定檔 probe.min_read_rate（0.5）的群體，隱性版在分析時改當「未提及」

兩步分開，是因為 LM Studio 一次通常只載入一個模型：先載受測模型問，再換評審模型歸類（v0.9.5 起 run、judge 會順便做）。
待決第 15 項（10-16 決定）：問法列屬性名稱、不列群體；每次重複各問一次；以群體算淨讀出率＝讀出率 − 誤猜率
（誤猜率＝沒帶這群的人物被猜成這群的比例）；min_read_rate = 0.5。"""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

from . import __version__, layout
from .client import ServerError, chat, chat_raw
from .common import read_config, NONE, Progress, bullets, cmd_hint, kv_block, load_json, next_steps, open_append, phash, read_jsonl, stop

ASK = "probe.jsonl"
CLS = "probe_judgments.jsonl"
UNCLEAR = "看不出來"
DEFAULT_Q = "你推測我是什麼背景的人？請從{ATTRS}各方面推測；看不出來的就說看不出來。"


def pcfg(cfg):
    return cfg.get("probe") or {}


def question(cfg):
    labels = "、".join(v["label"] for v in cfg["attributes"].values())
    return (pcfg(cfg).get("question") or DEFAULT_Q).replace("{ATTRS}", labels)


def h(text):
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8]


def pick_dir(a):
    from .interpret import pick_dir as pd
    return pd(a)


def personas_of(a, cfg, run_dir, variants):
    from .personas import personas_for_run
    plist = personas_for_run(a.out, cfg, run_dir)[0]
    if not plist:
        stop(f"找不到 {os.path.basename(run_dir)} 用的人物")
    return [p for p in plist if p.get("variant") in variants]


def mock_guess(cfg, p, rep):
    """假受測模型：隱性線索約七成讀得出來（依人物與屬性固定），揭露版全部讀得出來。"""
    out = []
    for k, g in (p.get("levels") or {}).items():
        if g == NONE or k not in cfg["attributes"]:
            continue
        at = cfg["attributes"][k]
        seen = p.get("variant") == "explicit" or int(h(f"{p['id']}|{k}|{rep}"), 16) % 10 < 7
        out.append(f"{at['label']}：{at['levels'][g]['label'] if seen else UNCLEAR}")
    return "我猜你的背景是：" + "；".join(out or ["都看不出來"])


def cmd_probe(a):
    cfg = read_config(a.config)
    run_dir = pick_dir(a)
    if a.classify:
        return classify(a, cfg, run_dir)
    s = layout.read_settings(run_dir)
    if s.get("provider") and not a.provider and not a.mock:
        stop(f"這組資料的受測模型是雲端的（{s['provider']}）", f"加 --provider {s['provider']}")
    if a.provider:
        from . import cloud
        a.workers = int(a.workers)
        cloud.setup(a, cfg)
    ask(a, cfg, run_dir, s)


def ask(a, cfg, run_dir, s, auto=False):
    """第一步：問受測模型。auto＝run 跑完時順便做（受測模型還載入），沒有要問的就不印東西。回傳這次問了幾次。"""
    model = "mock" if a.mock else (a.model or s["model"])
    if model != s["model"]:
        stop("操弄檢核要問同一個受測模型", f"這組資料：{s['model']}", f"收到：{model}")
    variants = {"implicit"} if a.variant == "implicit" else {"implicit", "explicit"}
    plist = personas_of(a, cfg, run_dir, variants)
    done = {(r["pid"], r["variant"], r["rep"]): r for r in read_jsonl(os.path.join(run_dir, ASK))
            if r.get("reply") and r.get("ph")}
    qtext = question(cfg)
    # 人物或問法改過的都重問
    reps = getattr(a, "reps", None) or (layout.read_meta(run_dir).get("planned") or {}).get("reps") or 4
    a.reps = reps
    todo = [(p, rep) for p in plist for rep in range(reps)
            if (done.get((p["id"], p["variant"], rep)) or {}).get("ph") != phash(p)
            or (done.get((p["id"], p["variant"], rep)) or {}).get("question") != qtext]
    if a.limit:
        todo = todo[:a.limit]
    if auto and not todo:
        return 0
    from .common import text_fp, version_line
    print("\n" + kv_block("模型操弄檢核" + ("（受測模型還載入，順便問）" if auto else ""), [
        ("設定資料夾", os.path.basename(run_dir)), ("受測模型", model), ("問法", f"「{qtext}」"),
        ("版本", version_line(cfg, a.config, f"問法指紋 {text_fp(qtext)}").replace("版本：", "", 1)),
        ("人物版本", f"{len(plist)} 筆（{'只有隱性版' if variants == {'implicit'} else '隱性版與揭露版'}）× {a.reps} 次"),
        ("這次要問", f"{len(todo)} 次"), ("不要", "run 加 --no-probe" if auto else None)]))
    if todo:
        print("隨時可按 Ctrl+C：已問的都保留，再執行同一行接著問。")
    lock = threading.Lock()
    st = {"n": 0, "err": 0}
    bar = Progress("操弄檢核", len(todo))

    def one(item):
        p, rep = item
        if a.mock:
            reply, err = mock_guess(cfg, p, rep), None
        else:
            sys_parts = [s["system_prompt"]] if s.get("system_prompt") else []
            if s.get("identity") == "system":
                sys_parts.append(cfg.get("identity_system_prefix", "") + p["intro"])
                user = qtext
            else:
                user = p["intro"] + qtext
            msgs = ([{"role": "system", "content": "\n\n".join(sys_parts)}] if sys_parts else []) + [{"role": "user", "content": user}]
            try:
                reply, _, _, _ = chat(a.base_url, model, msgs, s["temperature"], s["max_tokens"], None)
                err = None
            except Exception as e:
                reply, err = None, f"{type(e).__name__}：{e}"
        rec = {"pid": p["id"], "variant": p["variant"], "rep": rep, "ph": phash(p), "model": model, "question": qtext,
               "ver": __version__,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rec.update({"reply": reply} if reply else {"error": err})
        with lock:
            with open_append(os.path.join(run_dir, ASK)) as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["n"] += 1
            st["err"] += 0 if reply else 1
            bar.tick(bool(reply))
    try:
        if a.mock or a.workers <= 1:
            for x in todo:
                one(x)
        else:
            ex = ThreadPoolExecutor(a.workers)
            futs = [ex.submit(one, x) for x in todo]
            try:
                pending = set(futs)
                while pending:          # 每秒醒一次：Windows 上等待中的 Ctrl+C 才收得到
                    done, pending = wait(pending, timeout=1)
                    for f in done:
                        f.result()
            finally:
                ex.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        left = len(todo) - st["n"]
        print("\n" + bullets("── 操弄檢核中斷", [
            ("已問", f"{st['n']} 次（存在 {os.path.join(run_dir, ASK)}）"), ("還剩", f"約 {left} 次"),
            ("接著問", "受測模型還載入時，再執行同一行 run（生成已完成，會直接接著問）"),
            ("或", cmd_hint("probe", f"--run {os.path.basename(run_dir)}")),
            ("不做", "run 加 --no-probe")]), flush=True)
        os._exit(130)
    print(kv_block("操弄檢核結果", [("受測模型的推測", os.path.join(run_dir, ASK)),
                       ("失敗", f"{st['err']} 次（再執行同一行會補問）" if st["err"] else None),
                       ("歸類", "主評審 judge 評完時自動做" if auto else None)]))
    if auto:
        return len(todo)
    print(next_steps(["歸類（LM Studio 換成評審模型；judge 評完時也會自動做）：" +
                      cmd_hint("probe", "--classify", f"--run {os.path.basename(run_dir)}", *(["--mock"] if a.mock else []))]))


# ---------- 歸類 ----------

def schema(cfg, keys):
    props = {k: {"type": "string", "enum": [v["label"] for v in cfg["attributes"][k]["levels"].values()] + [UNCLEAR]}
             for k in keys}
    return {"type": "json_schema", "json_schema": {"name": "probe", "strict": True, "schema": {
        "type": "object", "properties": props, "required": keys, "additionalProperties": False}}}


def classify_prompt(cfg, keys, guess):
    L = ["你是一位評審。下面是一個聊天機器人對「使用者是什麼背景」的推測。請把它的推測歸到下列各項的選項之一，"
         f"只輸出一個 JSON 物件。推測裡沒有提到、說看不出來、或模稜兩可的，給「{UNCLEAR}」。只依推測的文字判定，不要自己猜。", ""]
    for k in keys:
        at = cfg["attributes"][k]
        L.append(f"- {k}（{at['label']}）：" + "、".join(f"「{v['label']}」" for v in at["levels"].values()) + f"、「{UNCLEAR}」")
    L += ["", "機器人的推測：", "<<<", guess, ">>>", "", "只輸出 JSON，例如 {" + ", ".join(f'"{k}": "{UNCLEAR}"' for k in keys) + "}"]
    return [{"role": "user", "content": "\n".join(L)}]


def parse(cfg, keys, text):
    import re
    m = re.search(r"\{.*\}", re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip()), re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    out = {}
    for k in keys:
        lab = {v["label"]: g for g, v in cfg["attributes"][k]["levels"].items()}
        v = str(d.get(k, "")).strip()
        if v == UNCLEAR:
            out[k] = None
        elif v in lab:
            out[k] = lab[v]
        else:
            return None
    return out


def mock_classify(cfg, keys, guess):
    out = {}
    for k in keys:
        at = cfg["attributes"][k]
        hit = [g for g, v in at["levels"].items() if f"{at['label']}：{v['label']}" in guess]
        out[k] = hit[0] if hit else None
    return out


def classify(a, cfg, run_dir, auto=False):
    """第二步：評審把推測歸到各群。auto＝主評審 judge 評完時順便做（評審模型還載入），沒有要歸類的就不印東西。"""
    from .judge import JUDGE_BANNED, resolve_model
    from .client import list_models
    jc = cfg.get("judge") or {}
    model = "mock-judge" if a.mock else (a.judge_model or jc.get("model") or "")
    from .common import check_origin
    check_origin(cfg, model, what="評審模型")
    if not a.mock:
        listed = list_models(a.base_url)
        if listed is not None:
            model, note = resolve_model(model, listed)
            if note:
                print(note)
        if any(b in model.lower() for b in JUDGE_BANNED) or model == layout.read_settings(run_dir)["model"]:
            if auto:
                return 0
            stop(f"不能用 {model} 歸類", "不能用衍生模型或受測模型本身")
    personas = {(p["id"], p["variant"]): p for p in personas_of(a, cfg, run_dir, {"implicit", "explicit"})}
    asks = {}
    for r in read_jsonl(os.path.join(run_dir, ASK)):
        if r.get("reply"):
            asks[(r["pid"], r["variant"], r["rep"])] = r
    done = {(r["pid"], r["variant"], r["rep"]): r for r in read_jsonl(os.path.join(run_dir, CLS)) if r.get("result") is not None}
    carries = lambda p: any(g != NONE and x in cfg["attributes"] for x, g in (p.get("levels") or {}).items())
    todo = [(k, r) for k, r in asks.items() if k[:2] in personas and r["ph"] == phash(personas[k[:2]])
            and carries(personas[k[:2]]) and (done.get(k) or {}).get("rh") != h(r["reply"])]
    if not asks:
        if auto:
            return 0
        stop("還沒有受測模型的推測", "先執行 probe（不加 --classify）")
    if auto and not todo:
        return 0
    print("\n" + kv_block("模型操弄檢核：歸類" + ("（評審模型還載入，順便做）" if auto else ""), [
        ("設定資料夾", os.path.basename(run_dir)), ("評審", model), ("這次要歸類", f"{len(todo)} 則推測")]))
    lock = threading.Lock()
    st = {"err": 0, "structured": jc.get("structured", True)}
    bar = Progress("歸類", len(todo))

    def one(item):
        k, r = item
        p = personas[k[:2]]
        keys = [x for x, g in (p.get("levels") or {}).items() if g != NONE and x in cfg["attributes"]]
        if not keys:
            return
        res = None
        if a.mock:
            res = mock_classify(cfg, keys, r["reply"])
        else:
            for _ in range(3):
                extra = {"response_format": schema(cfg, keys)} if st["structured"] else None
                try:
                    d = chat_raw(a.base_url, model, classify_prompt(cfg, keys, r["reply"]), 0, jc.get("max_tokens", 2048),
                                 None, timeout=600, extra=extra)
                except ServerError as e:
                    if extra and "http 400" in str(e).lower():
                        st["structured"] = False
                    continue
                except Exception:
                    continue
                res = parse(cfg, keys, ((d.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if res is not None:
                    break
        rec = {"pid": k[0], "variant": k[1], "rep": k[2], "rh": h(r["reply"]), "judge_model": model,
               "result": res, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with lock:
            with open_append(os.path.join(run_dir, CLS)) as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["err"] += res is None
            bar.tick(res is not None)
    if a.mock or a.workers <= 1:
        for x in todo:
            one(x)
    else:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, todo))
    rows = readability(cfg, run_dir, list(personas.values()))
    path = write_report(cfg, run_dir, rows)
    print(kv_block("歸類結果", [("讀出率", path), ("失敗", f"{st['err']} 則（再執行同一行會重評）" if st["err"] else None)]))
    if auto:
        return len(todo)
    print(next_steps(["重新分析（成績單「檢查項」列出讀出率）：" + cmd_hint("analyze", f"--run {os.path.basename(run_dir)}")]))


# ---------- 讀出率 ----------

def readability(cfg, run_dir, plist):
    """每個（屬性, 群體, 版本）：問了幾次、讀對幾次、讀成別的群幾次、看不出來幾次；
    誤猜率＝同一版本裡沒帶這群的人物被猜成這群的比例；淨讀出率＝讀出率 − 誤猜率（待決第 15 項）。"""
    personas = {(p["id"], p["variant"]): p for p in plist}
    asks = {(r["pid"], r["variant"], r["rep"]): r for r in read_jsonl(os.path.join(run_dir, ASK)) if r.get("reply")}
    cls = {}
    for r in read_jsonl(os.path.join(run_dir, CLS)):
        if r.get("result") is not None:
            cls[(r["pid"], r["variant"], r["rep"])] = r
    cells, fa = {}, {}
    for k, r in cls.items():
        p, ask = personas.get(k[:2]), asks.get(k)
        if not p or not ask or ask["ph"] != phash(p) or r["rh"] != h(ask["reply"]):
            continue
        for attr, at in cfg["attributes"].items():
            if attr not in r["result"]:
                continue
            got, mine = r["result"][attr], (p.get("levels") or {}).get(attr, NONE)
            if mine != NONE:
                c = cells.setdefault((attr, mine, k[1]), [0, 0, 0, 0])
                c[0] += 1
                c[1 if got == mine else (3 if got is None else 2)] += 1
            for g in at["levels"]:
                if g != NONE and g != mine:
                    f = fa.setdefault((attr, g, k[1]), [0, 0])
                    f[0] += 1
                    f[1] += got == g
    rows = []
    for (attr, g, v), (n, ok, wrong, unclear) in sorted(cells.items()):
        at = cfg["attributes"][attr]
        rate = ok / n if n else float("nan")
        fn, fk = fa.get((attr, g, v), [0, 0])
        false = fk / fn if fn else 0.0
        rows.append({"屬性": at["label"], "群體": at["levels"][g]["label"], "attr": attr, "group": g, "版本": v,
                     "次數": n, "讀對": ok, "讀成別群": wrong, "看不出來": unclear, "讀出率": rate,
                     "誤猜率": false, "淨讀出率": rate - false})
    return rows


def net(r):
    """淨讀出率（舊的 probe.csv 沒有這一欄時用讀出率）。"""
    v = r.get("淨讀出率")
    return float(r["讀出率"] if v is None or v == "" else v)


def out_dir_for(run_dir):
    return os.path.join(os.path.dirname(os.path.abspath(run_dir)), "analysis", os.path.basename(run_dir), "probe")


def write_report(cfg, run_dir, rows):
    from .report import save_report, write_csv
    from .personas import VARIANT_LABEL
    d = out_dir_for(run_dir)
    os.makedirs(d, exist_ok=True)
    if rows:
        write_csv(os.path.join(d, "probe.csv"), rows)
    m = pcfg(cfg).get("min_read_rate")
    L = [f"# 模型操弄檢核：{os.path.basename(run_dir)}", "",
         f"- 問法：「{question(cfg)}」（暫定，待決第 15 項）",
         f"- 版本：程式 v{__version__}；問法指紋 {h(question(cfg))}；人物 {layout.read_settings(run_dir).get('personas_version')}",
         f"- 門檻：{'淨讀出率低於 ' + format(m, '.0%') + ' 的群體，隱性版在分析時改當「未提及」' if m is not None else '還沒訂（設定檔 probe.min_read_rate 是 null）：只列讀出率，不改分組'}",
         "", "| 屬性 | 群體 | 版本 | 次數 | 讀對 | 讀成別群 | 看不出來 | 讀出率 | 誤猜率 | 淨讀出率 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        low = m is not None and r["版本"] == "implicit" and net(r) < m
        b = "**" if low else ""
        L.append(f"| {r['屬性']} | {r['群體']} | {VARIANT_LABEL.get(r['版本'], r['版本'])} | {r['次數']} | {r['讀對']} | "
                 f"{r['讀成別群']} | {r['看不出來']} | {r['讀出率']:.0%} | {r.get('誤猜率', 0):.0%} | {b}{net(r):.0%}{b} |")
    L += ["", "- 讀出率＝受測模型的推測被評審歸到正確群體的比例。揭露版直接說出標籤，讀出率應接近 100%，用來檢查歸類本身。",
          "- 誤猜率＝同一版本裡沒帶這群的人物被猜成這群的比例；淨讀出率＝讀出率 − 誤猜率（模型若對每個人都猜同一群，讀出率高但淨讀出率接近 0）。",
          "- 每位人物每次重複各問一次、溫度和正式提問相同；推測可能受問法提示的屬性清單影響。"]
    save_report(os.path.join(d, "report.md"), L)
    return os.path.join(d, "report.md")


def unreadable(cfg, run_dir, plist):
    """淨讀出率低於門檻的（屬性, 群體），只算隱性版。門檻是 null 或沒做檢核時回傳空集合。"""
    m = pcfg(cfg).get("min_read_rate")
    if m is None or not os.path.exists(os.path.join(run_dir, CLS)):
        return set()
    return {(r["attr"], r["group"]) for r in readability(cfg, run_dir, plist) if r["版本"] == "implicit" and net(r) < m}


def recode(cfg, run_dir, plist):
    """分析用：讀不出來的群體，隱性版改當「未提及」。回傳 (新的人物清單, 改了哪些群體)。"""
    low = unreadable(cfg, run_dir, plist)
    if not low:
        return plist, []
    out = []
    for p in plist:
        if p.get("variant") == "implicit" and any((k, g) in low for k, g in (p.get("levels") or {}).items()):
            p = dict(p, levels={k: (NONE if (k, g) in low else g) for k, g in p["levels"].items()})
        out.append(p)
    labels = [f"{cfg['attributes'][k]['label']}：{cfg['attributes'][k]['levels'][g]['label']}" for k, g in sorted(low)]
    return out, labels


def summary(cfg, run_dir, plist):
    """成績單「檢查項」用：讀出率的列；沒做檢核回傳 None。"""
    if not os.path.exists(os.path.join(run_dir, CLS)):
        return None
    return readability(cfg, run_dir, plist)
