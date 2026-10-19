"""資料夾的安排：人物所有模型共用，回覆依「模型與設定」各放一個子資料夾，不同設定的資料不會混在一起。

  out/
  ├─ personas.jsonl、personas_review.md     人物（所有模型共用同一批，結果才能互相比較）
  ├─ tuning.json                            各台機器測得的並行數
  └─ <模型>__<身分位置>__t<溫度>__max<上限>__turns<輪數>__<題庫版本>__<人物版本>[__mod-<對照模組>]/
     ├─ settings.json                        這組設定的完整內容
     ├─ run_meta.json                        測試條件與每次執行的實測耗時
     ├─ responses.jsonl                      回覆
     └─ report.md、results.csv、embeddings.json

會改變回覆內容的設定都寫進資料夾名稱；系統提示太長，用它的指紋（__sys 加 6 碼）代表。
重複次數、並行數、時間上限不影響回覆內容，所以不分資料夾。"""
import json
import os
import re
import time

from . import __version__
from .common import load_json, open_append, read_jsonl, save_json

SETTINGS = "settings.json"
RESPONSES = "responses.jsonl"
LEGACY_EXTRAS = ("report.md", "results.csv", "embeddings.json")


def sys_hash(prompt):
    import hashlib
    return hashlib.md5((prompt or "").encode("utf-8")).hexdigest()[:6] if prompt else ""


def settings_from_args(a, cfg, personas_version, identity=None):
    gen = cfg.get("generation", {})
    s = {"model": a.model, "identity": identity or a.identity, "temperature": float(a.temperature),
            "max_tokens": int(a.max_tokens), "turns": a.rounds,
            "system_prompt": gen.get("system_prompt") or "", "thinking": gen.get("thinking"),
            "questions_version": cfg.get("version", {}).get("questions", ""), "personas_version": personas_version,
            "plan_version": cfg.get("version", {}).get("plan", "")}
    if getattr(a, "module", None):
        s["module"] = a.module
    return s


def settings_of_record(r):
    """從一筆回覆紀錄推回它的設定。舊紀錄沒有記系統提示，當作空白（之前的版本預設都是空白）。"""
    return {"model": r.get("model") or "unknown", "identity": r.get("identity") or "message",
            "temperature": float(r.get("temperature", 0.7)), "max_tokens": int(r.get("max_tokens") or 0),
            "turns": len(r.get("replies") or []) or 3, "sysh": r.get("sysh", ""),
            "questions_version": r.get("qver", ""), "personas_version": r.get("pver", "")}


def run_name(s):
    model = re.sub(r"[^0-9A-Za-z._-]+", "_", str(s["model"])).strip("_.") or "model"
    name = f"{model}__{s['identity']}__t{s['temperature']:g}__max{s['max_tokens']}__turns{s['turns']}"
    sysh = s["sysh"] if "sysh" in s else sys_hash(s.get("system_prompt"))
    name += f"__sys{sysh}" if sysh else ""
    for v in (s.get("questions_version"), s.get("personas_version")):   # 題庫與人物版本不同，回覆就分開放
        name += f"__{re.sub(r'[^0-9A-Za-z._-]+', '_', v)}" if v else ""
    if s.get("module"):                       # 對照模組（簡繁、標點、無關擾動）另放一個資料夾
        name += f"__mod-{s['module']}"
    return name


def caps(s):
    """這組設定用過的每則上限（由舊到新）。調高過上限的資料夾在 settings.json 記有 max_tokens_history。"""
    h = s.get("max_tokens_history") or []
    return [int(x["max_tokens"]) for x in h] or [int(s["max_tokens"])]


def qvers(s):
    """這組設定用過的題庫版本（由舊到新）。換過題目的資料夾在 settings.json 記有 questions_version_history。"""
    h = s.get("questions_version_history") or []
    return [x["questions_version"] for x in h] or [s.get("questions_version", "")]


def folder_name(s):
    """資料夾名稱：用建立時的每則上限與題庫版本。上限調高、換題後接續寫進同一個資料夾，名稱不變。"""
    return run_name(dict(s, max_tokens=caps(s)[0], questions_version=qvers(s)[0]))


def same_but_cap(a, b):
    """除了每則上限與題庫版本，其餘設定都相同。"""
    return run_name(dict(a, max_tokens=0, questions_version="")) == run_name(dict(b, max_tokens=0, questions_version=""))


def raise_qver(d, new):
    """題庫版本升級（例如 I1 換成新題，q1.0 → q1.1）：沿用同一個設定資料夾，記下歷程。
    沒改的題目照常沿用；換過的題目依新的指紋補跑，舊版的回覆保留、只描述（questions[].previous）。"""
    p = os.path.join(d, SETTINGS)
    s = load_json(p)
    now = time.strftime("%Y-%m-%d %H:%M")
    hist = s.get("questions_version_history") or [{"questions_version": s.get("questions_version", ""),
                                                   "since": s.get("created"), "version": s.get("version")}]
    hist.append({"questions_version": new, "since": now, "version": __version__})
    s.update(questions_version=new, questions_version_history=hist)
    save_json(p, s)
    update_meta(d, {"questions_version_history": hist})
    return hist


def raise_cap(d, new):
    """每則上限調高：沿用同一個設定資料夾，settings.json 與 run_meta.json 記下變更歷程。
    未截斷的回覆保留；被截斷的要用 continue 續寫到新上限加 10%，續寫完才能評審（pending_continuation）。"""
    p = os.path.join(d, SETTINGS)
    s = load_json(p)
    now = time.strftime("%Y-%m-%d %H:%M")
    hist = s.get("max_tokens_history") or [{"max_tokens": int(s["max_tokens"]), "since": s.get("created"),
                                            "version": s.get("version")}]
    hist.append({"max_tokens": int(new), "since": now, "version": __version__})
    s.update(max_tokens=int(new), max_tokens_history=hist)
    save_json(p, s)
    update_meta(d, {"max_tokens_history": hist})
    return hist


CONTINUE_EXTRA = 0.10     # 續寫的總上限＝每則上限 × 1.1（作者 10-16：續寫到同一個上限常常還是寫不完，續寫就沒有意義）


def continue_target(d, s=None):
    """這組設定被截斷的回覆要續寫到的總上限（每則，含思考）：每則上限加 10%（無條件進位）；
    之前用更高的值續寫過，就沿用那個值（總上限只能調高，所有被截斷的回覆一律相同）。"""
    s = s or read_settings(d)
    base = -(-int(s["max_tokens"]) * int(round(CONTINUE_EXTRA * 100)) // 100) + int(s["max_tokens"])
    prev = (read_meta(d).get("continuation") or {}).get("to") or 0
    return max(base, int(prev))


def pending_continuation(d, recs=None):
    """還沒續寫的段：有回覆被截斷，而且生成或續寫時的總上限低於續寫的總上限（continue_target）。"""
    s = read_settings(d)
    to = continue_target(d, s)
    if recs is None:
        recs = read_jsonl(os.path.join(d, RESPONSES))
    return [r for r in recs if "length" in (r.get("finish") or [])
            and int(r.get("max_tokens_total") or r.get("max_tokens") or s["max_tokens"]) < to]


def open_run(out_dir, s, say=print):
    """回傳這組設定的資料夾（沒有就建立），並寫下 settings.json。
    只有每則上限比既有資料夾高時，接續寫進那個資料夾（上限只能調高；調低會另開資料夾）。"""
    d = os.path.join(out_dir, run_name(s))
    if not os.path.isdir(d):
        for old in run_dirs(out_dir):
            try:
                os_ = read_settings(old)
                if same_but_cap(os_, s) and int(os_["max_tokens"]) <= int(s["max_tokens"]):
                    if os_.get("questions_version") != s.get("questions_version"):
                        raise_qver(old, s.get("questions_version"))
                        from .common import bullets
                        say(bullets(f"題庫版本從 {os_.get('questions_version')} 換成 {s.get('questions_version')}：接續寫進同一個設定資料夾",
                                    [os.path.basename(old), "沒改的題目沿用", "換過的題目補跑新版",
                                     "舊版的回覆保留在資料夾裡，只在報告裡描述"]))
                    if int(os_["max_tokens"]) < int(s["max_tokens"]):
                        raise_cap(old, s["max_tokens"])
                        n = len(pending_continuation(old))
                        from .common import bullets
                        say(bullets(f"每則上限從 {os_['max_tokens']} 提高到 {s['max_tokens']}：接續寫進同一個設定資料夾",
                                    [f"{os.path.basename(old)}（名稱保留建立時的上限）", "沒有截斷的回覆照常使用",
                                     f"被截斷的 {n} 段：續寫到 {continue_target(old)}（新上限加 10%），評審前要做完"]))
                    return old
            except (KeyError, TypeError, ValueError, OSError):
                continue
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, SETTINGS)
    if not os.path.exists(p):
        keep = {k: v for k, v in s.items() if k != "sysh"}
        save_json(p, dict(keep, created=time.strftime("%Y-%m-%d %H:%M"), version=__version__))
    return d


def drop_if_empty(d):
    """這次一段都沒跑成：把剛建立的空資料夾收掉，免得打錯模型名稱就留下一個空資料夾。"""
    names = set(os.listdir(d)) if os.path.isdir(d) else set()
    rp = os.path.join(d, RESPONSES)
    if names <= {SETTINGS, RESPONSES, META} and (not os.path.exists(rp) or os.path.getsize(rp) == 0):
        for n in names:
            os.remove(os.path.join(d, n))
        os.rmdir(d)


def run_dirs(out_dir):
    """所有設定資料夾，依名稱排序。"""
    if not os.path.isdir(out_dir):
        return []
    return sorted(os.path.join(out_dir, n) for n in os.listdir(out_dir)
                  if os.path.isfile(os.path.join(out_dir, n, SETTINGS)))


def read_settings(d):
    return load_json(os.path.join(d, SETTINGS))


# settings.json 的欄位 → 回覆紀錄裡對應的欄位（取值方式）與給人看的名稱
SETTING_FIELDS = [
    ("model", "模型", lambda r: r.get("model")),
    ("identity", "身分位置", lambda r: r.get("identity")),
    ("temperature", "溫度", lambda r: None if r.get("temperature") is None else float(r["temperature"])),
    ("max_tokens", "每則上限（max_tokens）", lambda r: None if r.get("max_tokens") is None else int(r["max_tokens"])),
    ("turns", "每段輪數", lambda r: len(r["replies"]) if r.get("replies") else None),
    ("sysh", "系統提示（指紋）", lambda r: r.get("sysh")),
    ("questions_version", "題庫版本", lambda r: r.get("qver")),
    ("personas_version", "人物版本", lambda r: r.get("pver")),
    ("module", "對照模組", lambda r: r.get("module") or ""),       # 主流程的紀錄沒有這個欄位
]


def check_settings(d, recs=None):
    """settings.json 有沒有被手動改過：和資料夾名稱、回覆紀錄比對，回傳對不上的地方（給人看的句子）。
    回覆紀錄記的是實際送給模型的設定，對不上時以回覆紀錄為準。舊紀錄沒記的欄位不比。"""
    from collections import Counter
    s = read_settings(d)
    probs = []
    try:
        want = folder_name(s)
    except (KeyError, TypeError, ValueError) as e:
        return [f"settings.json 缺少欄位或格式不對（{e}），可能被手動改過"]
    if want != os.path.basename(d):
        probs.append(f"資料夾名稱和 settings.json 對不上：照 settings.json 應該叫 {want}")
    if recs is None:
        recs = read_jsonl(os.path.join(d, RESPONSES))
    cur = dict(s, sysh=sys_hash(s.get("system_prompt")), module=s.get("module") or "")
    for key, label, get in SETTING_FIELDS:
        seen = Counter(v for v in (get(r) for r in recs) if v is not None)
        mine = cur.get(key)
        if key == "temperature" and mine is not None:
            mine = float(mine)
        ok = set(caps(s)) if key == "max_tokens" else set(qvers(s)) if key == "questions_version" else {mine}
        # 調高上限前、換題前跑的段照常算數
        other = {v: n for v, n in seen.items() if v not in ok}
        if other:
            show = lambda v: "空白" if v in ("", None) else v
            vals = "、".join(f"{show(v)}（{n} 段）" for v, n in sorted(other.items(), key=lambda x: -x[1]))
            probs.append(f"{label}：settings.json 是 {show(mine)}，回覆紀錄是 {vals}")
    return probs


def settings_warning(d, recs=None, indent="  "):
    """check_settings 的結果排成提示；沒有問題時回傳空字串。"""
    probs = check_settings(d, recs)
    if not probs:
        return ""
    from .common import bullets
    return bullets(f"{indent}⚠ settings.json 和這組資料對不上（可能被手動改過）",
                   list(probs) + ["回覆紀錄記的是實際送給模型的設定：settings.json 改回和回覆紀錄一致",
                                  "要換設定：用指令的參數（程式會另開一個設定資料夾）"], indent=len(indent) + 3)


def describe(s):
    """給人看的一行設定說明。"""
    where = "系統提示" if s.get("identity") == "system" else "提問前"
    extra = "｜系統提示：有" if s.get("system_prompt") else ""
    ver = "／".join(x for x in (s.get("questions_version"), s.get("personas_version")) if x)
    from .styles import MODULE_LABEL
    mod = f"｜對照模組：{MODULE_LABEL.get(s['module'], s['module'])}" if s.get("module") else ""
    hist = caps(s)
    cap = f"{s['max_tokens']} token" + (f"（由 {'→'.join(map(str, hist[:-1]))} 調高）" if len(hist) > 1 else "")
    return (f"{s['model']}｜身分位置：{where}｜溫度 {s['temperature']:g}｜每則上限 {cap}｜"
            f"每段 {s['turns']} 輪{extra}" + (f"｜題庫／人物版本 {ver}" if ver else "") + mod)


IDENTITY_LABEL = {"message": "寫在第一句提問前", "system": "放在系統提示（模擬記憶功能）"}


def run_arg(out_dir, d, required=False):
    """下一步提示裡的 --run：out 只有這一組設定時不用給（required 的指令照給）；否則用最短又只對到它的名稱（例如 p1.2）。"""
    dirs = run_dirs(out_dir)
    if len(dirs) <= 1 and not required:
        return ""
    name = os.path.basename(d)
    tail = name.rsplit("__", 1)[-1]
    if pick_one([x for x in dirs if tail in os.path.basename(x)], tail) == [d] or (required and len(dirs) <= 1):
        return f"--run {tail}"
    return f"--run {name}"


def pick_one(dirs, text):
    """--run 的文字對到好幾個資料夾時，名稱以這段文字結尾的優先（例如 p1.2 對到篩檢、不對到 p1.2-c-health）。"""
    if len(dirs) > 1 and text:
        tail = [d for d in dirs if os.path.basename(d).endswith(text)]
        if len(tail) == 1:
            return tail
    return dirs


def settings_rows(s):
    """一組設定排成「項目：內容」的列，給 kv_block 用。名稱和報告、文件一致。"""
    from .styles import MODULE_LABEL
    ver = "／".join(x for x in (s.get("questions_version"), s.get("personas_version")) if x)
    return [("模型", s["model"]),
            ("身分位置", IDENTITY_LABEL.get(s.get("identity"), s.get("identity"))),
            ("溫度", f"{s['temperature']:g}"),
            ("每則上限", f"{s['max_tokens']} token（max_tokens，含思考）"),
            ("每段輪數", f"{s['turns']} 輪"),
            ("系統提示", f"有（指紋 {sys_hash(s['system_prompt'])}）" if s.get("system_prompt") else "無"),
            ("題庫／人物版本", ver or None),
            ("對照模組", MODULE_LABEL.get(s["module"], s["module"]) if s.get("module") else None)]


META = "run_meta.json"


def read_meta(d):
    p = os.path.join(d, META)
    return load_json(p) if os.path.isfile(p) else {}


def add_session(d, base, session):
    """把這次執行的條件與實測耗時寫進 run_meta.json；每次執行附加一筆 sessions。"""
    meta = read_meta(d)
    meta.update(base)
    meta.setdefault("sessions", []).append(session)
    save_json(os.path.join(d, META), meta)


def update_meta(d, fields):
    """把幾個欄位寫進 run_meta.json（覆蓋同名欄位）。"""
    meta = read_meta(d)
    meta.update(fields)
    save_json(os.path.join(d, META), meta)


def add_judge_session(d, base, session):
    """評審的條件與實測耗時寫進 run_meta.json 的 judge（主評審）或 judge_<tag>（例如第二評審）。"""
    meta = read_meta(d)
    key = "judge" + (f"_{base['tag']}" if base.get("tag") else "")
    j = meta.get(key, {})
    j.update({k: v for k, v in base.items() if k != "tag"})
    j.setdefault("sessions", []).append(dict(session, model=base["model"]))
    meta[key] = j
    save_json(os.path.join(d, META), meta)


def machine_label(rec, envs=None):
    """報告與提示裡的「機器」：機器代號加顯示卡型號（同一台電腦換了顯示卡也分得開），例如 machine-A（RTX 3090）。
    生成紀錄的顯示卡查 run_meta.json 的 envs；評審紀錄自己記了 gpu（v0.9.5 起）。"""
    from .envinfo import gpu_model, short_gpu
    m = rec.get("machine") or "未記錄"
    g = rec.get("gpu") or gpu_model(((envs or {}).get(rec.get("env")) or {}).get("gpu"))
    g = gpu_model(g) if g else None
    return f"{m}（{short_gpu(g)}）" if g else m


BACKUPS = "backups"


def backup_path(path, tag=""):
    """備份檔的位置：out/backups/<設定資料夾>/。設定資料夾裡只留現用的資料，打包或壓縮時才不會多出好幾份大檔。"""
    d = os.path.dirname(os.path.abspath(path))
    dest = os.path.join(os.path.dirname(d), BACKUPS, os.path.basename(d))
    os.makedirs(dest, exist_ok=True)
    return os.path.join(dest, os.path.basename(path) + time.strftime(".%Y%m%d-%H%M%S") + (f".{tag}" if tag else "") + ".bak")


def tidy_backups(out_dir, say=print):
    """舊版留在設定資料夾裡的 .bak 備份移到 out/backups/<設定資料夾>/（內容不動）。回傳移了幾個。"""
    n = 0
    for d in run_dirs(out_dir):
        for f in os.listdir(d):
            if f.endswith(".bak") and os.path.isfile(os.path.join(d, f)):
                dest = os.path.join(out_dir, BACKUPS, os.path.basename(d))
                os.makedirs(dest, exist_ok=True)
                target = os.path.join(dest, f)
                if os.path.exists(target):
                    target += time.strftime(".%Y%m%d-%H%M%S")
                os.replace(os.path.join(d, f), target)
                n += 1
    if n:
        say(f"整理：{n} 個備份檔（.bak）移到 {os.path.join(out_dir, BACKUPS)}（內容不變）")
    return n


def response_files(out_dir):
    """所有有內容的回覆檔，包含還沒搬移的舊格式檔。"""
    paths = [os.path.join(d, RESPONSES) for d in run_dirs(out_dir)] + [os.path.join(out_dir, RESPONSES)]
    return [p for p in paths if os.path.isfile(p) and os.path.getsize(p) > 0]


def migrate_legacy(out_dir):
    """v0.2 以前回覆直接放在 out/responses.jsonl。依每筆紀錄自己的設定分到對應的資料夾。
    舊的報告、向量快取與備份檔：只有一組設定時一起搬過去，有好幾組時放進備份資料夾。原檔改名留存。"""
    legacy = os.path.join(out_dir, RESPONSES)
    if not os.path.isfile(legacy):
        return
    groups = {}
    for r in read_jsonl(legacy):
        s = settings_of_record(r)
        groups.setdefault(run_name(s), (s, []))[1].append(r)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for s, rs in groups.values():
        d = open_run(out_dir, s)
        with open_append(os.path.join(d, RESPONSES)) as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if len(groups) == 1:
        dest = os.path.join(out_dir, next(iter(groups)))
    else:
        dest = os.path.join(out_dir, f"_舊格式備份_{stamp}")
        os.makedirs(dest, exist_ok=True)
    for n in os.listdir(out_dir):
        if n in LEGACY_EXTRAS or (n.startswith(RESPONSES + ".") and n.endswith(".bak")):
            target = os.path.join(dest, n)
            if os.path.exists(target):
                target += f".{stamp}"
            os.replace(os.path.join(out_dir, n), target)
    os.replace(legacy, os.path.join(dest, f"{RESPONSES}.{stamp}.舊格式.bak"))
    from .common import bullets
    print(bullets("已把舊格式的回覆依設定分開存放", [f"{name}（{len(rs)} 段）" for name, (_, rs) in sorted(groups.items())]
                  or ["舊檔是空的：已改名留存"]))
