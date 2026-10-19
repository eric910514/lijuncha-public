"""共用的小工具：常數、讀寫檔、人物指紋、時間格式、讀取可用的回覆紀錄、提示的排版與名詞。"""
import hashlib
import json
import os
import sys
import time

NONE = "未提及"            # 人物沒有提到某個屬性時的群體名稱
MODES = ["揭露", "隱性"]    # 身分的兩種說法：直接講明，或只給線索


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_BAD_WARNED = set()


def read_jsonl(path):
    """一行一筆的紀錄檔。寫到一半被中斷（斷電、強制關閉、寫入中複製）的半行會略過並提醒一次，不讓整個指令當掉；
    那幾段會被當成沒做過，下次執行時重跑或重評。"""
    if not os.path.exists(path):
        return []
    out, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    bad += 1
    if bad and path not in _BAD_WARNED:
        _BAD_WARNED.add(path)
        print(bullets(f"⚠ {path}：{bad} 行讀不出來（略過）", ["多半是寫到一半被中斷", "那幾段會重跑或重評"]))
    return out


def open_append(path):
    """接著寫入紀錄檔。上次寫到一半被中斷、最後一行沒有換行時，先補一個換行，新紀錄才不會黏在半行後面。"""
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last != b"\n":
            with open(path, "ab") as f:
                f.write(b"\n")
    return open(path, "a", encoding="utf-8")


def phash(p):
    """人物的指紋。人物檔被重新產生後，舊回覆對不上指紋就不會被沿用。"""
    return hashlib.md5(p["intro"].encode("utf-8")).hexdigest()[:8]


def fmt_dur(sec):
    sec = int(max(sec, 0))
    h, m = sec // 3600, sec % 3600 // 60
    return f"{h} 小時 {m} 分" if h else (f"{m} 分 {sec % 60} 秒" if m else f"{sec} 秒")


def fmt_secs(sec):
    """「約每幾秒完成一段」用：不到 10 秒留一位小數，免得印成「每 0 秒」。"""
    return f"{sec:.1f}" if sec < 10 else f"{sec:.0f}"


def report_block(title, rows):
    """執行中的進度報告排成條列。rows 是 (項目, 內容)；內容是空的（例如讀不到顯示卡）就不列。"""
    from .i18n import label, sep
    rows = [(k, v) for k, v in rows if v]
    return "\n".join([f"  ── {label(title)}"] + [f"     {i}. {label(k)}{sep()}{v}" for i, (k, v) in enumerate(rows, 1)])


def remain_rows(per, left, all_left, unit="段"):
    """「預估剩餘」「預計完成」兩列。這次有 --limit 或 --minutes 限制時，這次與全部分開列。"""
    now = time.time()
    at = lambda n: time.strftime("%m/%d %H:%M", time.localtime(now + per * n))
    if all_left != left:
        return [("預估剩餘", f"這次 {fmt_dur(per * left)}；全部 {all_left} {unit}約 {fmt_dur(per * all_left)}"),
                ("預計完成", f"這次 {at(left)}；全部 {at(all_left)}")]
    return [("預估剩餘", fmt_dur(per * left)), ("預計完成", at(left))]


def gpu_row(g):
    return ("顯示卡", f"{g[0]}°C，使用率 {g[1]}%") if g else ("顯示卡", None)


# ---------- 名詞與排版：程式提示、報告、文件用同一套說法 ----------
# 段＝一段對話（某人物、某題、第幾次重複）；則＝一則回覆；每則上限＝max_tokens（含思考）；
# 上下文長度＝LM Studio 載入模型時的 Context Length（並行的各段平分）；設定資料夾＝out 底下每組設定的資料夾。
# 「怎麼改」：每項一行，用 bullets／note_block 列出；設定檔的項目寫出檔案位置與行號（config_spot）
def config_spot(config_path, section, key):
    """設定檔裡某一項的位置，給「怎麼改」用：config\\config.json 第 1649 行（"judge" 裡的 "max_tokens"）。
    路徑在目前資料夾底下時用相對路徑；找不到那一行時只給檔名與鍵。"""
    path = os.path.abspath(config_path)
    try:
        rel = os.path.relpath(path)
        shown = rel if not rel.startswith("..") else path
    except ValueError:                  # Windows：不同磁碟機
        shown = path
    line, indent = None, None
    try:
        with open(path, encoding="utf-8") as f:
            for i, raw in enumerate(f, 1):
                body = raw.lstrip()
                ind = len(raw) - len(body)
                if indent is None:
                    if body.startswith(f'"{section}"') and ":" in body:
                        indent = ind
                elif ind <= indent and body[:1] in "}\"":     # 區塊結束
                    break
                elif body.startswith(f'"{key}"'):
                    line = i
                    break
    except OSError:
        pass
    where = f'"{section}" 裡的 "{key}"'
    return f"{shown} 第 {line} 行（{where}）" if line else f"{shown}（{where}）"


def how_max_tokens(config_path):
    return (f"每則上限：改 {config_spot(config_path, 'generation', 'max_tokens')}（整個研究相同；建議值用 calibrate 算）",
            "只試流程：加 --draft，再用 --max-tokens 臨時指定")


def how_judge_max_tokens(config_path):
    return (f"評審的每則上限（含思考）：改 {config_spot(config_path, 'judge', 'max_tokens')}，存檔後再執行 judge",
            "改了不用全部重評：只有被截斷（judge_error）的會重評，評完沒截斷的結果不受影響")

HOW_CONTEXT = ("上下文長度：在 LM Studio 的模型載入設定改 Context Length，改完重新載入模型",
               "各機器一致：正式值寫在設定檔 generation.context_length")
HOW_JUDGE_CONTEXT = ("評審的上下文長度：在 LM Studio 的評審模型載入設定改 Context Length，改完重新載入（只影響能並行幾段）",)

def text_width(s):
    """終端機上的顯示寬度：中文字與全形標點算 2 格。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def kv_block(title, rows, indent=2):
    """一節「標題＋對齊的項目：內容」。rows 是 (項目, 內容)；內容是 None 或空字串就不列；
    內容可以是多行文字或清單，續行會對齊到冒號後面。"""
    from .i18n import label, sep
    rows = [(label(k), v) for k, v in rows if v not in (None, "", [])]
    w = max((text_width(k) for k, _ in rows), default=0)
    out = [f"── {label(title)}"] if title else []
    for k, v in rows:
        lines = v if isinstance(v, list) else str(v).split("\n")
        head = " " * indent + k + " " * (w - text_width(k)) + sep()
        out.append(head + lines[0])
        out += [" " * text_width(head) + x for x in lines[1:]]
    return "\n".join(out)


def note_block(title, items, indent=2):
    """一節條列的提醒（每項一行，前面加「・」）；沒有項目就回傳空字串。"""
    items = [x for x in items if x]
    if not items:
        return ""
    from .i18n import is_en, label
    return "\n".join([f"── {label(title)}"] + [" " * indent + ("- " if is_en() else "・") + x for x in items])


def bullets(head, items, indent=2):
    """一行標題＋每項一行（前面加「・」）。項目是空的就略過；項目可以是 (項目, 內容)，排成對齊的兩欄。
    排版規則（程式所有提示都照這個）：一行只說一件事；狀態用「項目：內容」；理由與做法分開列；指令自己一行。"""
    from .i18n import is_en, label, sep
    items = [x for x in items if x and not (isinstance(x, tuple) and x[1] in (None, "", []))]
    mark = "- " if is_en() else "・"
    pairs = [x for x in items if isinstance(x, tuple)]
    w = max((text_width(label(k)) for k, _ in pairs), default=0)
    out = [head] if head else []
    for x in items:
        if isinstance(x, tuple):
            k = label(x[0])
            out.append(" " * indent + mark + k + " " * (w - text_width(k)) + sep() + str(x[1]))
        else:
            out.append(" " * indent + mark + str(x))
    return "\n".join(out)


def stop(head, *items):
    """停下來不做：第一行「✗ 停在哪裡」，下面條列原因與怎麼做。"""
    raise SystemExit(bullets("✗ " + head, items))


def warn(head, *items):
    """要注意、但照樣繼續：第一行「⚠ 什麼事」，下面條列細節與怎麼做。"""
    print(bullets("⚠ " + head, items), flush=True)


class Progress:
    """長時間的批次工作（續寫、操弄檢核）：每完成約 5%（或每 2 分鐘）印一行
      [續寫 ██████·············· 12/40 30%]｜已用 1 分 2 秒｜約剩 2 分 30 秒｜失敗 1"""

    def __init__(self, label, total, every=120, width=20):
        self.label, self.total, self.every, self.width = label, max(int(total), 0), every, width
        self.n = self.err = 0
        self.t0 = self.last = time.time()
        self.step = max(1, -(-self.total // 20))          # 約每 5% 一行

    def line(self):
        n, tot = self.n, max(self.total, 1)
        fill = int(self.width * n / tot)
        el = time.time() - self.t0
        left = self.total - n
        return (f"  [{self.label} {'█' * fill}{'·' * (self.width - fill)} {n}/{self.total} {n / tot:.0%}]"
                f"｜已用 {fmt_dur(el)}" + (f"｜約剩 {fmt_dur(el / max(n, 1) * left)}" if left and n else "")
                + (f"｜失敗 {self.err}" if self.err else ""))

    def tick(self, ok=True, k=1):
        before = self.n
        self.n += k
        self.err += 0 if ok else k
        now = time.time()
        if self.n >= self.total or self.n // self.step != before // self.step or now - self.last >= self.every:
            self.last = now
            print(self.line(), flush=True)


def ask(prompt):
    """在終端機問作者一句，回傳去掉空白、轉小寫的回答；讀不到輸入（EOF）回傳 None。
    呼叫前先用 sys.stdin.isatty() 判斷能不能問；但 Windows 的 NUL（例如背景行程的輸入）也算終端機，所以還要接 EOF。"""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return None


def next_steps(items):
    """每個階段結束（完成或中途停下）時印出的「下一步」：接下來要執行的指令。"""
    return note_block("下一步", items)


def cmd_hint(*args):
    """下一步裡的指令，前面固定加 python lijuncha.py。"""
    return "python lijuncha.py " + " ".join(str(x) for x in args if x)


def followup_texts(q, rounds):
    """這一題在 rounds 輪對話中會送出的追問：兩輪只送 specify，三輪再加 extend。
    舊格式的追問是純文字清單，依序取用。"""
    fs = q.get("followups", [])
    if fs and isinstance(fs[0], str):
        return list(fs)[:max(rounds - 1, 0)]
    kinds = ["specify"] if rounds <= 2 else ["specify", "extend"]
    return [f["text"] for f in fs if f.get("type") in kinds][:max(rounds - 1, 0)]


def qhash(q, turns):
    """題目的指紋：首輪提問加上實際會問到的追問。改了題目文字但沿用題號時，舊回覆就不會被當成這題的答案。"""
    text = "\n".join([q["q"]] + followup_texts(q, turns))
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def retired_hashes(questions, turns):
    """換過題目文字的舊版（設定檔 questions[].previous）：{(題號, 舊指紋): 舊版題庫版本}。
    這些回覆保留在資料夾裡、不算過期也不重跑，只在報告「只描述」一節列出。"""
    out = {}
    for q in questions or []:
        for prev in q.get("previous") or []:
            out[(q["id"], qhash(prev, turns))] = prev.get("questions_version", "舊版")
    return out


def load_retired(path, questions, turns):
    """舊版題目的回覆紀錄（見 retired_hashes）。"""
    old = retired_hashes(questions, turns)
    return [r for r in read_jsonl(path) if (r.get("qid"), r.get("qh")) in old] if old else []


def done_key(r):
    """一段對話的鍵：人物、版本、題號、重複序號、寫法。"""
    return (*pkey(r), r["qid"], r["rep"], r.get("style", "orig"))


def pkey(r):
    """人物紀錄或回覆紀錄的鍵：同一位人物的揭露、隱性兩版是不同的鍵。"""
    return (r.get("pid", r.get("id")), r.get("variant"))


def load_valid(path, personas, turns, questions=None, expect=None):
    """回傳 (可用紀錄, 空回覆筆數, 人物或題目不符筆數)。
    給了 questions 就同時檢查題目指紋；舊紀錄沒有指紋的照常採用。
    expect：{(題號, 寫法): 指紋}，對照模組換了寫法時用；沒給就是原文的指紋。"""
    ph = {pkey(p): phash(p) for p in personas}
    qh = expect if expect is not None else {(q["id"], "orig"): qhash(q, turns) for q in questions or []}
    old = retired_hashes(questions, turns)
    good, empty, stale = [], 0, 0
    for r in read_jsonl(path):
        if old and (r.get("qid"), r.get("qh")) in old:
            continue                     # 舊版題目：不是好紀錄，也不算過期（不會被清掉）
        reps = r.get("replies") or []
        if len(reps) < turns or any(not (t or "").strip() for t in reps):
            empty += 1
        elif pkey(r) not in ph or ("ph" in r and r["ph"] != ph[pkey(r)]):
            stale += 1
        elif "qh" in r and (r["qid"], r.get("style", "orig")) in qh and r["qh"] != qh[(r["qid"], r.get("style", "orig"))]:
            stale += 1
        else:
            good.append(r)
    return good, empty, stale


def save_json(path, obj, indent=1):
    """先寫到暫存檔再換名：寫到一半斷電或被關掉時，原本的檔案仍然完整。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


# ---------- 版本 ----------

def file_fp(path):
    """檔案內容的指紋（8 碼）；讀不到回傳「不明」。設定檔或提示詞改過，指紋就會不同。"""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except OSError:
        return "不明"


def text_fp(text):
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8]


def persona_gen_version(pdir):
    """人物檔產生時的程式版本（personas_meta.json）；沒有紀錄回傳 None。"""
    p = os.path.join(pdir, "personas_meta.json")
    if not os.path.exists(p):
        return None
    b = load_json(p).get("batches") or []
    vs = list(dict.fromkeys(f"v{x['program_version']}" for x in b if x.get("program_version")))
    return "、".join(vs) or None


# ---------- 設定檔與保留題 ----------
# 保留題（企劃書 10-19）：題目全文、追問、同義改寫與標準答案不公開，放在不進版本控制的私有檔；
# 設定檔只記題號、檔名與 SHA-256（公開版 README 的雜湊承諾）。有這個檔就合併進題庫，沒有就只用公開題。
RESERVED_FILE = os.path.join("private", "reserved_questions.json")


def text_sha256(path):
    """文字檔內容的 SHA-256；換行一律當成 \\n 算（Windows 的 Git 取出時可能換成 \\r\\n，內容其實一樣）。"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def reserved_path(cfg, config_path):
    r = cfg.get("reserved_questions") or {}
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), r.get("file") or RESERVED_FILE)


def read_config(config_path):
    """讀設定檔，有保留題私有檔就把保留題合併進 questions（依 question_order 排回原本的順序）。
    cfg["_reserved"] 記下：保留題號、私有檔位置、有沒有讀到、實際的 SHA-256 與設定檔記的承諾值、英文題目標題。"""
    cfg = load_json(config_path)
    r = cfg.get("reserved_questions")
    if not r:
        return cfg
    p = reserved_path(cfg, config_path)
    info = {"ids": list(r.get("ids") or []), "path": p, "loaded": False, "sha256": None,
            "expected": r.get("sha256"), "terms_en": {}}
    if os.path.isfile(p):
        priv = load_json(p)
        have = {q["id"] for q in cfg["questions"]}
        merged = cfg["questions"] + [q for q in priv.get("questions") or [] if q["id"] not in have]
        order = cfg.get("question_order") or []
        merged.sort(key=lambda q: order.index(q["id"]) if q["id"] in order else len(order))
        cfg["questions"] = merged
        info.update(loaded=True, sha256=text_sha256(p), terms_en=priv.get("terms_en") or {})
    cfg["_reserved"] = info
    return cfg


def question_scope(cfg):
    """題庫範圍的說明：公開幾題、保留幾題；沒有保留題檔時寫明只用公開題。"""
    r = cfg.get("_reserved")
    n = len(cfg.get("questions") or [])
    if not r:
        return f"{n} 題"
    ids = "、".join(r["ids"])
    if not r["loaded"]:
        return f"只有公開的 {n} 題（這台沒有保留題檔：{ids} 不跑、不評、不分析）"
    return f"{n} 題（公開 {n - len(r['ids'])} 題＋保留 {len(r['ids'])} 題：{ids}）"


def question_scope_en(cfg):
    """question_scope 的英文版（英文報告用）。"""
    r = cfg.get("_reserved")
    n = len(cfg.get("questions") or [])
    if not r:
        return f"{n} questions"
    ids = ", ".join(r["ids"])
    if not r["loaded"]:
        return f"public questions only ({n}; reserved file not present: {ids} not used)"
    return f"{n} questions ({n - len(r['ids'])} public + {len(r['ids'])} reserved: {ids})"


def reserved_notes(cfg):
    """保留題要提醒的事：沒有私有檔、或私有檔和設定檔記的雜湊不同。"""
    r = cfg.get("_reserved")
    if not r:
        return []
    if not r["loaded"]:
        return [f"沒有保留題檔（{r['path']}）：只用公開題；作者的資料要放好這個檔再跑"]
    if r["expected"] and r["sha256"] != r["expected"]:
        return [f"保留題檔的 SHA-256（{r['sha256'][:12]}…）和設定檔記的承諾值（{r['expected'][:12]}…）不同："
                "題目被改過，或不是定稿的那一份"]
    return []


def guard_reserved(cfg, path, what):
    """回覆檔裡有保留題、這台卻沒有保留題檔時停下（只掃 qid，不解析整個檔）。"""
    r = cfg.get("_reserved")
    if not r or r["loaded"] or not os.path.isfile(path):
        return
    import re
    pat = re.compile(r'"qid":\s*"(' + "|".join(re.escape(x) for x in r["ids"]) + r')"')
    found = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                found.add(m.group(1))
                if len(found) == len(r["ids"]):
                    break
    require_reserved(cfg, found, what)


def require_reserved(cfg, qids, what):
    """資料裡有保留題的回覆、這台卻沒有保留題檔時停下：不然那幾題會被默默略過，結果只剩一部分。"""
    r = cfg.get("_reserved")
    if not r or r["loaded"]:
        return
    used = sorted({q for q in qids if q in set(r["ids"])})
    if used:
        stop(f"這組資料有保留題（{'、'.join(used)}）的回覆，但這台沒有保留題檔，{what}會漏掉這幾題",
             f"把作者給的 reserved_questions.json 放到 {r['path']}",
             "只想用公開題：另外用一個新的 out 資料夾")


def version_rows(cfg, config_path, pver=None, pdir=None, sentences=None, extra=()):
    """執行提示的「版本」一節：程式、設定檔、企劃書、題庫、人物、人物句，以及各指令自己的（例如評審提示詞）。
    pver：這次用的人物版本（不給就用設定檔的）；sentences：人物句版本（用舊版人物時沿用那組資料記的）。"""
    from . import __version__
    v = cfg.get("version") or {}
    pv = pver or v.get("personas")
    gen = persona_gen_version(pdir) if pdir else None
    rows = [("程式", f"v{__version__}"),
            ("設定檔", f"{os.path.basename(config_path)}（指紋 {file_fp(config_path)}）"),
            ("企劃書", v.get("plan") or "未記錄"),
            ("題庫", f"{v.get('questions') or '未記錄'}｜{question_scope(cfg)}"),
            ("人物", (pv or "未記錄") + (f"（人物檔由 {gen} 產生）" if gen else "")
             + ("（配對確認用）" if pv and "-c-" in pv else
                "（不是設定檔目前的版本）" if pv and v.get("personas") and pv != v.get("personas") else "")),
            ("人物句", sentences or v.get("persona_sentences") or "未記錄")]
    return rows + list(extra)


def version_line(cfg, config_path, *extra):
    """一行的版本摘要（interpret、probe、confirm 用）。"""
    from . import __version__
    v = cfg.get("version") or {}
    parts = [f"程式 v{__version__}", f"設定檔指紋 {file_fp(config_path)}", f"題庫 {v.get('questions') or '未記錄'}",
             f"人物句 {v.get('persona_sentences') or '未記錄'}", *extra]
    return "版本：" + "｜".join(x for x in parts if x)


DEFAULT_BLOCKED = ["qwen", "qwq", "deepseek", "glm", "chatglm", "baichuan", "internlm", "minimax", "kimi", "moonshot",
                   "hunyuan", "ernie", "doubao", "yi", "bge", "gte", "m3e", "text2vec"]


def blocked_origin(cfg, name):
    """模型名稱裡有中國開發的模型家族就回傳那個名稱（設定檔 model_origin.blocked），否則 None。
    名稱以 / - _ . : 與空白分段，任一段等於名單裡的名稱、或名稱後面直接接數字（例如 qwen3、glm4）就算。"""
    import re
    if not name or name == "mock":
        return None
    blocked = (cfg.get("model_origin") or {}).get("blocked", DEFAULT_BLOCKED)
    for part in re.split(r"[/\-_.:\s@]+", name.lower()):
        for b in blocked:
            if part == b or (part.startswith(b) and part[len(b):].isdigit()):
                return b
    return None


def check_origin(cfg, *names, what="模型"):
    """有中國開發的模型就停（作者 10-16 決定：不使用中國開發的模型、框架、嵌入與套件）。"""
    for n in names:
        b = blocked_origin(cfg, n)
        if b:
            stop(f"{what}「{n}」屬於 {b} 家族（中國開發）", "作者 10-16 決定：不使用中國開發的模型、框架與嵌入",
                 "換別的模型", "名單：設定檔 model_origin.blocked")


# ---------- 長時間執行：不讓電腦睡著、睡著了也不算進速度 ----------
class PauseClock:
    """偵測程式被暫停（電腦休眠、筆電闔上）。主迴圈每秒至少醒一次；兩次之間的時間跳超過 GAP 秒，就是整台被暫停了。
    暫停的時間不算進已耗時、速度與預估；跨過暫停的請求，耗時不能拿來測速（作者 10-19：休眠後預估變成 602 小時）。"""
    GAP = 60

    def __init__(self):
        self.last = time.time()
        self.paused = 0.0          # 這次執行累計暫停了幾秒
        self.resumed_at = 0.0      # 最近一次暫停結束的時間

    def tick(self):
        """主迴圈每一圈呼叫。剛從暫停醒來時回傳暫停的秒數，否則回傳 0。"""
        now = time.time()
        gap, self.last = now - self.last, now
        if gap > self.GAP:
            self.paused += gap
            self.resumed_at = now
            return gap
        return 0.0

    def active(self, t0):
        """從 t0 到現在、扣掉暫停的秒數。"""
        return max(0.0, time.time() - t0 - self.paused)

    def spans_pause(self, secs):
        """一個剛結束、花了 secs 秒的請求是不是跨過了暫停（它的耗時含睡著的時間）。"""
        return bool(self.resumed_at) and time.time() - secs < self.resumed_at


class keep_awake:
    """執行期間不讓系統進入睡眠（螢幕照樣可以關）；指令結束、Ctrl+C 或當掉都會恢復原本的電源設定。
    Windows 用 SetThreadExecutionState；macOS 用系統內建的 caffeinate（筆電闔上蓋子仍會睡）；其他系統不做事。"""
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __init__(self, enabled=True):
        self.enabled, self.how, self._proc = enabled, None, None

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            if os.name == "nt":
                import ctypes
                if ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED):
                    self.how = "Windows 電源設定"
            elif sys.platform == "darwin":
                import shutil
                import subprocess
                exe = shutil.which("caffeinate")
                if exe:
                    self._proc = subprocess.Popen([exe, "-i", "-w", str(os.getpid())],
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.how = "caffeinate（筆電闔上蓋子仍會睡）"
        except Exception:                 # 做不到就算了：只影響會不會睡著，不影響結果
            self.how = None
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt" and self.how:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            if self._proc:
                self._proc.terminate()
        except Exception:
            pass
        return False

    def row(self):
        """給開頭設定表的一列。"""
        if not self.enabled:
            return ("防止休眠", None)              # kv_block 不列內容是 None 的項目
        return ("防止休眠", f"開（{self.how}；結束就恢復，螢幕照樣可以關）" if self.how
                else "這台做不到：長時間執行請在系統設定關掉自動睡眠")
