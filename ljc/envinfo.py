"""執行環境的紀錄：機器代號、GPU、LM Studio 與 llama.cpp 執行環境版本、模型檔雜湊值、模型格式。

跨機器分工時，各機器必須用同一個模型檔（同量化、同雜湊值）與同一組生成設定，所以每段都要記下來。
問不到的欄位記「不明」，不會中斷執行；合併（merge）時再比對。

隱私：模型檔只記檔名，不記完整路徑（路徑裡常有使用者名稱）；這些紀錄都在 out/，不上傳。"""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess

from .common import load_json, save_json

UNKNOWN = "不明"


def machine_code(a=None):
    """機器代號：--machine 指定；或環境變數 LJC_MACHINE；都沒有就用電腦名稱。"""
    code = getattr(a, "machine", None) or os.environ.get("LJC_MACHINE") or platform.node() or "unknown"
    return re.sub(r"[^0-9A-Za-z._-]+", "-", code).strip("-") or "unknown"


def _run(cmd, timeout=8):
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        r = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip() or None
    except Exception:
        return None


def gpu_name():
    """顯示卡名稱：NVIDIA 用 nvidia-smi；Mac 用晶片名稱。"""
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if out:
        return "; ".join(l.strip() for l in out.splitlines() if l.strip())
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return UNKNOWN


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_lms_version(text):
    """從 lms version 的輸出取版本。輸出開頭是一大塊 ASCII 圖案（還帶顏色碼），舊版程式誤把圖案的第一行當成版本。
    先找 JSON 的 version 欄位，再找像 0.3.31 的版本號，再找「commit」那一行；都沒有就回傳 None。"""
    if not text:
        return None
    text = ANSI.sub("", text)
    try:
        d = json.loads(text)
        v = d.get("version") or d.get("appVersion") or d.get("cliVersion")
        if v:
            return str(v)
    except (ValueError, AttributeError):
        pass
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if re.search(r"[0-9A-Za-z]{2,}", l) and not re.fullmatch(r"[\s_/\\|()`'.,-]+", l)]
    for l in lines:
        m = re.search(r"\bv?(\d+\.\d+\.\d+(?:[-+.][\w.]+)?)", l)
        if m:
            return m.group(1)
    for l in lines:
        m = re.search(r"commit[:\s]+([0-9a-f]{6,40})", l, re.I)
        if m:
            return f"lms commit {m.group(1)}"
    return None


def clean_lms_field(v):
    """舊紀錄裡的 lmstudio 欄位：抓到 ASCII 圖案的，顯示成「未正確記錄」。"""
    if not v or v in (UNKNOWN, "mock"):
        return v
    v2 = ANSI.sub("", v)
    return v if re.search(r"\d", v2) and not re.search(r"__|/_", v2) else "未正確記錄（舊版程式抓到 lms 的圖案）"


def lmstudio_versions():
    """LM Studio 與 llama.cpp 執行環境的版本：用 LM Studio 的 lms 指令查。沒有安裝 lms 時記「不明」。"""
    ver = parse_lms_version(_run(["lms", "version", "--json"])) or parse_lms_version(_run(["lms", "version"]))
    rt = _run(["lms", "runtime", "ls"])
    runtime = UNKNOWN
    if rt:
        # 只留 llama.cpp 那幾行；有標示使用中的（✓、*、selected）優先
        lines = [l.strip() for l in rt.splitlines() if "llama" in l.lower()]
        sel = [l for l in lines if any(m in l for m in ("✓", "*", "selected", "(selected)"))]
        runtime = " | ".join(sel or lines)[:300] or UNKNOWN
    if rt:
        runtime = ANSI.sub("", runtime)
    return {"lmstudio": ver or UNKNOWN, "runtime": runtime}


def _lms_paths():
    """lms ls --json 列出的模型與檔案路徑（新版 LM Studio 才有）。"""
    out = _run(["lms", "ls", "--json"], timeout=15)
    if not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    items = data if isinstance(data, list) else data.get("models", []) if isinstance(data, dict) else []
    res = []
    for m in items:
        if isinstance(m, dict):
            key = m.get("modelKey") or m.get("path") or ""
            path = m.get("path") or ""
            res.append((key, path))
    return res


MODEL_DIRS = [os.path.join("~", ".lmstudio", "models"), os.path.join("~", ".cache", "lm-studio", "models")]


def find_model_file(model, quantization=None, explicit=None):
    """找受測模型的 GGUF 檔。優先序：--model-file 指定 → lms ls 的路徑 → 在 LM Studio 的模型資料夾裡比對名稱（找到恰好一個才算）。
    回傳 (路徑或 None, 說明)。"""
    if explicit:
        return (explicit, "--model-file 指定") if os.path.isfile(explicit) else (None, f"--model-file 找不到檔案：{explicit}")
    parts = [p for p in re.split(r"[/\\]", model.lower()) if p]
    tail = parts[-1] if parts else model.lower()
    q = (quantization or "").lower().replace("_", "")
    cands = []
    roots = [os.path.expanduser(d) for d in MODEL_DIRS]
    for key, path in _lms_paths():
        if tail in key.lower() or tail in path.lower():
            for r in roots:
                full = os.path.join(r, path)
                if os.path.isfile(full) and full.lower().endswith(".gguf"):
                    cands.append(full)
                elif os.path.isdir(full):
                    cands += [os.path.join(full, f) for f in os.listdir(full) if f.lower().endswith(".gguf")]
    if not cands:
        for r in roots:
            if not os.path.isdir(r):
                continue
            for dp, _, fs in os.walk(r):
                for f in fs:
                    low = os.path.join(dp, f).lower()
                    if f.lower().endswith(".gguf") and tail in low and "mmproj" not in f.lower():
                        cands.append(os.path.join(dp, f))
    cands = sorted(set(cands))
    if len(cands) > 1 and q:
        narrowed = [c for c in cands if q in os.path.basename(c).lower().replace("_", "")]
        cands = narrowed or cands
    if len(cands) == 1:
        return cands[0], "在 LM Studio 的模型資料夾找到"
    if not cands:
        return None, "找不到模型檔（可用 --model-file 指定 GGUF 檔的位置）"
    return None, f"找到 {len(cands)} 個可能的模型檔，無法確定是哪一個（請用 --model-file 指定）"


def file_sha256(path, cache_path=None):
    """模型檔的 SHA-256。檔案很大，算一次要幾十秒；依（檔名、大小、修改時間）快取在 out/hashes.json。"""
    st = os.stat(path)
    key = f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        try:
            cache = load_json(cache_path)
        except Exception:
            cache = {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    digest = h.hexdigest()
    if cache_path:
        cache[key] = digest
        try:
            save_json(cache_path, cache)
        except OSError:
            pass
    return digest


def gpu_memory_gb():
    """顯示記憶體（GB）：NVIDIA 用 nvidia-smi；Mac 是統一記憶體，回傳 None（不比）。讀不到也回傳 None。"""
    out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    try:
        return sum(float(x) for x in out.splitlines() if x.strip()) / 1024 if out else None
    except ValueError:
        return None


LINK_HINT = "可能是 LM Link 接來的別台模型：請求會送去別台，紀錄卻記成這台"


def fit_notes(model, quantization=None, explicit=None, what="模型"):
    """開跑前的提醒：這台找不到模型檔（可能是 LM Link 接來的）、模型檔比顯示記憶體大（會很慢）。回傳給人看的句子。"""
    path, _ = find_model_file(model, quantization, explicit)
    if not path:
        return [f"這台找不到{what} {model} 的模型檔：{LINK_HINT}；確定是這台的模型，可用 --model-file 指定 GGUF 檔"]
    size, vram = os.path.getsize(path) / 1e9, gpu_memory_gb()
    if vram and size > vram * 0.9:
        return [f"{what}檔 {size:.1f} GB，顯示記憶體 {vram:.0f} GB：放不下的部分由 CPU 算，可能慢很多；正式跑之前先用 --limit 30 試速度"]
    return []


def collect(a, model, info, out_dir, mock=False):
    """這次執行的環境紀錄。info 是 LM Studio /api/v0/models 對這個模型的資料（可能是 None）。"""
    if mock:
        return {"machine": machine_code(a), "gpu": "mock", "lmstudio": "mock", "runtime": "mock",
                "model_file": "mock", "model_sha256": "mock", "quantization": "mock", "format": "mock"}
    info = info or {}
    env = {"machine": machine_code(a), "gpu": gpu_name(), **lmstudio_versions(),
           "quantization": info.get("quantization") or UNKNOWN,
           "format": info.get("compatibility_type") or UNKNOWN, "arch": info.get("arch") or UNKNOWN}
    path, why = find_model_file(model, info.get("quantization"), getattr(a, "model_file", None))
    env["model_file"] = os.path.basename(path) if path else UNKNOWN
    env["model_file_note"] = why
    if path:
        print(f"計算模型檔雜湊值：{os.path.basename(path)}（第一次要幾十秒，之後會快取）…")
        env["model_sha256"] = file_sha256(path, os.path.join(out_dir, "hashes.json"))
    else:
        env["model_sha256"] = UNKNOWN
    return env


# LM Studio 的載入設定：會影響回覆內容的要各機器一致（規格「跨機器分工」）；只影響速度的只記錄。
LOAD_CONTENT = {"flash_attention": "Flash Attention", "speculative": "推測解碼（Speculative Decoding）",
                "k_cache": "K 快取量化", "v_cache": "V 快取量化", "unified_kv": "Unified KV Cache"}
LOAD_SPEED = ("context_length", "parallel", "eval_batch_size", "physical_batch_size", "offload_kv_cache_to_gpu")


def _norm(key, v):
    if key in ("flash_attention", "speculative", "unified_kv"):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "on", "yes", "開")
    return "f16" if v in (None, False, "", "none", "off") else str(v).strip().lower()


def load_settings(api, declared=None):
    """這次載入的設定：LM Studio /api/v1/models 讀得到的（Flash Attention、推測解碼、上下文、並行……），
    讀不到的（K／V 快取量化、Unified KV Cache）用設定檔 generation.lmstudio_load 的聲明。都沒有回傳 None。
    回傳 {"content": {鍵: {"value", "from"}}, "speed": {...}}。"""
    api = api or {}
    content = {}
    if "flash_attention" in api:
        content["flash_attention"] = api["flash_attention"]
    spec = [k for k in ("speculative_draft_mtp", "speculative_draft_simple", "speculative_draft_model") if k in api]
    if spec:
        content["speculative"] = any(bool(api[k]) for k in spec)
    for k, v in api.items():             # 新版 LM Studio 若回傳 KV 快取量化或 Unified KV 的欄位，就直接用
        lk = k.lower()
        if "unified" in lk:
            content["unified_kv"] = v
        elif "cache" in lk and ("quant" in lk or "type" in lk):
            if "k_" in lk or "_k" in lk or "key" in lk:
                content["k_cache"] = v
            elif "v_" in lk or "_v" in lk or "value" in lk:
                content["v_cache"] = v
    out = {k: {"value": _norm(k, v), "from": "LM Studio"} for k, v in content.items()}
    for k, v in (declared or {}).items():
        if k in LOAD_CONTENT and k not in out:
            out[k] = {"value": _norm(k, v), "from": "設定檔聲明"}
    speed = {k: api[k] for k in LOAD_SPEED if k in api}
    return {"content": out, "speed": speed} if out or speed else None


def load_text(load):
    """一行給人看的載入設定。"""
    if not load:
        return "讀不到（LM Studio 沒有 /api/v1/models，設定檔也沒有 generation.lmstudio_load）"
    show = lambda k, v: (("開" if v else "關") if isinstance(v, bool) else str(v).upper())
    parts = [f"{LOAD_CONTENT[k]} {show(k, x['value'])}" + ("（設定檔聲明）" if x["from"] == "設定檔聲明" else "")
             for k, x in load.get("content", {}).items()]
    sp = load.get("speed") or {}
    if sp.get("parallel"):
        parts.append(f"Max Concurrent {sp['parallel']}")
    return "｜".join(parts) or "—"


def load_lines(load):
    """載入設定，一項一行（執行前的「環境」一節用）。"""
    if not load:
        return "讀不到（LM Studio 沒有 /api/v1/models，設定檔也沒有 generation.lmstudio_load）"
    show = lambda v: ("開" if v else "關") if isinstance(v, bool) else str(v).upper()
    out = [f"{LOAD_CONTENT[k]}：{show(x['value'])}" + ("（設定檔聲明）" if x["from"] == "設定檔聲明" else "")
           for k, x in load.get("content", {}).items()]
    sp = load.get("speed") or {}
    if sp.get("parallel"):
        out.append(f"Max Concurrent：{sp['parallel']}")
    return out or "—"


def load_conflicts(prev_loads, load):
    """會影響內容的載入設定和先前不同 → 給人看的句子。prev_loads：[(機器, {鍵: {"value"}})]。"""
    cur = (load or {}).get("content") or {}
    probs = []
    for k, label in LOAD_CONTENT.items():
        if k not in cur:
            continue
        seen = {}
        for who, content in prev_loads:
            if k in (content or {}):
                seen.setdefault(json.dumps(content[k]["value"]), []).append(who)
        if seen and json.dumps(cur[k]["value"]) not in seen:
            before = "；".join(f"{'、'.join(sorted(set(w)))} 是 {json.loads(v)}" for v, w in seen.items())
            probs.append(f"LM Studio 的{label}不同：先前 {before}，這台是 {cur[k]['value']}（會影響回覆內容，各機器要一致）")
    return probs


def env_id(env):
    """環境紀錄的短代號，寫進每段對話；完整內容放在 run_meta.json 的 envs。"""
    keep = {k: v for k, v in env.items() if k != "model_file_note"}
    return hashlib.md5(json.dumps(keep, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


SKIP_SHA = {UNKNOWN, "mock", "cloud", None, ""}


def env_conflicts(prev_envs, env):
    """換機器接續時，這台的環境和這組資料先前的環境比：模型檔雜湊值、量化方式要一樣（規格「跨機器分工」）。
    回傳 (不能接受的差異, 只提醒的差異)，都是給人看的句子。"""
    probs, notes = [], []
    prev = list(prev_envs.values())
    shas = {e.get("model_sha256") for e in prev} - SKIP_SHA
    if shas and env.get("model_sha256") not in SKIP_SHA and env["model_sha256"] not in shas:
        probs.append(f"模型檔雜湊值不同：先前 {'、'.join(s[:12] for s in sorted(shas))}，這台 {env['model_sha256'][:12]}"
                     "（各機器要用同一個 GGUF 檔）")
    elif shas and env.get("model_sha256") in SKIP_SHA:
        notes.append("這台找不到模型檔、無法比對雜湊值；請確認和先前是同一個 GGUF 檔（--model-file 可指定）")
        notes.append(LINK_HINT)
    quants = {e.get("quantization") for e in prev} - SKIP_SHA
    if quants and env.get("quantization") not in SKIP_SHA and env["quantization"] not in quants:
        probs.append(f"量化方式不同：先前 {'、'.join(sorted(quants))}，這台 {env['quantization']}")
    runtimes = {e.get("runtime") for e in prev} - SKIP_SHA
    if runtimes and env.get("runtime") not in SKIP_SHA and env["runtime"] not in runtimes:
        notes.append(f"LM Studio 的執行環境版本不同：先前 {'、'.join(sorted(map(str, runtimes)))}，這台 {env['runtime']}（記在紀錄裡，報告會列出）")
    return probs, notes


def gpu_model(text):
    """顯示卡型號（去掉驅動程式版本）：「NVIDIA GeForce RTX 3090, 617.42」→「NVIDIA GeForce RTX 3090」。不明時回傳 None。"""
    if not text or text in SKIP_SHA or text.startswith("—"):
        return None
    return "; ".join(p.split(",")[0].strip() for p in str(text).split(";") if p.strip()) or None


_GPU = {}


def current_gpu():
    """這台電腦現在的顯示卡型號（只查一次）。同一台電腦換了顯示卡，規格視為不同機器。"""
    if "v" not in _GPU:
        _GPU["v"] = gpu_model(gpu_name())
    return _GPU["v"]


def short_gpu(g):
    return (g or "顯示卡未記錄").replace("NVIDIA GeForce ", "")


def same_model_name(a, b):
    """模型名稱只差前面的發行者也算同一個（例如 gpt-oss-20b 與 openai/gpt-oss-20b）。"""
    return bool(a) and bool(b) and a.split("/")[-1].lower() == b.split("/")[-1].lower()
