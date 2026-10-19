"""與模型伺服器溝通（OpenAI 相容 API），以及讀取顯示卡狀態。這個檔案不含任何實驗邏輯。"""
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np


class ServerError(RuntimeError):
    """伺服器回了錯誤。capacity=True 表示是容量問題（上下文不夠、太忙），降低並行可能就會好。"""
    def __init__(self, msg, capacity=False, context=False):
        super().__init__(msg)
        self.capacity = capacity
        self.context = context


# 雲端 API 的金鑰：只放在記憶體裡（從環境變數讀進來，見 cloud.py），不寫進任何檔案或紀錄
_KEYS = {}
# 雲端服務不接受的欄位：遇到一次就記下來，之後同一個網址都照改（例如 max_tokens 改名、不收 seed）
_COMPAT = {}
CLOUD_RETRIES = 6


def set_key(base, key):
    _KEYS[base.rstrip("/")] = key


def compat(base):
    """這個網址自動改寫過哪些欄位（記進 run_meta.json）。"""
    return dict(_COMPAT.get(base.rstrip("/"), {}))


def is_cloud(base):
    return base.rstrip("/") in _KEYS


def _hide(text, base):
    """錯誤訊息裡萬一出現金鑰，換成 ***。"""
    k = _KEYS.get(base.rstrip("/"))
    return text.replace(k, "***") if k else text


def _headers(base):
    return {"Content-Type": "application/json", "Authorization": "Bearer " + _KEYS.get(base.rstrip("/"), "local")}


def chat_raw(base, model, messages, temperature, max_tokens, seed, timeout=900, extra=None):
    """extra：其他要送的欄位，例如評審用的 response_format（結構化輸出）。
    雲端 API（有金鑰的網址）：碰到 429 依 Retry-After 等一下再送；不收某個欄位時自動改寫後重送一次。"""
    key = base.rstrip("/")
    for attempt in range(CLOUD_RETRIES if key in _KEYS else 1):
        fix = _COMPAT.get(key, {})
        body = {"model": model, "messages": messages,
                ("max_completion_tokens" if fix.get("rename_max") else "max_tokens"): max_tokens}
        if not fix.get("no_temperature"):      # 推理模型不收溫度：改用供應商預設值（記在 run_meta.json 的 compat）
            body["temperature"] = temperature
        if seed is not None and not fix.get("no_seed"):
            body["seed"] = seed
        body.update(extra or {})
        try:
            return _post(base, body, timeout)
        except ServerError as e:
            if key not in _KEYS:
                raise
            low = str(e).lower()
            if "http 429" in low or "http 503" in low or "http 529" in low:
                time.sleep(min(getattr(e, "retry_after", None) or 2 ** (attempt + 1), 60))
                continue
            # 依這次實際送出的欄位判斷（不看共用的旗標）：並行時別的請求可能剛設好旗標，這則還是用舊寫法送的，要重送
            if "http 400" in low and "max_completion_tokens" in low and "max_tokens" in body:
                _COMPAT.setdefault(key, {})["rename_max"] = True
                continue
            if "http 400" in low and "temperature" in low and "temperature" in body:
                _COMPAT.setdefault(key, {})["no_temperature"] = True
                continue
            if "http 400" in low and "seed" in low and "seed" in body:
                _COMPAT.setdefault(key, {})["no_seed"] = True
                continue
            raise
    raise ServerError("雲端 API 一直回「太忙或超過速率限制」（HTTP 429／503），重試幾次都沒成功；請稍後再跑或降低 --workers。", True)


def _post(base, body, timeout):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers=_headers(base))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = ""
        retry_after = None
        try:
            retry_after = float(e.headers.get("Retry-After") or 0) or None
        except (TypeError, ValueError, AttributeError):
            pass
        try:
            raw = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(raw).get("error", raw)
                detail = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            except (ValueError, AttributeError):
                detail = raw
        except Exception:
            pass
        finally:
            e.close()          # 錯誤回應也要關掉連線，否則 Python 會提出 ResourceWarning
        detail = _hide(detail, base)
        low = detail.lower()
        ctx = "context" in low or "kv cache" in low or "n_ctx" in low
        cap = ctx or e.code in (429, 500, 502, 503, 504) or "channel error" in low or "busy" in low
        # LM Studio 在上下文不足時回 400，有時不帶說明；並行時的 400 先當作容量問題
        if e.code == 400 and not detail.strip():
            cap = True
        err = ServerError(f"HTTP {e.code}：{detail.strip()[:300] or '伺服器沒有說明原因'}", cap, ctx)
        err.retry_after = retry_after
        raise err from None
    except (TimeoutError, ConnectionResetError) as e:
        raise ServerError(f"連線逾時或被重設：{e}", True) from None


def chat(base, model, messages, temperature, max_tokens, seed):
    """回傳 (回覆文字, 結束原因, usage, 思考字數)。回覆是空的就直接報錯，不默默存檔。
    思考字數是伺服器另外回傳的思考內容（reasoning_content）長度；伺服器沒有把思考分開回傳時是 0。"""
    d = chat_raw(base, model, messages, temperature, max_tokens, seed)
    ch = d["choices"][0]
    msg = ch.get("message", {})
    text = (msg.get("content") or "").strip()
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    finish = ch.get("finish_reason")
    if not text:
        why = (f"模型把 {max_tokens} 個 token 全用在思考（思考內容 {len(think)} 字）"
               if think else "伺服器沒有回傳任何文字")
        raise RuntimeError(f"回覆是空的：{why}，結束原因 {finish}。"
                           "請加大 --max-tokens，或在 LM Studio 關閉該模型的思考功能；"
                           "可先用 check 指令看原始回應")
    return text, finish, d.get("usage") or {}, len(think)


def loaded_context(base, model):
    """問 LM Studio 這個模型載入時的上下文長度。問不到（例如 Ollama）就回傳 None。"""
    root = base.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(root + "/api/v0/models", timeout=5) as r:
            for m in json.load(r).get("data", []):
                if m.get("id") == model and m.get("loaded_context_length"):
                    return int(m["loaded_context_length"]), int(m.get("max_context_length") or 0)
    except urllib.error.HTTPError as e:
        e.close()                # 伺服器沒有這個網址（例如 Ollama）：關掉回應，免得出現 ResourceWarning
    except Exception:
        pass
    return None


def load_config(base, model):
    """LM Studio REST API v1（/api/v1/models）裡這個模型「正在跑」的載入設定（loaded_instances[].config），
    例如 flash_attention、speculative_draft_*、context_length、parallel。問不到（舊版 LM Studio、不是 LM Studio）回傳 None。"""
    root = base.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(root + "/api/v1/models", timeout=5) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        e.close()
        return None
    except Exception:
        return None
    tail = lambda s: str(s or "").split("/")[-1].lower()
    for m in (d.get("models") or d.get("data") or []) if isinstance(d, dict) else []:
        for inst in m.get("loaded_instances") or []:
            ids = {x for x in (inst.get("id"), inst.get("identifier"), m.get("key"), m.get("id")) if x}
            if model in ids or any(tail(x) == tail(model) for x in ids):
                return dict(inst.get("config") or {})
    return None


def list_models(base):
    """LM Studio 的模型清單（/api/v0/models）；問不到（例如不是 LM Studio）回傳 None。"""
    root = base.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(root + "/api/v0/models", timeout=5) as r:
            return json.load(r).get("data", [])
    except urllib.error.HTTPError as e:
        e.close()
        return None
    except Exception:
        return None


def model_info(base, model):
    """LM Studio 的模型資訊（/api/v0/models）：量化方式、是否已載入、上下文長度。問不到就回傳 None。"""
    root = base.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(root + "/api/v0/models", timeout=5) as r:
            data = json.load(r).get("data", [])
    except urllib.error.HTTPError as e:
        e.close()
        return None
    except Exception:
        return None
    for m in data:
        if m.get("id") == model:
            return m
    return {}


def api_embed(base, model, texts, timeout=600):
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/embeddings", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return [np.array(x["embedding"]) for x in sorted(d["data"], key=lambda x: x["index"])]


_SMI = {"ok": None}


def gpu_status():
    """回傳 (溫度°C, 使用率%)；沒有 NVIDIA 顯示卡或 nvidia-smi 就回傳 None。"""
    if _SMI["ok"] is False:
        return None
    exe = shutil.which("nvidia-smi")
    if not exe:
        _SMI["ok"] = False
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=4).stdout.strip().splitlines()
        vals = [tuple(int(float(x)) for x in line.split(",")) for line in out if line.strip()]
        _SMI["ok"] = True
        return max(v[0] for v in vals), max(v[1] for v in vals)
    except Exception:
        _SMI["ok"] = False
        return None
