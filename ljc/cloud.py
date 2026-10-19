"""雲端模型（F 批）：經各家的 OpenAI 相容 API 提問（規格「雲端模型」）。

  run --provider openai --model gpt-4.1-mini            金鑰從環境變數 OPENAI_API_KEY 讀
  run --provider custom --base-url <網址> --key-env 變數名稱

- 金鑰只從環境變數讀，只放在記憶體：不寫進設定檔、settings.json、run_meta.json、回覆紀錄、輸出資料夾或版本控制，
  錯誤訊息裡萬一出現也會換成 ***。
- 不做 LM Studio 專屬的檢查（上下文長度、模型檔雜湊值、量化方式）；並行數固定（設定檔 cloud.providers.<名稱>.workers）。
- 碰到 429（超過速率限制）依 Retry-After 等一下再送；服務不收 max_tokens 或 seed 時自動改寫（記在 run_meta.json）。
- 記錄呼叫次數與 token 用量；費用依設定檔 cloud.prices 的單價算。
- 預算上限：設定檔 cloud.budget_usd（預設總額 150 美元，所有雲端設定資料夾合計）。到上限就不再派新的對話，
  手上的跑完就停；沒填單價（美元）的模型不能跑，因為算不出花了多少。
- 不用中國開發的模型（作者 10-16 決定）：設定檔沒有 DeepSeek；--model 是 model_origin.blocked 裡的家族時拒絕執行。
- 大量測試前要先讀各家服務條款，確認允許評測用途（企劃書）；讀過後在 cloud.providers.<名稱>.tos_checked 填日期。"""
import os

from . import client
from .common import stop, warn

ENV_HINT = {"nt": "set {var}=你的金鑰（PowerShell：$env:{var}=\"你的金鑰\"）", "posix": "export {var}=你的金鑰"}


def providers(cfg):
    return {k: v for k, v in ((cfg.get("cloud") or {}).get("providers") or {}).items() if not k.startswith("_")}


def setup(a, cfg):
    """--provider 有給時：換成那家的網址、從環境變數讀金鑰、固定並行數。回傳服務名稱（沒給時 None）。"""
    name = getattr(a, "provider", None)
    if not name:
        return None
    if getattr(a, "mock", False):
        stop("--provider 不能和 --mock 一起用")
    ps = providers(cfg)
    if name == "custom":
        if not getattr(a, "key_env", ""):
            stop("--provider custom 少了設定", "--key-env <放金鑰的環境變數名稱>", "--base-url <網址>")
        p = {"base_url": a.base_url, "key_env": a.key_env}
    elif name in ps:
        p = dict(ps[name])
        if getattr(a, "key_env", ""):
            p["key_env"] = a.key_env
    else:
        stop(f"設定檔 cloud.providers 裡沒有「{name}」", f"可用：{'、'.join(ps) or '（空的）'}", "或用 custom")
    var = p["key_env"]
    key = os.environ.get(var, "").strip()
    if not key:
        stop(f"找不到環境變數 {var}", "金鑰只從環境變數讀，不放設定檔",
             "先在這個終端機執行：" + ENV_HINT.get(os.name, ENV_HINT["posix"]).format(var=var))
    a.base_url = p["base_url"]
    client.set_key(a.base_url, key)
    if str(getattr(a, "workers", "auto")).lower() == "auto":
        a.workers = str(p.get("workers", 4))
    a.cloud = {"provider": name, "base_url": a.base_url, "tos_checked": p.get("tos_checked")}
    if not p.get("tos_checked"):
        warn(f"還沒記錄讀過 {name} 的服務條款", "大量測試前先確認允許評測用途",
             f"讀過後：設定檔 cloud.providers.{name}.tos_checked 填上日期（企劃書「模型」一節）")
    return name


def env_record(a, provider):
    """雲端模型的環境紀錄（取代 LM Studio 的模型檔、量化與顯示卡）。"""
    from .envinfo import machine_code
    return {"machine": machine_code(a), "gpu": "—（雲端）", "lmstudio": "—（雲端）", "runtime": "—",
            "model_file": f"雲端 API（{provider}）", "model_sha256": "cloud", "quantization": "不公開", "format": "cloud",
            "model_file_note": "雲端模型沒有模型檔"}


def usage(recs):
    """回覆紀錄的呼叫次數與 token 用量：{calls, input, output}。每一輪是一次呼叫，輸入含前面幾輪的對話。"""
    calls = inp = out = 0
    for r in recs:
        calls += len(r.get("replies") or [])
        inp += sum(x or 0 for x in r.get("prompt_tokens") or [])
        out += sum(x or 0 for x in r.get("turn_tokens") or [])
    return {"calls": calls, "input": inp, "output": out}


def cost(cfg, model, u):
    """依設定檔 cloud.prices.<模型> 的單價（每百萬 token）算費用。沒填單價回傳 None。"""
    pr = ((cfg.get("cloud") or {}).get("prices") or {}).get(model)
    if not pr or pr.get("input") is None or pr.get("output") is None:
        return None
    return {"amount": u["input"] / 1e6 * float(pr["input"]) + u["output"] / 1e6 * float(pr["output"]),
            "currency": pr.get("currency", "USD"), "as_of": pr.get("as_of")}


def usage_line(cfg, model, recs):
    u = usage(recs)
    c = cost(cfg, model, u)
    s = f"呼叫 {u['calls']} 次｜輸入 {u['input']:,} token｜輸出 {u['output']:,} token"
    if c:
        s += f"｜費用約 {c['amount']:.2f} {c['currency']}（設定檔 cloud.prices，單價日期 {c['as_of'] or '未填'}）"
    else:
        s += f"｜費用：設定檔 cloud.prices 沒有 {model} 的單價"
    return s, u, c


def budget(cfg):
    b = (cfg.get("cloud") or {}).get("budget_usd")
    return None if b is None else float(b)


def rec_cost(cfg, model, rec):
    c = cost(cfg, model, usage([rec]))
    return c["amount"] if c else 0.0


def spent(out_dir, cfg):
    """所有雲端設定資料夾到目前的花費（美元）。回傳 (金額, 沒有單價的模型)。"""
    from . import layout
    from .common import read_jsonl
    total, unpriced = 0.0, set()
    for d in layout.run_dirs(out_dir):
        s = layout.read_settings(d)
        if not s.get("provider"):
            continue
        recs = read_jsonl(os.path.join(d, layout.RESPONSES))
        by = {}
        for r in recs:
            by.setdefault(r.get("model") or s.get("model"), []).append(r)
        for m, rs in by.items():
            c = cost(cfg, m, usage(rs))
            if c is None or c.get("currency", "USD") != "USD":
                unpriced.add(m)
            else:
                total += c["amount"]
    return total, sorted(unpriced)


def check_budget(a, cfg, model):
    """執行前：這個模型要有美元單價；已花的錢還沒到上限。回傳 (上限, 已花)；沒設上限回傳 (None, 已花)。"""
    b = budget(cfg)
    pr = ((cfg.get("cloud") or {}).get("prices") or {}).get(model) or {}
    if b is not None and (pr.get("input") is None or pr.get("output") is None or pr.get("currency", "USD") != "USD"):
        stop(f"設定檔 cloud.prices 沒有 {model} 的美元單價", f"算不出花費：預算上限（{b:g} 美元）沒辦法執行",
             "查官方價目表，填上 input、output（每百萬 token）與 as_of")
    used, unpriced = spent(a.out, cfg)
    if unpriced:
        warn("先前的花費沒算進預算（沒有美元單價）", *unpriced)
    if b is not None and used >= b:
        stop("雲端預算已用完", f"已花：約 {used:.2f} 美元", f"上限：{b:g} 美元（設定檔 cloud.budget_usd）")
    return b, used
