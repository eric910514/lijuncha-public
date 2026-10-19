"""實測 LM Studio 能不能回傳 token 機率（驗證與深化計畫第六節）。

在作者的機器上執行：LM Studio 載入受測模型（例如 google/gemma-4-e4b），按 Start Server，然後
    python tools/logprobs_check.py --model google/gemma-4-e4b
程式對同一個簡短的問題各試三種端點，每種送兩次，回報：
- 有沒有回傳 token 機率、每個 token 給了幾個候選；
- 兩次的機率差多少（計畫的標準：同一題重算兩次差距 < 0.001）。
原始回應存在 out/logprobs_check/<時間>/，把整個資料夾打包給程式對話看。只送題目，不送人物資料。"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTION = "台灣的法定成年年齡是幾歲？只回答阿拉伯數字。"


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, {"error": e.read().decode("utf-8", "replace")[:2000]}
        finally:
            e.close()
    except Exception as e:                  # 連不上、逾時
        return None, {"error": f"{type(e).__name__}: {e}"}


def chat_tokens(d):
    """chat/completions：choices[0].logprobs.content = [{token, logprob, top_logprobs: [...]}, ...]"""
    lp = ((d.get("choices") or [{}])[0].get("logprobs") or {})
    return [(t.get("token"), t.get("logprob"), len(t.get("top_logprobs") or [])) for t in (lp.get("content") or [])]


def completion_tokens(d):
    """舊式 completions：choices[0].logprobs = {tokens, token_logprobs, top_logprobs}"""
    lp = ((d.get("choices") or [{}])[0].get("logprobs") or {})
    toks, vals, tops = lp.get("tokens") or [], lp.get("token_logprobs") or [], lp.get("top_logprobs") or []
    return [(t, v, len(tops[i]) if i < len(tops) and tops[i] else 0) for i, (t, v) in enumerate(zip(toks, vals))]


def responses_tokens(d):
    """Responses API：output[*].content[*].logprobs = [{token, logprob, top_logprobs}, ...]"""
    out = []
    for item in d.get("output") or []:
        for c in item.get("content") or []:
            for t in c.get("logprobs") or []:
                out.append((t.get("token"), t.get("logprob"), len(t.get("top_logprobs") or [])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="LM Studio 裡的模型識別名稱")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(PROG), "out"))
    ap.add_argument("--top", type=int, default=5, help="每個 token 要幾個候選")
    a = ap.parse_args(argv)
    base = a.base_url.rstrip("/")
    msgs = [{"role": "user", "content": QUESTION}]
    tries = [
        ("chat/completions（OpenAI 相容）", base + "/chat/completions",
         {"model": a.model, "messages": msgs, "temperature": 0, "max_tokens": 64, "logprobs": True, "top_logprobs": a.top},
         chat_tokens),
        ("completions（舊式）", base + "/completions",
         {"model": a.model, "prompt": QUESTION + "\n答：", "temperature": 0, "max_tokens": 16, "logprobs": a.top},
         completion_tokens),
        ("responses（Open Responses API）", base + "/responses",
         {"model": a.model, "input": QUESTION, "temperature": 0, "max_output_tokens": 64,
          "top_logprobs": a.top, "include": ["message.output_text.logprobs"]},
         responses_tokens),
    ]
    folder = os.path.join(a.out, "logprobs_check", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    print(f"── token 機率實測：{a.model}（{base}）")
    summary = []
    for i, (name, url, body, parse) in enumerate(tries, 1):
        runs = []
        for k in (1, 2):
            code, d = post(url, body)
            with open(os.path.join(folder, f"{i}_{url[len(base) + 1:].replace('/', '_')}_{k}.json"), "w", encoding="utf-8") as f:
                json.dump({"request": body, "status": code, "response": d}, f, ensure_ascii=False, indent=1)
            runs.append((code, d, parse(d) if code == 200 else []))
        (c1, d1, t1), (c2, _, t2) = runs
        if c1 != 200:
            line = f"✗ HTTP {c1}：{str(d1.get('error'))[:160]}"
        elif not t1:
            line = "✗ 回應裡沒有 token 機率（欄位不存在或是空的）"
        else:
            diffs = [abs(x[1] - y[1]) for x, y in zip(t1, t2) if isinstance(x[1], (int, float)) and isinstance(y[1], (int, float))
                     and x[0] == y[0]]
            same = len(t1) == len(t2) and all(x[0] == y[0] for x, y in zip(t1, t2))
            first = t1[0]
            line = (f"✓ {len(t1)} 個 token 有機率，每個 {max(x[2] for x in t1)} 個候選；"
                    f"第一個 token「{first[0]}」p = {math.exp(first[1]):.4f}；"
                    + (f"兩次最大差距 {max(diffs):.6f}" if diffs and same else "兩次產生的 token 不一樣"))
        summary.append({"端點": name, "結果": line})
        print(f"  {name}：{line}")
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"原始回應：{folder}")
    print("把這個資料夾（或上面幾行）交給程式對話。")


if __name__ == "__main__":
    sys.exit(main())
