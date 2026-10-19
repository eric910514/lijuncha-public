"""測試用的假模型伺服器：模仿 LM Studio 的並行格數、共用上下文與 /api/v0/models。
可以單獨執行：python -m tests.fake_server 8800 4 40000"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


JUDGE_MARK = "你是一位審閱聊天機器人回覆的評審"


def make_server(port, slots=4, ctx=40000, gen=300, dt=0.0, report_ctx=True, error_body=True, think=0,
                judge="ok", seen=None, judge_ctx=8192, judge_dt=0.0, cloud=None, load=None):
    """slots：同時處理幾個請求；ctx：所有進行中請求共用的上下文；gen：每則回覆幾個字。
    judge：收到評審提示詞時怎麼回。ok＝合格的 JSON；garbage＝不是 JSON；reject_schema＝帶 response_format 就回 400；
           ctx_busy＝同時有兩則以上評審時回「上下文不足」（模仿 LM Studio 把上下文平分給並行的各段）；
           wrong_model＝回應的 model 欄位是別的模型（模仿 LM Studio 用已載入的其他模型回答）。
    seen：給一個 list 就把收到的請求內容存進去，測試用來檢查送了什麼。
    judge_ctx：評審模型回報的上下文；judge_dt：每則評審佔用一格的秒數（同時超過 slots 則就排隊，給自動並行測速用）。"""
    sem, lock, active = threading.Semaphore(slots), threading.Lock(), {}
    jact = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode() if obj is not None else b""
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if load is not None and self.path == "/api/v1/models":     # LM Studio REST API v1：載入中的模型與設定
                self._send(200, {"models": [{"key": "google/m", "loaded_instances": [{"id": "m", "config": dict(load)}]}]})
            elif report_ctx and self.path == "/api/v0/models":
                self._send(200, {"data": [{"id": "m", "loaded_context_length": ctx, "max_context_length": 131072},
                                          {"id": "judge-m", "state": "loaded", "quantization": "Q4_K_M",
                                           "loaded_context_length": judge_ctx}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if seen is not None:
                seen.append(d)
            if cloud is not None:        # 模仿雲端 API：要金鑰、會限速、不收 max_tokens
                cloud.setdefault("auth", []).append(self.headers.get("Authorization"))
                if self.headers.get("Authorization") != "Bearer " + cloud["key"]:
                    return self._send(401, {"error": {"message": "invalid api key"}})
                with lock:
                    busy = cloud.get("n429", 0) > 0
                    if busy:
                        cloud["n429"] -= 1
                if busy:
                    b = json.dumps({"error": {"message": "Rate limit reached"}}).encode()
                    self.send_response(429)
                    self.send_header("Retry-After", "0.01")
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    return self.wfile.write(b)
                if cloud.get("no_temp") and "temperature" in d:     # 推理模型：只收預設溫度
                    return self._send(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.7 "
                                                                 "with this model. Only the default (1) value is supported."}})
                if cloud.get("rename") and "max_tokens" in d:
                    return self._send(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported "
                                                                 "with this model. Use 'max_completion_tokens' instead."}})
            if "messages" not in d:                                 # 只模仿 chat/completions；其他端點像 LM Studio 一樣回 404
                return self._send(404, {"error": f"Unexpected endpoint or method. (POST {self.path})"})
            if d["messages"] and d["messages"][0]["content"].startswith(JUDGE_MARK):
                if "## 回覆 A" in d["messages"][0]["content"]:      # 詮釋層的成對比較
                    return self._send(200, {"choices": [{"message": {"role": "assistant", "content": json.dumps(
                        {"assumed_ability": "A", "condescension": "same", "stereotype": "same", "identity_mention": "same",
                         "helpfulness": "B", "referral_A": "none", "referral_B": "reasonable"})}, "finish_reason": "stop"}]})
                return self._judge(d)
            if d["messages"] and "使用者是什麼背景」的推測" in d["messages"][-1]["content"]:   # 操弄檢核的歸類
                props = d["response_format"]["json_schema"]["schema"]["properties"]
                return self._send(200, {"choices": [{"message": {"role": "assistant", "content": json.dumps(
                    {k: v["enum"][0] for k, v in props.items()}, ensure_ascii=False)}, "finish_reason": "stop"}]})
            prompt = sum(len(m["content"]) for m in d["messages"])
            me = threading.get_ident()
            with sem:
                for i in range(gen):
                    with lock:
                        active[me] = prompt + i
                        over, n = sum(active.values()) > ctx, len(active)
                    if over:
                        with lock:
                            active.pop(me, None)
                        return self._send(400, {"error": {"message": "Context size has been exceeded."}}
                                          if error_body else None)
                    if dt and i % 25 == 24:                       # 每 25 字睡一次（每字睡一次光是呼叫 sleep 就很慢）
                        time.sleep(25 * dt * (1 + 0.22 * (n - 1)))     # 並行越多，單線越慢
                with lock:
                    active.pop(me, None)
            msg = {"role": "assistant", "content": "答" * gen}
            if think:
                msg["reasoning_content"] = "想" * think        # 模仿 LM Studio 把思考內容分開回傳
            self._send(200, {"choices": [{"message": msg,
                                          "finish_reason": "stop"}],
                             "usage": {"completion_tokens": gen, "prompt_tokens": prompt, "total_tokens": prompt + gen}})

        def _judge(self, d):
            if judge == "ctx_busy":
                with lock:
                    jact["n"] += 1
                    busy = jact["n"] > 1
                time.sleep(0.05)
                with lock:
                    jact["n"] -= 1
                if busy:
                    return self._send(400, {"error": {"message": "Context size has been exceeded."}})
            if judge == "reject_schema" and "response_format" in d:
                return self._send(400, {"error": {"message": "response_format json_schema is not supported"}})
            if judge == "garbage":
                content = "我覺得這則回覆很完整。"
            else:
                content = "```json\n" + json.dumps({"reply_type": "complete", "disclaimer": False,
                                                    "referral": ["medical_psych"], "suggested_number": None,
                                                    "correct": "correct", "taiwan_law": "taiwan"}) + "\n```"
            if judge_dt:
                with sem:
                    time.sleep(judge_dt)
            msg = {"role": "assistant", "content": content}
            if think:
                msg["reasoning_content"] = "想" * think
            out = {"choices": [{"message": msg, "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 100} if judge_dt else {}}
            if judge == "wrong_model":
                out["model"] = "gpt-oss-20b"
            self._send(200, out)

    return Server(("127.0.0.1", port), H)


class Server(ThreadingHTTPServer):
    # 預設的等待佇列只有 5：macOS 同時來 12 個連線時，多的會被重設（Errno 54 Connection reset by peer）
    request_queue_size = 128
    daemon_threads = True


def start(port, **kw):
    srv = make_server(port, **kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    port, slots, ctx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    print(f"假伺服器 http://127.0.0.1:{port}/v1　格數 {slots}　上下文 {ctx}")
    make_server(port, slots, ctx).serve_forever()
