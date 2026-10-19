"""python -m tests 的內容：跑全部測試，顯示進度條。

- 同時開幾個行程分頭跑（預設 CPU 核心數，最多 4 個；--jobs 1 就是一個一個跑）。
- 每個行程用不同的埠開假伺服器，不會互相干擾。
- 上次各測試的耗時記在 tests/.timings.json（不進版本控制），這次先派最慢的，總時間比較短。
- 只跑名稱含某段文字的測試：python -m tests continue（可以給好幾段）。
python -m unittest 照樣可以用（沒有進度條，一個一個跑）。"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import os
import sys
import time
import traceback
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TIMINGS = os.path.join(HERE, ".timings.json")
PORT_BASE = 18000
PORT_STEP = 250          # 每個行程分到的埠：18000、18250、18500……（測試用的偏移量最多到一百多）


def all_ids(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from all_ids(t)
        else:
            yield t.id()


def init_worker(counter):
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    os.environ["LJC_TEST_PORT"] = str(PORT_BASE + idx * PORT_STEP)
    # 每個行程的 numpy 只用一條執行緒：好幾個行程各自再開多條執行緒，反而互相搶 CPU、變慢
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    # 工作行程的輸入接到空的：Windows 上 multiprocessing 給的 NUL 會被 isatty() 當成終端機，程式就會停下來等人回答
    sys.stdin = io.StringIO("")


def run_one(tid):
    """在工作行程裡跑一個測試。回傳 (代號, 結果, 秒數, 錯誤說明)；測試印的東西不顯示，失敗時附上。"""
    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            suite = unittest.defaultTestLoader.loadTestsFromName(tid)
            res = unittest.TestResult()
            suite.run(res)
    except Exception:
        return tid, "error", time.time() - t0, traceback.format_exc()
    bad = res.errors + res.failures
    if bad:
        detail = "\n".join(tb for _, tb in bad)
        out = buf.getvalue().strip()
        if out:
            detail += "\n── 測試印出的內容（最後 40 行）\n" + "\n".join(out.splitlines()[-40:])
        return tid, "fail", time.time() - t0, detail
    if res.skipped:
        return tid, "skip", time.time() - t0, res.skipped[0][1]
    return tid, "ok", time.time() - t0, ""


def fmt(secs):
    secs = int(round(secs))
    return f"{secs // 60} 分 {secs % 60:02d} 秒" if secs >= 60 else f"{secs} 秒"


def bar(done, total, bad, t0, est_left, width=28):
    filled = int(width * done / max(total, 1))
    left = f"，約剩 {fmt(est_left)}" if est_left is not None and done < total else ""
    mark = f"｜失敗 {bad}" if bad else ""
    return f"[{'█' * filled}{'·' * (width - filled)}] {done}/{total}｜已用 {fmt(time.time() - t0)}{left}{mark}"


def main():
    ap = argparse.ArgumentParser(prog="python -m tests", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help="只跑名稱含這段文字的測試")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="同時跑幾個行程")
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from ljc.common import text_width
    ids = list(all_ids(unittest.defaultTestLoader.discover(HERE, top_level_dir=ROOT)))
    if a.only:
        ids = [t for t in ids if any(k.lower() in t.lower() for k in a.only)]
    if not ids:
        sys.exit("沒有符合的測試。")
    try:
        with open(TIMINGS, encoding="utf-8") as f:
            past = json.load(f)
    except (OSError, ValueError):
        past = {}
    guess = sorted(past.values())[len(past) // 2] if past else 2.0
    ids.sort(key=lambda t: -past.get(t, guess))            # 慢的先派
    jobs = max(1, min(a.jobs, len(ids)))
    tty = sys.stdout.isatty()
    print(f"測試：{len(ids)} 個｜同時 {jobs} 個行程｜不接 LM Studio（假資料與假伺服器）")
    t0 = time.time()
    results, times = [], dict(past)
    remaining = sum(past.get(t, guess) for t in ids)
    last_pct = -1
    shown = 0                                # 上一行進度的顯示寬度：新的一行比較短時補空白蓋掉，免得留下上一行的尾巴
    ctx = mp.get_context("spawn")            # Windows、macOS 本來就是 spawn；Linux 也用它，三種系統行為一樣
    counter = ctx.Value("i", 0)
    ex = ProcessPoolExecutor(jobs, mp_context=ctx, initializer=init_worker, initargs=(counter,))
    try:
        futs = [ex.submit(run_one, t) for t in ids]
        for f in as_completed(futs):
            try:
                tid, status, secs, detail = f.result()
            except BrokenProcessPool:
                tid, status, secs, detail = ids[futs.index(f)], "error", 0.0, "工作行程意外結束（測試裡呼叫了 os._exit 或當掉）"
            results.append((tid, status, secs, detail))
            times[tid] = round(secs, 2)
            remaining -= past.get(tid, guess)
            bad = sum(r[1] in ("fail", "error") for r in results)
            line = bar(len(results), len(ids), bad, t0, max(remaining, 0) / jobs if past else None)
            if tty:
                w = text_width(line)                 # 中文字佔 2 格：用顯示寬度算要補幾格
                print("\r" + line + " " * max(shown - w, 0), end="", flush=True)
                shown = w
            else:
                pct = len(results) * 10 // len(ids)
                if pct != last_pct:
                    print(line, flush=True)
                    last_pct = pct
    except KeyboardInterrupt:
        print(f"\n收到中斷：跑完 {len(results)}/{len(ids)} 個。")
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    ex.shutdown()
    if tty:
        print()
    bad = [r for r in results if r[1] in ("fail", "error")]
    skipped = [r for r in results if r[1] == "skip"]
    slow = sorted(results, key=lambda r: -r[2])[:5]
    print("── 結果")
    print(f"  通過：{len(results) - len(bad) - len(skipped)}/{len(ids)}")
    if skipped:
        print(f"  略過：{len(skipped)}（" + "；".join(f"{r[0].split('.')[-1]}：{r[3]}" for r in skipped) + "）")
    print(f"  失敗：{len(bad)}")
    print(f"  總時間：{fmt(time.time() - t0)}（最慢：" + "、".join(f"{r[0].split('.')[-1]} {r[2]:.0f} 秒" for r in slow) + "）")
    for tid, status, _, detail in bad:
        print(f"\n✗ {tid}\n{detail}")
    if not a.only or len(ids) == len(past):
        try:
            with open(TIMINGS, "w", encoding="utf-8") as f:
                json.dump(times, f, ensure_ascii=False, indent=0, sort_keys=True)
        except OSError as e:
            # 例如 Windows「受控資料夾存取」擋住 Python 寫進「文件」資料夾：只是下次不能慢的先派，結果照樣有效
            print(f"\n⚠ 沒辦法記下各測試的耗時（{e}）\n  ・{TIMINGS}\n  ・不影響測試結果；但同一個原因可能也讓程式寫不進 out/，見 README「下載與安裝」")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
