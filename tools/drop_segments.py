"""把某台機器多跑的重複移出回覆檔（例如 10-17 的 3090 沒給 --rep-range，多跑了 M1 Pro 負責的第 4 次 8 段）。

為什麼要移：合併時兩台跑了同一段會「保留這邊的」，M1 Pro 那幾段就被丟掉，第 4 次變成兩台混著跑。
先把這台多跑的移出，合併後第 4 次就全部是 M1 Pro 的。移出的段放進 out/backups/，run_meta.json 記下來。

用法（在程式資料夾，Windows 那台）：
    python tools/drop_segments.py --run p1.2 --rep 4 --machine machine-A          先看會移出哪些（不改檔）
    python tools/drop_segments.py --run p1.2 --rep 4 --machine machine-A --yes    真的移出

--machine 是回覆紀錄裡的機器代號（status「機器」一列括號前的名稱）。"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ljc import layout  # noqa: E402
from ljc.common import bullets, read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out"))
    ap.add_argument("--run", required=True, help="設定資料夾名稱裡的一段（例如 p1.2）")
    ap.add_argument("--rep", type=int, required=True, help="要移出的重複序號（從 1 起算，例如 4）")
    ap.add_argument("--machine", required=True, help="只移出這台機器跑的（回覆紀錄的 machine 欄）")
    ap.add_argument("--yes", action="store_true", help="真的移出；沒加只列出")
    a = ap.parse_args(argv)
    dirs = layout.pick_one([d for d in layout.run_dirs(a.out) if a.run in os.path.basename(d)], a.run)
    if len(dirs) != 1:
        sys.exit("✗ --run 要剛好對到一個設定資料夾\n  ・對到：" + ("、".join(os.path.basename(d) for d in dirs) or "沒有"))
    d = dirs[0]
    path = os.path.join(d, layout.RESPONSES)
    recs = read_jsonl(path)
    drop = [r for r in recs if r.get("rep") == a.rep - 1 and r.get("machine") == a.machine]
    keep = [r for r in recs if not (r.get("rep") == a.rep - 1 and r.get("machine") == a.machine)]
    other = sum(1 for r in keep if r.get("rep") == a.rep - 1)
    print(bullets(f"■ {os.path.basename(d)}", [
        ("移出", f"第 {a.rep} 次重複、機器 {a.machine} 的 {len(drop)} 段"),
        ("留下", f"{len(keep)} 段（第 {a.rep} 次其他機器的 {other} 段不動）"),
        ("時間", f"{min(r['time'] for r in drop)} 到 {max(r['time'] for r in drop)}" if drop else None)]))
    if not drop:
        print("沒有要移出的段。")
        return
    if not a.yes:
        print("（只列出，沒有改檔；確定要移出加 --yes）")
        return
    bak = layout.backup_path(path, "dropped")
    with open(bak, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in drop))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    os.replace(tmp, path)
    meta = layout.read_meta(d)
    times = [r["time"] for r in drop]
    for ss in meta.get("sessions", []):
        # 這次執行跑的段全部被移出：標記起來，run 判斷「這台上一次跑到第幾次」時不算這次。
        # 認法：同一台、重複範圍包含這一次、執行期間移出的段數不少於這次執行的段數
        rr = ss.get("rep_range") or [0, 0]
        n_in = sum(ss.get("start", "") <= t <= ss.get("end", "") for t in times)
        if ss.get("machine") == a.machine and rr[0] <= a.rep <= rr[1] and ss.get("segments") and n_in >= ss["segments"]:
            ss["dropped_all"] = True
    meta.setdefault("dropped", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rep": a.rep, "machine": a.machine,
                                          "segments": len(drop), "backup": os.path.relpath(bak, a.out)})
    layout.update_meta(d, meta)
    print(bullets("已移出", [("移出的段", bak), ("記錄", "run_meta.json 的 dropped"),
                           ("下一步", "M1 Pro 的第 4 次跑完後再 merge；status 的「機器」一列第 4 次只會剩 M1 Pro")]))


if __name__ == "__main__":
    main()
