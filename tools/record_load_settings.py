"""把人工比對過的 LM Studio 載入設定記進設定資料夾的 run_meta.json（load_manual）。

v0.9.6 以前程式讀不到 Flash Attention、KV 快取量化這些設定，換機器時只能由作者看畫面比對。這個工具把比對結果
記下來：報告「各機器」一節會列出，之後換機器接續時也會拿來比（會影響內容的不同就停下）。

用法（在程式資料夾，Windows 那台）：
    python tools/record_load_settings.py --run p1.2 --preset 2026-10-16      記下作者截圖比對的紀錄（私有檔裡的）
    python tools/record_load_settings.py --run p1.2 --machine "machine-A（RTX 3080 Ti）" --flash-attention on \\
        --speculative off --k-cache f16 --v-cache f16 --unified-kv on --source "10-20 作者截圖"

--machine 要和報告「各機器」一節的寫法一樣（機器代號加括號裡的顯示卡型號）。"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ljc import layout  # noqa: E402

PRIVATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "private", "load_presets.json")


def load_presets(path=PRIVATE):
    """作者的比對紀錄有真實電腦名稱，放在不進版本控制的 config/private/load_presets.json；沒有就沒有預設可用。"""
    if not os.path.exists(path):
        return {}, {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("presets") or {}, d.get("source") or {}


ONOFF = {"on": True, "off": False}


def main():
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)   # --preset 不能被當成 --presets-file 的縮寫
    pre.add_argument("--presets-file", default=PRIVATE)
    PRESETS, SOURCE = load_presets(pre.parse_known_args()[0].presets_file)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets-file", default=PRIVATE, help="比對紀錄檔（預設 config/private/load_presets.json）")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out"))
    ap.add_argument("--run", required=True, help="設定資料夾名稱裡的一段（例如 p1.2）")
    ap.add_argument("--preset", choices=sorted(PRESETS) or None, help="記下已知的人工比對結果（config/private/load_presets.json）")
    ap.add_argument("--machine", default="")
    ap.add_argument("--flash-attention", choices=ONOFF)
    ap.add_argument("--speculative", choices=ONOFF)
    ap.add_argument("--unified-kv", choices=ONOFF)
    ap.add_argument("--k-cache", default=None, help="沒勾量化填 f16")
    ap.add_argument("--v-cache", default=None)
    ap.add_argument("--source", default="作者人工比對")
    a = ap.parse_args()
    dirs = layout.pick_one([d for d in layout.run_dirs(a.out) if a.run in os.path.basename(d)], a.run)
    if len(dirs) != 1:
        sys.exit("--run 要剛好對到一個設定資料夾；對到：" + ("、".join(os.path.basename(d) for d in dirs) or "沒有"))
    if a.preset:
        recs = {m: dict(v, _source=SOURCE[a.preset]) for m, v in PRESETS[a.preset].items()}
    else:
        if not a.machine:
            sys.exit("沒有 --preset 時要給 --machine。")
        rec = {"flash_attention": ONOFF.get(a.flash_attention), "speculative": ONOFF.get(a.speculative),
               "unified_kv": ONOFF.get(a.unified_kv), "k_cache": a.k_cache, "v_cache": a.v_cache}
        recs = {a.machine: dict({k: v for k, v in rec.items() if v is not None}, _source=a.source)}
    for v in recs.values():
        v["_recorded"] = time.strftime("%Y-%m-%d %H:%M")
    meta = layout.read_meta(dirs[0])
    manual = dict(meta.get("load_manual") or {})
    manual.update(recs)
    layout.update_meta(dirs[0], {"load_manual": manual})
    print(f"已記進 {os.path.join(dirs[0], 'run_meta.json')}（load_manual）：" + "、".join(recs))


if __name__ == "__main__":
    main()
