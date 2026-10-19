"""姓名資料檔的清單（config/names/manifest.json）：每個檔的 SHA-256、筆數、各來源的筆數與授權。

清單進版本控制。拿到姓名資料檔的人可以比對自己的檔和作者用的是不是同一份，
再用 tools/verify_personas.py 重新產生人物、驗證和研究用的人物一模一樣。

用法（在程式資料夾）：
    python tools/names_manifest.py            依 config/names/ 現在的檔重寫清單（授權欄保留先前填的）
    python tools/names_manifest.py --check    只比對，不改檔：檔案和清單不同就列出來，結束代碼 1"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ljc.common import read_config, load_json, read_jsonl  # noqa: E402
from ljc.names import FILES, _src, names_dir  # noqa: E402
from ljc.personas import file_sha256  # noqa: E402

PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = "manifest.json"


def source_id(r):
    s = r.get("source") or {}
    return (s.get("source_id") if isinstance(s, dict) else None) or _src(r).split(" ")[0] or "未標示"


def scan(d):
    out = {}
    for f in FILES:
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            continue
        recs = read_jsonl(p)
        by = {}
        for r in recs:
            k = source_id(r)
            by[k] = by.get(k, 0) + 1
        out[f] = {"sha256": file_sha256(p), "records": len(recs), "sources": dict(sorted(by.items()))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(PROG, "config", "config.json"))
    ap.add_argument("--check", action="store_true", help="只比對，不改檔")
    a = ap.parse_args(argv)
    d = names_dir(read_config(a.config), a.config)
    if not os.path.isdir(d):
        sys.exit(f"✗ 沒有姓名資料夾：{d}")
    path = os.path.join(d, MANIFEST)
    old = load_json(path) if os.path.exists(path) else {"files": {}}
    now = scan(d)
    if a.check:
        bad, missing = [], []
        for f, info in old.get("files", {}).items():
            got = now.get(f)
            if not got:
                missing.append(f)
            elif got["sha256"] != info["sha256"]:
                bad.append(f"{f}：內容和清單不同（{got['records']} 筆；清單是 {info['records']} 筆）")
        for f in now:
            if f not in old.get("files", {}):
                bad.append(f"{f}：清單裡沒有")
        for f in missing:
            print(f"⚠ 這台沒有 {f}（不進版本控制的檔要另外取得；沒有它，用到這類姓名的人物產生不了）")
        if bad:
            print("✗ 姓名資料和清單不同\n" + "\n".join(f"  ・{x}" for x in bad))
            sys.exit(1)
        print(f"✓ 這台有的 {len(now)} 個姓名檔都和清單相同")
        return
    for f, info in now.items():
        info["license"] = old.get("files", {}).get(f, {}).get("license", "【待填：各來源的授權】")
    for f, info in old.get("files", {}).items():
        if f not in now:              # 不進版本控制的檔（原住民族名）：這台沒有，保留作者那台寫的紀錄
            now[f] = info
            print(f"{f}：這台沒有這個檔，保留清單裡的紀錄")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_說明": "姓名資料檔的指紋與來源；由 tools/names_manifest.py 產生，license 欄由作者填", "files": now},
                  fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    for f, info in now.items():
        print(f"{f}：{info['records']} 筆，SHA-256 {info['sha256'][:12]}…，來源 " +
              "、".join(f"{k} {v}" for k, v in info["sources"].items()))
    print(f"已寫入 {path}")


if __name__ == "__main__":
    main()
