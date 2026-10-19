"""驗證人物檔可以重現：用人物檔記下的種子與人數，在暫存資料夾重新 gen 一次，逐筆比對。

同一版程式、同一份設定檔與姓名資料、同一個種子，產生的人物應該一模一樣（測試在 Linux 與 Windows 都檢查）。
v0.10.1 起 personas_meta.json 記下種子、人數、設定檔指紋與姓名資料的 SHA-256；更早的人物檔沒有記錄，
只能用目前的程式猜（種子 1），不同也不一定是錯（例如 v0.8.6 起多了屬性平衡互換）。

用法（在程式資料夾）：
    python tools/verify_personas.py --personas p1.2"""
import argparse
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ljc import __version__  # noqa: E402
from ljc.cli import main as cli_main  # noqa: E402
from ljc.common import read_config, file_fp, load_json, read_jsonl  # noqa: E402
from ljc.names import FILES, names_dir  # noqa: E402
from ljc.personas import file_sha256  # noqa: E402

PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(PROG), "out"))
    ap.add_argument("--config", default=os.path.join(PROG, "config", "config.json"))
    ap.add_argument("--personas", required=True, help="人物版本（out/personas/ 底下的資料夾名稱，例如 p1.2）")
    a = ap.parse_args(argv)
    pdir = os.path.join(a.out, "personas", a.personas)
    recs = read_jsonl(os.path.join(pdir, "personas.jsonl"))
    if not recs:
        sys.exit(f"✗ 找不到人物檔：{pdir}")
    cfg = read_config(a.config)
    want_ver = (cfg.get("version") or {}).get("personas")
    if want_ver != a.personas:
        sys.exit(f"✗ 設定檔的人物版本是 {want_ver}，不是 {a.personas}：用產生那批人物時的設定檔（--config）")
    meta = load_json(os.path.join(pdir, "personas_meta.json")) if os.path.exists(os.path.join(pdir, "personas_meta.json")) else None
    notes = []
    if meta and len(meta.get("batches", [])) == 1:
        b = meta["batches"][0]
        seed, n, args = b.get("seed", 1), b.get("total_people"), b.get("args") or {}
        if b.get("program_version") != __version__:
            notes.append(f"人物由 v{b.get('program_version')} 產生，現在是 v{__version__}：不同時，切回那一版（git checkout）再驗證")
        if b.get("config_fp") and b["config_fp"] != file_fp(a.config):
            notes.append("設定檔和產生時不同（指紋不同）：不同時，用產生時的設定檔")
        nd = names_dir(cfg, a.config)
        for f, sha in (b.get("names_sha256") or {}).items():
            p = os.path.join(nd, f)
            if not os.path.isfile(p):
                notes.append(f"這台沒有 {f}（產生時有）")
            elif file_sha256(p) != sha:
                notes.append(f"{f} 和產生時不同")
    elif meta:
        sys.exit("✗ 這批人物是分幾次接續加人產生的：不支援自動驗證")
    else:
        seed, n, args = 1, len({r["id"] for r in recs}), {}
        notes.append("人物檔沒有產生紀錄（v0.9.1 以前）：用種子 1、目前的程式重算；不同也不一定是錯")
    cmd = ["--config", a.config, "--seed", str(seed), "--out", "", "gen", "--n", str(n)]
    if args.get("mention_k"):
        cmd += ["--mention-k", str(args["mention_k"])]
    if args.get("min_per_group") is not None:
        cmd += ["--min-per-group", str(args["min_per_group"])]
    with tempfile.TemporaryDirectory() as tmp:
        cmd[cmd.index("--out") + 1] = tmp
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                cli_main(cmd)
            except SystemExit as e:
                sys.exit(f"✗ 重新產生失敗：{e}")
        again = read_jsonl(os.path.join(tmp, "personas", a.personas, "personas.jsonl"))
    key = lambda r: (r["id"], r.get("variant"))
    mine, theirs = {key(r): r for r in recs}, {key(r): r for r in again}
    diff = [k for k in mine if mine[k] != theirs.get(k)]
    print(f"── 驗證人物 {a.personas}")
    print(f"  重新產生：種子 {seed}、{n} 位人物（v{__version__}）")
    for x in notes:
        print(f"  ⚠ {x}")
    if not diff and len(mine) == len(theirs):
        print(f"  ✓ {len(mine)} 筆人物版本完全相同：這批人物可以重現")
        return
    print(f"  ✗ {len(diff)}/{len(mine)} 筆不同" + (f"；筆數 {len(mine)} 對 {len(theirs)}" if len(mine) != len(theirs) else ""))
    if diff:
        k = diff[0]
        fields = sorted(f for f in set(mine[k]) | set(theirs.get(k) or {}) if mine[k].get(f) != (theirs.get(k) or {}).get(f))
        print(f"  第一筆不同：{k[0]}（{k[1]}），欄位 {'、'.join(fields)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
