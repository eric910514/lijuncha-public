"""pack 指令：把某一步的產出打成一個 zip，直接傳給別人（或程式對話）看，不必自己挑檔案。

  python lijuncha.py pack run --run p1.2        生成：responses.jsonl、settings.json、run_meta.json、人物檔
  python lijuncha.py pack judge --run p1.2      評審：judgments*.jsonl（主評審、第二評審）、操弄檢核的歸類
  python lijuncha.py pack analysis --run p1.2   分析：analysis/<設定資料夾>/ 底下全部（各範圍、讀出率、詮釋層）
  python lijuncha.py pack annotate --run p1.2   人工標註：annotation/ 底下這組的行為層與詮釋層
  python lijuncha.py pack all --run p1.2        以上全部

zip 放在 out/packages/，裡面的路徑和 out/ 一樣（解壓到另一個 out/ 就接得上），附一份「打包說明.txt」
列出程式版本、每個檔案的大小與指紋，以及給 unpack 驗證用的 manifest.json（每個檔的 SHA-256）。
不放：備份檔（.bak）、tuning.json、hashes.json（各機器自己的紀錄）。

換電腦（v0.10）：舊電腦 pack all，新電腦 unpack <zip>：先驗證每個檔，再解到這台的 out/；
這台已經有同名但內容不同的檔就停下（那是兩台各跑了一部分，要用 merge）。"""
import json
import hashlib
import os
import time
import zipfile

from . import __version__, layout
from .common import bullets, stop

STEPS = {
    "personas": "步驟 0 產生人物：personas/<人物版本>/",
    "run": "步驟 2–3 生成與續寫：設定資料夾的 responses.jsonl、settings.json、run_meta.json、probe.jsonl，加上用到的人物檔",
    "judge": "步驟 4、7 評審：judgments.jsonl（主評審）、judgments__second.jsonl（第二評審）、probe_judgments.jsonl（操弄檢核的歸類）",
    "analysis": "步驟 5、8 分析：analysis/<設定資料夾>/ 底下全部",
    "annotate": "步驟 6 人工標註：annotation/behavior 與 annotation/interpretation 底下這組的檔案",
    "interpret": "步驟 10 詮釋層：interpretations*.jsonl、analysis/<設定資料夾>/interpretation/、annotation/interpretation/",
    "all": "以上全部",
}
SKIP = {"tuning.json", "hashes.json"}
MANIFEST = "manifest.json"


def _walk(root):
    if not os.path.isdir(root):
        return []
    out = []
    for base, _, files in os.walk(root):
        out += [os.path.join(base, f) for f in files if not f.endswith(".bak") and f not in SKIP]
    return sorted(out)


def files_for(out_dir, run_dir, step):
    """這一步、這組設定要打包的檔案（絕對路徑）。"""
    name = os.path.basename(run_dir)
    s = layout.read_settings(run_dir)
    pick = lambda *fs: [os.path.join(run_dir, f) for f in fs if os.path.isfile(os.path.join(run_dir, f))]
    tagged = lambda prefix: sorted(os.path.join(run_dir, f) for f in os.listdir(run_dir)
                                   if f.startswith(prefix) and f.endswith(".jsonl"))
    got = []
    if step in ("personas", "run", "all"):
        got += _walk(os.path.join(out_dir, "personas", s.get("personas_version") or ""))
    if step in ("run", "all"):
        got += pick(layout.SETTINGS, layout.META, layout.RESPONSES, "probe.jsonl")
    if step in ("judge", "all"):
        got += pick(layout.SETTINGS, layout.META, "probe_judgments.jsonl") + tagged("judgments")
    if step in ("analysis", "all"):
        got += _walk(os.path.join(out_dir, "analysis", name))
    if step in ("annotate", "interpret", "all"):
        got += _walk(os.path.join(out_dir, "annotation", "interpretation", name))
    if step in ("annotate", "all"):
        got += _walk(os.path.join(out_dir, "annotation", "behavior", name))
    if step in ("interpret", "all"):
        got += pick(layout.SETTINGS, layout.META) + tagged("interpretations")
        got += _walk(os.path.join(out_dir, "analysis", name, "interpretation"))
    return list(dict.fromkeys(got))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8]


def cmd_pack(a):
    out_dir = os.path.abspath(a.out)
    layout.tidy_backups(out_dir)
    dirs = [d for d in layout.run_dirs(out_dir) if os.path.basename(d).startswith("mock__") == bool(a.mock)]
    if a.run:
        dirs = layout.pick_one([d for d in dirs if a.run in os.path.basename(d)], a.run)
    if not dirs:
        stop("沒有符合的設定資料夾" + (f"：名稱含「{a.run}」" if a.run else ""))
    files = []
    for d in dirs:
        files += files_for(out_dir, d, a.step)
    files = list(dict.fromkeys(files))
    if not files:
        stop(f"這一步（{a.step}）還沒有產出", STEPS[a.step])
    tag = os.path.basename(dirs[0]).split("__")[0] + "_" + (layout.read_settings(dirs[0]).get("personas_version") or "") \
        if len(dirs) == 1 else f"{len(dirs)}組"
    os.makedirs(os.path.join(out_dir, "packages"), exist_ok=True)
    zpath = os.path.join(out_dir, "packages", f"{time.strftime('%Y%m%d-%H%M')}_{a.step}_{tag}.zip")
    rows, manifest = [], {}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = os.path.relpath(f, out_dir).replace(os.sep, "/")
            z.write(f, rel)
            rows.append(f"{rel}\t{os.path.getsize(f):,} bytes\t指紋 {md5(f)}")
            manifest[rel] = {"bytes": os.path.getsize(f), "sha256": sha256(f)}
        z.writestr(MANIFEST, json.dumps({"version": __version__, "step": a.step, "files": manifest},
                                        ensure_ascii=False, indent=1))
        note = [f"離均差打包：{a.step}（{STEPS[a.step]}）", f"程式 v{__version__}｜打包時間 {time.strftime('%Y-%m-%d %H:%M')}",
                "設定資料夾：" + "、".join(os.path.basename(d) for d in dirs),
                "路徑和 out/ 一樣：解壓到另一個 out/ 就接得上。不含備份檔、tuning.json、hashes.json。", "", "檔案："] + rows
        z.writestr("打包說明.txt", "\n".join(note) + "\n")
    size = os.path.getsize(zpath)
    print(bullets(f"打包 {a.step}：{zpath}（{len(files)} 個檔案，{size / 1e6:.1f} MB）",
                  [r.split("\t")[0] for r in rows[:12]]
                  + ([f"……另外 {len(rows) - 12} 個（清單在 zip 裡的「打包說明.txt」）"] if len(rows) > 12 else [])))


def cmd_unpack(a):
    """換電腦：驗證 pack 的 zip，再解到這台的 out/。同名檔內容相同就略過；不同就停下（要用 merge）。"""
    from .common import kv_block, next_steps, cmd_hint
    out_dir = os.path.abspath(a.out)
    try:
        z = zipfile.ZipFile(a.zip)
    except (OSError, zipfile.BadZipFile) as e:
        stop(f"打不開 {a.zip}", str(e))
    with z:
        bad = z.testzip()
        if bad:
            stop("zip 損壞，檔案沒解開", f"第一個壞掉的檔：{bad}", "重新複製或重新打包一次")
        if MANIFEST not in z.namelist():
            stop("這個 zip 沒有 manifest.json（v0.10 以前的 pack，或不是 pack 產生的）", "在舊電腦更新程式後重新 pack all")
        files = json.loads(z.read(MANIFEST).decode("utf-8"))["files"]
        new, same, differ = [], [], []
        for rel, info in files.items():
            parts = rel.split("/")
            if rel.startswith("/") or ".." in parts or ":" in parts[0]:
                stop("zip 裡有不安全的路徑，檔案沒解開", rel)
            data = z.read(rel)
            if hashlib.sha256(data).hexdigest() != info["sha256"] or len(data) != info["bytes"]:
                stop("zip 裡的檔案和打包時不一樣，檔案沒解開", rel, "重新複製或重新打包一次")
            dest = os.path.join(out_dir, *parts)
            if not os.path.exists(dest):
                new.append((dest, data))
            elif sha256(dest) == info["sha256"]:
                same.append(rel)
            else:
                differ.append(rel)
        if differ:
            stop("這台的 out 已經有同名但內容不同的檔，檔案沒解開", *differ[:8],
                 "兩台各跑了一部分：把 zip 解到別的資料夾，再用 merge 合併",
                 cmd_hint("merge", "<解開的資料夾>/<設定資料夾>"))
        for dest, data in new:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
    print(kv_block("解開", [("驗證", f"✓ {len(files)} 個檔的 SHA-256 都和打包時相同"),
                            ("新增", f"{len(new)} 個檔"), ("已經一樣", f"{len(same)} 個檔（略過）" if same else None),
                            ("位置", out_dir)]))
    print(next_steps(["看進度與下一步：" + cmd_hint("status")]))
