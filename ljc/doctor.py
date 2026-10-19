"""doctor：換電腦或更新程式後，一次檢查這台能不能跑（不需要 LM Studio，也不改任何檔案）。

每一項標 ✓（沒問題）、⚠（要注意，但不擋）或 ✗（要先處理）。檢查的是先前真的踩過的坑：
程式不是最新版、資料夾寫不進去（Windows 的受控資料夾存取）、out 放在雲端同步資料夾、姓名資料沒複製、
LM Studio 連不上或載錯模型。"""
import os
import shutil
import subprocess
import sys
import uuid

from . import __version__, layout
from .common import read_config, kv_block, load_json, next_steps

OK, WARN, BAD = "✓", "⚠", "✗"
SYNC_DIRS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud", "proton drive")
MIN_FREE_GB = 5


def _git(prog, *args):
    exe = shutil.which("git")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-C", prog, *args], capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def check_program(prog):
    rows = [("Python", (OK if sys.version_info >= (3, 9) else BAD) + f" {sys.version.split()[0]}"
             + ("" if sys.version_info >= (3, 9) else "（要 3.9 以上）"))]
    if not os.path.isdir(os.path.join(prog, ".git")):
        rows.append(("程式", [f"{WARN} v{__version__}", "不是用 git clone 取得：沒辦法確認是不是最新版"]))
        return rows
    head = _git(prog, "rev-parse", "--short", "HEAD")
    if head is None:
        rows.append(("程式", [f"{WARN} v{__version__}", "git 讀不到 .git 資料夾（可能不完整或寫不進去）"]))
        return rows
    lines = [l for l in (_git(prog, "status", "--porcelain") or "").splitlines() if l.strip()]
    path = lambda l: l.split(maxsplit=1)[-1]          # _git 去掉了整段開頭的空白：第一行的狀態碼可能少一格
    new = [path(l) for l in lines if l.startswith("??")]
    dirty = [path(l) for l in lines if not l.startswith("??")]
    branch = _git(prog, "rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch not in ("main", "HEAD"):
        # 先前踩過：留在自己建的分支上，git pull 只更新那條分支，程式一直停在舊版
        rows.append(("分支", [f"{WARN} {branch}（不是 main）", "git pull 只會更新目前的分支，拿不到 main 的新版程式",
                              "切回 main：git switch main，再 git pull"]))
    behind = _git(prog, "rev-list", "--count", "HEAD..origin/main") or _git(prog, "rev-list", "--count", "HEAD..@{u}")
    if behind and behind != "0":
        state = [f"{WARN} v{__version__}（{head}）", f"比 GitHub 的 main 舊 {behind} 個 commit"
                 + ("：git pull" if branch == "main" else "")]
    else:
        state = [f"{OK} v{__version__}（{head}）", "和上次 git fetch 時 GitHub 的 main 一樣（要確認最新：先 git fetch）"]
    rows.append(("程式", state))
    if dirty:
        rows.append(("改過的檔", [f"{WARN} {len(dirty)} 個：" + "、".join(dirty[:3]) + ("…" if len(dirty) > 3 else ""),
                                "git pull 前要先處理（git status 看全部）"]))
    if new:
        rows.append(("新檔", [f"{len(new)} 個沒進版本控制：" + "、".join(new[:3]) + ("…" if len(new) > 3 else ""),
                            "不影響 git pull；不要 git add 研究資料"]))
    return rows


def check_packages():
    rows = []
    try:
        import numpy
        rows.append(("numpy", f"{OK} {numpy.__version__}"))
    except Exception:
        rows.append(("numpy", f"{BAD} 沒有安裝：python -m pip install -r requirements.txt"))
    try:
        import lmstudio  # noqa: F401
        rows.append(("lmstudio", f"{OK} 已安裝"))
    except Exception:
        rows.append(("lmstudio", f"{WARN} 沒有安裝（只有 analyze --embed 要用）"))
    return rows


def check_names(cfg, config_path):
    from .names import names_dir
    d = names_dir(cfg, config_path)
    have = [f for f in ("names_general.jsonl", "names_indigenous.jsonl", "names_foreign.jsonl") if os.path.isfile(os.path.join(d, f))]
    if len(have) >= 2:
        return [("姓名資料", f"{OK} " + "、".join(have))]
    return [("姓名資料", [f"{WARN} {len(have)} 個檔（{'、'.join(have) or '沒有'}）",
                          "只有 gen 產生新人物時要用：已有人物就不影響",
                          "換電腦時從舊電腦複製到 config/names/"])]


def check_reserved(cfg):
    """保留題私有檔：作者的機器要有（不然那幾題的資料評不了也分析不了）；公開版的使用者沒有是正常的。"""
    from .common import question_scope, reserved_notes
    r = cfg.get("_reserved")
    if not r:
        return []
    if not r["loaded"]:
        return [("保留題", [f"{WARN} 沒有私有檔（{r['path']}）", "公開版的使用者：正常，只用公開題",
                          "作者的機器：從作者那裡複製 reserved_questions.json 放到這個位置"])]
    notes = reserved_notes(cfg)
    return [("保留題", [f"{WARN} {question_scope(cfg)}", *notes] if notes else f"{OK} {question_scope(cfg)}，雜湊和承諾值相同")]


def check_out(out):
    rows = []
    path = os.path.abspath(out)
    low = path.lower()
    if any(s in low for s in SYNC_DIRS):
        rows.append(("位置", [f"{WARN} {path}", "在雲端同步資料夾裡：同步可能只傳一半或產生衝突副本", "建議搬出來"]))
    else:
        rows.append(("位置", f"{OK} {path}"))
    target = path if os.path.isdir(path) else os.path.dirname(path)
    probe = os.path.join(target, f".ljc-doctor-{uuid.uuid4().hex[:8]}.tmp")
    try:
        if os.path.exists(path) and not os.path.isdir(path):
            raise OSError("out 是一個檔案，不是資料夾")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        rows.append(("寫入", f"{OK} 可以寫入"))
    except OSError as e:
        hint = "Windows：可能是「受控資料夾存取」擋住 Python，見 README「下載與安裝」" if os.name == "nt" else "檢查資料夾權限"
        rows.append(("寫入", [f"{BAD} 寫不進去（{e}）", hint]))
    try:
        free = shutil.disk_usage(target).free / 1e9
        rows.append(("剩餘空間", (OK if free >= MIN_FREE_GB else WARN) + f" {free:.0f} GB"
                     + ("" if free >= MIN_FREE_GB else f"（少於 {MIN_FREE_GB} GB）")))
    except OSError:
        pass
    dirs = layout.run_dirs(path)
    rows.append(("設定資料夾", f"{OK} {len(dirs)} 個" if dirs else
                 [f"{WARN} 沒有", "新電腦要先把舊的 out 放到這裡", "out 底下要直接是設定資料夾（不要多一層 out）"]))
    return rows


def check_machine(a):
    from .envinfo import current_gpu, machine_code
    env = os.environ.get("LJC_MACHINE")
    return [("代號", f"{OK} {machine_code(a)}" + ("（環境變數 LJC_MACHINE）" if env else "（電腦名稱）")),
            ("顯示卡", f"{OK} {current_gpu()}" if current_gpu() else f"{WARN} 讀不到（沒有 nvidia-smi）")]


def check_lmstudio(a, cfg):
    from .client import list_models
    models = list_models(a.base_url)
    if models is None:
        return [("連線", [f"{WARN} 連不上 {a.base_url}", "只有 run、judge、continue、probe 需要",
                         "LM Studio 的 Developer 分頁按 Start Server"])]
    loaded = [m.get("id") for m in models if m.get("state") == "loaded"]
    jc = cfg.get("judge", {})
    rows = [("連線", f"{OK} {a.base_url}"),
            ("已載入", "、".join(loaded) if loaded else f"{WARN} 沒有載入任何模型")]
    roles = [(jc.get("model"), "主評審"), (jc.get("second_model"), "第二評審")]
    for mid, role in roles:
        if mid and mid in loaded:
            rows.append((role, f"{OK} {mid} 已載入"))
    from .envinfo import LINK_HINT, fit_notes
    for m in models:
        if m.get("state") == "loaded" and m.get("type") in ("llm", "vlm", None):
            for note in fit_notes(m.get("id"), m.get("quantization")):
                remote = "LM Link" in note
                rows.append((m.get("id"), [f"{WARN} 這台找不到模型檔", LINK_HINT] if remote else f"{WARN} {note}"))
    if len(loaded) > 1:
        rows.append(("提醒", [f"{WARN} 同時載入 {len(loaded)} 個模型：都在這台時會互相搶顯示記憶體",
                             "LM Studio 標著別台電腦名稱的是 LM Link 接過來的，在那台算，不佔這台：可以忽略，也不要按它的 Eject"]))
    return rows


def cmd_doctor(a):
    cfg = read_config(a.config)
    prog = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blocks = [("程式", check_program(prog) + check_packages() + check_names(cfg, a.config) + check_reserved(cfg)),
              ("out 資料夾", check_out(a.out)),
              ("機器", check_machine(a)),
              ("LM Studio", check_lmstudio(a, cfg))]
    for title, rows in blocks:
        print(kv_block(title, rows))
    marks = [(v[0] if isinstance(v, list) else str(v))[:1] for _, rows in blocks for _, v in rows]
    bad, warns = marks.count(BAD), marks.count(WARN)
    print(kv_block("結果", [("要先處理", f"{bad} 項（✗）"), ("要注意", f"{warns} 項（⚠）")]))
    print(next_steps(["處理完 ✗ 的項目，再執行一次 python lijuncha.py doctor" if bad else None,
                      "跑自動測試：python -m tests",
                      "看資料進度：python lijuncha.py status"]))
    if bad:
        raise SystemExit(1)
