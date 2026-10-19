"""公開版快照：從這個（私有）儲存庫挑出可以公開的檔案，放到一個新資料夾，再逐項檢查（企劃書 10-19）。

只複製白名單裡、而且進了版本控制的檔案（git ls-files），所以 out/、config/private/、沒加進 git 的檔都不會帶過去。
複製完逐項檢查，任何一項不過就結束代碼 1，快照不能拿去發布：
  1. 保留題：快照裡不能出現保留題私有檔的任何文字（逐 10 字比對題目、追問、同義改寫、標準答案、英文標題）；
  2. 原住民族名：沒有 names_indigenous*.jsonl，也沒有族語發展會資料庫（ILRDF）的紀錄；
  3. 研究資料：沒有 out/、回覆檔、評審檔、人物檔；
  4. 私人字串：沒有 config/private/denylist.txt 列的字（真實電腦名稱等；這個檔不進版本控制）；
  5. 金鑰：內建的規則掃一次（API 金鑰、權杖、私鑰）；有安裝 detect-secrets（Yelp 開發）就再用它掃一次；
  6. 測試：在快照裡跑 python -m tests（沒有保留題檔，只用公開題）。

用法（在程式資料夾）：
    python tools/public_snapshot.py --dest ../lijuncha-public-snapshot --code-license MIT --holder "作者名"
    python tools/public_snapshot.py --dest ... --check-only        只檢查已有的快照資料夾
需要 config/private/reserved_questions.json 才能做第 1 項檢查（沒有就停下：沒辦法確認沒有外流）。"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time

PROG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROG)
from ljc.common import load_json, text_sha256  # noqa: E402

# 可以公開的檔案（相對於程式資料夾；fnmatch 寫法）。不在這裡的一律不帶。
WHITELIST = [
    "lijuncha.py", "requirements.txt", "README.md", "CHANGELOG.md", "DEVELOPMENT.md",
    ".gitignore", ".gitattributes", ".github/workflows/*.yml",
    "ljc/*.py", "tools/*.py", "tests/*.py", "tests/fixtures/*.json", "tests/fixtures/names_files/*.jsonl",
    "config/config.json", "config/i18n/*.json", "config/judge_prompts/*.txt", "config/script/*.json",
    "config/names/README.md", "config/names/manifest.json",
    "config/names/names_general.jsonl", "config/names/names_foreign.jsonl",
]
# 白名單裡也不能有的（雙重保險）
NEVER = ["config/private/*", "out/*", "*/out/*", "config/names/names_indigenous*", "config/names/raw/*",
         "*.docx", "*.pdf", ".env", "*.env", "tuning.json", "hashes.json"]
DATA_NAMES = {"responses.jsonl", "judgments.jsonl", "personas.jsonl", "run_meta.json", "settings.json"}
# 金鑰的內建規則：常見服務的格式，加上「key／token／secret = 一長串」的寫法
SECRET_RULES = [
    ("Anthropic 金鑰", r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    ("OpenAI 式金鑰", r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}"),
    ("GitHub 權杖", r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"),
    ("Google API 金鑰", r"\bAIza[0-9A-Za-z_\-]{35}"),
    ("AWS 金鑰", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("Slack 權杖", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    ("Hugging Face 權杖", r"\bhf_[A-Za-z0-9]{30,}"),
    ("私鑰", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("寫死的金鑰", r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
]
NGRAM = 10


def git_files():
    out = subprocess.run(["git", "-C", PROG, "ls-files"], capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l]


def pick(files):
    keep = [f for f in files if any(fnmatch.fnmatch(f, w) for w in WHITELIST)
            and not any(fnmatch.fnmatch(f, n) for n in NEVER)]
    return sorted(keep)


def reserved_texts(priv):
    """保留題私有檔裡所有夠長的文字（題號、題型這種短代碼不算）。"""
    texts = []

    def walk(x, key=None):
        if isinstance(x, str):
            if key not in ("id", "type", "answer_form") and len(x) >= NGRAM:
                texts.append(x)
        elif isinstance(x, list):
            for y in x:
                walk(y, key)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
    walk({k: v for k, v in priv.items() if not k.startswith("_")})
    return texts


def walk_files(root):
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for n in names:
            p = os.path.join(base, n)
            yield os.path.relpath(p, root).replace(os.sep, "/"), p


def read_text(p):
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def check(root, priv_path, denylist_path, run_tests=True):
    """逐項檢查快照資料夾。回傳 [(項目, 通過與否, 說明)]。"""
    results = []
    files = list(walk_files(root))
    texts = {rel: read_text(p) for rel, p in files}

    # 1. 保留題
    if not os.path.isfile(priv_path):
        results.append(("保留題", False, f"找不到保留題私有檔（{priv_path}），沒辦法確認沒有外流"))
    else:
        grams = {t[i:i + NGRAM] for t in reserved_texts(load_json(priv_path)) for i in range(len(t) - NGRAM + 1)}
        # 只有英文字母、數字、底線的片段多半是欄位名稱（例如 suggested_number），不算
        grams = {g for g in grams if not re.fullmatch(r"[A-Za-z0-9_ ]+", g)}
        # 公開題本來就有的句子（例如備註「標準答案尚未對照官方條文」）不算外流
        public = json.dumps(load_json(os.path.join(PROG, "config", "config.json")).get("questions") or [], ensure_ascii=False)
        grams = {g for g in grams if g not in public}
        hits = {rel: sorted(g for g in grams if g in t)[:3] for rel, t in texts.items() if t and any(g in t for g in grams)}
        results.append(("保留題", not hits, "沒有保留題的文字" if not hits else
                        "；".join(f"{rel}（例如「{'」「'.join(v)}」）" for rel, v in hits.items())))

    # 2. 原住民族名
    bad = [rel for rel, _ in files if fnmatch.fnmatch(rel, "config/names/names_indigenous*")]
    bad += [rel for rel, t in texts.items() if t and "ILRDF" in t and rel.endswith(".jsonl")]
    results.append(("原住民族名", not bad, "沒有原住民族名檔與族語發展會資料庫的紀錄" if not bad else "、".join(sorted(set(bad)))))

    # 3. 研究資料
    bad = [rel for rel, _ in files if rel.startswith("out/") or "/out/" in rel or os.path.basename(rel) in DATA_NAMES
           or any(fnmatch.fnmatch(rel, n) for n in NEVER)]
    results.append(("研究資料", not bad, "沒有 out/、回覆、評審、人物檔" if not bad else "、".join(bad)))

    # 4. 私人字串
    words = []
    if os.path.isfile(denylist_path):
        with open(denylist_path, encoding="utf-8") as f:
            words = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    hits = [f"{rel}：{w}" for rel, t in texts.items() if t for w in words if w.lower() in t.lower()]
    results.append(("私人字串", not hits, (f"沒有 denylist 的 {len(words)} 個字" if words else "沒有 denylist（config/private/denylist.txt），略過")
                    if not hits else "、".join(hits)))

    # 5. 金鑰
    hits = []
    for rel, t in texts.items():
        if not t:
            continue
        for name, pat in SECRET_RULES:
            for m in re.finditer(pat, t):
                hits.append(f"{rel}：{name}（{m.group(0)[:12]}…）")
    ext = scan_detect_secrets(root)
    ok = not hits and ext[0] is not False
    msg = "內建規則沒有發現" if not hits else "、".join(hits)
    msg += "；" + ext[1]
    results.append(("金鑰", ok, msg))

    # 6. 測試
    if run_tests:
        r = subprocess.run([sys.executable, "-m", "tests"], cwd=root, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        last = [l for l in (r.stdout + r.stderr).splitlines() if "通過" in l or "失敗" in l][-2:]
        results.append(("測試", r.returncode == 0, "、".join(x.strip() for x in last) or f"結束代碼 {r.returncode}"))
    return results


def scan_detect_secrets(root):
    """有安裝 detect-secrets（Yelp, Inc. 開發，美國；Apache-2.0）就用它再掃一次。回傳 (通過與否或 None, 說明)。"""
    exe = shutil.which("detect-secrets")
    if not exe:
        return None, "沒有安裝 detect-secrets（python -m pip install detect-secrets），只用內建規則"
    r = subprocess.run([exe, "scan", "--all-files", "--exclude-files", r"\.git/"], cwd=root, capture_output=True, text=True)
    try:
        found = json.loads(r.stdout).get("results") or {}
    except ValueError:
        return False, f"detect-secrets 執行失敗：{(r.stderr or r.stdout)[:200]}"
    # 測試用的假字串與雜湊值（SHA-256、指紋）會被當成高熵字串：只有真的像金鑰的類型才算
    real = {f: [x["type"] for x in v if x["type"] not in ("Hex High Entropy String", "Base64 High Entropy String")]
            for f, v in found.items()}
    real = {f: v for f, v in real.items() if v}
    if real:
        return False, "detect-secrets 發現：" + "、".join(f"{f}（{'、'.join(sorted(set(v)))}）" for f, v in real.items())
    skipped = sum(len(v) for v in found.values())
    return True, "detect-secrets 沒有發現金鑰" + (f"（另有 {skipped} 個高熵字串，是雜湊值或測試資料，已逐一略過）" if skipped else "")


MIT = """MIT License

Copyright (c) {year} {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

APACHE_NOTICE = """Copyright {year} {holder}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def write_licenses(dest, code_license, holder):
    year = time.strftime("%Y")
    if code_license == "MIT":
        body = MIT.format(year=year, holder=holder)
    else:
        body = APACHE_NOTICE.format(year=year, holder=holder) + \
            "\nThe full license text: https://www.apache.org/licenses/LICENSE-2.0.txt\n"
    with open(os.path.join(dest, "LICENSE"), "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    man = load_json(os.path.join(PROG, "config", "names", "manifest.json"))
    lines = ["# 授權（Licenses）", "",
             f"- 程式碼（`ljc/`、`tools/`、`tests/`、`lijuncha.py`）：{code_license}，見 `LICENSE`。",
             "- 文件（`README.md`、`CHANGELOG.md`、`DEVELOPMENT.md`、`config/names/README.md`）與作者自製的資料"
             "（題目、人物句、評審提示詞、作者整理的姓名清單）：CC BY 4.0（https://creativecommons.org/licenses/by/4.0/），"
             f"標示出處：{holder}，離均差（Lijuncha）。",
             "- 姓名檔：依各來源的授權，逐一列在下面（也記在 `config/names/manifest.json`）。原住民族名檔授權確認前不公開。",
             "- 保留題不在這個倉庫：見 README「保留題」。", ""]
    for f, info in man.get("files", {}).items():
        if not os.path.exists(os.path.join(dest, "config", "names", f)):
            continue
        lines.append(f"## {f}")
        for src, lic in (info.get("license") or {}).items():
            lines.append(f"- {src}（{info['sources'].get(src, '?')} 筆）：{lic}")
        lines.append("")
    with open(os.path.join(dest, "LICENSES.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="快照要放的新資料夾（不能已經有東西）")
    ap.add_argument("--code-license", choices=("MIT", "Apache-2.0"))
    ap.add_argument("--holder", default="", help="著作權人名稱（LICENSE 用）")
    ap.add_argument("--check-only", action="store_true", help="不複製，只檢查 --dest 已有的快照")
    ap.add_argument("--no-tests", action="store_true", help="不在快照裡跑測試")
    a = ap.parse_args(argv)
    dest = os.path.abspath(a.dest)
    priv = os.path.join(PROG, "config", "private", "reserved_questions.json")
    deny = os.path.join(PROG, "config", "private", "denylist.txt")
    if not a.check_only:
        if not a.code_license or not a.holder:
            sys.exit("✗ 要給 --code-license（MIT 或 Apache-2.0）與 --holder（著作權人）")
        if os.path.exists(dest) and os.listdir(dest):
            sys.exit(f"✗ {dest} 已經有東西：換一個新資料夾")
        if os.path.commonpath([dest, PROG]) == PROG:
            sys.exit("✗ 快照不能放在程式資料夾裡面")
        files = pick(git_files())
        for f in files:
            os.makedirs(os.path.dirname(os.path.join(dest, f)) or dest, exist_ok=True)
            shutil.copy2(os.path.join(PROG, f), os.path.join(dest, f))
        write_licenses(dest, a.code_license, a.holder)
        print(f"── 公開快照：{dest}（{len(files)} 個檔，另加 LICENSE、LICENSES.md）")
    results = check(dest, priv, deny, run_tests=not a.no_tests)
    for name, ok, msg in results:
        print(f"  {'✓' if ok else '✗'} {name}：{msg}")
    if os.path.isfile(priv):
        print(f"  保留題私有檔 SHA-256：{text_sha256(priv)}（README「保留題」寫的要是這個值）")
    if not all(ok for _, ok, _ in results):
        print("✗ 有項目沒通過：這個快照不能發布")
        sys.exit(1)
    print("✓ 全部通過：可以交給作者檢查")


if __name__ == "__main__":
    main()
