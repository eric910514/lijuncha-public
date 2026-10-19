"""產生簡繁模組的逐字對照表：config/script/unihan_t2s.json（〈簡繁詞表〉v0.3：不用 OpenCC，改用 Unicode Unihan）。

只取 kSimplifiedVariant（繁轉簡）。一個字有兩個以上候選時，取第一個和原字不同的候選（例如裡→里、著→着、徵→征），
所有候選另存在 multi，script_review.md 會把用到的這些字標出來。Unihan 沒收的字（遊、週）放在設定檔
modules.script.extra_chars，不寫進這個表。

用法（在程式資料夾）：
    python tools/make_unihan_table.py                      從 unicode.org 下載最新的 Unihan.zip
    python tools/make_unihan_table.py --zip Unihan.zip     用已下載的 Unihan.zip（例如 17.0）
    python tools/make_unihan_table.py --txt kSimplifiedVariant.txt   用單一屬性檔（GitHub unicode-org/unihan-database 的格式）

輸出會記下來源、檔案指紋與產生時間；和舊表不同時列出差異的字數。Unicode 資料依 Unicode License v3 可再散布，
表頭附版權聲明。"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "script", "unihan_t2s.json")
LICENSE = ("Unicode Unihan Database © 1991-2025 Unicode, Inc. Distributed under the Unicode License v3 "
           "(https://www.unicode.org/license.txt).")


def parse(lines):
    """kSimplifiedVariant 的每一行 → {字: [候選字…]}。接受 Unihan.zip 的「U+88E1<TAB>kSimplifiedVariant<TAB>U+88E1 U+91CC」
    和 GitHub 屬性檔的「U+88E1 裡<TAB>kSimplifiedVariant<TAB>…」兩種格式。"""
    out = {}
    for line in lines:
        if not line.startswith("U+"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[1] != "kSimplifiedVariant":
            continue
        ch = chr(int(parts[0].split()[0][2:], 16))
        out[ch] = [chr(int(u[2:].split("<")[0], 16)) for u in parts[2].split() if u.startswith("U+")]
    return out


def build(variants):
    table, multi = {}, {}
    for ch, cands in sorted(variants.items()):
        pick = next((c for c in cands if c != ch), None)
        if pick:
            table[ch] = pick
        if len(cands) > 1:
            multi[ch] = cands
    return table, multi


def read_source(a):
    if a.txt:
        data = open(a.txt, "rb").read()
        return data, data.decode("utf-8").splitlines(), {"file": os.path.basename(a.txt)}
    if a.zip:
        data = open(a.zip, "rb").read()
        src = {"file": os.path.basename(a.zip)}
    else:
        print(f"下載 {URL} …")
        with urllib.request.urlopen(URL, timeout=120) as r:
            data = r.read()
        src = {"url": URL}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = z.read("Unihan_Variants.txt").decode("utf-8")
    return data, text.splitlines(), src


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="", help="已下載的 Unihan.zip")
    ap.add_argument("--txt", default="", help="單一屬性檔 kSimplifiedVariant.txt")
    ap.add_argument("--version", default="", help="Unihan 版本（例如 17.0），記在表頭")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    data, lines, src = read_source(a)
    variants = parse(lines)
    if not variants:
        sys.exit("來源裡找不到 kSimplifiedVariant")
    table, multi = build(variants)
    src.update({"sha256": hashlib.sha256(data).hexdigest(), "unihan_version": a.version or None})
    old = {}
    if os.path.exists(a.out):
        old = json.load(open(a.out, encoding="utf-8")).get("map", {})
    doc = {"_說明": "繁轉簡逐字對照（Unicode Unihan kSimplifiedVariant）。map＝每個字要換成的字（多個候選時取第一個和原字不同的）；"
                   "multi＝有兩個以上候選的字與全部候選。由 tools/make_unihan_table.py 產生，不要手改；補充字放設定檔 "
                   "modules.script.extra_chars。",
           "_license": LICENSE, "source": src, "built": time.strftime("%Y-%m-%d %H:%M"),
           "n": len(table), "map": table, "multi": multi}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=0, sort_keys=False)
        f.write("\n")
    diff = sum(1 for k in set(old) | set(table) if old.get(k) != table.get(k))
    print(f"寫出 {a.out}：{len(table)} 個字（其中 {len(multi)} 個有多個候選）"
          + (f"；和舊表不同 {diff} 個字" if old else ""))


if __name__ == "__main__":
    main()
