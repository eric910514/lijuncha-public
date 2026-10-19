"""WGND 2.0 名字檔的存放與篩選（〈姓名資料〉第五節）。

原始檔與篩完的檔都放在 config/names/raw/wgnd/，整個 config/names/raw/ 不進版本控制（.gitignore）。

用法（在程式資料夾）：
    python tools/filter_wgnd.py --move ~/Downloads/wgnd_2_0_name-gender-code.csv    把下載的檔搬進來，印欄位與前 5 列
    python tools/filter_wgnd.py                                                      篩選

篩選規則：
- 只留需要 WGND 的國家：外籍身分的原國籍裡沒有官方名字統計的 9 國（越南、印尼、菲律賓、柬埔寨、泰國、緬甸、
  日本、韓國、馬來西亞）。--all-origin 改成留 origin_countries 全部 14 國（美、加、英、澳、法有官方資料，只當備用）。
- 某一性別的 wgt ≥ 0.9（--min-wgt）；同一名字在同一國兩種性別都沒達到的，當作「性別不確定」排除。
- 每國每性別取 nobs 前 100 名是 gen 的規則（names_top_n），這裡先不截，全部留下。
- 實際的 name-gender-code 檔只有 name、code、gender、wgt 四欄，沒有 nobs（〈姓名資料〉第五節寫的五欄和檔案不符）；
  沒有 nobs 時輸出的 nobs 欄留空，排序方式待企劃書決定（待決第 24 項）。
- 名字前後的引號（例如 "baby"、'aisyah）去掉；去掉後沒有字母的略過。

資料：Raffo, Julio, 2021, "WGND 2.0", https://doi.org/10.7910/DVN/MSEGSJ, Harvard Dataverse, V1（CC0 1.0）。"""
import argparse
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "config", "names", "raw", "wgnd")
SRC_NAME = "wgnd_2_0_name-gender-code.csv"
OUT_NAME = "wgnd_2_0_filtered.csv"
# 原國籍（中文，和 origin_countries 一字不差）→ ISO 3166-1 二碼
CODES = {"越南": "VN", "印尼": "ID", "菲律賓": "PH", "柬埔寨": "KH", "泰國": "TH", "緬甸": "MM", "日本": "JP",
         "韓國": "KR", "馬來西亞": "MY", "美國": "US", "加拿大": "CA", "英國": "GB", "澳大利亞": "AU", "法國": "FR"}
NEEDED = ["越南", "印尼", "菲律賓", "柬埔寨", "泰國", "緬甸", "日本", "韓國", "馬來西亞"]
# 欄位名稱的候選（WGND 說明寫的是 name、code、gender、wgt、nobs；大小寫或別名不同也認得）
COLS = {"name": ["name", "given_name", "firstname"], "code": ["code", "country", "iso2", "country_code"],
        "gender": ["gender", "sex"], "wgt": ["wgt", "weight", "prob"], "nobs": ["nobs", "n", "count"]}
GENDER = {"m": "male", "f": "female"}
OPTIONAL = {"nobs"}          # 實際檔案沒有這欄（2026-10-16 作者本機實測）


def open_csv(path):
    f = open(path, encoding="utf-8-sig", newline="")
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return f, csv.reader(f, dialect)


def find_cols(header):
    low = [h.strip().lower() for h in header]
    got = {}
    for k, cands in COLS.items():
        for c in cands:
            if c in low:
                got[k] = low.index(c)
                break
    return got


def clean(name):
    """去掉名字前後的引號與空白；沒有任何字母就回傳空字串。"""
    name = name.strip().strip("'\"`‘’“”").strip()
    return name if any(ch.isalpha() for ch in name) else ""


def peek(path, n=5):
    f, rd = open_csv(path)
    with f:
        header = next(rd)
        print("欄位：" + "、".join(header))
        for i, row in enumerate(rd):
            if i >= n:
                break
            print(f"  {i + 1}: " + " | ".join(row))
    cols = find_cols(header)
    missing = [k for k in COLS if k not in cols and k not in OPTIONAL]
    if missing:
        print("認不出這些欄位：" + "、".join(missing) + "。請把上面的欄位名稱貼給程式對話。")
    elif "nobs" not in cols:
        print("注意：沒有 nobs 欄（出現次數）。照樣篩選，輸出的 nobs 留空；每國每性別怎麼取前 100 名待企劃書決定。")
    return header, cols, missing


def filter_file(path, out, countries, min_wgt):
    want = {CODES[c]: c for c in countries}
    f, rd = open_csv(path)
    n_all = n_keep = n_unsure = 0
    per = {}
    with f, open(out, "w", encoding="utf-8", newline="") as g:
        header = next(rd)
        cols = find_cols(header)
        if any(k not in cols for k in COLS if k not in OPTIONAL):
            sys.exit("欄位認不出來，先執行 --peek 看欄位名稱。")
        w = csv.writer(g)
        w.writerow(["name", "code", "origin_country", "gender_lean", "wgt", "nobs"])
        for row in rd:
            n_all += 1
            try:
                code = row[cols["code"]].strip().upper()
                if code not in want:
                    continue
                lean = GENDER.get(row[cols["gender"]].strip().lower()[:1])
                wgt = float(row[cols["wgt"]])
                nobs = int(float(row[cols["nobs"]] or 0)) if "nobs" in cols else ""
            except (IndexError, ValueError):
                continue
            name = clean(row[cols["name"]])
            if not name or lean is None or wgt < min_wgt:
                n_unsure += lean is not None
                continue
            w.writerow([name, code, want[code], lean, wgt, nobs])
            n_keep += 1
            per[(want[code], lean)] = per.get((want[code], lean), 0) + 1
    return n_all, n_keep, n_unsure, per


def show_sample(path, codes, n=15):
    """篩完的檔裡，指定國家每個性別的前 n 個名字（看寫法：拉丁字母、漢字或韓文，有沒有雜項）。"""
    if not os.path.isfile(path):
        sys.exit(f"找不到 {path}，先篩選。")
    got = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["code"] in codes:
                got.setdefault((r["code"], r["gender_lean"]), []).append(r["name"])
    for c in codes:
        for lean, zh in (("female", "女"), ("male", "男")):
            names = got.get((c, lean), [])
            print(f"{c} {zh}（{len(names)}）：" + "、".join(names[:n]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move", default="", help="把下載的原始檔搬進 config/names/raw/wgnd/（例如 ~/Downloads/" + SRC_NAME + "）")
    ap.add_argument("--peek", action="store_true", help="只印欄位名稱與前 5 列，不篩選")
    ap.add_argument("--min-wgt", type=float, default=0.9)
    ap.add_argument("--sample", default="", help="看篩完的檔裡某幾國的名字（國碼，逗號分隔，例如 JP,KR,VN），每國每性別印前 15 個")
    ap.add_argument("--all-origin", action="store_true", help="留 origin_countries 全部 14 國（預設只留沒有官方資料的 9 國）")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    src = os.path.join(RAW, SRC_NAME)
    if a.sample:
        return show_sample(os.path.join(RAW, OUT_NAME), [c.strip().upper() for c in a.sample.split(",") if c.strip()])
    if a.move:
        p = os.path.expanduser(a.move)
        if not os.path.isfile(p):
            sys.exit(f"找不到 {p}")
        if os.path.exists(src):
            sys.exit(f"{src} 已經有了；確定要換掉的話先刪掉舊的。")
        shutil.move(p, src)
        print(f"已搬到 {src}（{os.path.getsize(src) / 1e6:.1f} MB）")
    if not os.path.isfile(src):
        sys.exit(f"找不到 {src}。先用 --move 把下載的檔搬進來。")
    _, _, missing = peek(src)
    if a.peek or a.move or missing:
        if not missing:
            print("欄位都認得。確認無誤後再執行一次（不加參數）就會篩選。")
        return
    countries = list(CODES) if a.all_origin else NEEDED
    out = os.path.join(RAW, OUT_NAME)
    n_all, n_keep, n_unsure, per = filter_file(src, out, countries, a.min_wgt)
    print(f"\n原始 {n_all:,} 列 → 留下 {n_keep:,} 列（{len(countries)} 國、某一性別 wgt ≥ {a.min_wgt}；"
          f"這幾國裡性別不確定而排除 {n_unsure:,} 列）→ {out}")
    print("各國各性別的名字數（gen 會再取 nobs 前 names_top_n 名）：")
    for c in countries:
        print(f"  {c}（{CODES[c]}）：女 {per.get((c, 'female'), 0):,}、男 {per.get((c, 'male'), 0):,}")


if __name__ == "__main__":
    main()
