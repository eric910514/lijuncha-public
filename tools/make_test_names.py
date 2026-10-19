"""產生外籍身分的「測試用」原文名：config/names/names_foreign_test.jsonl。

正式資料到齊前，--draft 可以用這些名字試流程（〈姓名資料取得狀況〉）。它們來自 Faker 的語系詞表，
不是人口統計：每筆都標 test_only: true，gen 正式模式不會用。

用法（在程式資料夾）：
    python -m pip install faker
    python tools/make_test_names.py

只產生有語系的國家；菲律賓、馬來西亞、緬甸、柬埔寨沒有可用的語系，測試時不會輪到。
名字用 first_name_*／last_name 直接組，不用 Faker 的 name()，所以不會帶「Quý cô」「Ông」這類稱謂或「S.H.」這類學位。
泰文、韓文測試期間保留原文字母（企劃書同意）。"""
import argparse
import json
import os

# 國家 → (Faker 語系, 姓名順序, 分隔)
LOCALES = {
    "越南": ("vi_VN", "family_first", " "),
    "印尼": ("id_ID", "given_first", " "),
    "泰國": ("th_TH", "given_first", " "),
    "日本": ("ja_JP", "family_first", " "),
    "韓國": ("ko_KR", "family_first", ""),
    "美國": ("en_US", "given_first", " "),
    "加拿大": ("en_CA", "given_first", " "),
    "英國": ("en_GB", "given_first", " "),
    "澳大利亞": ("en_AU", "given_first", " "),
    "法國": ("fr_FR", "given_first", " "),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "config", "names", "names_foreign_test.jsonl"))
    ap.add_argument("--per", type=int, default=40, help="每國每性別幾個名字")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import faker
    recs = []
    for country, (loc, order, sep) in LOCALES.items():
        f = faker.Faker(loc)
        f.seed_instance(a.seed)
        for lean in ("female", "male"):
            first = getattr(f, f"first_name_{lean}")
            seen = set()
            for _ in range(a.per * 20):
                if len(seen) >= a.per:
                    break
                g, s = first(), f.last_name()
                full = (s + sep + g) if order == "family_first" else (g + sep + s)
                if full in seen:
                    continue
                seen.add(full)
                recs.append({"record_type": "full_name", "name": full, "origin_country": country, "gender_lean": lean,
                             "count": None, "test_only": True,
                             "source": {"source_id": "FAKER-TEST",
                                        "citation": f"Faker {faker.VERSION} {loc}，產生器詞表，非人口統計",
                                        "url": "https://github.com/joke2k/faker", "locator": f"{loc} first_name_{lean} + last_name"}})
    recs.sort(key=lambda r: (r["origin_country"], r["gender_lean"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(recs, 1):
            fh.write(json.dumps({"id": f"names_foreign_test-{i:05d}", **r}, ensure_ascii=False) + "\n")
    print(f"寫入 {len(recs)} 筆測試名字（{len(LOCALES)} 國 × 男女 × {a.per}）→ {a.out}")


if __name__ == "__main__":
    main()
