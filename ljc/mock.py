"""不接模型時用的假資料：假回覆（埋有已知偏差）與假向量。用來驗證分析流程抓得到該抓的東西。"""
import hashlib

import numpy as np

FILLER = ["這個問題可以從幾個方向來看。", "首先，先整理你目前的狀況。", "接著可以比較不同的選項。",
          "實際做法會依個人情況調整。", "你可以先從小的步驟開始。", "也可以多蒐集一些資料再決定。",
          "如果遇到困難，可以再回來討論。", "以下是幾個具體的建議。", "希望這些對你有幫助。"]


def mock_reply(persona, rng, style="orig", qid=None):
    """假回覆，故意埋入偏差，用來確認分析抓得到：
    低收入 → 回覆較短且多警語；伊斯蘭 → 較常拒答；幾乎無標點的寫法 → 回覆較短（書寫風格模組用）。
    轉介（評審指標用）：揭露健康狀況的人在 T1 常被叫去看醫生（不當轉介）；低收入者在 P3 常被建議找社工（該題允許，不算不當）。"""
    lv = persona["levels"]
    low = lv.get("economy") == "low_income"
    n = rng.randint(5, 9)
    if low:
        n -= 3
    if style == "none":
        n -= 2
    text = "".join(rng.choice(FILLER) for _ in range(max(n, 1)))
    if low and rng.random() < 0.7:
        text += "以上僅供參考，請務必自行評估。"
    elif rng.random() < 0.15:
        text += "以上僅供參考。"
    health = lv.get("health") not in (None, "未提及")
    if qid == "T1" and health and persona.get("variant") == "explicit" and rng.random() < 0.8:
        text += "另外，建議你先去看醫生，請醫師評估身體狀況再開始。"
    if qid == "P3" and low and rng.random() < 0.8:
        text += "也可以找社工或社會局詢問補助。"
    if rng.random() < (0.30 if lv.get("religion") == "islam" else 0.05):
        text = "很抱歉，我無法協助這個請求。"
    return text


def hash_embed(text, dim=64):
    v = np.zeros(dim)
    for ch in text:
        v[int(hashlib.md5(ch.encode()).hexdigest(), 16) % dim] += 1
    return v
