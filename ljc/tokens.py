"""嵌入前的 token 檢查（規格「詞彙層」）：每段用 LM Studio 載入的那個嵌入模型自己的分詞器實數 token（含前綴），
超過設定檔 embedding.max_tokens（預設 510）就把該段對半切。

數 token 走 LM Studio 的官方 Python 套件 lmstudio（Element Labs，美國）：python -m pip install lmstudio。
它問的是 LM Studio 裡實際載入的模型，所以 e5 與 xVITA 各用各的分詞器，不必另外下載分詞器檔。
--mock 時用字元數代替（只測流程）。"""
from urllib.parse import urlparse

from .common import stop

INSTALL = "python -m pip install lmstudio"


class LMStudioCounter:
    def __init__(self, base_url, model):
        try:
            import lmstudio
        except ImportError:
            stop("缺 LM Studio 的 Python 套件（lmstudio）", "詞彙層用嵌入模型自己的分詞器數 token（超過上限的段再切）",
                 f"安裝：{INSTALL}")
        host = urlparse(base_url).netloc or "localhost:1234"
        try:
            self.client = lmstudio.Client(host)
            self.handle = self.client.embedding.model(model)
        except Exception as e:      # 連不上、模型名稱不符
            stop(f"用 LM Studio 數 token 失敗（{host}，模型 {model}）：{e}", "LM Studio 已開啟伺服器", "嵌入模型已載入",
                 "名稱和 --embed-model／設定檔 embedding.models 一致")
        self.name = "LM Studio 分詞器"

    def count(self, texts):
        return [int(self.handle.count_tokens(t)) for t in texts]

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class MockCounter:
    name = "字元數（--mock）"

    def count(self, texts):
        return [len(t) for t in texts]

    def close(self):
        pass


def counter(base_url, model, use_mock):
    return MockCounter() if use_mock else LMStudioCounter(base_url, model)


def kind_of(text):
    """再切的段是哪一類：英文多、數字多、符號或空白多、一般中文。給報告說明是什麼樣的回覆造成的。"""
    n = max(len(text), 1)
    latin = sum(ch.isascii() and ch.isalpha() for ch in text) / n
    digit = sum(ch.isdigit() for ch in text) / n
    other = sum(not ch.isalnum() for ch in text) / n
    if latin >= 0.3:
        return "英文多"
    if digit >= 0.2:
        return "數字多"
    if other >= 0.3:
        return "符號或空白多（表格、清單）"
    return "一般中文"
