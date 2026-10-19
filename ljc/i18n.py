"""介面語言：預設中文；--lang en 或環境變數 LJC_LANG=en 改成英文（公開版的內部測試）。

做法：程式裡的中文字串都不改，印出時（終端機與 report.md）逐行比對 EN_LINES 的句型換成英文，
再把剩下的固定詞（EN_TERMS，例如欄位名稱、單位）換掉。沒有對應的句子照樣印中文。
不翻的東西：送給模型的人物句、題目與回覆（研究內容），results.csv 等資料檔，-h 的說明。
之後的新功能預設只做中文；作者要求時再補英文句型。"""
import os
import re
import sys

STATE = {"lang": (os.environ.get("LJC_LANG") or "zh").strip().lower()}


def set_lang(code):
    STATE["lang"] = (code or "zh").strip().lower()


def is_en():
    return STATE["lang"].startswith("en")


EN_LINES = []          # [(regex, 英文)]：整句或句子開頭，依序比對
EN_TERMS = []          # [(中文, 英文)]：整句換完後剩下的固定詞，長的先換


def sep():
    """標籤和內容之間的冒號：中文用全形，英文用半形加空格。"""
    return ": " if is_en() else "："


def label(zh):
    """區塊標題與標籤：英文時查 LABELS（含「進度報告（14:39）」這種帶時間的標題），查不到就整句翻。"""
    if not is_en() or not zh:
        return zh
    from .i18n_en import LABELS
    if zh in LABELS:
        return LABELS[zh]
    m = re.fullmatch(r"(.+?)（(\d\d:\d\d)）", zh)
    if m and m.group(1) in LABELS:
        return f"{LABELS[m.group(1)]} ({m.group(2)})"
    return _line(zh)


def translate(text):
    if not is_en() or not text:
        return text
    return "\n".join(_line(x) for x in text.split("\n"))


def translate_doc(text):
    return translate(text)


_CJK = re.compile(r"[㐀-鿿！-～　-〿]")


def _line(line):
    if not _CJK.search(line) or line.lstrip().startswith(("[第 ", "[系統提示]")):
        return line
    for pat, rep in EN_LINES:
        line = pat.sub(rep, line)          # rep 可以是字串（\1 代表第一組）或函式
        if not _CJK.search(line):
            return line
    done = line
    for zh, en in EN_TERMS:
        if zh in done:
            done = done.replace(zh, en)
    # 固定詞只在整行都能翻完時才採用：翻一半會出現「none法」這種中英夾雜，不如保留中文
    return line if _CJK.search(done) else done


class _Writer:
    """包住 sys.stdout／sys.stderr：湊滿一行再翻譯後寫出。"""

    def __init__(self, raw):
        self.raw, self.buf = raw, ""

    def write(self, s):
        self.buf += s
        if "\n" in self.buf:
            done, self.buf = self.buf.rsplit("\n", 1)
            self.raw.write(translate(done) + "\n")
        return len(s)

    def flush(self):
        if self.buf:
            self.raw.write(translate(self.buf))
            self.buf = ""
        self.raw.flush()

    def __getattr__(self, k):
        return getattr(self.raw, k)


class output:
    """with i18n.output(): 這段期間的 print 都經過翻譯（英文時）；中文時什麼都不做。結束時還原。"""

    def __enter__(self):
        self.saved = None
        if is_en() and not isinstance(sys.stdout, _Writer):
            self.saved = (sys.stdout, sys.stderr)
            sys.stdout, sys.stderr = _Writer(sys.stdout), _Writer(sys.stderr)
        return self

    def __exit__(self, *exc):
        if self.saved:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = self.saved
        return False


def _rules():
    from .i18n_en import LINES, TERMS
    EN_LINES[:] = [(re.compile(p), r) for p, r in LINES]
    EN_TERMS[:] = sorted(TERMS.items(), key=lambda x: -len(x[0]))


_rules()
