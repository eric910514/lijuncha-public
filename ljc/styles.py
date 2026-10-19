"""對照模組：同一批人物與題目，只換送給模型的「寫法」，量出寫法本身造成的差異。

- script（簡繁）：人物介紹、提問、追問全部轉成簡體（流程見〈簡繁詞表〉v0.3；不用 OpenCC）：
    1. 「保留」清單的詞（法律名稱、法律用語、台灣制度名稱）先換成佔位符；
       「替換」清單的詞（一般名詞）也先換成佔位符，以繁體原詞比對，長片語優先。
    2. 其餘文字用 Unicode Unihan 的 kSimplifiedVariant 逐字轉簡體（config/script/unihan_t2s.json；
       補充字在設定檔 modules.script.extra_chars）；一個字有多個候選時取第一個和原字不同的；沒有對照的字照舊。
    3. 佔位符換回：保留詞只轉字形（同第 2 步，例如勞基法→劳基法），替換詞換成大陸用語（例如捷運→地铁）。
    4. 引號「」改為“”。
  提示文字只轉換一次：python lijuncha.py script 轉好後寫出 script_review.md 與固定資料 script_simp.json
  （放在人物資料夾），作者校對後執行時直接讀，不在執行時轉換。
  詞彙指標：繁體組的回覆（含遮罩字眼）逐字轉成簡體，兩組都在簡體上比；簡體回覆不轉回繁體。字數用原文字元數。
- punctuation（標點）：全形台灣慣用、半形、幾乎不加標點並以空格斷句；只改標點，其餘文字逐字相同。
- paraphrase（無關擾動）：首輪提問換成 q_paraphrase 的同義改寫，和主流程同一組設定的回覆對照。

報告把簡繁與標點稱為「書寫風格」，只描述寫法不同時回覆有什麼不同，不推論使用者是誰。"""
import hashlib
import json
import os
import re

STYLE_LABEL = {"orig": "原文", "trad": "繁體（台灣）", "simp": "簡體（大陸用語）",
               "full": "全形標點", "half": "半形標點", "none": "幾乎無標點", "para": "同義改寫"}
MODULES = {"script": ["trad", "simp"], "punctuation": ["full", "half", "none"], "paraphrase": ["para"]}
MODULE_LABEL = {"script": "簡繁", "punctuation": "標點", "paraphrase": "無關擾動"}
# 半形標點的對照（設定檔 modules.punctuation.half 可以覆寫）
HALF = {"，": ",", "。": ".", "？": "?", "！": "!", "：": ":", "；": ";", "、": ",",
        "「": '"', "」": '"', "『": "'", "』": "'", "（": "(", "）": ")", "…": "...", "～": "~"}
QUOTES = {"「": "“", "」": "”", "『": "‘", "』": "’"}


def module_questions(cfg, module=None):
    """這個模組要問的題目。主流程與無關擾動用 modules 含 main 的題目；簡繁、標點用各自標記的題目。"""
    key = "main" if module in (None, "", "main", "paraphrase") else module
    return [q for q in cfg["questions"] if key in q.get("modules", ["main"])]


T2S_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "script",
                        "unihan_t2s.json")
TABLE = "script_simp.json"          # 人物資料夾裡的固定資料：轉好、校對過的簡體提示文字
REVIEW = "script_review.md"
_T2S = {}


def script_cfg(cfg):
    return (cfg.get("modules") or {}).get("script", {})


class T2S:
    """逐字繁轉簡（Unihan kSimplifiedVariant）。char(text) 轉整段；multi_in(text) 列出用到的多候選字。"""

    def __init__(self, cfg=None):
        path = script_cfg(cfg or {}).get("unihan_table") or T2S_PATH
        if path not in _T2S:
            if not os.path.isfile(path):
                from .common import stop
                stop(f"找不到繁簡字對照表 {path}", "產生：python tools/make_unihan_table.py（從 unicode.org 下載 Unihan.zip）")
            with open(path, encoding="utf-8") as f:
                _T2S[path] = json.load(f)
        doc = _T2S[path]
        self.map = dict(doc["map"])
        self.extra = dict(script_cfg(cfg or {}).get("extra_chars", {}))
        self.map.update(self.extra)
        self.multi = doc.get("multi", {})
        self.source = doc.get("source", {})
        self.built = doc.get("built")
        # 「著」保留詞（詞表 v0.4）：念 zhù 的詞裡「著」不轉，其他字照常轉；其餘的「著」轉「着」
        self.zhu = sorted(script_cfg(cfg or {}).get("zhu_keep", []), key=len, reverse=True)
        self.fp = hashlib.md5(json.dumps([sorted(self.map.items()), self.zhu], ensure_ascii=False).encode()).hexdigest()[:8]

    def _plain(self, text):
        m = self.map
        return "".join(m.get(ch, ch) for ch in text or "")

    def char(self, text):
        """逐字轉簡體。「著」保留詞先換成佔位符（\ue002 編號 \ue003，和詞表佔位符分開），轉完再換回。"""
        if not self.zhu or "著" not in (text or ""):
            return self._plain(text)
        words = []
        for w in self.zhu:
            if w in text:
                text = text.replace(w, f"\ue002{len(words)}\ue003")
                words.append("".join(ch if ch == "著" else self.map.get(ch, ch) for ch in w))
        return re.sub("\ue002(\\d+)\ue003", lambda mt: words[int(mt.group(1))], self._plain(text))

    def multi_in(self, text):
        """(原字, 用的字, 全部候選)；只列真的有歧義的字：常用字範圍（BMP）裡有兩個以上候選（例如裡→裡／里、著→着／著）。
        另一個候選是擴充區罕用字的（例如闆→板／𬮳）不列。"""
        seen = []
        for ch in dict.fromkeys(text or ""):
            cands = self.multi.get(ch) or []
            if ch in self.map and ch not in self.extra and sum(ord(c) < 0x10000 for c in cands) >= 2:
                seen.append((ch, self.map[ch], cands))
        return seen

    def rare_in(self, text):
        """轉成擴充區罕用字（BMP 以外）的字：[(原字, 轉成的字)]。很多字型顯示不出來，模型也可能讀不懂。"""
        return [(ch, self.map[ch]) for ch in dict.fromkeys(text or "") if ch in self.map and ord(self.map[ch]) >= 0x10000]

    def describe(self):
        s = self.source
        where = s.get("unihan_version") or s.get("file") or s.get("url") or "來源不明"
        return f"Unihan kSimplifiedVariant（{where}；{len(self.map)} 字，指紋 {self.fp}）"


class ScriptConverter:
    """把繁體提示文字轉成簡體（只在 script 指令用一次；執行時讀轉好的固定資料）。"""

    def __init__(self, cfg):
        sc = script_cfg(cfg)
        self.t2s = T2S(cfg)
        keep = [(w, self.t2s.char(w), "保留") for w in sc.get("keep", [])]
        repl = [(a, b, "替換") for a, b in (tuple(t) for t in sc.get("replace", sc.get("terms", [])))]
        # 先保留、後替換；各自長的先比對（例如「全國性公民投票」先於「公民投票」）。
        # 替換詞可以寫成「在…打工」→「在…兼职」：中間是任意幾個字（不跨標點），只換兩頭，中間照常逐字轉換。
        self.rules, self.out = [], []
        for a, b, kind in sorted(keep, key=lambda t: -len(t[0])) + sorted(repl, key=lambda t: -len(t[0])):
            if "…" in a:
                a1, a2 = a.split("…", 1)
                b1, b2 = (b.split("…", 1) + [""])[:2]
                rx = re.compile(re.escape(a1) + r"([^\s，。？！；：、「」（）,.?!;:]{1,12}?)" + re.escape(a2))
                self.rules.append((a, b, kind, rx, [len(self.out), len(self.out) + 1]))
                self.out += [b1, b2]
            else:
                self.rules.append((a, b, kind, None, [len(self.out)]))
                self.out.append(b)
        self.quotes = sc.get("quotes", QUOTES)

    def _protect(self, text):
        """把清單裡的詞換成佔位符（私用區字元加編號，逐字轉換不會動），以繁體原文比對。回傳 (文字, 用到的規則)。"""
        used = []
        ph = lambda i: f"\ue000{i}\ue001"
        for k, (a, b, kind, rx, idx) in enumerate(self.rules):
            if rx is None:
                if a in text:
                    text = text.replace(a, ph(idx[0]))
                    used.append(k)
            else:
                text, n = rx.subn(lambda m: ph(idx[0]) + m.group(1) + ph(idx[1]), text)
                if n:
                    used.append(k)
        return text, used

    def _restore(self, text):
        return re.sub("\ue000(\\d+)\ue001", lambda m: self.out[int(m.group(1))], text)

    def convert(self, text):
        t, _ = self._protect(text)
        t = self._restore(self.t2s.char(t))
        for a, b in self.quotes.items():
            t = t.replace(a, b)
        return t

    def explain(self, text):
        """審閱用：(轉換後, 用到的清單詞 [(原詞, 結果, 保留／替換)], 逐字轉換用到的多候選字 [(原字, 用的字, 候選)],
        轉成擴充區罕用字的字 [(原字, 轉成的字)])。"""
        t, used = self._protect(text)
        return self.convert(text), [self.rules[k][:3] for k in used], self.t2s.multi_in(t), self.t2s.rare_in(t)


class MissingScript(Exception):
    pass


def table_path(pdir):
    return os.path.join(pdir, TABLE)


def load_table(pdir):
    """人物資料夾裡轉好的簡體提示文字：{"texts": [{where, orig, auto, simp}], "reviewed": bool, ...}；沒有時 None。"""
    p = table_path(pdir)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Styler:
    """把一段文字換成某種寫法。orig／trad／full／para 不改文字（para 換的是題目，在組訊息時處理）。
    simp 不在執行時轉換：查 script 指令轉好、校對過的固定資料（table）；查不到就丟 MissingScript。"""

    def __init__(self, cfg, style, table=None):
        self.style = style
        mc = (cfg.get("modules") or {})
        if style == "simp":
            self.table = {t["orig"]: t["simp"] for t in (table or {}).get("texts", [])}
            self.reviewed = bool((table or {}).get("reviewed"))
            self.sig = "simp|table"
        elif style in ("half", "none"):
            self.half = mc.get("punctuation", {}).get("half", HALF)
            self.marks = set(self.half) | set("．‧·—")
            self.sig = f"{style}|{sorted(self.half.items())}"
        else:
            self.sig = style

    def __call__(self, text):
        if self.style == "simp":
            if not text:
                return text or ""
            if text not in self.table:
                raise MissingScript(text)
            return self.table[text]
        if self.style == "half":
            return "".join(self.half.get(ch, ch) for ch in text)
        if self.style == "none":
            t = "".join(" " if ch in self.marks else ch for ch in text)
            return re.sub(r"\s+", " ", t).strip()
        return text

    def fingerprint(self, texts):
        return hashlib.md5("\n".join([self.sig] + [self(t) for t in texts]).encode("utf-8")).hexdigest()[:8]


def stylers_for(cfg, styles, out_dir, settings):
    """某組設定資料夾的各寫法；簡繁要讀那一版人物資料夾裡的固定資料。"""
    table = None
    if "simp" in styles:
        table = load_table(os.path.join(out_dir, "personas", settings.get("personas_version") or ""))
        if table is None:
            from .common import stop
            stop(f"找不到簡繁模組的固定資料（人物 {settings.get('personas_version')} 的 {TABLE}）", "先執行：python lijuncha.py script")
    return {st: Styler(cfg, st, table) for st in styles}


