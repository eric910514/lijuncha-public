"""自動決定同時跑幾段對話。只做判斷，不碰網路也不碰檔案，所以可以單獨測試。

規則：
1. 測速：每換一個並行數，先丟掉「並行數」段當暖機，再收 max(3, 並行數) 段的速度。
2. 往上試：依 STEPS 的階梯加一階；整體速度至少要拿到理想增幅的四分之一才採用，
   採用了就繼續往上，否則退回上一階並定案。
3. 定案後持續看速度：
   - 每隔 explore_gap（預設 15 分鐘）再試高一階，有變快就採用並繼續往上。
     同一台機器常常同時在做別的事（瀏覽網頁、算圖、玩遊戲），能用的算力會變。
   - 整體速度掉超過 20%，退一階重新測（兩次至少隔 retune_gap）。
4. 伺服器回報容量不足就降一段。"""
import time
from collections import deque

TUNE_RULE = 4   # 判斷規則的版本；規則改了，各機器記下的舊結果就重測


class Tuner:
    STEPS = [1, 2, 3, 4, 6, 8, 12, 16]

    def __init__(self, start, cap, fixed, say, retune_gap=20 * 60, explore_gap=15 * 60):
        self.cap = cap
        self.level = min(start, cap)
        self.fixed = fixed            # 使用者指定了固定數字
        self.probing = not fixed
        self.best = None              # (level, 整體速度)
        self.samples = []
        self.warm = self.level        # 換到新的並行數後，先丟掉這麼多段不計（暖機）
        self.cap_errors = deque(maxlen=10)
        self.say = say
        self.retune_gap, self.explore_gap = retune_gap, explore_gap
        now = time.time()
        self.watch, self.ref, self.last_retune, self.last_explore = [], None, now, now
        self.measured = {}            # 這次測速量到的整體速度 {並行數: 速度}，只用來顯示與存檔

    # ---------- 存檔與接續 ----------
    def needed(self):
        """目前這一階要收幾個樣本才能判斷。"""
        return max(3, self.level)

    def snapshot(self):
        """測速進行中的進度，給 runner 存進 tuning.json，下次接著測。
        不在測速、或還沒量到任何東西時回傳 None。"""
        if self.fixed or not self.probing or not (self.best or self.samples or self.measured):
            return None
        return {"level": self.level, "cap": self.cap,
                "best": [self.best[0], round(self.best[1], 3)] if self.best else None,
                "samples": [round(x, 4) for x in self.samples],
                "measured": {str(k): round(v, 1) for k, v in self.measured.items()}}

    def resume(self, snap):
        """接續上次中斷的測速：已量到的最佳值與這一階已收的樣本都沿用。
        暖機要重做：這次起跑時伺服器又是空的，最先完成的幾段同樣會偏快。"""
        self.cap = max(1, min(self.cap, int(snap.get("cap") or self.cap)))
        best = snap.get("best")
        self.best = (int(best[0]), float(best[1])) if best else None
        if self.best and self.best[0] > self.cap:
            self.best = None                      # 上限變小了，舊的最佳值已經用不到
        self.measured = {int(k): float(v) for k, v in (snap.get("measured") or {}).items()}
        level, samples = int(snap["level"]), [float(x) for x in snap.get("samples") or []]
        if level > self.cap:
            if self.best:
                self.settle(self.best[0], f"上次測到一半的 {level} 段超過這次的上限")
                return
            level, samples = self.cap, []
        self.probing, self.level, self.samples, self.warm = True, level, samples, level

    # ---------- 判斷 ----------
    def _next_up(self):
        for x in self.STEPS:
            if x > self.level:
                return x if x <= self.cap else None
        return None

    def _go(self, level):
        self.probing, self.level, self.samples, self.warm = True, level, [], level

    def settle(self, level, why):
        self.level = max(1, min(level, self.cap))
        self.probing = False
        self.samples, self.watch = [], []
        self.ref = self.best[1] if self.best and self.best[0] == self.level else None
        self.last_explore = time.time()
        self.say(f"  ◆ 定為 {self.level} 段：{why}")

    def set_cap(self, cap, why):
        if cap < self.cap:
            self.cap = max(1, cap)
            if self.level > self.cap:
                self.level = self.cap
                self.samples = []
                self.say(f"  ◆ 降為 {self.level} 段：{why}")

    def raise_cap(self, cap, why):
        """實際用量比估計小，上下文容得下更多段：放寬上限，並繼續往上試。"""
        if cap <= self.cap or self.fixed:
            return
        was_capped = not self.probing and self.best is not None and self.level == self.cap
        self.cap = cap
        if was_capped:
            up = self._next_up()
            if up:
                self.say(f"  ◆ {why}，上限放寬到 {cap} 段，試 {up} 段")
                self._go(up)

    def need_gain(self, lo, hi):
        """從 lo 段加到 hi 段，理想上會快 (hi-lo)/lo；至少要拿到其中四分之一才算值得。
        例：4 段到 6 段理想是快 50%，所以要快 12.5% 以上才留在 6 段。"""
        return 1 + 0.25 * (hi - lo) / lo

    def on_ok(self, rate, at_level):
        """rate：這段對話每秒生成的 token 數；at_level：它開始時的並行數。"""
        self.cap_errors.append(0)
        if at_level != self.level:
            return
        if not self.probing:
            self._watch(rate)
            return
        if self.warm > 0:
            # 剛換並行數時，最先完成的幾段不用排隊、會顯得特別快；先丟掉一輪不計
            self.warm -= 1
            return
        self.samples.append(rate)
        if len(self.samples) < self.needed():
            return
        lv = self.level
        thr = lv * sum(self.samples) / len(self.samples)
        self.measured[lv] = thr
        if self.best is None or thr >= self.best[1] * self.need_gain(self.best[0], lv):
            gain = "" if self.best is None else f"，比 {self.best[0]} 段快 {thr / self.best[1] - 1:.0%}"
            self.best = (lv, thr)
            up = self._next_up()
            if up is None:
                self.settle(lv, f"每秒 {thr:.0f} token{gain}，已到上限")
            else:
                self.say(f"  ◆ {lv} 段：每秒 {thr:.0f} token{gain}。試 {up} 段"
                         f"（要快 {self.need_gain(lv, up) - 1:.0%} 才換）")
                self._go(up)
        else:
            b = self.best
            self.settle(b[0], f"{lv} 段沒有更快（每秒 {thr:.0f} token，{thr / b[1] - 1:+.0%}，"
                              f"門檻 {self.need_gain(b[0], lv) - 1:.0%}）")

    def _watch(self, rate):
        """定案之後持續看速度：掉很多就退一階重測；隔一段時間就試高一階。"""
        if self.fixed:
            return
        self.watch.append(rate)
        if len(self.watch) < max(6, 2 * self.level):
            return
        thr = self.level * sum(self.watch) / len(self.watch)
        self.watch.clear()
        now = time.time()
        if self.ref is not None and thr < self.ref * 0.80 and now - self.last_retune > self.retune_gap \
                and self.level > 1:
            down = max(x for x in self.STEPS if x < self.level)
            self.say(f"  ◆ 速度掉了（每秒 {self.ref:.0f} → {thr:.0f} token），從 {down} 段重新測")
            self.last_retune = now
            self.best, self.ref, self.measured = None, None, {}
            self._go(down)
            return
        self.ref = thr if self.ref is None or thr >= self.ref else max(self.ref * 0.9, thr)  # 慢慢跟著現況走
        up = self._next_up()
        if self.explore_gap and up and now - self.last_explore >= self.explore_gap:
            # 以剛量到的速度當比較基準：機器狀況隨時在變，不拿很久以前的數字比
            self.say(f"  ◆ 試試 {up} 段會不會更快（{self.level} 段目前每秒 {thr:.0f} token）")
            self.best, self.measured = (self.level, thr), {self.level: thr}
            self.last_explore = now
            self._go(up)

    def on_pause(self):
        """電腦睡著過：睡前收的樣本、定案後的速度紀錄都作廢（跨過暫停的那幾段耗時含睡著的時間），重收一輪。"""
        self.samples, self.watch = [], []
        self.warm = self.level
        now = time.time()
        self.last_retune = self.last_explore = now

    def on_capacity_error(self):
        self.cap_errors.append(1)
        if self.level <= 1:
            return
        if self.probing:
            self.cap = self.level - 1
            if self.best:
                self.settle(self.best[0], f"{self.level} 段時伺服器容量不足")
            else:                      # 起跑的段數就太多：降一段再量
                self.say(f"  ◆ 伺服器容量不足，降為 {self.cap} 段")
                self._go(self.cap)
        elif sum(self.cap_errors) >= 2:
            self.cap_errors.clear()
            self.cap = self.level - 1
            self.level = self.cap
            self.say(f"  ◆ 伺服器容量不足，降為 {self.level} 段")
