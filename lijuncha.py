#!/usr/bin/env python3
"""離均差原型的入口。程式本體在 ljc/ 資料夾，說明見 README.md。

  python lijuncha.py gen                      產生人物
  python lijuncha.py check --model <模型名>    確認模型回覆正常
  python lijuncha.py run --model <模型名>      對本機模型提問（可中斷續跑）
  python lijuncha.py status                   查看進度
  python lijuncha.py analyze                  統計並輸出報告

每個指令加 -h 可以看選項，例如 python lijuncha.py run -h
"""
from ljc.cli import main

if __name__ == "__main__":
    main()
