# -*- coding: utf-8 -*-
"""v0.93 对照臂入口（评审 5——同工不同损：768 步 [LM,LM,LM]——唯一差异无意图损失）
直接复用 lora_train.Runner(mode='control')——批调度/步数/批大小/lr/LoRA 配置与主臂逐项一致。
"""
import sys
sys.argv = [sys.argv[0], '--mode', 'control'] + [a for a in sys.argv[1:] if a not in ('--mode', 'control')]
from lora_train import main

if __name__ == '__main__':
    main()
