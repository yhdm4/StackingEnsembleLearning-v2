"""文件内容与用途：论文4.3（旧编号42）比较基预测与基预测加原始特征，复用论文4.2联合输入模型。"""

from pathlib import Path
import pandas as pd
from main import write_csv
from train_4_1 import run_section41, run_pure_input


def run_section42(name, run_dir):
    """复用本轮论文4.2联合输入结果并训练 base-only，输出六种输入/模型组合。"""
    joint = run_section41(name, run_dir)
    base = run_pure_input(name, run_dir, True)
    write_csv(
        Path(run_dir) / "datasets" / name / "section42" / "metrics.csv",
        pd.concat([base, joint], ignore_index=True),
    )


if __name__ == "__main__":
    from main import main

    main("42")
