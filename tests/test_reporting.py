# -*- coding: utf-8 -*-
"""文件内容与用途：验证只汇总入口不启动训练、保留逐折记录，并先逐折开平方再汇总RMSE曲线。"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import main as core
import train_4_1 as fitting


class ReportingTests(unittest.TestCase):
    """以正式逐折记录和可区分两种开平方顺序的样例检查复现入口。"""

    def test_saved_record_cli_preserves_inputs_and_never_trains(self):
        """运行真实汇总命令的分支，要求原CSV不变且训练、冻结与选参均不被调用。"""
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);(output/'summary').mkdir()
            before={}
            for section in ['41','42','43','44']:
                filename=f'section{section}_all_folds.csv';target=output/'summary'/filename
                shutil.copyfile(core.ROOT/'results/summary'/filename,target)
                before[filename]=hashlib.sha256(target.read_bytes()).hexdigest()
            original=core.publication_analysis

            def short_analysis(path):
                """仅缩小测试中的置换次数，保留实际汇总与统计计算。"""
                return original(path,permutations=31,concordance_permutations=31)

            with mock.patch('sys.argv',['main.py','--stage','summarize','--output',str(output)]), \
                 mock.patch.object(core,'publication_analysis',side_effect=short_analysis), \
                 mock.patch.object(core,'train_unified_job',side_effect=AssertionError('不应启动训练')), \
                 mock.patch.object(core,'train_mlp',side_effect=AssertionError('不应训练神经模型')), \
                 mock.patch.object(core,'select_mlp_configs',side_effect=AssertionError('不应重新选择参数')), \
                 mock.patch.object(core,'freeze_unified_models',side_effect=AssertionError('不应冻结新模型')), \
                 mock.patch.object(fitting,'plot_fivefold_curves'),mock.patch.object(core,'plot_publication_fit'):
                core.main()
            for filename,digest in before.items():
                self.assertEqual(hashlib.sha256((output/'summary'/filename).read_bytes()).hexdigest(),digest)
            records=json.loads((output/'statistics/friedman.json').read_text(encoding='utf-8'))
            self.assertEqual([(r['section'],r['metric']) for r in records],
                             [('41','RMSE'),('42','MSE'),('43','RMSE'),('44','RMSE')])
            for row,expected in zip(records,[12,30,58.43589743589746,39.87179487179486]):
                self.assertAlmostEqual(row['friedman_statistic'],expected,places=10)
            for section,expected in [('43',1.3),('44',3.8666666666666667)]:
                ranks=pd.read_csv(output/f'summary/section{section}_overall_average_ranks.csv',encoding='utf-8-sig')
                self.assertIn('RMSE_rank',ranks)
                self.assertNotIn('MSE_rank',ranks)
                self.assertAlmostEqual(float(ranks.loc[ranks.method_key=='Proposed','mean_rank'].iloc[0]),expected)
            self.assertTrue(json.loads((output/'statistics/verification.json').read_text(encoding='utf-8'))['no_retraining'])

    def test_rmse_curves_take_square_root_per_trial(self):
        """用五折MSE为1、4、9、16、25的样例，要求曲线均值为3而非sqrt(11)。"""
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);source=output/'curves';source.mkdir()
            for dataset in core.DATASETS:
                with (source/(dataset+'.csv')).open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['outer_fold','epoch','eval_mse','LR_mse','Ridge_mse'])
                    writer.writeheader()
                    for fold,value in enumerate([1,4,9,16,25]):
                        for epoch in [1,2]:writer.writerow(dict(outer_fold=fold,epoch=epoch,eval_mse=value,LR_mse=value,Ridge_mse=value))
            captured=[]

            def capture_rows(figure,axes,destination,prefix):
                """取得实际绘制的数据，避免以文件是否存在替代数值验证。"""
                captured.extend(ax.lines[0].get_ydata().copy() for ax in axes.flat)

            with mock.patch.object(Figure,'savefig'),mock.patch.object(core,'save_paper_figure_rows',side_effect=capture_rows):
                result=fitting.plot_fivefold_curves(source,output)
            self.assertEqual(len(captured),6)
            for curve in captured:np.testing.assert_allclose(curve,[3,3],rtol=0,atol=1e-14)
            np.testing.assert_allclose(result.final_mean_rmse,3,rtol=0,atol=1e-14)
            np.testing.assert_allclose(result.final_sd_rmse,np.std([1,2,3,4,5],ddof=1),rtol=0,atol=1e-14)
            self.assertNotAlmostEqual(float(result.final_mean_rmse.iloc[0]),np.sqrt(11))


if __name__=='__main__':
    unittest.main()
