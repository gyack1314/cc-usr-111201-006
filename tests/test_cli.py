"""命令行子命令的端到端冒烟测试。"""
import io
import json
import unittest
from contextlib import redirect_stdout

from analysis_data.cli import main


class CliTests(unittest.TestCase):
    def _run(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main.__globals__  # 显式引用，保持测试可读性
            import sys
            old = sys.argv
            sys.argv = ['cli', *argv]
            try:
                main()
            finally:
                sys.argv = old
        return json.loads(buf.getvalue())

    def test_inspect(self):
        payload = self._run('inspect', 'data/normal.json')
        self.assertIn('source_sha256', payload)
        self.assertEqual(payload['summary']['nodes'], 7)

    def test_solve_with_verification(self):
        payload = self._run('solve', 'data/wait-value.json', '--verify')
        self.assertEqual(payload['report']['status'], 'optimal')
        self.assertTrue(payload['verification']['independently_checked'])

    def test_solve_loss_limit_override(self):
        payload = self._run('solve', 'data/cvar-choice.json',
                            '--loss-limit', '100')
        # 放宽上限后现在买成为期望最优（与文件内默认上限0的等待政策对照）
        root = next(r for r in payload['report']['node_policy']
                    if r['node'] == 'root')
        self.assertEqual(root['action'], 1)

    def test_verify_infeasible(self):
        payload = self._run('verify', 'data/infeasible-risk.json')
        self.assertEqual(payload['verification']['status'], 'infeasible')

    def test_not_completed_serializable(self):
        import copy
        from analysis_data.io import load
        raw, _ = load('data/normal.json')
        raw['max_policies'] = 3
        from analysis_data.report import build_report
        report = build_report(raw)
        # 报告整体必须能被JSON序列化（含分数串，无Fraction泄漏）
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(encoded)['status'], 'not_completed')


if __name__ == '__main__':
    unittest.main()
