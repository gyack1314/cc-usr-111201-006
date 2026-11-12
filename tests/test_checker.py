"""独立检查器的对抗性测试：篡改报告必须被发现，正常报告必须通过。"""
import copy
import unittest
from fractions import Fraction

from analysis_data.checker import CheckError, verify
from analysis_data.io import load
from analysis_data.report import build_report

F = Fraction


class CheckerPassTests(unittest.TestCase):
    def test_all_fixtures_pass_independent_verification(self):
        for name in ('normal', 'zero-probability', 'infeasible-risk',
                     'wait-value', 'cvar-choice'):
            with self.subTest(file=name):
                raw, _ = load(f'data/{name}.json')
                report = build_report(raw)
                checked = verify(raw, report)
                self.assertTrue(checked['independently_checked'])

    def test_not_completed_is_checked_without_claim(self):
        raw, _ = load('data/normal.json')
        raw['max_policies'] = 7
        report = build_report(raw)
        checked = verify(raw, report)
        self.assertEqual(checked['status'], 'not_completed')

    def test_foresight_gap_non_negative(self):
        raw, _ = load('data/wait-value.json')
        checked = verify(raw, build_report(raw))
        self.assertGreaterEqual(checked['foresight_gap'], 0)
        # wait-value示例中上界严格高于可实施最优
        self.assertEqual(checked['foresight_gap'], F(17, 11))


class CheckerTamperTests(unittest.TestCase):
    def setUp(self):
        self.raw, _ = load('data/normal.json')
        self.report = build_report(self.raw)

    def _assert_rejected(self, report):
        with self.assertRaises(CheckError):
            verify(self.raw, report)

    def test_tampered_expected_value_rejected(self):
        report = copy.deepcopy(self.report)
        report['summary']['expected_npv'] = '999999'
        self._assert_rejected(report)

    def test_tampered_cvar_rejected(self):
        report = copy.deepcopy(self.report)
        report['summary']['cvar_alpha_loss'] = '0'
        self._assert_rejected(report)

    def test_tampered_policy_action_rejected(self):
        report = copy.deepcopy(self.report)
        # 把根动作从1改成0，独立重算会发现政策与报告NPV对不上
        for row in report['node_policy']:
            if row['node'] == 'root':
                row['action'] = 0
        self._assert_rejected(report)

    def test_tampered_leaf_npv_rejected(self):
        report = copy.deepcopy(self.report)
        report['leaf_outcomes'][0]['npv'] = '12345'
        report['leaf_outcomes'][0]['loss'] = '-12345'
        self._assert_rejected(report)

    def test_tampered_cash_flow_detail_rejected(self):
        report = copy.deepcopy(self.report)
        path = report['leaf_outcomes'][0]['cash_flow_path']
        path[0]['discounted_cash_flow'] = '42'
        self._assert_rejected(report)

    def test_tampered_budget_check_rejected(self):
        report = copy.deepcopy(self.report)
        # 选一个实际发生采购（动作非0）的节点篡改其预算占用
        target = next(row for row in report['budget_check'] if row['action'] > 0)
        target['budget_used'] = '0'
        self._assert_rejected(report)

    def test_hindsight_mislabeled_implementable_rejected(self):
        report = copy.deepcopy(self.report)
        report['hindsight_foresight_upper_bound']['implementable'] = True
        self._assert_rejected(report)

    def test_hindsight_value_tamper_rejected(self):
        report = copy.deepcopy(self.report)
        report['hindsight_foresight_upper_bound']['expected_npv'] = '1'
        self._assert_rejected(report)

    def test_missing_node_in_policy_rejected(self):
        report = copy.deepcopy(self.report)
        report['node_policy'] = report['node_policy'][:-1]
        self._assert_rejected(report)

    def test_infeasible_claim_with_hidden_feasible_policy_rejected(self):
        raw, _ = load('data/normal.json')
        report = build_report(raw)
        report['status'] = 'infeasible'
        report['counts']['risk_feasible_policies'] = 0
        self._assert_rejected(report)

    def test_policy_space_miscount_rejected(self):
        report = copy.deepcopy(self.report)
        report['counts']['legal_policy_space'] += 1
        self._assert_rejected(report)


class CheckerReplayTests(unittest.TestCase):
    def test_budget_and_capacity_violation_caught(self):
        from analysis_data.checker import CheckError, _replay_policy
        from analysis_data.model import build
        raw, _ = load('data/normal.json')
        model = build(raw)
        report = build_report(raw)
        policy = {row['node']: row['action'] for row in report['node_policy']}
        # 超容量
        policy['root'] = 99
        with self.assertRaises(CheckError):
            _replay_policy(model, policy)
        # 不在允许动作集
        policy['root'] = -1
        with self.assertRaises(CheckError):
            _replay_policy(model, policy)


if __name__ == '__main__':
    unittest.main()
