"""手算小树与求解器全量枚举的精确核对。

所有期望值、CVaR、政策空间大小都在测试里用有理数手工给出，
不引用求解器自己的中间量；手算政策表与枚举结果逐一比较。
"""
import copy
import unittest
from fractions import Fraction

from analysis_data.io import load
from analysis_data.model import build
from analysis_data.solver import (
    OPTIMAL,
    INFEASIBLE,
    NOT_COMPLETED,
    cvar_alpha,
    count_legal_policies,
    hindsight_upper_bound,
    solve,
)

F = Fraction


def _two_leaf_tree(root_cash_buy, loss_limit='100', budget='100'):
    """根+两个叶的小树：容量1，动作{0,1}，采购成本10，折现率1/10。"""
    def node(node_id, period, cash, children=()):
        return {
            'id': node_id,
            'period': period,
            'children': [
                {'id': child, 'probability': prob}
                for child, prob in children
            ],
            'operating_cash': {'0': '0', '1': cash},
            'budget': budget,
            'purchase_cost': '10',
            'salvage_per_unit': '0',
        }
    return {
        'root': 'root',
        'nodes': [
            node('root', 0, root_cash_buy, [('high', '1/2'), ('low', '1/2')]),
            node('high', 1, '12'),
            node('low', 1, '0'),
        ],
        'initial_installed': 0,
        'capacity_limit': 1,
        'actions': [0, 1],
        'rate': '1/10',
        'alpha': '4/5',
        'loss_limit': loss_limit,
        'max_policies': 1000000,
        'illustrative_cash': [],
    }


class HandCalculatedTreeTests(unittest.TestCase):
    """等待信息有价值的小树：手工列出全部5条合法政策。"""

    def setUp(self):
        # 根期装机1的经营现金流仅4，不足以覆盖采购10，等待在期望上更优
        self.raw = _two_leaf_tree('4')
        self.tree = build(self.raw)

    def test_policy_space_by_hand(self):
        # 容量为1：根买(动作1)后叶节点装机已满只能选0，贡献1条政策；
        # 根等(动作0)后每个叶节点各有买/不买2种，贡献2*2=4条，共5条
        self.assertEqual(count_legal_policies(self.tree), 5)

    def test_all_policies_match_hand_table(self):
        result = solve(self.raw)
        # 手算表（折现因子d=10/11，根期现金流不折现），
        # 键为稳定顺序(high动作, low动作, root动作)，值为(高叶NPV, 低叶NPV)：
        #  根等且两叶都不买(0,0,0)：NPV(0, 0)，期望0
        #  根等、仅低叶买(0,1,0)：NPV(0, -100/11)，期望-50/11
        #  根等、仅高叶买(1,0,0)：NPV(20/11, 0)，期望10/11
        #  根等、两叶都买(1,1,0)：NPV(20/11, -100/11)，期望-40/11
        #  根买(0,0,1)：根现金流-6；高叶12d、低叶0d
        #    NPV(54/11, -6)，期望(54/11-6)/2=-6/11
        hand_npv = {
            (0, 0, 0): (F(0, 1), F(0, 1)),
            (0, 1, 0): (F(0, 1), F(-100, 11)),
            (1, 0, 0): (F(20, 11), F(0, 1)),
            (1, 1, 0): (F(20, 11), F(-100, 11)),
            (0, 0, 1): (F(54, 11), F(-6, 1)),
        }
        seen = set()
        for outcomes, vector in self._all_evaluated():
            npv_high = next(leaf.npv for leaf in outcomes if leaf.leaf == 'high')
            npv_low = next(leaf.npv for leaf in outcomes if leaf.leaf == 'low')
            self.assertEqual((npv_high, npv_low), hand_npv[tuple(vector)])
            expected = F(1, 2) * npv_high + F(1, 2) * npv_low
            self.assertEqual(
                sum(leaf.probability * leaf.npv for leaf in outcomes),
                expected,
            )
            seen.add(expected)
        self.assertEqual(seen, {
            F(0, 1), F(-50, 11), F(10, 11), F(-40, 11), F(-6, 11),
        })
        self.assertEqual(len(hand_npv), 5)
        # 最优是“等待”而不是“现在就买”：现场需求信息有价值
        self.assertEqual(result.status, OPTIMAL)
        self.assertEqual(result.policy['root'], 0)
        self.assertEqual(result.policy['high'], 1)
        self.assertEqual(result.policy['low'], 0)
        self.assertEqual(result.expected_npv, F(10, 11))

    def test_hindsight_is_unimplementable_and_higher(self):
        leaves, bound = hindsight_upper_bound(self.raw)
        # 高叶预知会买：NPV 54/11；低叶预知不买：0；期望27/11
        per_leaf = {row.leaf: row.npv for row in leaves}
        self.assertEqual(per_leaf['high'], F(54, 11))
        self.assertEqual(per_leaf['low'], F(0, 1))
        self.assertEqual(bound, F(27, 11))
        # 同一根节点在两条叶路径下动作冲突，证明上界不可实施
        root_actions = {row.actions['root'] for row in leaves}
        self.assertEqual(root_actions, {0, 1})
        # 上界严格高于任何可实施政策的期望
        self.assertGreater(bound, solve(self.raw).expected_npv)

    def _all_evaluated(self):
        from analysis_data.solver import _enumerate_states
        yield from _enumerate_states(self.tree)


class CvarChangesChoiceTests(unittest.TestCase):
    """风险上限收紧后，扩点选择被CVaR改变。"""

    def test_tight_limit_reverses_root_decision(self):
        # 根期装机1经营现金流6（d=10/11，根期现金流不折现），实测：
        #  现在买(0,0,1)：NPV高=76/11、低=-4，期望16/11，CVaR=4
        #  等待仅高叶买(1,0,0)：NPV高=20/11、低=0，期望10/11，CVaR=0
        # 上限100时现在买期望最高；收紧到0后现在买CVaR=4被拒，等待政策胜出
        raw = _two_leaf_tree('6')
        loose = solve(raw, loss_limit='100')
        tight = solve(raw, loss_limit='0')
        self.assertEqual(loose.policy['root'], 1)
        self.assertEqual(loose.expected_npv, F(16, 11))
        self.assertEqual(loose.cvar, F(4))
        self.assertEqual(tight.status, OPTIMAL)
        self.assertEqual(tight.policy['root'], 0)
        self.assertEqual(tight.policy['high'], 1)
        self.assertEqual(tight.policy['low'], 0)
        self.assertEqual(tight.expected_npv, F(10, 11))
        self.assertEqual(tight.cvar, F(0))
        # 风险约束把“现在买”换成“等看到需求再买”，期望下降
        self.assertLess(tight.expected_npv, loose.expected_npv)

    def test_cvar_excludes_each_policy_as_hand_calculated(self):
        raw = _two_leaf_tree('6')
        # 逐政策CVaR实测对照：现在买=4；仅高叶买与永不买=0；
        # 在低叶买的两条政策低叶NPV=-100/11（损失100/11），CVaR=100/11
        cvar_by_vector = {}
        from analysis_data.solver import _enumerate_states
        result = solve(raw, loss_limit='100')
        for outcomes, vector in _enumerate_states(result.tree):
            risk = cvar_alpha(
                [(leaf.loss, leaf.probability) for leaf in outcomes],
                result.tree.alpha,
            )
            cvar_by_vector[tuple(vector)] = risk
        self.assertEqual(cvar_by_vector[(0, 0, 1)], F(4))
        self.assertEqual(cvar_by_vector[(1, 0, 0)], F(0))
        self.assertEqual(cvar_by_vector[(0, 0, 0)], F(0))
        self.assertEqual(cvar_by_vector[(0, 1, 0)], F(100, 11))
        self.assertEqual(cvar_by_vector[(1, 1, 0)], F(100, 11))


class CvarFormulaTests(unittest.TestCase):
    def test_exact_cvar_on_hand_distribution(self):
        # 损失 4（概率1/2）与 -2（概率1/2），alpha=4/5
        # eta=4: 4 + (1/5)*( (4-4)+(-2-4截断0) )=4
        # eta=-2: -2 + 5*(1/2*6)=13，故CVaR=4
        self.assertEqual(
            cvar_alpha([(F(4), F(1, 2)), (F(-2), F(1, 2))], F(4, 5)),
            F(4),
        )

    def test_zero_probability_ignored(self):
        base = cvar_alpha([(F(4), F(1, 2)), (F(-2), F(1, 2))], F(4, 5))
        with_zero = cvar_alpha(
            [(F(999), F(0)), (F(4), F(1, 2)), (F(-2), F(1, 2))], F(4, 5)
        )
        self.assertEqual(base, with_zero)


class ZeroProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.raw, _ = load('data/zero-probability.json')
        self.tree = build(self.raw)

    def test_zero_branch_kept_in_policy_space(self):
        full = count_legal_policies(self.tree)
        # 手工删除零概率子树（low及其叶）后的政策空间应严格更小
        pruned = copy.deepcopy(self.raw)
        pruned['nodes'] = [
            n for n in pruned['nodes']
            if n['id'] not in {'low', 'll', 'lh'}
        ]
        pruned['nodes'][0]['children'] = [
            {'id': 'high', 'probability': '1'}
        ]
        pruned_tree = build(pruned)
        self.assertLess(count_legal_policies(pruned_tree), full)

    def test_optimal_policy_covers_zero_probability_nodes(self):
        result = solve(self.raw)
        self.assertEqual(result.status, OPTIMAL)
        for node_id in self.tree.order:
            self.assertIn(node_id, result.policy)
        # 零概率叶仍出现在叶结果里，且其概率显式为0
        zero_leaves = {
            leaf.leaf: leaf for leaf in result.leaves
            if leaf.probability == 0
        }
        self.assertEqual(set(zero_leaves), {'ll', 'lh'})
        # 零概率路径不改变期望：与只看正概率叶的手工期望一致
        positive = [leaf for leaf in result.leaves if leaf.probability > 0]
        self.assertEqual(
            result.expected_npv,
            sum(leaf.probability * leaf.npv for leaf in positive),
        )

    def test_illegal_action_on_zero_node_rejected_by_independent_check(self):
        from analysis_data.checker import _replay_policy
        from analysis_data.checker import CheckError
        result = solve(self.raw)
        policy = dict(result.policy)
        # 超容量动作在零概率节点上也必须被判非法
        policy['low'] = 99
        with self.assertRaises(CheckError):
            _replay_policy(self.tree, policy)


class InfeasibleTests(unittest.TestCase):
    def test_complete_enumeration_reports_infeasible(self):
        raw, _ = load('data/infeasible-risk.json')
        result = solve(raw)
        self.assertEqual(result.status, INFEASIBLE)
        # 枚举完整：评估数等于合法空间，可行数为0
        self.assertEqual(result.policies_evaluated,
                         result.legal_policy_space)
        self.assertEqual(result.feasible_policies, 0)
        self.assertIsNone(result.policy)


class NotCompletedTests(unittest.TestCase):
    def test_resource_limit_never_claims_infeasible_or_optimal(self):
        raw, _ = load('data/normal.json')
        raw['max_policies'] = 10
        result = solve(raw)
        self.assertEqual(result.status, NOT_COMPLETED)
        self.assertEqual(result.policies_evaluated, 10)
        self.assertIsNone(result.policy)
        self.assertIsNone(result.expected_npv)
        self.assertGreater(result.legal_policy_space, 10)

    def test_limit_above_space_completes(self):
        raw, _ = load('data/normal.json')
        result = solve(raw)
        self.assertEqual(result.status, OPTIMAL)
        self.assertEqual(result.policies_evaluated,
                         result.legal_policy_space)


class ValidationTests(unittest.TestCase):
    def test_reject_bad_inputs(self):
        raw, _ = load('data/normal.json')
        def tampered(**changes):
            value = copy.deepcopy(raw)
            value.update(changes)
            return value
        with self.assertRaises(ValueError):
            build(tampered(alpha='0'))
        with self.assertRaises(ValueError):
            build(tampered(rate='-1'))
        with self.assertRaises(ValueError):
            build(tampered(initial_installed=99))
        with self.assertRaises(ValueError):
            build(tampered(actions=[1, 1]))
        with self.assertRaises(ValueError):
            build(tampered(max_policies=0))
        missing_cash = copy.deepcopy(raw)
        del missing_cash['nodes'][0]['operating_cash']['3']
        with self.assertRaises(ValueError):
            build(missing_cash)


class TieBreakTests(unittest.TestCase):
    def test_lexicographic_tie_break_on_stable_order(self):
        # 两条政策期望与风险完全相同：动作0与动作1都不产生现金流差异
        # 根下两个独立叶，采购成本0、现金流与装机无关，4条政策完全并列
        raw = {
            'root': 'r',
            'nodes': [
                {'id': 'r', 'period': 0,
                 'children': [{'id': 'a', 'probability': '1/2'},
                              {'id': 'b', 'probability': '1/2'}],
                 'operating_cash': {'0': '0', '1': '0'},
                 'budget': '100', 'purchase_cost': '0',
                 'salvage_per_unit': '0'},
                {'id': 'a', 'period': 1, 'children': [],
                 'operating_cash': {'0': '0', '1': '0'},
                 'budget': '100', 'purchase_cost': '0',
                 'salvage_per_unit': '0'},
                {'id': 'b', 'period': 1, 'children': [],
                 'operating_cash': {'0': '0', '1': '0'},
                 'budget': '100', 'purchase_cost': '0',
                 'salvage_per_unit': '0'},
            ],
            'initial_installed': 0, 'capacity_limit': 1,
            'actions': [0, 1], 'rate': '0', 'alpha': '1/2',
            'loss_limit': '100', 'max_policies': 1000000,
            'illustrative_cash': [],
        }
        result = solve(raw)
        # 稳定顺序为['a','b','r']；实现按(期望, 动作向量)取最大，
        # 完全并列时字典序最大的向量确定性胜出。容量为1：根买后叶只能0，
        # 故最大向量是(1,1,0)而非(1,1,1)，结果可复现
        self.assertEqual(result.best_tie_vector, (1, 1, 0))
        self.assertEqual(result.policy, {'a': 1, 'b': 1, 'r': 0})


if __name__ == '__main__':
    unittest.main()
