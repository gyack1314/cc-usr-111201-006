"""随机小树的性质测试：两套独立枚举必须一致，报告必须通过独立核验。

这是对“局部风险剪枝若丢掉可行政策必须能被小树对照发现”的自动化兜底：
随机生成结构、概率（含零概率）、金额与预算各异的小树，
求解器与检查器各自独立枚举，数量、最优政策、风险口径逐项对照。
"""
import random
import unittest

from analysis_data.checker import verify
from analysis_data.model import build
from analysis_data.report import build_report
from analysis_data.solver import count_legal_policies, solve
from analysis_data.checker import _all_policies


def _collect(rng, max_depth=2):
    rng.random()
    nodes = []
    capacity = rng.randint(1, 3)

    def gen(depth):
        is_leaf = depth == max_depth or (depth > 0 and rng.random() < 0.3)
        node_id = 'root' if not nodes else f'n{len(nodes)}'
        node = {'id': node_id, 'period': depth, 'children': []}
        nodes.append(node)
        if not is_leaf:
            degree = rng.randint(1, 3)
            weights = [rng.choice([0, 1, 1, 2, 3]) for _ in range(degree)]
            if sum(weights) == 0:
                weights[0] = 1
            total = sum(weights)
            for w in weights:
                child_id = gen(depth + 1)
                node['children'].append(
                    {'id': child_id, 'probability': f'{w}/{total}'}
                )
        # 收入表宽度统一对齐顶层容量，保证每个合法装机数都有金额
        node['operating_cash'] = {
            str(k): str(rng.choice([-6, -2, 0, 4, 8, 12, 16]))
            for k in range(capacity + 1)
        }
        node['budget'] = str(rng.choice([0, 5, 12, 24, 100]))
        node['purchase_cost'] = str(rng.choice([4, 10, 12]))
        node['salvage_per_unit'] = str(rng.choice([0, 1, 3]))
        return node_id

    root = gen(0)
    return {
        'root': root,
        'nodes': nodes,
        'initial_installed': 0,
        'capacity_limit': capacity,
        'actions': list(range(capacity + 1)),
        'rate': rng.choice(['0', '1/10', '1/4']),
        'alpha': rng.choice(['1/2', '4/5', '9/10']),
        'loss_limit': str(rng.choice([-100, -10, 0, 5, 25, 100])),
        'max_policies': 100000,
        'illustrative_cash': [],
    }


class RandomTreePropertyTests(unittest.TestCase):
    def test_enumerations_agree_and_reports_verify(self):
        rng = random.Random(20261112)
        checked = 0
        for _ in range(60):
            raw = _collect(rng)
            try:
                tree = build(raw)
            except ValueError:
                continue
            solver_count = count_legal_policies(tree)
            checker_policies = list(_all_policies(tree))
            # 两套独立实现的政策空间必须完全一致
            self.assertEqual(len(checker_policies), solver_count)
            self.assertTrue(
                all(set(policy) == set(tree.nodes)
                    for policy in checker_policies)
            )
            result = solve(raw)
            self.assertIn(result.status,
                          ('optimal', 'infeasible', 'not_completed'))
            if result.status == 'not_completed':
                continue
            report = build_report(raw)
            verification = verify(raw, report)
            self.assertTrue(verification['independently_checked'])
            if result.status == 'optimal':
                self.assertEqual(
                    verification['legal_policies'], solver_count
                )
            checked += 1
        self.assertGreater(checked, 40)


if __name__ == '__main__':
    unittest.main()
