"""独立检查器：拿报告中的政策重新遍历树，不信任求解器缓存的NPV。

检查器只复用 :mod:`analysis_data.model` 的资料解析（纯语法/结构校验），
现金流遍历、政策枚举、期望与CVaR全部在此重新实现一遍，
与 :mod:`analysis_data.solver` 没有任何函数共享。任何不一致抛
:class:`CheckError`，通过则返回独立重算结果供留痕。
"""
from fractions import Fraction

from .model import build
from .solver import OPTIMAL, INFEASIBLE, NOT_COMPLETED


class CheckError(AssertionError):
    """报告与独立重算结果不一致。"""


def _frac(value):
    return Fraction(value)


def _discount(rate, period):
    return Fraction(1, 1) / (1 + rate) ** period


def _legal(model, node_id, installed):
    node = model.nodes[node_id]
    out = []
    for action in model.actions:
        if installed + action <= model.capacity_limit and \
                action * node.purchase_cost <= node.budget:
            out.append(action)
    return out


def _replay_policy(model, policy):
    """按政策从根独立遍历，逐叶重建现金流路径与NPV。"""
    results = []

    def walk(node_id, installed, prefix):
        node = model.nodes[node_id]
        if node_id not in policy:
            raise CheckError(f'政策缺少节点{node_id}的动作（零概率节点也必须给出）')
        action = policy[node_id]
        if action not in model.actions:
            raise CheckError(f'节点{node_id}的动作{action}不在允许配置中')
        if action not in _legal(model, node_id, installed):
            raise CheckError(
                f'节点{node_id}动作{action}违反容量或本期预算约束'
            )
        after = installed + action
        cost = action * node.purchase_cost
        salvage = node.salvage_per_unit * after if node.is_leaf else Fraction(0)
        operating = node.operating_cash[after]
        cash = -cost + operating + salvage
        detail = {
            'node': node_id, 'period': node.period, 'action': action,
            'installed_before': installed, 'installed_after': after,
            'action_cost': cost, 'operating_cash': operating,
            'salvage': salvage, 'node_cash_flow': cash,
            'discounted_cash_flow': cash * _discount(model.rate, node.period),
            'budget_used': cost, 'budget': node.budget,
        }
        path = prefix + [detail]
        if node.is_leaf:
            npv = sum(item['discounted_cash_flow'] for item in path)
            results.append({
                'leaf': node_id,
                'probability': model.reach_probability[node_id],
                'npv': npv,
                'loss': -npv,
                'path': path,
            })
        else:
            for edge in node.children:
                walk(edge.child, after, path)

    walk(model.root, model.initial_installed, [])
    return results


def _cvar(pairs, alpha):
    active = [(loss, prob) for loss, prob in pairs if prob > 0]
    if sum(prob for _, prob in active) != 1:
        raise CheckError('正概率叶概率和不为1')
    values = []
    for eta in sorted({loss for loss, _ in active}):
        tail = sum(prob * max(loss - eta, 0) for loss, prob in active)
        values.append(eta + tail / (1 - alpha))
    return min(values)


def _all_policies(model):
    """独立完整枚举（与solver不同的无状态递归写法），产出完整政策字典。

    每个递归层级返回该子树所有完整赋值的不可变快照序列，
    不依赖共享可变字典的恢复时序。
    """

    def gen(node_id, installed):
        node = model.nodes[node_id]
        completed = []
        for action in _legal(model, node_id, installed):
            if node.is_leaf:
                completed.append(((node_id, action),))
                continue
            child_alternatives = []
            for edge in node.children:
                child_alternatives.append(
                    gen(edge.child, installed + action)
                )
            # 各子树赋值做笛卡尔积，再与本节点动作合并
            combinations = [()]
            for alternatives in child_alternatives:
                combinations = [
                    prefix + choice
                    for prefix in combinations
                    for choice in alternatives
                ]
            completed.extend(
                ((node_id, action),) + rest for rest in combinations
            )
        return completed

    for assignment in gen(model.root, model.initial_installed):
        yield dict(assignment)


def _evaluate(model, policy):
    leaves = _replay_policy(model, policy)
    expected = sum(item['probability'] * item['npv'] for item in leaves)
    risk = _cvar([(item['loss'], item['probability']) for item in leaves],
                 model.alpha)
    return leaves, expected, risk


def _hindsight(model):
    """独立重算逐叶预知上界：每条根-叶路径独立枚举动作组合。"""
    total = Fraction(0)
    per_leaf = {}
    for leaf in model.leaves:
        chain = list(model.ancestors[leaf]) + [leaf]
        best_npv = None

        def rec(index, installed, discounted_sum):
            nonlocal best_npv
            node_id = chain[index]
            node = model.nodes[node_id]
            for action in _legal(model, node_id, installed):
                after = installed + action
                cost = action * node.purchase_cost
                salvage = node.salvage_per_unit * after if node.is_leaf \
                    else Fraction(0)
                cash = -cost + node.operating_cash[after] + salvage
                value = discounted_sum + cash * _discount(model.rate, node.period)
                if index == len(chain) - 1:
                    if best_npv is None or value > best_npv:
                        best_npv = value
                else:
                    rec(index + 1, after, value)

        rec(0, model.initial_installed, Fraction(0))
        per_leaf[leaf] = best_npv
        total += model.reach_probability[leaf] * best_npv
    return per_leaf, total


def verify(raw, report):
    """对照原始资料独立核验整份报告，返回独立重算的关键量。"""
    model = build(raw)
    status = report['status']
    limit = _frac(report['parameters']['loss_limit'])

    if status == NOT_COMPLETED:
        if report['counts']['policies_evaluated'] > model.max_policies:
            raise CheckError('未完成状态的枚举数超过资源限额')
        if report['summary'].get('expected_npv') is not None:
            raise CheckError('未完成枚举不得给出最优值')
        return {'status': status, 'independently_checked': True}

    # 完整枚举：不使用solver的任何结果
    feasible = []
    total = 0
    for policy in _all_policies(model):
        total += 1
        leaves, expected, risk = _evaluate(model, policy)
        if risk <= limit:
            vector = tuple(policy[nid] for nid in model.order)
            feasible.append((expected, vector, policy, leaves, risk))

    if total != report['counts']['legal_policy_space']:
        raise CheckError(
            f'合法政策空间不一致：报告{report["counts"]["legal_policy_space"]}，'
            f'独立枚举{total}'
        )

    independent_hindsight, independent_bound = _hindsight(model)

    if status == INFEASIBLE:
        if feasible:
            raise CheckError('报告称不可行，但独立枚举找到了风险可行政策')
        if report['counts']['risk_feasible_policies'] != 0:
            raise CheckError('不可行报告中的可行政策计数应为0')
        return {
            'status': status,
            'legal_policies': total,
            'hindsight_expected_npv': independent_bound,
            'independently_checked': True,
        }

    if status != OPTIMAL:
        raise CheckError(f'未知状态{status}')
    if not feasible:
        raise CheckError('报告称最优，但独立枚举未找到任何风险可行政策')

    best_expected, best_vector, _, _, best_risk = max(
        feasible, key=lambda item: (item[0], item[1])
    )
    reported_policy = {
        row['node']: row['action'] for row in report['node_policy']
    }
    if set(reported_policy) != set(model.nodes):
        raise CheckError('逐节点政策没有覆盖全树节点')
    leaves, expected, risk = _evaluate(model, reported_policy)

    if expected != best_expected or risk != best_risk:
        raise CheckError('报告政策不是独立枚举下的风险约束最优')
    if tuple(reported_policy[nid] for nid in model.order) != best_vector:
        raise CheckError('同期望下的稳定顺序决胜结果不一致')
    if _frac(report['summary']['expected_npv']) != expected:
        raise CheckError('期望NPV与独立重算不符')
    if _frac(report['summary']['cvar_alpha_loss']) != risk:
        raise CheckError('CVaR与独立重算不符')
    if risk > limit:
        raise CheckError('选中政策不满足风险上限')

    reported_leaves = {row['leaf']: row for row in report['leaf_outcomes']}
    if set(reported_leaves) != {item['leaf'] for item in leaves}:
        raise CheckError('叶结果集合与独立遍历不符')
    for item in leaves:
        row = reported_leaves[item['leaf']]
        if _frac(row['probability']) != item['probability'] \
                or _frac(row['npv']) != item['npv'] \
                or _frac(row['loss']) != item['loss']:
            raise CheckError(f'叶{item["leaf"]}的概率/NPV/损失与独立重算不符')
        if len(row['cash_flow_path']) != len(item['path']):
            raise CheckError(f'叶{item["leaf"]}现金流路径长度不符')
        for reported_flow, own_flow in zip(row['cash_flow_path'], item['path']):
            for key in ('action_cost', 'operating_cash', 'salvage',
                        'node_cash_flow', 'discounted_cash_flow',
                        'budget_used', 'budget'):
                if _frac(reported_flow[key]) != own_flow[key]:
                    raise CheckError(
                        f'叶{item["leaf"]}路径上的{key}与独立重算不符'
                    )
            if reported_flow['action'] != own_flow['action']:
                raise CheckError('路径动作与独立遍历不符')

    for row in report['budget_check']:
        node = model.nodes[row['node']]
        used = row['action'] * node.purchase_cost
        if _frac(row['budget_used']) != used or _frac(row['budget']) != node.budget:
            raise CheckError(f'节点{row["node"]}预算核对错误')
        if row['within_budget'] != (used <= node.budget):
            raise CheckError(f'节点{row["node"]}预算合规标记错误')

    bound_section = report['hindsight_foresight_upper_bound']
    if bound_section.get('implementable') is not False:
        raise CheckError('预知上界必须明确标记为不可实施')
    if _frac(bound_section['expected_npv']) != independent_bound:
        raise CheckError('预知上界期望与独立重算不符')
    for row in bound_section['leaves']:
        if _frac(row['npv']) != independent_hindsight[row['leaf']]:
            raise CheckError(f'叶{row["leaf"]}的预知NPV与独立重算不符')
    # 可实施最优的期望不可能超过预知上界
    if expected > independent_bound:
        raise CheckError('可实施政策期望超过预知上界，逻辑矛盾')

    return {
        'status': OPTIMAL,
        'legal_policies': total,
        'feasible_policies': len(feasible),
        'expected_npv': expected,
        'cvar_alpha_loss': risk,
        'policy': reported_policy,
        'hindsight_expected_npv': independent_bound,
        'foresight_gap': independent_bound - expected,
        'independently_checked': True,
    }
