"""把求解结果整理成可序列化报告（全部分数保留精确字符串）。"""
from .model import build
from .solver import (
    OPTIMAL,
    INFEASIBLE,
    SolveResult,
    hindsight_upper_bound,
    solve,
)

DISCLAIMER = '随附金额和需求均为虚构分析资料，不是投资建议，也不调用任何付款或采购接口。'


def _q(value):
    return str(value) if value is not None else None


def _flow_row(flow):
    return {
        'node': flow.node,
        'period': flow.period,
        'action': flow.action,
        'installed_before': flow.installed_before,
        'installed_after': flow.installed_after,
        'action_cost': _q(flow.action_cost),
        'operating_cash': _q(flow.operating_cash),
        'salvage': _q(flow.salvage),
        'node_cash_flow': _q(flow.node_cash_flow),
        'discounted_cash_flow': _q(flow.discounted_cash_flow),
        'budget': _q(flow.budget),
        'budget_used': _q(flow.budget_used),
        'within_budget': flow.budget_used <= flow.budget,
    }


def _installed_before(tree, policy, node_id):
    installed = tree.initial_installed
    for ancestor in tree.ancestors[node_id]:
        installed += policy[ancestor]
    return installed


def policy_rows(result: SolveResult):
    tree = result.tree
    rows = []
    for node_id in tree.order:
        node = tree.nodes[node_id]
        row = {
            'node': node_id,
            'period': node.period,
            'reach_probability': _q(tree.reach_probability[node_id]),
        }
        if result.status == OPTIMAL:
            action = result.policy[node_id]
            before = _installed_before(tree, result.policy, node_id)
            cost = action * node.purchase_cost
            row.update({
                'action': action,
                'installed_before': before,
                'installed_after': before + action,
                'action_cost': _q(cost),
                'budget': _q(node.budget),
                'budget_used': _q(cost),
                'within_budget': cost <= node.budget,
            })
        else:
            row['action'] = None
        rows.append(row)
    return rows


def leaf_rows(result: SolveResult):
    rows = []
    for outcome in result.leaves:
        rows.append({
            'leaf': outcome.leaf,
            'probability': _q(outcome.probability),
            'npv': _q(outcome.npv),
            'loss': _q(outcome.loss),
            'cash_flow_path': [_flow_row(flow) for flow in outcome.path],
        })
    return rows


def budget_check(result: SolveResult):
    """逐节点核对本期预算（每期独立授信，无资金滚存）。"""
    if result.status != OPTIMAL:
        return []
    tree = result.tree
    rows = []
    for node_id in tree.order:
        node = tree.nodes[node_id]
        action = result.policy[node_id]
        used = action * node.purchase_cost
        rows.append({
            'node': node_id,
            'period': node.period,
            'action': action,
            'budget': _q(node.budget),
            'budget_used': _q(used),
            'headroom': _q(node.budget - used),
            'within_budget': used <= node.budget,
        })
    return rows


def hindsight_section(raw):
    leaves, expected = hindsight_upper_bound(raw)
    from .model import build
    model = build(raw)
    # 同一节点在不同叶路径下动作不一致，正是不可实施的直接证据
    seen = {node_id: {} for node_id in model.order}
    for item in leaves:
        for node_id, action in item.actions.items():
            seen[node_id].setdefault(action, set()).add(item.leaf)
    conflicts = [
        {'node': node_id, 'actions': sorted(actions.keys()),
         'leaves_per_action': {str(action): sorted(leaf_set)
                               for action, leaf_set in actions.items()}}
        for node_id, actions in seen.items() if len(actions) > 1
    ]
    return {
        'implementable': False,
        'reason': '逐叶单独最大化NPV，同一节点会按未来结局选不同动作，违反非预知约束',
        'expected_npv': _q(expected),
        'leaves': [
            {
                'leaf': item.leaf,
                'probability': _q(item.probability),
                'actions': item.actions,
                'npv': _q(item.npv),
                'cash_flow_path': [_flow_row(flow) for flow in item.path],
            }
            for item in leaves
        ],
        'node_action_conflicts': conflicts,
    }


def build_report(raw, *, loss_limit=None):
    result = solve(raw, loss_limit=loss_limit)
    tree = result.tree
    report = {
        'status': result.status,
        'parameters': {
            'alpha': _q(tree.alpha),
            'loss_limit': _q(result.loss_limit),
            'discount_rate': _q(tree.rate),
            'initial_installed': tree.initial_installed,
            'capacity_limit': tree.capacity_limit,
            'actions_allowed': list(tree.actions),
            'max_policies': tree.max_policies,
        },
        'counts': {
            'legal_policy_space': result.legal_policy_space,
            'policies_evaluated': result.policies_evaluated,
            'risk_feasible_policies': result.feasible_policies,
        },
        'node_policy': policy_rows(result),
        'leaf_outcomes': leaf_rows(result),
        'budget_check': budget_check(result),
        'hindsight_foresight_upper_bound': hindsight_section(raw),
        'disclaimer': DISCLAIMER,
    }
    if result.status == OPTIMAL:
        report['summary'] = {
            'expected_npv': _q(result.expected_npv),
            'cvar_alpha_loss': _q(result.cvar),
            'risk_limit_satisfied': result.cvar <= result.loss_limit,
            'tie_break_order': list(tree.order),
            'policy_action_vector': list(result.best_tie_vector),
        }
    elif result.status == INFEASIBLE:
        report['summary'] = {
            'expected_npv': None,
            'cvar_alpha_loss': None,
            'risk_limit_satisfied': False,
            'note': '枚举已完整：存在合法政策，但没有任何政策满足CVaR损失上限',
        }
    else:
        report['summary'] = {
            'expected_npv': None,
            'cvar_alpha_loss': None,
            'risk_limit_satisfied': None,
            'note': '达到max_policies硬资源限额，枚举未完成；不能据此声称不可行或最优',
        }
    return report
