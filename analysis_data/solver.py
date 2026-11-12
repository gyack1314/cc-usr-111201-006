"""确定性非随机化政策的精确枚举与风险约束选优。

方法是对全树合法动作做完整枚举：政策在每个节点给出一个动作，
动作选择不允许依赖该节点之后才发生的分支。零概率节点同样必须
给出合法动作，枚举从不删枝。``max_policies``是硬资源限额，
枚举超限只报告``not_completed``，绝不报不可行或最优。

所有金额、概率、折现率都用 :class:`fractions.Fraction` 精确运算，
风险界附近的判断不受浮点误差影响。
"""
from dataclasses import dataclass
from fractions import Fraction

from .model import Tree, build as build_tree

OPTIMAL = 'optimal'
INFEASIBLE = 'infeasible'
NOT_COMPLETED = 'not_completed'


@dataclass(frozen=True)
class NodeFlow:
    """政策沿一条叶路径到达某节点时的逐节点现金流明细。"""
    node: str
    period: int
    action: int
    installed_before: int
    installed_after: int
    action_cost: Fraction
    operating_cash: Fraction
    salvage: Fraction
    node_cash_flow: Fraction
    discounted_cash_flow: Fraction
    budget: Fraction
    budget_used: Fraction


@dataclass(frozen=True)
class LeafOutcome:
    leaf: str
    probability: Fraction
    npv: Fraction
    loss: Fraction
    path: tuple[NodeFlow, ...]


@dataclass
class SolveResult:
    status: str
    tree: Tree
    policy: dict[str, int] | None
    leaves: tuple[LeafOutcome, ...]
    expected_npv: Fraction | None
    cvar: Fraction | None
    loss_limit: Fraction
    policies_evaluated: int
    feasible_policies: int
    legal_policy_space: int
    best_tie_vector: tuple[int, ...] | None


def discount_factor(tree: Tree, period: int) -> Fraction:
    return Fraction(1, 1) / (1 + tree.rate) ** period


def cvar_alpha(losses: list[tuple[Fraction, Fraction]], alpha: Fraction) -> Fraction:
    """离散分布的精确CVaR。

    ``losses``是``(损失, 概率)``序列，概率为零的项不参与统计。
    对凸分段线性函数，在损失样本点取eta即可取到最小值，
    因此直接在去重后的损失点上精确枚举eta。
    """
    active = [(loss, prob) for loss, prob in losses if prob > 0]
    total_probability = sum(prob for _, prob in active)
    if total_probability != 1:
        raise ValueError('正概率叶概率和不为1，分布不完整')
    candidates = sorted({loss for loss, _ in active})
    best = None
    for eta in candidates:
        excess = sum(prob * max(loss - eta, Fraction(0)) for loss, prob in active)
        value = eta + excess / (1 - alpha)
        if best is None or value < best:
            best = value
    return best


def _node_flow(tree: Tree, node_id: str, installed: int, action: int) -> NodeFlow:
    node = tree.nodes[node_id]
    after = installed + action
    cost = action * node.purchase_cost
    salvage = node.salvage_per_unit * after if node.is_leaf else Fraction(0)
    operating = node.operating_cash[after]
    cash_flow = -cost + operating + salvage
    return NodeFlow(
        node=node_id,
        period=node.period,
        action=action,
        installed_before=installed,
        installed_after=after,
        action_cost=cost,
        operating_cash=operating,
        salvage=salvage,
        node_cash_flow=cash_flow,
        discounted_cash_flow=cash_flow * discount_factor(tree, node.period),
        budget=node.budget,
        budget_used=cost,
    )


def _enumerate_states(tree: Tree):
    """深度优先枚举完整合法政策并逐政策独立遍历求值。

    政策在每个节点选定一个动作后，整棵树赋值完成才做一次独立的
    现金流遍历，避免不同政策的叶路径结果相互串扰。每次产出
    ``(叶结果元组, 稳定顺序动作向量)``。
    """
    policy = {}
    tie_index = {nid: i for i, nid in enumerate(tree.order)}
    tie_vector = [0] * len(tree.order)

    def evaluate():
        outcomes = []

        def traverse(node_id, installed, prefix):
            action = policy[node_id]
            flow = _node_flow(tree, node_id, installed, action)
            node = tree.nodes[node_id]
            if node.is_leaf:
                path = prefix + (flow,)
                npv = sum(item.discounted_cash_flow for item in path)
                outcomes.append(
                    LeafOutcome(
                        leaf=node_id,
                        probability=tree.reach_probability[node_id],
                        npv=npv,
                        loss=-npv,
                        path=path,
                    )
                )
            else:
                for edge in node.children:
                    traverse(edge.child, installed + action, prefix + (flow,))

        traverse(tree.root, tree.initial_installed, ())
        return tuple(outcomes), tuple(tie_vector)

    def assign_edges(edges, index, installed_after, done):
        # 按固定顺序给一个节点的各子树赋政策，全部完成才回调
        if index == len(edges):
            done()
            return
        assign_node(
            edges[index].child,
            installed_after,
            lambda: assign_edges(edges, index + 1, installed_after, done),
        )

    def assign_node(node_id, installed, done):
        node = tree.nodes[node_id]
        for action in tree.legal_actions(node, installed):
            policy[node_id] = action
            tie_vector[tie_index[node_id]] = action
            if node.is_leaf:
                done()
            else:
                assign_edges(node.children, 0, installed + action, done)
        del policy[node_id]

    results = []
    assign_node(tree.root, tree.initial_installed,
                lambda: results.append(evaluate()))
    yield from results


def count_legal_policies(tree: Tree) -> int:
    """统计合法政策空间大小（零概率节点同样计入，不删枝）。"""
    memo = {}

    def ways(node_id, installed):
        key = (node_id, installed)
        if key in memo:
            return memo[key]
        node = tree.nodes[node_id]
        total = 0
        for action in tree.legal_actions(node, installed):
            if node.is_leaf:
                total += 1
            else:
                sub = 1
                for edge in node.children:
                    sub *= ways(edge.child, installed + action)
                total += sub
        memo[key] = total
        return total

    return ways(tree.root, tree.initial_installed)


def solve(raw, *, loss_limit=None):
    """枚举全部合法政策，返回风险约束下的期望NPV最优政策。

    目标层级：CVaR不超过损失上限的政策中期望NPV最大；
    期望相同按稳定节点顺序的动作向量字典序决胜。
    """
    tree = build_tree(raw)
    limit = tree.loss_limit if loss_limit is None else Fraction(loss_limit)
    legal_space = count_legal_policies(tree)

    evaluated = 0
    feasible_count = 0
    best = None
    truncated = False

    for outcomes, tie_vector in _enumerate_states(tree):
        evaluated += 1
        expected = sum(leaf.probability * leaf.npv for leaf in outcomes)
        risk = cvar_alpha(
            [(leaf.loss, leaf.probability) for leaf in outcomes], tree.alpha
        )
        if risk <= limit:
            feasible_count += 1
            candidate = (expected, tie_vector, outcomes, risk)
            if best is None or (expected, tie_vector) > (best[0], best[1]):
                best = candidate
        if evaluated >= tree.max_policies:
            truncated = True
            break

    if truncated:
        return SolveResult(
            status=NOT_COMPLETED,
            tree=tree,
            policy=None,
            leaves=(),
            expected_npv=None,
            cvar=None,
            loss_limit=limit,
            policies_evaluated=evaluated,
            feasible_policies=feasible_count,
            legal_policy_space=legal_space,
            best_tie_vector=None,
        )

    if best is None:
        return SolveResult(
            status=INFEASIBLE,
            tree=tree,
            policy=None,
            leaves=(),
            expected_npv=None,
            cvar=None,
            loss_limit=limit,
            policies_evaluated=evaluated,
            feasible_policies=0,
            legal_policy_space=legal_space,
            best_tie_vector=None,
        )

    expected, tie_vector, outcomes, risk = best
    chosen = {nid: tie_vector[i] for i, nid in enumerate(tree.order)}
    return SolveResult(
        status=OPTIMAL,
        tree=tree,
        policy=chosen,
        leaves=tuple(sorted(outcomes, key=lambda item: item.leaf)),
        expected_npv=expected,
        cvar=risk,
        loss_limit=limit,
        policies_evaluated=evaluated,
        feasible_policies=feasible_count,
        legal_policy_space=legal_space,
        best_tie_vector=tie_vector,
    )


@dataclass(frozen=True)
class HindsightLeaf:
    leaf: str
    probability: Fraction
    actions: dict[str, int]
    npv: Fraction
    path: tuple[NodeFlow, ...]


def hindsight_upper_bound(raw):
    """预知未来的逐叶最优（乐观上界，不可实施）。

    对每条叶路径单独最大化NPV：相当于到达叶之前就知道结局，
    同一节点在不同叶路径下可以选不同动作，违反非预知政策约束。
    它不受风险上限约束，只用于展示原表格事后挑选的乐观偏差。
    """
    tree = build_tree(raw)
    rows = []
    for leaf in tree.leaves:
        path = _best_path_to_leaf(tree, leaf)
        npv = sum(flow.discounted_cash_flow for flow in path)
        rows.append(
            HindsightLeaf(
                leaf=leaf,
                probability=tree.reach_probability[leaf],
                actions={flow.node: flow.action for flow in path},
                npv=npv,
                path=path,
            )
        )
    expected = sum(row.probability * row.npv for row in rows)
    return tuple(sorted(rows, key=lambda item: item.leaf)), expected


def _best_path_to_leaf(tree: Tree, leaf: str):
    """对单条根到叶路径做动作组合枚举，返回NPV最大的现金流序列。"""
    chain = list(tree.ancestors[leaf]) + [leaf]
    best_path = None
    best_npv = None

    def recurse(index, installed, prefix):
        nonlocal best_path, best_npv
        node_id = chain[index]
        node = tree.nodes[node_id]
        for action in tree.legal_actions(node, installed):
            flow = _node_flow(tree, node_id, installed, action)
            if index == len(chain) - 1:
                path = prefix + (flow,)
                npv = sum(item.discounted_cash_flow for item in path)
                if best_npv is None or npv > best_npv:
                    best_npv, best_path = npv, path
            else:
                recurse(index + 1, installed + action, prefix + (flow,))

    recurse(0, tree.initial_installed, ())
    if best_path is None:
        raise ValueError(f'叶{leaf}不存在合法动作组合')
    return best_path
