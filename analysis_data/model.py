"""情景树的精确分数模型与结构校验。

模型只负责把JSON资料翻译成``Fraction``精确表示并验证资料契约，
不搜索政策、不评价整棵树（评价逻辑在 :mod:`analysis_data.solver`）。
"""
from dataclasses import dataclass
from fractions import Fraction

from .io import unique


def _fraction(value, what):
    """把分数串或整数转成``Fraction``，拒绝布尔与浮点。"""
    if isinstance(value, bool) or not isinstance(value, (str, int, Fraction)):
        raise ValueError(f'{what}必须是分数字符串或整数')
    try:
        return Fraction(value)
    except (ZeroDivisionError, ValueError, TypeError) as exc:
        raise ValueError(f'{what}不是合法有理数') from exc


def _non_negative_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{what}必须是非负整数')
    return value


@dataclass(frozen=True)
class Edge:
    child: str
    probability: Fraction


@dataclass(frozen=True)
class Node:
    id: str
    period: int
    children: tuple[Edge, ...]
    operating_cash: dict[int, Fraction]
    budget: Fraction
    purchase_cost: Fraction
    salvage_per_unit: Fraction

    @property
    def is_leaf(self):
        return not self.children


@dataclass(frozen=True)
class Tree:
    root: str
    nodes: dict[str, Node]
    initial_installed: int
    capacity_limit: int
    actions: tuple[int, ...]
    rate: Fraction
    alpha: Fraction
    loss_limit: Fraction
    max_policies: int
    # node id字典序，政策决胜向量与稳定输出都用这个顺序
    order: tuple[str, ...]
    parent: dict[str, str | None]
    ancestors: dict[str, tuple[str, ...]]
    reach_probability: dict[str, Fraction]
    leaves: tuple[str, ...]

    def legal_actions(self, node, installed):
        """返回该到达装机数下配置与预算都允许的动作（按升序）。"""
        result = []
        for action in self.actions:
            after = installed + action
            if after > self.capacity_limit:
                continue
            if action * node.purchase_cost > node.budget:
                continue
            result.append(action)
        return result


def build(raw):
    """校验资料并构造:class:`Tree`，任何违约都抛``ValueError``。"""
    if not isinstance(raw, dict):
        raise ValueError('资料顶层必须是对象')
    ids = unique(raw['nodes'], 'id')
    root = raw['root']
    if root not in ids:
        raise ValueError('根不存在')

    rate = _fraction(raw['rate'], '折现率')
    if rate <= -1:
        raise ValueError('折现率必须大于-1')
    alpha = _fraction(raw['alpha'], 'alpha')
    if not (0 < alpha < 1):
        raise ValueError('alpha必须严格位于0和1之间')
    loss_limit = _fraction(raw['loss_limit'], '损失上限')
    capacity = _non_negative_int(raw['capacity_limit'], 'capacity_limit')
    initial = _non_negative_int(raw['initial_installed'], 'initial_installed')
    if initial > capacity:
        raise ValueError('初始装机数不能超过容量上限')
    if not isinstance(raw['actions'], list) or not raw['actions']:
        raise ValueError('actions必须是非空整数列表')
    actions = tuple(sorted(_non_negative_int(a, '动作') for a in raw['actions']))
    if len(set(actions)) != len(actions):
        raise ValueError('actions不能重复')
    max_policies = raw['max_policies']
    if isinstance(max_policies, bool) or not isinstance(max_policies, int) or max_policies <= 0:
        raise ValueError('max_policies必须是正整数')

    nodes = {}
    for row in raw['nodes']:
        node_id = row['id']
        period = _non_negative_int(row['period'], 'period')
        children = tuple(
            Edge(child=edge['id'], probability=_fraction(edge['probability'], '条件概率'))
            for edge in row['children']
        )
        if any(edge.probability < 0 for edge in children):
            raise ValueError(f'节点{node_id}存在负概率')
        if children and sum(edge.probability for edge in children) != 1:
            raise ValueError(f'节点{node_id}的条件概率和不为1')
        if len({edge.child for edge in children}) != len(children):
            raise ValueError(f'节点{node_id}的子节点重复')
        if any(edge.child not in ids for edge in children):
            raise ValueError(f'节点{node_id}引用了不存在的子节点')

        cash_raw = row['operating_cash']
        if not isinstance(cash_raw, dict):
            raise ValueError(f'节点{node_id}的operating_cash必须是对象')
        cash = {}
        for key, amount in cash_raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'节点{node_id}的收入表索引必须是整数') from exc
            cash[index] = _fraction(amount, f'节点{node_id}装机数{key}的经营现金流')
        for index in range(capacity + 1):
            if index not in cash:
                raise ValueError(f'节点{node_id}缺少装机数{index}的经营现金流')

        nodes[node_id] = Node(
            id=node_id,
            period=period,
            children=children,
            operating_cash=cash,
            budget=_fraction(row['budget'], f'节点{node_id}预算'),
            purchase_cost=_fraction(row['purchase_cost'], f'节点{node_id}采购成本'),
            salvage_per_unit=_fraction(row['salvage_per_unit'], f'节点{node_id}残值'),
        )
    if nodes[root].period != 0:
        raise ValueError('根period必须为0')

    order = tuple(sorted(ids))
    parent = {root: None}
    # 结构遍历：每非根节点恰一父、无循环、period逐层加一
    stack = [root]
    seen = {root}
    while stack:
        nid = stack.pop()
        node = nodes[nid]
        for edge in node.children:
            child = nodes[edge.child]
            if edge.child in seen:
                raise ValueError(f'节点{edge.child}被重复到达，树不能合流或循环')
            if child.period != node.period + 1:
                raise ValueError(f'节点{edge.child}的period必须比父节点多1')
            seen.add(edge.child)
            parent[edge.child] = nid
            stack.append(edge.child)
    if seen != ids:
        raise ValueError('存在孤立节点，所有节点必须从根可达')

    ancestors = {root: ()}
    reach = {root: Fraction(1, 1)}
    # 从根按拓扑层计算祖先链与到达概率，不能按id字典序（父id未必更小）
    frontier = [root]
    while frontier:
        nid = frontier.pop(0)
        for edge in nodes[nid].children:
            ancestors[edge.child] = ancestors[nid] + (nid,)
            reach[edge.child] = reach[nid] * edge.probability
            frontier.append(edge.child)

    leaves = tuple(nid for nid in order if nodes[nid].is_leaf)
    if not leaves:
        raise ValueError('有限树必须至少有一个叶节点')

    return Tree(
        root=root,
        nodes=nodes,
        initial_installed=initial,
        capacity_limit=capacity,
        actions=actions,
        rate=rate,
        alpha=alpha,
        loss_limit=loss_limit,
        max_policies=max_policies,
        order=order,
        parent=parent,
        ancestors=ancestors,
        reach_probability=reach,
        leaves=leaves,
    )
