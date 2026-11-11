"""读取情景节点并对明确现金流做精确折现。"""
from fractions import Fraction
from .io import unique
def discounted(cash,period,rate):
    rate=Fraction(rate)
    if rate<=-1 or type(period) is not int or period<0:raise ValueError('折现参数错误')
    return Fraction(cash)/(1+rate)**period
def summarize(value):
    ids=unique(value['nodes'],'id')
    if value['root'] not in ids:raise ValueError('根不存在')
    for node in value['nodes']:
        children=node['children']
        if any(x['id'] not in ids or Fraction(x['probability'])<0 for x in children):raise ValueError('子节点或概率错误')
        if children and sum(Fraction(x['probability']) for x in children)!=1:raise ValueError('条件概率和错误')
    return {'nodes':len(ids),'root':value['root'],'discount_rate':str(Fraction(value['rate'])),'illustrative_cash_npv':str(sum(discounted(x['amount'],x['period'],value['rate']) for x in value['illustrative_cash']))}
