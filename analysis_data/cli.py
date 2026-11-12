"""命令行：资料摘要、情景树求解与独立核验，不改变原始资料。"""
import argparse
import json
from fractions import Fraction

from .checker import verify
from .domain import summarize
from .io import load
from .report import build_report


def _default(value):
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(f'不能序列化类型{type(value).__name__}')


def _print(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2,
                     allow_nan=False, default=_default))


def main():
    parser = argparse.ArgumentParser(
        description='机器人扩点情景树风险约束决策（虚构分析资料）'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    inspect_parser = sub.add_parser('inspect', help='展示资料基本统计')
    inspect_parser.add_argument('file')

    solve_parser = sub.add_parser('solve', help='求解风险约束最优确定性政策')
    solve_parser.add_argument('file')
    solve_parser.add_argument('--loss-limit', default=None,
                              help='覆盖资料中的损失上限（有理数字符串）')
    solve_parser.add_argument('--verify', action='store_true',
                              help='求解后立即运行独立检查器')

    verify_parser = sub.add_parser('verify', help='只运行独立检查器核验求解报告')
    verify_parser.add_argument('file')

    args = parser.parse_args()
    value, digest = load(args.file)

    if args.command == 'inspect':
        _print({'source_sha256': digest, 'summary': summarize(value)})
        return

    if args.command == 'verify':
        report = build_report(value)
        checked = verify(value, report)
        _print({'source_sha256': digest, 'verification': checked})
        return

    report = build_report(value, loss_limit=args.loss_limit)
    payload = {'source_sha256': digest, 'report': report}
    if args.verify:
        payload['verification'] = verify(value, report)
    _print(payload)


if __name__ == '__main__':
    main()
