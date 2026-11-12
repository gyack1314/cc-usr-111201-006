# 机器人扩点投入的情景树风险约束决策

这是现场机器人应用分析的合成资料项目。`data`保存业务记录，`analysis_data.io`负责严格JSON读取和原始字节指纹，`analysis_data.domain`提供已有的基础汇总。字段与数学口径见[资料约定](docs/data-contract.md)。所有厂区、设备、收益与结果均为合成，不代表任何实际项目的效果。

决策分析把“等看到现场需求再扩点”和“现在就买设备”的价值分开，避免原表格对每种未来行情事后挑最赚钱动作的乐观偏差：

- `analysis_data/model.py`：分数精确（`Fraction`）的有限根树模型与严格结构校验。
- `analysis_data/solver.py`：枚举全部确定性非随机化政策（零概率节点同样要求合法动作、不删枝），精确计算逐叶NPV、期望与`CVaR_alpha`，在风险上限内最大化期望NPV；`max_policies`是硬资源限额，超限只报`not_completed`，完整枚举无可行政策才报`infeasible`。另给逐叶“预知未来”上界并标记为不可实施。
- `analysis_data/checker.py`：独立检查器，用与求解器不同的实现重新遍历和枚举，不信任求解器缓存的NPV，篡改任一会额、政策或风险结论都会被`CheckError`拦下。
- `analysis_data/report.py`：逐节点政策、各叶现金流路径与概率、期望、CVaR、逐节点预算核对与预知上界报告。

Python 3.11及以上版本可直接运行，当前无第三方Python依赖。项目根目录中的常用命令：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
python -m analysis_data.cli solve data/normal.json --verify
python -m analysis_data.cli verify data/normal.json
```

`solve`支持`--loss-limit`覆盖资料中的CVaR损失上限（有理数字符串），`--verify`在求解后立即运行独立检查器。示例资料覆盖四类情形：`wait-value.json`中等待现场需求的政策严格优于现在采购且预知上界更高；`cvar-choice.json`中收紧损失上限会把根节点动作从“现在买”改成“等待”；`zero-probability.json`保留零概率分支的合法政策要求；`infeasible-risk.json`在完整枚举后确认无风险可行政策。`normal`是含残值的三期主资料。

当前命令展示原始资料摘要与基础统计。结果写入其他目录时应保留输入来源指纹，原始资料不覆盖。更多边界资料与normal采用同样的JSON结构，invalid前缀文件保存不合法输入。

容器可执行 `docker build -t field-analysis .` 后用 `docker run --rm field-analysis` 查看资料，或 `docker run --rm field-analysis python -m unittest discover -s tests -v` 运行现有检查。运行期间无外网或账号要求。全部金额与需求均为虚构分析资料，不是投资建议，也不调用任何付款或采购接口。
