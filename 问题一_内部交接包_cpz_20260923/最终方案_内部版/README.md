# 问题一内部方案交付

本目录是问题一交接包，包含[论文内部稿](问题一_论文内部稿.md)、[章节—数据—文献映射](章节_数据_文献映射.md)、[代码与运行说明](代码与运行说明.md)、可重建的 [SQLite 证据库](data/q1_evidence.sqlite) 与 [构建程序](build_database.py)。`problem/` 是题面、数据说明和原项目指南，`code/` 是两套实验的代码快照，`literature/` 是六篇原题 PDF 与笔记，`evidence/` 是关键小型实验结果，`notes/` 是过程报告；全部复制关系记录在 [包清单](bundle_manifest.json)。原位置的代码和文件没有移动。

数据库只保留分析所需结构化数据；A1–A3 全文仍在 `real_attachments/A_data_value/`，库内 `source_file` 记录原始/派生文件相对路径、大小、SHA-256 和证据性质。

## 重建

在本项目根目录运行：

```powershell
python '问题一研究\最终方案_内部版\build_database.py'
python '问题一研究\最终方案_内部版\organize_bundle.py'
python '问题一研究\最终方案_内部版\verify_bundle.py'
python '问题一研究\最终方案_内部版\export_bundle.py'
```

数据库脚本先写 `data/q1_evidence.building.sqlite`，完成完整性、外键、行数和关键冲突/重合检验后再替换正式库。组织脚本复制 44 份题面、代码、文献、关键结果和笔记并写 SHA-256 清单；校验脚本核对所有快照与数据库核心约束。`export_bundle.py` 在上一级生成 `问题一_内部交接包_20260923.zip`，排除临时检查文件并做 ZIP CRC 校验。这些整理脚本只需要 Python 标准库；模型重算依赖 [`requirements.txt`](requirements.txt)。运行环境的 Python 为 3.13；库为通用 SQLite 格式。

## 表与口径

| 表/视图 | 用途 | 关键口径 |
|---|---|---|
| `source_file` | 36 个输入文件的角色、证据状态、大小和 SHA-256 | 路径相对项目根目录，原始 CSV 与论文 PDF 均有记录 |
| `domain_mapping` | A16 的 17 域质量—配比对应 | 3 直接、3 近似、11 推断/无同名质量域；空质量域为 `NULL` |
| `quality_record` | F_Q1-1 的 A1–A3 逐条分数，272,505 行 | `(source,id)` 为主键；含四组等权、平权、中位数、结构 PCA、初版四对冲突 |
| `quality_recheck` | F_Q1-2 的逐条独立复算，272,505 行 | 三对语义冲突由三个原始成对标记取并集；结构分歧单列 |
| `ai_blind_rating` | 135 条 AI 片段盲评 | 仅前 1500 字符，非人工真值 |
| `mixture_run` | 1,214 个按 `index` 成对连接的训练/测试/估算实验 | `observed=0` 仅 10B/70B；`overlaps_train=1` 包含 512 训练行及 126 估算行 |
| `mixture_weight` | 20,638 个比例值 | 同存原比例与按行和归一化比例，17 域/组 |
| `validation_loss` | 15,782 个 Loss | 13 域/组，估算状态由 `mixture_run` 关联判断 |
| `model_route`, `model_metric` | 两套已运行实验的模型设置和标量结果 | `train_cv` 与 `test_*` 分开，测试比较属事后探索 |
| `transfer_effect`, `pair_effect` | Pile-CC 减少 1 个百分点的局部情景 | 模型情景，不是随机干预或因果效应 |
| `bibliography` | 原题六篇本地 PDF 与问题一用途 | RegMix 正式版本勘误为 Liu 等、ICLR 2025 |
| `v_quality_summary`, `v_conflict_summary`, `v_mixture_summary` | 常用汇总视图 | 可直接查询各来源、域和尺度 |

## 查询示例

```sql
SELECT * FROM v_mixture_summary;
SELECT source, SUM(semantic_conflict_three) AS n FROM quality_recheck GROUP BY source;
SELECT route,sample,metric,value FROM model_metric
WHERE route IN ('kernel_ilr_eps_0.001','extra_trees')
  AND metric IN ('domain_standardized_mse','overall_mean_loss_mae');
SELECT path,evidence_state,sha256 FROM source_file WHERE role LIKE '%mixture%';
```

## 核验结果

构建时 `PRAGMA integrity_check=ok` 且外键检查为空；质量分 A1/A2/A3 行数为 51,230/17,523/203,752；三对语义冲突分别为 965/0/4,173；配比 512+256+256+64+63+63=1,214 组；17 项比例与 13 域 Loss 均无缺行；估算 10B/70B 与训练配方重合数各 63。正文所有训练与测试模型数字为已有工作流结果的归档，构建数据库没有重新训练模型。
