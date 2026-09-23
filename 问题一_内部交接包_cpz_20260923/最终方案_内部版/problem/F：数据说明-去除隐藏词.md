## 数据说明

对应赛题《算力约束下提升大语言模型能力的资源配置建模》。本说明给出各数据文件的结构、字段与跨附件兼容性；各文件均使用英文文件名，赛题正文中以编号 A1–A18、B1–B12、C1–C10 指代，编号与英文文件名的对照见《编号与读取速查》。

本赛题以真实公开数据为主，并包含经真实数据校准、已明确标注的半合成补充数据。全部附件位于 real\_attachments/ 目录下，本文所有路径均相对于该目录。按实际文件占用的存储空间（即磁盘上的字节数）统计，总体积约 0.55 GB（压缩后，原始约 1.66 GB），其中绝大部分为必用数据（见《强制使用清单》）。参赛队伍可在个人电脑上使用 Python、Matlab 或 R 完成全部处理，无须 GPU。

如本文档、source\_manifest.json、attachment\_size\_summary.csv 与实际文件出现差异，以实际附件为准。

## 编号与读取速查

本节集中给出编号索引、各文件绑定问题与读取要求，便于与赛题正文分开查阅。各文件的完整字段与规模见后文《附件 A/B/C》各节。

## 数据总表

下表为各数据文件的编号—英文文件名对照，并标出每个文件绑定的问题与在解题中的角色。

性质栏口径统一如下：真实 = 直接观测或公开数据；半合成 = 基于真实数据校准并叠加噪声；外推/估算= 由模型外推或预估，非直接观测；插值 = 由检查点插值生成；子集 = 取自其他表的子集，非独立实验；混合 = 来源口径不完全统一，须分层使用；参考/说明 = 辅助文件。

附件 A（问题一）A\_data\_value/

<table><tr><td>编号</td><td>英文文件名</td><td>中文说明</td><td>性质</td><td>绑定问题·角色</td></tr><tr><td>A1</td><td>A_data_value/slimpajama_quality_signal_sample.jsonl.xz</td><td>质量信号抽样集(51,230条,27字段;来自SlimPajama-Meta-rater 按域抽样)</td><td>真实</td><td>问题一·质量主样本</td></tr><tr><td>A2</td><td>A_data_value/slimpajama_quality_extended/arxiv_*.jsonl.xz</td><td>arxiv 扩展质量信号</td><td>真实</td><td>问题一·必用</td></tr><tr><td>A3</td><td>A_data_value/slimpajama_quality_extended/github_*.jsonl.xz</td><td>github 扩展质量信号</td><td>真实</td><td>问题一·必用</td></tr><tr><td>A4</td><td>A_data_value/regmix_tables/train_mixture_1m.csv</td><td>训练配比(512组)</td><td>真实</td><td>问题一·配方·训练</td></tr><tr><td>A5</td><td>A_data_value/regmix_tables/train_pile_loss_1m.csv</td><td>训练 Loss</td><td>真实</td><td>问题一·Loss·训练</td></tr><tr><td>A6</td><td>A_data_value/regmix_tables/test_mixture_1m.csv</td><td>检验配比 1M(256 组)</td><td>真实</td><td>问题一·配方·检验</td></tr><tr><td>A7</td><td>A_data_value/regmix_tables/test_pile_loss_1m.csv</td><td>检验 Loss 1M</td><td>真实</td><td>问题一·Loss·检验</td></tr><tr><td>A8</td><td>A_data_value/regmix_tables/test_mixture_60m.csv</td><td>检验配比 60M</td><td>真实</td><td>问题一·配方·检验</td></tr><tr><td>A9</td><td>A_data_value/regmix_tables/test_pile_loss_60m.csv</td><td>检验 Loss 60M</td><td>真实</td><td>问题一·Loss·检验</td></tr><tr><td>A10</td><td>A_data_value/regmix_tables/test_mixture_1B.csv</td><td>检验配比 1B(64组)</td><td>真实</td><td>问题一·配方·检验</td></tr><tr><td>A11</td><td>A_data_value/regmix_tables/test_pile_loss_1B.csv</td><td>检验 Loss 1B</td><td>真实</td><td>问题一·Loss·检验</td></tr><tr><td>A12</td><td>A_data_value/regmix_tables/est_mixture_10b.csv</td><td>外推配比 10B(train 子集)</td><td>子集</td><td>问题一·配方·外推</td></tr><tr><td>A13</td><td>A_data_value/regmix_tables/est_pile_loss_10b.csv</td><td>外推 Loss 10B</td><td>外推</td><td>问题一·Loss·外推</td></tr><tr><td>A14</td><td>A_data_value/regmix_tables/est_mixture_70b.csv</td><td>外推配比 70B(train 子集)</td><td>子集</td><td>问题一·配方·外推</td></tr><tr><td>A15</td><td>A_data_value/regmix_tables/est_pile_loss_70b.csv</td><td>外推 Loss 70B</td><td>外推</td><td>问题一·Loss·外推</td></tr><tr><td>A16</td><td>A_data_value/domain_mapping_guide.csv</td><td>17 域 → 质量域映射表</td><td>参考</td><td>问题一·域映射</td></tr><tr><td>A17</td><td>A_data_value/regmix_domain_summary.csv</td><td>域摘要</td><td>真实</td><td>问题一·辅助(本版未接入求解,供扩展分析)</td></tr><tr><td>A18</td><td>A_data_value/regmix_domain_sample.jsonl.xz</td><td>域原始文本样例</td><td>真实</td><td>可选·辅助</td></tr></table>

说明：A4–A5 为训练表（train），A6–A11 为检验表（test），A12–A15 为外推表（est，即 estimated）。配方表与对应 Loss 表各为一个文件，按 index 列对应。

附件 B（问题二）B\_scaling\_laws/

<table><tr><td>编号</td><td>英文文件名</td><td>中文说明</td><td>性质</td><td>绑定问题·角色</td></tr><tr><td>B1</td><td>B_scaling_laws/pythia_training_log_existing.csv</td><td>Pythia 训练日志(1,176 行)</td><td>真实</td><td>问题二·主拟合</td></tr><tr><td>B2</td><td>B_scaling_laws/cerebras_training_log.csv</td><td>Cerebras 训练日志(1,029 行)</td><td>半合成</td><td>问题二·族外验证</td></tr><tr><td>B3</td><td>B_scaling_laws/training_trajectories/</td><td>Pythia 插值轨迹(8×500)</td><td>插值</td><td>问题二·轨迹验证</td></tr><tr><td>B4</td><td>B_scaling_laws/scaling_baseline.csv</td><td>跨族收敛点(57行,12 族)</td><td>真实</td><td>问题二·跨族验证</td></tr><tr><td>B5</td><td>B_scaling_laws/published_scaling_data.csv</td><td>文献标度律数据(44 行)</td><td>真实</td><td>问题二·文献验证</td></tr><tr><td>B6</td><td>B_scaling_laws/supplementary_NQ_experiment.csv</td><td>NQ 半合成实验(360 点)</td><td>半合成</td><td>问题二·质量 Q·基础</td></tr><tr><td>B7</td><td>B_scaling_laws/supplementary_NQ_experiment_expanded.csv</td><td>NQ 半合成实验(450 点)</td><td>半合成</td><td>问题二·质量 Q·扩展</td></tr><tr><td>B8</td><td>B_scaling_laws/supplementary_NQ_experiment_large.csv</td><td>NQ 半合成实验(1,704 点,含外推)</td><td>半合成</td><td>问题二·质量 Q·大规模</td></tr><tr><td>B9</td><td>B_scaling_laws/supplementary_large_models.csv</td><td>大模型元数据(100B-10000B)</td><td>真实</td><td>问题二·大模型参数</td></tr><tr><td>B10</td><td>B_scaling_laws/supplementary_large_baseline.csv</td><td>大模型预估 Loss(100B-10000B)</td><td>估算</td><td>问题二·大模型外推</td></tr><tr><td>B11</td><td>B_scaling_laws/open_model_family_metadata.csv</td><td>模型族元数据</td><td>真实</td><td>问题二·辅助</td></tr><tr><td>B12</td><td>B_scaling_laws/pythia_checkpoint_index.csv</td><td>Pythia 检查点索引</td><td>真实</td><td>问题二·辅助</td></tr></table>

说明：B1 为主拟合数据；B2/B3 用于模型族外或插值轨迹验证；B4/B5 用于跨族或文献验证；B6–B8 是唯一含质量 Q 的数据，用于把 Q 纳入标度律；B9/B10 用于百亿参数以上外推。

问题二在问题一输出的质量评分 Q 与配比 p 基础上构建广义标度律；附件 A 为同一数据集，本问不直接重复读取其原始文件，质量与配比信息通过问题一结果引入。

附件 C（问题三、四）C\_efficiency\_evolution/

<table><tr><td>编号</td><td>英文文件名</td><td>中文说明</td><td>性质</td><td>绑定问题·角色</td></tr><tr><td>C1</td><td>C_efficiency_evolution/leaderboard_cleaned.csv</td><td>排行榜(4,576行,6维)</td><td>真实</td><td>问题四·主用</td></tr><tr><td>C2</td><td>C_efficiency_evolution/leaderboard_enhanced.csv</td><td>排行榜增强</td><td>真实</td><td>问题四·可替代C1</td></tr><tr><td>C3</td><td>C_efficiency_evolution/leaderboard_extended_timeseries.csv</td><td>排行榜时序</td><td>混合</td><td>问题四·时序</td></tr><tr><td>C4</td><td>C_efficiency_evolution/epoch_all_ai_models.csv</td><td>全模型元数据</td><td>真实</td><td>问题四·宏观元数据</td></tr><tr><td>C5</td><td>C_efficiency_evolution/loss_benchmark_bridge.csv</td><td>Loss-Benchmark桥接(43条)</td><td>混合</td><td>问题四·桥接·可对照</td></tr><tr><td>C6</td><td>C_efficiency_evolution/loss_benchmark_bridge_expanded.csv</td><td>桥接扩展(75条)</td><td>混合</td><td>问题四·桥接·主用</td></tr><tr><td>C7</td><td>C_efficiency_evolution/model_architecture_metadata.csv</td><td>架构元数据(含上下文长度)</td><td>真实</td><td>问题三·主用;问题四</td></tr><tr><td>C8</td><td>C_efficiency_evolution/detailed_results/</td><td>逐任务评测JSON(约0.23GB)</td><td>真实</td><td>问题四·必用</td></tr><tr><td>C9</td><td>C_efficiency_evolution/data/*.parquet</td><td>Leaderboard原始Parquet</td><td>真实</td><td>问题四·与C1等价(本版未使用,供扩展分析)</td></tr><tr><td>C10</td><td>C_efficiency_evolution/pythia_*_eval_details/</td><td>评测说明目录(仅README)</td><td>说明</td><td>参考·不可替代C8</td></tr></table>

说明：问题三除 C7 外，另需使用问题一、二的输出（质量评分、配比、标度律参数）。

## 数据读取说明

数据可能存在缺失、噪声与冗余，须由参赛队伍自行预处理。大文件已提供 xz 压缩版（.jsonl.xz），可用Python 的 lzma.open() 直接读取，无须解压。

• 流式与分块：流式即逐行读取，分块即按固定行数批量读取，目的都是避免将大文件一次性载入内存。附件 A 质量信号 JSONL（A1–A3）宜流式或分块读取；不得以内存为由放弃大文件。

• 大文本字段：content / text 字段若仅用于统计特征，可不全文入库，仅在线提取统计量。

• 附件 A：A4 与 A5 通过 index 列对应；配比列为 train\_the\_pile\_\*（17 列），Loss 列为 metric/the\_pile\_\*\_val\_loss（13 列）。

• 附件 B：B1 每行记录某模型在某时刻的 N\_params\_B、D\_tokens\_B 与 val\_loss。

• 附件 C：C1 含 IFEval、BBH、MATH Lvl 5、GPQA、MUSR、MMLU-PRO 六维得分；C8 下每个子目录对应一个模型的逐任务 JSON。

• 单位换算（全文通用）：N\_params\_B、D\_tokens\_B 以十亿为单位（如 1.04 表示 1.04 × 10<sup>9</sup>），代入 C = 6ND 时须乘以 10<sup>9</sup>；C\_FLOPs\_1e21 的单位为 10<sup>21</sup> FLOPs。

## 强制使用清单

<table><tr><td>优先级</td><td>路径</td><td>约计体积</td><td>绑定问题</td><td>最低使用方式</td></tr><tr><td>必用</td><td>A_data_value/slimpajama_quality_extended/*.jsonl.xz</td><td>约0.04 GB(压缩)</td><td>问题一</td><td>处理全文记录,估计arxiv/github域级Q,并与抽样集对照</td></tr><tr><td>必用</td><td>A_data_value/slimpajama_quality_signal_sample.jsonl.xz</td><td>~0.10 GB(压缩)</td><td>问题一</td><td>22维质量评价与冲突消解主样本</td></tr><tr><td>必用</td><td>C_efficiency_evolution/detailed_results/</td><td>~0.23 GB</td><td>问题四</td><td>至少一项逐任务聚合分析</td></tr><tr><td>必用</td><td>附件A/B/C全部主CSV(配方、Pythia、榜单、桥接、架构、Epoch等)</td><td>数十MB</td><td>各问</td><td>按赛题要求接入模型</td></tr><tr><td>任选其一</td><td>C_efficiency_evolution/data/*.parquet与leaderboard_*.csv</td><td>~1 MB</td><td>问题四</td><td>CSV与Parquet内容等价,不必重复计算</td></tr></table>

跨附件兼容性与表头速查

<table><tr><td>附件</td><td>核心文件</td><td>关键字段</td><td>兼容性与连接键</td></tr><tr><td>A</td><td>A_data_value/regmix_tables/train_mixture_1m.csv</td><td>index, 17个域配比</td><td>通过 index 与 Loss 表对应</td></tr><tr><td>A</td><td>A_data_value/regmix_tables/train_pile_loss_1m.csv</td><td>index, 13个域 Loss</td><td>通过 index 与配方表对应</td></tr><tr><td>A</td><td>A_data_value/slimpajama_quality_signal_sample.jsonl.xz</td><td>_source_domain, 22个质量指标</td><td>需通过领域映射与 17 域体系建立联系</td></tr><tr><td>A</td><td>A_data_value/slimpajama_quality_extended/*.jsonl.xz</td><td>同质量指标字段</td><td>必用;与抽样集字段一致,覆盖 arxiv/github</td></tr><tr><td>A</td><td>A_data_value/regmix_domain_sample.jsonl.xz</td><td>text, _source_domain</td><td>可选;辅助跨域映射</td></tr><tr><td>A</td><td>A_data_value/domain_mapping_guide.csv</td><td>mixture_domain, quality_domain, mapping_type</td><td>17个配方域到质量域的参考映射</td></tr><tr><td>B</td><td>B_scaling_laws/pythia_training_log_existing.csv</td><td>N_params_B, D_tokens_B, val_loss</td><td>真实训练轨迹;标度律拟合核心数据</td></tr><tr><td>B</td><td>B_scaling_laws/scaling_baseline.csv</td><td>family, N_params_B, D_tokens_B, val_loss</td><td>跨族收敛点;泛化验证</td></tr><tr><td>B</td><td>B_scaling_laws/supplementary_NQ_experiment.csv</td><td>N_params_B, D_tokens_B, Q_score, val_loss</td><td>半合成 N-D-Q 实验点</td></tr><tr><td>C</td><td>C_efficiency_evolution/leaderboard_cleaned.csv</td><td>Model, #Params (B), Submission Date, 6 维得分</td><td>核心评测表</td></tr><tr><td>C</td><td>C_efficiency_evolution/loss_benchmark_bridge_expanded.csv</td><td>Model, Val_Loss, LB_*, Loss_Comparability</td><td>Loss 到 Benchmark 桥接;按可比性分层</td></tr><tr><td>C</td><td>C_efficiency_evolution/model_architecture_metadata.csv</td><td>model_name, n_layers, n_heads, max_position_embeddings</td><td>架构和上下文长度分析</td></tr><tr><td>C</td><td>C_efficiency_evolution/detailed_results/</td><td>逐任务 JSON</td><td>必用;细粒度评测聚合</td></tr></table>

## 附件 A：数据配方实验与质量信号

## 数据来源

• 配方实验数据：RegMix 项目（ICML 2024），https://github.com/sail-sg/regmix

• 质量信号数据与扩展质量信号（arxiv、github 两域完整数据文件）：SlimPajama-Meta-rater，https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater；其中 A1 抽样集为该数据源各质量信号域的按域抽样记录，非另行生成的独立实验数据

## 文件结构与说明

<table><tr><td>文件</td><td>格式</td><td>规模</td><td>性质</td><td>说明</td></tr><tr><td>A_data_value/regmix_tables/train_mixture_1m.csv</td><td>CSV</td><td>512 行 × 18列</td><td>真实</td><td>512 个训练配方,17 个The Pile 子领域配比</td></tr><tr><td>A_data_value/regmix_tables/train_pile_loss_1m.csv</td><td>CSV</td><td>512 行 × 14列</td><td>真实</td><td>对应配方的 13 个验证域交叉熵损失</td></tr><tr><td>A_data_value/regmix_tables/test_mixture_1m.csv</td><td>CSV</td><td>256 行 × 18列</td><td>真实</td><td>1M 参数规模检验配方</td></tr><tr><td>A_data_value/regmix_tables/test_pile_loss_1m.csv</td><td>CSV</td><td>256 行 × 14列</td><td>真实</td><td>1M 检验配方对应 Loss</td></tr><tr><td>A_data_value/regmix_tables/test_mixture_60m.csv</td><td>CSV</td><td>256 行 × 18列</td><td>真实</td><td>60M 参数规模检验配方</td></tr><tr><td>A_data_value/regmix_tables/test_pile_loss_60m.csv</td><td>CSV</td><td>256 行 × 14列</td><td>真实</td><td>60M 检验配方对应 Loss</td></tr><tr><td>A_data_value/regmix_tables/test_mixture_1B.csv</td><td>CSV</td><td>64 行 × 18列</td><td>真实</td><td>1B 参数规模检验配方</td></tr><tr><td>A_data_value/regmix_tables/test_pile_loss_1B.csv</td><td>CSV</td><td>64 行 × 14列</td><td>真实</td><td>1B 检验配方对应 Loss</td></tr><tr><td>A_data_value/regmix_tables/est_pile_loss_10b.csv</td><td>CSV</td><td>63行×14列</td><td>外推</td><td>10B尺度预估配方Loss</td></tr><tr><td>A_data_value/regmix_tables/est_pile_loss_70b.csv</td><td>CSV</td><td>63行×14列</td><td>外推</td><td>70B尺度预估配方Loss</td></tr><tr><td>A_data_value/regmix_tables/est_mixture_10b.csv</td><td>CSV</td><td>63行×18列</td><td>子集</td><td>10B尺度配比(train子集)</td></tr><tr><td>A_data_value/regmix_tables/est_mixture_70b.csv</td><td>CSV</td><td>63行×18列</td><td>子集</td><td>70B尺度配比(train子集)</td></tr><tr><td>A_data_value/regmix_domain_summary.csv</td><td>CSV</td><td>17行×5列</td><td>真实</td><td>各领域文本抽样统计信息</td></tr><tr><td>A_data_value/domain_mapping_guide.csv</td><td>CSV</td><td>17行×4列</td><td>参考</td><td>17个配方域与质量信号域的映射指南</td></tr><tr><td>A_data_value/slimpajama_quality_signal_sample.jsonl.xz</td><td>JSONL (xz压缩)</td><td>51,230条×27字段</td><td>真实</td><td>22个质量指标,覆盖7个域;SlimPajama-Meta-rater质量信号按域抽样子集(必用)</td></tr><tr><td>A_data_value/slimpajama_quality_extended/arxiv_*.jsonl.xz</td><td>JSONL (xz压缩)</td><td>17,523条×24字段</td><td>真实</td><td>arxiv域扩展质量信号(必用,约3.5MB(压缩))</td></tr><tr><td>A_data_value/slimpajama_quality_extended/github_*.jsonl.xz</td><td>JSONL (xz压缩)</td><td>203,752条×24字段</td><td>真实</td><td>github域扩展质量信号(必用,约36MB(压缩))</td></tr><tr><td>A_data_value/regmix_domain_sample.jsonl.xz</td><td>JSONL (xz压缩)</td><td>138,034条</td><td>真实</td><td>RegMix各域原始文本抽样(可选,约0.16GB(xz压缩))</td></tr></table>

性质说明：真实 = 来自公开论文与数据集的直接观测值。外推 = est\_pile\_loss\_10b.csv、est\_pile\_loss\_70b.csv 从 1M/60M/1B 三尺度幂律外推，非直接观测。子集 = est\_mixture\_10b.csv、est\_mixture\_70b.csv 取自 train 配比子集，非独立实验。参考 =domain\_mapping\_guide.csv 为人工整理，非实验数据。

## 核心字段说明

配方表：index 为配方编号；train\_the\_pile\_\* 共 17 列，表示 17 个 The Pile 子领域的训练配比（即领域配比 p 的取值），每行和约为 1（存在千分位级舍入误差）。

17 个训练领域：arxiv，freelaw，nih\_exporter，pubmed\_central，wikipedia\_en，dm\_mathematics， github，philpapers，stackexchange，enron\_emails，gutenberg\_pg\_19，pile\_cc，ubuntu\_irc，europarl，hackernews，pubmed\_abstracts，uspto\_backgrounds。

Loss 表：index 对应配方编号；metric/the\_pile\_\*\_val\_loss 共 13 列。17 个训练域中，nih\_exporter、enron\_emails、europarl、philpapers 仅有训练配比列，无对应交叉熵损失列。参赛队伍可选择保留、合并、正则化处理或仅在训练配比模型中解释其间接影响，但需说明理由。

质量信号：A1 每行一个 JSON 对象，共 27 个字段，其中 22 个为质量指标字段（14 个标量指标 + 8 个多维列表型指标），其余 5 个为辅助字段（id、content、sub\_path、\_source\_domain、\_source\_path）。A2/A3 每行 24 个字段，为 22 个质量指标、id 与 sub\_path；已去除 content、\_source\_domain、\_source\_path，来源域由文件名或路径推断。

22 个质量指标：fineweb\_edu，fluency\_en，modernbert\_cleanliness，modernbert\_readability，modernbert\_reasoning，modernbert\_professionalism，dsir\_books，dsir\_wiki，dsir\_math，qurater， ad\_en，rps\_doc\_word\_count，rps\_doc\_num\_sentences，rps\_doc\_unigram\_entropy，rps\_doc\_frac\_unique\_words，rps\_doc\_frac\_no\_alph\_words，rps\_doc\_frac\_chars\_top\_2gram，rps\_doc\_frac\_chars\_top\_3gram，rps\_lines\_uppercase\_letter\_fraction，rps\_lines\_ending\_with\_terminal\_punctution\_mark，rps\_lines\_numerical\_chars\_fraction，rps\_doc\_mean\_word\_length。

其中 8 个列表型字段需先压缩为标量后再参与建模。扩展集与抽样集的 22 个质量指标字段一致，宜使用同一套预处理流程。

域映射：质量信号覆盖 7 个域（arxiv、book、c4、commoncrawl、github、stackexchange、wikipedia），配方实验涉及上述 17 个域。二者域名并不一一相同。domain\_mapping\_guide.csv 以配方域为行，给出到质量域的参考对应：direct / near\_direct / inferred。其中 c4 仅出现在质量信号侧，配方 17 域中无同名列。

## 附件 B：模型训练动态轨迹与标度律补充数据

## 数据来源

• Pythia 训练日志：EleutherAI Pythia Scaling Suite，https://github.com/EleutherAI/pythia

• Cerebras-GPT 轨迹：根据 Pythia 标度律和公开 Cerebras-GPT 训练设定校准生成

• 跨族基准点：从公开模型族文档与论文整理

• 大模型缩放参数：Epoch AI 数据库，https://epoch.ai/data/all\_ai\_models.csv

• 已发表标度律数据：Kaplan et al. (2020)、Hofmann et al. (2022) 等文献

## 文件结构与说明

<table><tr><td>文件</td><td>格式</td><td>规模</td><td>性质</td><td>说明</td></tr><tr><td>B_scaling_laws/pythia_training_log_existing.csv</td><td>CSV</td><td>1,176行×15列</td><td>真实</td><td>8个Pythia模型检查点的真实训练轨迹</td></tr><tr><td>B_scaling_laws/cerebras_training_log.csv</td><td>CSV</td><td>1,029行×15列</td><td>半合成</td><td>7个Cerebras模型训练轨迹(校准+噪声)</td></tr><tr><td>B_scaling_laws/scaling_baseline.csv</td><td>CSV</td><td>57行×5列</td><td>真实</td><td>12个模型族的跨族收敛基准点</td></tr><tr><td>B_scaling_laws/supplementary_NQ_experiment.csv</td><td>CSV</td><td>360行</td><td>半合成</td><td>N-D-Q实验点</td></tr><tr><td>B_scaling_laws/supplementary_NQ_experiment_expanded.csv</td><td>CSV</td><td>450行</td><td>半合成</td><td>扩展版</td></tr><tr><td>B_scaling_laws/supplementary_NQ_experiment_large.csv</td><td>CSV</td><td>1,704行</td><td>半合成</td><td>大规模矩阵,含更大参数外推</td></tr><tr><td>B_scaling_laws/supplementary_large_models.csv</td><td>CSV</td><td>132行</td><td>真实</td><td>100B以上开源模型缩放参数(Epoch AI公开报告值)</td></tr><tr><td>B_scaling_laws/supplementary_large_baseline.csv</td><td>CSV</td><td>128行</td><td>估算</td><td>100B以上模型预估Loss</td></tr><tr><td>B_scaling_laws/published_scaling_data.csv</td><td>CSV</td><td>44行</td><td>真实</td><td>已发表标度律基准点</td></tr><tr><td>B_scaling_laws/training_trajectories/*.csv</td><td>CSV</td><td>8×500=4,000行</td><td>插值</td><td>8个Pythia模型插值训练轨迹</td></tr><tr><td>B_scaling_laws/pythia_checkpoint_index.csv</td><td>CSV</td><td>1,386行</td><td>真实</td><td>Pythia检查点索引元数据</td></tr><tr><td>B_scaling_laws/open_model_family_metadata.csv</td><td>CSV</td><td>18行</td><td>真实</td><td>模型族仓库元数据</td></tr></table>

性质说明：真实 = 直接观测值或公开报告值。半合成 = cerebras\_training\_log.csv、supplementary\_NQ\_experiment\*.csv 基于真实标度律校准并叠加噪声，非直接实验观测。估算 = supplementary\_large\_baseline.csv 用已拟合标度律参数估算 Loss，非观测值。插值 = training\_trajectories/\*.csv 从 Pythia 检查点插值生成。

## 核心字段说明

pythia\_training\_log\_existing.csv：N\_params\_B 为参数量，D\_tokens\_B 为已训练数据量，C\_FLOPs\_1e21 为累计计算量，val\_loss 为验证集交叉熵损失。四者的单位换算统一见《数据读取说明》。

Pythia 日志中 C\_FLOPs\_1e21 与 6ND 的中位比值约为 1。Cerebras 半合成轨迹已统一 D\_tokens\_B 单位为十亿 tokens，并与 C\_FLOPs\_1e21 保持 6ND 量级一致。

pythia\_checkpoint\_index.csv 收录多个 Pythia 模型的检查点索引，而 pythia\_training\_log\_existing.csv 仅包含其中 8 个模型（约 70M–12B）的训练轨迹。

## 标度律数据汇总

<table><tr><td>数据集</td><td>N 范围</td><td>来源</td></tr><tr><td>Pythia 真实轨迹(1,176点)</td><td>0.07–12B</td><td>EleutherAI Pythia</td></tr><tr><td>Cerebras 半合成轨迹(1,029点)</td><td>0.11–13B</td><td>Pythia 标度律校准</td></tr><tr><td>跨族收敛基准(57点)</td><td>0.07–72B</td><td>12 个模型族公开数据</td></tr><tr><td>已发表基准(44点)</td><td>约 0.5–280B</td><td>公开论文整理</td></tr><tr><td>大模型缩放参数(132点)</td><td>100B 以上</td><td>Epoch AI 数据库</td></tr><tr><td>N-D-Q 半合成实验(360/450/1,704点)</td><td>含更大参数外推</td><td>校准 + 插值 + 外推</td></tr></table>

## 附件 C：模型评测与演进数据

## 数据来源

• Open LLM Leaderboard v2：https://huggingface.co/datasets/open-llm-leaderboard-old/results

• Epoch AI Models Dataset：https://epoch.ai/data/all\_ai\_models.csv

• 技术报告与模型配置：用于 Loss–Benchmark 桥接和架构元数据整理

• HuggingFace Open LLM Leaderboard 详细评测结果：https://huggingface.co/datasets/open-llmleaderboard/results

## 文件结构与说明

<table><tr><td>文件</td><td>格式</td><td>规模</td><td>性质</td><td>说明</td></tr><tr><td>C_efficiency_evolution/leaderboard_cleaned.csv</td><td>CSV</td><td>4,576行×12列</td><td>真实</td><td>6维 Benchmark评测得分</td></tr><tr><td>C_efficiency_evolution/leaderboard_enhanced.csv</td><td>CSV</td><td>4,576行×15列</td><td>真实</td><td>cleaned + Epoch AI发布日期等</td></tr><tr><td>C_efficiency_evolution/leaderboard_extended_timeseries.csv</td><td>CSV</td><td>4,599行×11列</td><td>混合</td><td>约2019-2025年时序(含历史模型)</td></tr><tr><td>C_efficiency_evolution/epoch_all_ai_models.csv</td><td>CSV</td><td>3,523行×57列</td><td>真实</td><td>Epoch AI全量模型元数据(不含评测得分)</td></tr><tr><td>C_efficiency_evolution/loss_benchmark_bridge.csv</td><td>CSV</td><td>43行</td><td>混合</td><td>原始Loss-Benchmark桥接</td></tr><tr><td>C_efficiency_evolution/loss_benchmark_bridge_expanded.csv</td><td>CSV</td><td>75行</td><td>混合</td><td>扩展桥接(高可比+中可比)</td></tr><tr><td>C_efficiency_evolution/model_architecture_metadata.csv</td><td>CSV</td><td>45行×7列</td><td>真实</td><td>主流开源模型架构元数据</td></tr><tr><td>C_efficiency_evolution/detailed_results/</td><td>JSON</td><td>1,863个模型子目录/1,958个JSON</td><td>真实</td><td>各模型6维 Benchmark逐任务详细评测结果(必用,约0.23GB;4个JSON截断,宜按目录取最新可解析记录)</td></tr><tr><td>C_efficiency_evolution/pythia_*_eval_details/</td><td>文本</td><td>7个目录</td><td>说明</td><td>目前仅含README(数据集说明),不能替代detailed_results/的逐任务JSON</td></tr><tr><td>C_efficiency_evolution/data/train-00000-of-00001.parquet</td><td>Parquet</td><td>4,576行×36列</td><td>真实</td><td>原始LeaderboardParquet;核心六维与C1等价,另含原始附加列,任选其一</td></tr></table>

性质说明：真实 = 来自公开 Leaderboard 或公开元数据（含 data/\*.parquet）。混合 =loss\_benchmark\_bridge\*.csv 的 Loss 来源与验证集不完全统一，须按可比性分层；leaderboard\_extended\_timeseries.csv 不同年份评测标准可能不完全一致。说明 = pythia\_\*\_eval\_details/ 仅含数据集说明文本。

## 核心字段说明

leaderboard\_cleaned.csv：关键字段包括 Model、#Params (B)、Submission Date、IFEval / BBH / MATH Lvl 5 / GPQA / MUSR / MMLU-PRO（6 维评测得分）、均分列、Hub License、Type。

leaderboard\_enhanced.csv：在 cleaned 基础上增加 Epoch AI 相关匹配列。

epoch\_all\_ai\_models.csv：不包含评测得分，主要提供发布时间、组织、参数量、训练计算量、训练数据量、开源权重等宏观元数据。部分文本字段可能包含换行，建议用标准 CSV 解析后计数。

Loss–Benchmark 桥接：loss\_benchmark\_bridge\_expanded.csv 包含 75 个模型同时具备交叉熵损失与 6维 Benchmark 得分。其中高可比样本主要来自 Pythia（Loss 可与附件 B 对照），其余为中可比样本（Loss来源于不同技术报告或论文）。Loss\_Comparability 字段标注可比性等级。

模型架构元数据：model\_architecture\_metadata.csv 包含层数、注意力头数、隐藏维度、词表大小、最大上下文长度（max\_position\_embeddings）和训练数据量等字段。模型名称与 Leaderboard 的 Model 字段不一定完全一致，匹配时需说明规则。

## 数据使用说明

1. 先满足问题绑定必用集，再展开方法设计；附录须附数据利用清单。

2. 半合成与外推数据（supplementary\_NQ\_experiment\*.csv、cerebras\_training\_log.csv、est\_pile\_loss\_\*.csv、supplementary\_large\_baseline.csv）非直接观测，应在论文中标注。

3. scaling\_baseline.csv 可用于跨族验证；若用作盲测，应说明其与训练日志的近邻关系。

4. 开源筛选可依据 Leaderboard 的 Hub License 或 Epoch AI 的 Open model weights?，须说明口径。

5. 三套附件数据源相互独立采集，四问建模逻辑需递进耦合；桥接关系主要通过 Loss、Model、(N, D) 与领域映射建立。

6. C8 共 1,863 个模型子目录、1,958 个 JSON；其中 4 个 JSON 截断，1,860 个目录含至少一个可解析JSON，1,854 个模型六维完整。逐任务聚合时宜按目录取最新可解析记录，跳过损坏文件并说明处理方式。