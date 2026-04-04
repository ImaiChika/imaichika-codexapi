# 消融实验对比总结

生成时间：2026-04-04 23:02:43

## 实验说明

- 数据集：`data/raw/1.json`、`data/raw/1_augmented.json`、`data/raw/2.json`、`data/raw/3.json`
- 共同流程：原始数据加载、Layer1 规则抽取、画像汇总、原报告模板渲染、结果保存
- 控制变量：只切换 `NLP增强`、`Layer3推理`、`身份关联` 三个能力开关，不改原有主流程代码
- 结果文件命名：先按原逻辑生成，再在文件名末尾追加 `__模式名` 便于对比

## 模式总览

| 模式 | 金标准PII召回 | 高风险消息 | 中高风险候选 | 关键角色准确率 | 嫌疑Top3命中 | 嫌疑Top5命中 | 线索链数 | 报告文件 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 只有规则 | 100.00% | 0 | 51 | 0.00% | 2/4 | 3/4 | 0 | `final_report_20260404_225959__rules_only.txt` |
| 规则 + NLP | 100.00% | 11 | 55 | 11.11% | 3/4 | 3/4 | 0 | `final_report_20260404_230010__rules_nlp.txt` |
| 规则 + NLP + 推理 | 100.00% | 27 | 58 | 11.11% | 2/4 | 4/4 | 0 | `final_report_20260404_230022__rules_nlp_reasoning.txt` |
| 规则 + NLP + 推理 + 身份关联 | 100.00% | 27 | 58 | 11.11% | 2/4 | 4/4 | 25 | `final_report_20260404_230131__full_stack.txt` |

## 关键观察

### 只有规则

- 报告文件：`E:\imaichika_whu_api_chroma\data\processed\final_report_20260404_225959__rules_only.txt`
- 主要嫌疑人：user_-1002394323226, XiaoTJiang, daili13812, tqkf009, dbkyi, xiaofnb, small_liu, dfc6f
- 主要受害者：xy_07777, qqcd001, user_6085607279
- 高风险消息筛选数：0；中高风险候选消息数：51；角色判断准确率（关键账号主角色）：0.00%
- 核心嫌疑人排序：Top3 命中 2/4，Top5 命中 3/4，平均排名 3.5
- 核心嫌疑人具体排名：user_-1002394323226=1，dbkyi=5，XiaoTJiang=2，xiaofnb=6
- 金标准PII召回：100.00% （命中 11/11）
- 跨群关联簇：0，线索链：0
- 嫌疑方消息识别比例：4/66=6.06%；仅看增强跨群样本为 4/45=8.89%
- 投诉方消息识别比例：1/20=5.00%；仅看售后群为 1/15=6.67%
- 关键账号主角色：user_-1002394323226=victim，dbkyi=other，XiaoTJiang=other，victim_awei=other，beizhaole666=other
- 核心线索链命中：WangKang=False，收款地址=False，QQ=False
- 报告是否出现关键投诉用户：victim_awei=True，beizhaole666=True，wq2025=True

### 规则 + NLP

- 报告文件：`E:\imaichika_whu_api_chroma\data\processed\final_report_20260404_230010__rules_nlp.txt`
- 主要嫌疑人：XiaoTJiang, user_-1002394323226, dbkyi, daili13812, tqkf009, small_liu, xiaofnb, dfc6f
- 主要受害者：xy_07777, qqcd001, user_6085607279
- 高风险消息筛选数：11；中高风险候选消息数：55；角色判断准确率（关键账号主角色）：11.11%
- 核心嫌疑人排序：Top3 命中 3/4，Top5 命中 3/4，平均排名 3.25
- 核心嫌疑人具体排名：user_-1002394323226=2，dbkyi=3，XiaoTJiang=1，xiaofnb=7
- 金标准PII召回：100.00% （命中 11/11）
- 跨群关联簇：0，线索链：0
- 嫌疑方消息识别比例：12/66=18.18%；仅看增强跨群样本为 12/45=26.67%
- 投诉方消息识别比例：1/20=5.00%；仅看售后群为 1/15=6.67%
- 关键账号主角色：user_-1002394323226=victim，dbkyi=other，XiaoTJiang=scammer，victim_awei=other，beizhaole666=scammer
- 核心线索链命中：WangKang=False，收款地址=False，QQ=False
- 报告是否出现关键投诉用户：victim_awei=True，beizhaole666=True，wq2025=True

### 规则 + NLP + 推理

- 报告文件：`E:\imaichika_whu_api_chroma\data\processed\final_report_20260404_230022__rules_nlp_reasoning.txt`
- 主要嫌疑人：user_-1002394323226, dbkyi, tqkf009, XiaoTJiang, xiaofnb, daili13812, dfc6f, small_liu
- 主要受害者：xy_07777, user_6085607279, qqcd001
- 高风险消息筛选数：27；中高风险候选消息数：58；角色判断准确率（关键账号主角色）：11.11%
- 核心嫌疑人排序：Top3 命中 2/4，Top5 命中 4/4，平均排名 3.0
- 核心嫌疑人具体排名：user_-1002394323226=1，dbkyi=2，XiaoTJiang=4，xiaofnb=5
- 金标准PII召回：100.00% （命中 11/11）
- 跨群关联簇：0，线索链：0
- 嫌疑方消息识别比例：4/66=6.06%；仅看增强跨群样本为 3/45=6.67%
- 投诉方消息识别比例：2/20=10.00%；仅看售后群为 2/15=13.33%
- 关键账号主角色：user_-1002394323226=scammer，dbkyi=other，XiaoTJiang=other，victim_awei=other，beizhaole666=other
- 核心线索链命中：WangKang=False，收款地址=False，QQ=False
- 报告是否出现关键投诉用户：victim_awei=True，beizhaole666=True，wq2025=True

### 规则 + NLP + 推理 + 身份关联

- 报告文件：`E:\imaichika_whu_api_chroma\data\processed\final_report_20260404_230131__full_stack.txt`
- 主要嫌疑人：user_-1002394323226, dbkyi, tqkf009, XiaoTJiang, xiaofnb, daili13812, dfc6f, small_liu
- 主要受害者：xy_07777, user_6085607279, qqcd001
- 高风险消息筛选数：27；中高风险候选消息数：58；角色判断准确率（关键账号主角色）：11.11%
- 核心嫌疑人排序：Top3 命中 2/4，Top5 命中 4/4，平均排名 3.0
- 核心嫌疑人具体排名：user_-1002394323226=1，dbkyi=2，XiaoTJiang=4，xiaofnb=5
- 金标准PII召回：100.00% （命中 11/11）
- 跨群关联簇：205，线索链：25
- 嫌疑方消息识别比例：4/66=6.06%；仅看增强跨群样本为 3/45=6.67%
- 投诉方消息识别比例：2/20=10.00%；仅看售后群为 2/15=13.33%
- 关键账号主角色：user_-1002394323226=scammer，dbkyi=other，XiaoTJiang=other，victim_awei=other，beizhaole666=other
- 核心线索链命中：WangKang=True，收款地址=True，QQ=True
- 报告是否出现关键投诉用户：victim_awei=True，beizhaole666=True，wq2025=True

## 结论

- 规则层对高价值隐私实体已有较强抓取能力，是整套系统的基础，因此各模式在 PII 召回上差异不大。
- 仅加入 NLP 后，高风险消息筛选数和嫌疑方消息识别比例会明显上升，说明关键词与语义特征对嫌疑内容过滤有帮助。
- 加入 Layer3 推理后，角色判断会发生再分配，部分账号的判定会更保守，但高风险消息筛选与投诉方识别会更接近“案件式研判”思路。
- 身份关联层本身不直接提高 PII 抽取，但会显著补强跨群证据链，是系统从“识别消息”走向“还原事件”的关键。
