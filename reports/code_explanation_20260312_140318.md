# 项目代码详细解释报告

- 生成时间: 2026-03-12 14:03:18
- 项目根目录: E:\imaichika_whu_api
- 扫描到 Python 文件数: 19

## 阅读说明

本报告对项目内的每个 Python 代码文件进行静态分析，列出：
- 文件职责与主要数据流
- 每个类/函数/方法的签名、用途、内部调用
- 在项目中被调用的位置（按行号列出，基于 AST 的 best-effort 匹配）

注意：Python 是动态语言，静态分析无法 100% 精确区分同名方法/函数的真实绑定；本报告的“被调用位置”为 best-effort 检索，适合作为审计与检索入口。

## 文件索引

- `local_qwen_chat.py`
- `main.py`
- `src\__init__.py`
- `src\analysis\__init__.py`
- `src\analysis\layer1_regex.py`
- `src\analysis\layer2_nlp.py`
- `src\analysis\layer3_reasoning.py`
- `src\config.py`
- `src\linkage\__init__.py` - Identity/linkage analyzers for multi-group tracing.
- `src\linkage\identity_resolver.py`
- `src\loader.py`
- `src\models\embedding.py`
- `src\models\llm_wrapper.py`
- `src\profiling\group_profile.py`
- `src\profiling\user_profile.py`
- `src\storage\__init__.py` - Storage adapters for local multi-database linkage.
- `src\storage\multi_db.py`
- `src\utils.py`
- `test.py`

## 运行入口与总体流程（基于代码结构推断）

- `main.py`：多文件(`data/raw/*.json`)处理入口，负责串联 L1/L2/L3 分析、画像聚合、跨群身份关联、SQLite 落库、报告输出。
- `src/analysis/*`：分层分析（正则/关键词 -> NLP/社交拓扑 -> 低成本规则 + LLM 推理）。
- `src/profiling/*`：用户/群聚合统计与角色判别（嫌疑/受害/水军）。
- `src/linkage/*`：跨群身份簇与“线索链”（软关联）生成。
- `src/storage/*`：SQLite（全局库 + 按账号库）持久化。

---

## 文件：`local_qwen_chat.py`

**主要依赖（imports）**：
- os
- torch
- from transformers import AutoModelForCausalLM, AutoTokenizer

**定义清单**：
- `module_function` `main()` (L14-L93)

### main()

- 类型: `module_function`
- 位置: `local_qwen_chat.py:14`
- 内部调用(去重): print, AutoTokenizer.from_pretrained, AutoModelForCausalLM.from_pretrained, history.append, tokenizer.apply_chat_template, tokenizer.to, input.strip, query.lower, torch.no_grad, model.generate, tokenizer.batch_decode, tokenizer, zip, input, len

**在项目中被调用的位置（按名称匹配）**：
- local_qwen_chat.py:96 | main | main()
- main.py:200 | main | main()

---

## 文件：`main.py`

**主要依赖（imports）**：
- os
- from datetime import datetime
- from pathlib import Path
- sys
- from typing import Dict, List
- from tqdm import tqdm
- from src.analysis.layer1_regex import RegexAnalyzer
- from src.analysis.layer2_nlp import InteractionNetwork, TextMiner
- from src.analysis.layer3_reasoning import ReasoningLayer
- from src.config import DATA_DB_DIR, DATA_PROC_DIR, DATA_RAW_DIR, HIGH_SIGNAL_KEYWORDS, LLM_MIN_L1_SCORE, LLM_MIN_L2_SCORE, REPORT_CLUE_CHAIN_TOP_K
- from src.linkage.identity_resolver import IdentityResolver
- from src.loader import load_json_data
- from src.profiling.group_profile import GroupProfiler
- from src.profiling.user_profile import UserProfiler
- from src.storage.multi_db import MultiDBManager
- from src.utils import save_json, setup_logger

**定义清单**：
- `module_function` `should_use_llm(message: dict)` (L32-L59) - 只有当消息被视为高风险或语义模糊时，才调用LLM，以节省token。
- `module_function` `iter_raw_files(raw_dir: Path)` (L62-L63)
- `module_function` `_format_clue_chain_lines(chains: List[Dict], top_k: int=REPORT_CLUE_CHAIN_TOP_K)` (L66-L86)
- `module_function` `main()` (L89-L196)

### should_use_llm(message: dict)

- 类型: `module_function`
- 位置: `main.py:32`
- Docstring(首行): 只有当消息被视为高风险或语义模糊时，才调用LLM，以节省token。
- 内部调用(去重): message.get, str, bool, any, float, len, text.strip

**在项目中被调用的位置（按名称匹配）**：
- main.py:128 | should_use_llm | if should_use_llm(msg):

### iter_raw_files(raw_dir: Path)

- 类型: `module_function`
- 位置: `main.py:62`
- 内部调用(去重): sorted, raw_dir.glob, p.is_file

**在项目中被调用的位置（按名称匹配）**：
- main.py:104 | iter_raw_files | raw_files = iter_raw_files(DATA_RAW_DIR)

### _format_clue_chain_lines(chains: List[Dict], top_k: int=REPORT_CLUE_CHAIN_TOP_K)

- 类型: `module_function`
- 位置: `main.py:66`
- 内部调用(去重): lines.append, join, c.get, pair_items.append, c.get.get, p.get

**在项目中被调用的位置（按名称匹配）**：
- main.py:184 | _format_clue_chain_lines | for line in _format_clue_chain_lines(clue_chains, REPORT_CLUE_CHAIN_TOP_K):

### main()

- 类型: `module_function`
- 位置: `main.py:89`
- 内部调用(去重): setup_logger, logger.info, datetime.now.strftime, RegexAnalyzer, TextMiner, InteractionNetwork, ReasoningLayer, UserProfiler, GroupProfiler, IdentityResolver, MultiDBManager, iter_raw_files, l2_net.build_from_data, l2_net.analyze_centrality, user_p.aggregate, user_p.finalize, group_p.get_summary_context, identity_resolver.resolve, identity_resolver.attach_cluster_labels, identity_resolver.build_trace_events, identity_resolver.summarize, identity_resolver.build_clue_chains, db.store_identity_clusters, db.store_trace_events, l3.generate_comprehensive_report, save_json, db.close, logger.warning, load_json_data, enumerate, msg.get, Path, open, f.write, _format_clue_chain_lines, datetime.now, tqdm, dict, l1.process_single_message, msg.update ?

**在项目中被调用的位置（按名称匹配）**：
- local_qwen_chat.py:96 | main | main()
- main.py:200 | main | main()

---

## 文件：`src\__init__.py`

（本文件未检测到可记录的类/函数定义，或仅包含常量/导入。）

---

## 文件：`src\analysis\__init__.py`

（本文件未检测到可记录的类/函数定义，或仅包含常量/导入。）

---

## 文件：`src\analysis\layer1_regex.py`

**主要依赖（imports）**：
- re
- from typing import Dict, List
- from src.config import KEYWORD_RULES, REGEX_PATTERNS

**定义清单**：
- `class` `RegexAnalyzer(class)` (L7-L148)
- `method` `RegexAnalyzer.__init__(self)` (L8-L9)
- `method` `RegexAnalyzer._normalize_matches(matches)` (L12-L20)
- `method` `RegexAnalyzer._clean_name_candidates(names: List[str])` (L23-L40)
- `method` `RegexAnalyzer._clean_address_candidates(addrs: List[str])` (L43-L51)
- `method` `RegexAnalyzer.scan_pii(self, text: str)` (L53-L78)
- `method` `RegexAnalyzer.match_keywords(self, text: str)` (L80-L85)
- `method` `RegexAnalyzer.detect_role_clues(self, text: str)` (L87-L102)
- `method` `RegexAnalyzer.process_single_message(self, message: Dict)` (L104-L148)

### RegexAnalyzer(class)

- 类型: `class`
- 位置: `src\analysis\layer1_regex.py:7`

**在项目中被调用的位置（按名称匹配）**：
- main.py:95 | RegexAnalyzer | l1 = RegexAnalyzer()

### RegexAnalyzer.__init__(self)

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:8`
- 内部调用(去重): re.compile, REGEX_PATTERNS.items

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### RegexAnalyzer._normalize_matches(matches)

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:12`
- 内部调用(去重): isinstance, str.strip, next, normalized.append, str

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer1_regex.py:57 | self._normalize_matches | matches = self._normalize_matches(pattern.findall(text or ""))

### RegexAnalyzer._clean_name_candidates(names: List[str])

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:23`
- 内部调用(去重): any, re.search, out.append, len

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer1_regex.py:62 | self._clean_name_candidates | matches = self._clean_name_candidates(matches)

### RegexAnalyzer._clean_address_candidates(addrs: List[str])

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:43`
- 内部调用(去重): out.append, len, any

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer1_regex.py:64 | self._clean_address_candidates | matches = self._clean_address_candidates(matches)

### RegexAnalyzer.scan_pii(self, text: str)

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:53`
- 内部调用(去重): self.patterns.items, set, self._normalize_matches, results.get, pattern.findall, self._clean_name_candidates, sorted, self._clean_address_candidates

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer1_regex.py:109 | self.scan_pii | pii_info = self.scan_pii(text)

### RegexAnalyzer.match_keywords(self, text: str)

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:80`
- 内部调用(去重): KEYWORD_RULES.items, any, detected_topics.append

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer1_regex.py:110 | self.match_keywords | topics = self.match_keywords(text)

### RegexAnalyzer.detect_role_clues(self, text: str)

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:87`
- 内部调用(去重): sum, re.search

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer1_regex.py:131 | self.detect_role_clues | role_clue = self.detect_role_clues(text)

### RegexAnalyzer.process_single_message(self, message: Dict)

- 类型: `method`
- 位置: `src\analysis\layer1_regex.py:104`
- 内部调用(去重): message.get, self.scan_pii, self.match_keywords, self.detect_role_clues, min, message.update, sum, l1_evidence.append, topic_weights.get, bool, len, pii_info.values, list, pii_info.keys

**在项目中被调用的位置（按名称匹配）**：
- main.py:125 | l1.process_single_message | msg = l1.process_single_message(msg)

---

## 文件：`src\analysis\layer2_nlp.py`

**主要依赖（imports）**：
- re
- from collections import Counter
- networkx
- from src.config import STOPWORDS, SYSTEM_MSG_KEYWORDS

**定义清单**：
- `class` `TextMiner(class)` (L15-L106) - NLP feature extraction for single messages.
- `method` `TextMiner.__init__(self)` (L18-L35)
- `method` `TextMiner.is_system_message(self, text: str)` (L37-L45)
- `method` `TextMiner._fallback_keywords(self, text: str, top_k: int=5)` (L47-L50)
- `method` `TextMiner.extract_keywords(self, text: str, top_k: int=5)` (L52-L66)
- `method` `TextMiner.process(self, message)` (L68-L106)
- `class` `InteractionNetwork(class)` (L109-L180) - Directed interaction graph with mention + adjacency edges.
- `method` `InteractionNetwork.__init__(self)` (L112-L115)
- `method` `InteractionNetwork.build_from_data(self, all_messages)` (L117-L136)
- `method` `InteractionNetwork._add_edge_weight(self, source, target, weight=1)` (L138-L142)
- `method` `InteractionNetwork.analyze_centrality(self)` (L144-L163)
- `method` `InteractionNetwork.identify_kols(self, stats, top_n=5)` (L165-L180)

### TextMiner(class)

- 类型: `class`
- 位置: `src\analysis\layer2_nlp.py:15`
- Docstring(首行): NLP feature extraction for single messages.

**在项目中被调用的位置（按名称匹配）**：
- main.py:96 | TextMiner | l2_nlp = TextMiner()

### TextMiner.__init__(self)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:18`

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### TextMiner.is_system_message(self, text: str)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:37`
- 内部调用(去重): str, any, re.search

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer2_nlp.py:53 | self.is_system_message | if not text or self.is_system_message(text):
- src\analysis\layer2_nlp.py:70 | self.is_system_message | is_sys = self.is_system_message(text)

### TextMiner._fallback_keywords(self, text: str, top_k: int=5)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:47`
- 内部调用(去重): re.split, len

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer2_nlp.py:57 | self._fallback_keywords | return self._fallback_keywords(text, top_k=top_k)

### TextMiner.extract_keywords(self, text: str, top_k: int=5)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:52`
- 内部调用(去重): jieba.analyse.extract_tags, self.is_system_message, self._fallback_keywords, len

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer2_nlp.py:71 | self.extract_keywords | keywords = self.extract_keywords(text) if not is_sys else []

### TextMiner.process(self, message)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:68`
- 内部调用(去重): str, self.is_system_message, self.extract_keywords, l2_evidence.append, any, min, message.get, len, list, jieba.cut

**在项目中被调用的位置（按名称匹配）**：
- main.py:126 | l2_nlp.process | msg.update(l2_nlp.process(msg))

### InteractionNetwork(class)

- 类型: `class`
- 位置: `src\analysis\layer2_nlp.py:109`
- Docstring(首行): Directed interaction graph with mention + adjacency edges.

**在项目中被调用的位置（按名称匹配）**：
- main.py:97 | InteractionNetwork | l2_net = InteractionNetwork()

### InteractionNetwork.__init__(self)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:112`
- 内部调用(去重): nx.DiGraph, Counter, re.compile

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### InteractionNetwork.build_from_data(self, all_messages)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:117`
- 内部调用(去重): msg.get, str, self.mention_pattern.findall, self._add_edge_weight

**在项目中被调用的位置（按名称匹配）**：
- main.py:139 | l2_net.build_from_data | l2_net.build_from_data(processed_msgs)

### InteractionNetwork._add_edge_weight(self, source, target, weight=1)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:138`
- 内部调用(去重): self.graph.has_edge, self.graph.add_edge

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer2_nlp.py:131 | self._add_edge_weight | self._add_edge_weight(user, target, weight=5)
- src\analysis\layer2_nlp.py:134 | self._add_edge_weight | self._add_edge_weight(user, last_user, weight=1)

### InteractionNetwork.analyze_centrality(self)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:144`
- 内部调用(去重): nx.pagerank, dict, self.graph.number_of_edges, self.graph.in_degree, set, self.user_activity.keys, self.user_activity.get, pagerank.get, in_degree.get, self.user_activity.items

**在项目中被调用的位置（按名称匹配）**：
- main.py:140 | l2_net.analyze_centrality | network_stats = l2_net.analyze_centrality()

### InteractionNetwork.identify_kols(self, stats, top_n=5)

- 类型: `method`
- 位置: `src\analysis\layer2_nlp.py:165`
- 内部调用(去重): sorted, stats.items, kol_list.append

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

---

## 文件：`src\analysis\layer3_reasoning.py`

**主要依赖（imports）**：
- from __future__ import annotations
- math
- re
- from collections import defaultdict
- from typing import Dict, List, Optional
- torch
- from src.config import REPORT_CORE_USER_TOP_K, REPORT_PROFILE_LINE_TOP_K
- from src.models.llm_wrapper import QwenWrapper

**定义清单**：
- `class` `ReasoningLayer(class)` (L14-L505) - Layer3: hybrid retrieval + LLM reasoning.
- `method` `ReasoningLayer.__init__(self, max_memory_size: int=3000, recent_window: int=16, semantic_top_k: int=12, context_top_k: int=6, max_context_chars: int=1400)` (L25-L46)
- `method` `ReasoningLayer._normalize(v: Optional[torch.Tensor])` (L49-L57)
- `method` `ReasoningLayer._safe_lower(x)` (L60-L61)
- `method` `ReasoningLayer._is_first_person(text: str)` (L64-L65)
- `method` `ReasoningLayer._risk_prior(self, msg: Dict)` (L67-L70)
- `method` `ReasoningLayer._extract_keywords(self, msg: Dict)` (L72-L76)
- `method` `ReasoningLayer._calc_keyword_overlap(self, a: List[str], b: List[str])` (L78-L86)
- `method` `ReasoningLayer._build_context(self, current_msg: Dict, current_vec: Optional[torch.Tensor])` (L88-L185)
- `method` `ReasoningLayer._append_memory(self, msg: Dict, vec: Optional[torch.Tensor])` (L187-L211)
- `method` `ReasoningLayer._prune_memory(self)` (L213-L225)
- `method` `ReasoningLayer._parse_result(self, raw: str)` (L227-L246)
- `method` `ReasoningLayer._apply_hard_rules(self, current_msg: Dict, parsed: Dict[str, str])` (L248-L347)
- `method` `ReasoningLayer.quick_analyze(self, current_msg: Dict)` (L349-L373) - Low-cost inference path (no LLM call), used for token saving on low-risk messages.
- `method` `ReasoningLayer.analyze(self, current_msg: Dict)` (L375-L410)
- `method` `ReasoningLayer.generate_comprehensive_report(self, group_stats, top_kols)` (L412-L505)
- `nested_function` `ReasoningLayer.generate_comprehensive_report.format_evidence(items: List[Dict])` (L445-L454)

### ReasoningLayer(class)

- 类型: `class`
- 位置: `src\analysis\layer3_reasoning.py:14`
- Docstring(首行): Layer3: hybrid retrieval + LLM reasoning.

**在项目中被调用的位置（按名称匹配）**：
- main.py:98 | ReasoningLayer | l3 = ReasoningLayer()

### ReasoningLayer.__init__(self, max_memory_size: int=3000, recent_window: int=16, semantic_top_k: int=12, context_top_k: int=6, max_context_chars: int=1400)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:25`
- 内部调用(去重): EmbeddingEngine, QwenWrapper, defaultdict

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### ReasoningLayer._normalize(v: Optional[torch.Tensor])

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:49`
- 内部调用(去重): torch.norm, isinstance, denom.item

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:111 | self._normalize | normalized_current = self._normalize(current_vec)
- src\analysis\layer3_reasoning.py:199 | self._normalize | "norm_vec": self._normalize(vec),

### ReasoningLayer._safe_lower(x)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:60`
- 内部调用(去重): str.strip.lower, str.strip, str

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:342 | self._safe_lower | username = self._safe_lower(current_msg.get("username"))
- src\analysis\layer3_reasoning.py:354 | self._safe_lower | username = self._safe_lower(current_msg.get("username"))
- src\analysis\layer3_reasoning.py:377 | self._safe_lower | username = self._safe_lower(current_msg.get("username"))

### ReasoningLayer._is_first_person(text: str)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:64`
- 内部调用(去重): any

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:252 | self._is_first_person | first_person = self._is_first_person(text)
- src\linkage\identity_resolver.py:66 | self._is_first_person | first = self._is_first_person(t)

### ReasoningLayer._risk_prior(self, msg: Dict)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:67`
- 内部调用(去重): float, min, msg.get

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:200 | self._risk_prior | "risk_prior": self._risk_prior(msg),

### ReasoningLayer._extract_keywords(self, msg: Dict)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:72`
- 内部调用(去重): kws.extend, str.strip, msg.get, str

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:93 | self._extract_keywords | cur_keywords = self._extract_keywords(current_msg)
- src\analysis\layer3_reasoning.py:193 | self._extract_keywords | keywords = self._extract_keywords(msg)

### ReasoningLayer._calc_keyword_overlap(self, a: List[str], b: List[str])

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:78`
- 内部调用(去重): set, len

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:138 | self._calc_keyword_overlap | kw_overlap = self._calc_keyword_overlap(cur_keywords, item.get("keywords", []))

### ReasoningLayer._build_context(self, current_msg: Dict, current_vec: Optional[torch.Tensor])

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:88`
- 内部调用(去重): current_msg.get, self._extract_keywords, set, max, candidate_idx.update, self.user_index.get, self._normalize, scored.sort, defaultdict, chosen.sort, range, self.keyword_index.get, enumerate, sem_scores.sort, math.exp, self._calc_keyword_overlap, item.get, scored.append, chosen.append, int, str.replace.strip, lines.append, len, join, torch.dot.item, sem_scores.append, round, str.replace, torch.dot, str

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:383 | self._build_context | context_str = self._build_context(current_msg, current_vec)

### ReasoningLayer._append_memory(self, msg: Dict, vec: Optional[torch.Tensor])

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:187`
- 内部调用(去重): msg.get, self._extract_keywords, len, self.memory_pool.append, self.user_index.append, self._normalize, self._risk_prior, self.keyword_index.append, self._prune_memory

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:372 | self._append_memory | self._append_memory(current_msg, current_vec)
- src\analysis\layer3_reasoning.py:409 | self._append_memory | self._append_memory(current_msg, current_vec)

### ReasoningLayer._prune_memory(self)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:213`
- 内部调用(去重): defaultdict, enumerate, item.get, new_user_index.append, new_keyword_index.append

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:211 | self._prune_memory | self._prune_memory()

### ReasoningLayer._parse_result(self, raw: str)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:227`
- 内部调用(去重): str.strip, txt.lower, any, re.search, intent_match.group.strip, str, intent_match.group

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:406 | self._parse_result | parsed = self._parse_result(raw_result)

### ReasoningLayer._apply_hard_rules(self, current_msg: Dict, parsed: Dict[str, str])

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:248`
- 内部调用(去重): str, text.lower, bool, self._is_first_person, any, self._safe_lower, current_msg.get, text.strip, re.search, username.startswith, parsed.get

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:360 | self._apply_hard_rules | parsed = self._apply_hard_rules(current_msg, base)
- src\analysis\layer3_reasoning.py:407 | self._apply_hard_rules | parsed = self._apply_hard_rules(current_msg, parsed)

### ReasoningLayer.quick_analyze(self, current_msg: Dict)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:349`
- Docstring(首行): Low-cost inference path (no LLM call), used for token saving on low-risk messages.
- 内部调用(去重): str.strip, self._safe_lower, self._apply_hard_rules, self.embedder.get_embedding, self._append_memory, current_msg.get, parsed.get, str, float

**在项目中被调用的位置（按名称匹配）**：
- main.py:132 | l3.quick_analyze | msg["llm_decision"] = l3.quick_analyze(msg)

### ReasoningLayer.analyze(self, current_msg: Dict)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:375`
- 内部调用(去重): str.strip, self._safe_lower, self.embedder.get_embedding, self._build_context, strip, self.llm.generate_response, self._parse_result, self._apply_hard_rules, self._append_memory, current_msg.get, str

**在项目中被调用的位置（按名称匹配）**：
- main.py:129 | l3.analyze | msg["llm_decision"] = l3.analyze(msg)

### ReasoningLayer.generate_comprehensive_report(self, group_stats, top_kols)

- 类型: `method`
- 位置: `src\analysis\layer3_reasoning.py:412`
- 内部调用(去重): set, group_stats.get, len, join, u.get, str, role_map.get, core_profiles.append, core_users.append, lines.append, min, format_evidence, chr.join, e.get, chr

**在项目中被调用的位置（按名称匹配）**：
- main.py:162 | l3.generate_comprehensive_report | report_content = l3.generate_comprehensive_report(group_summary, final_users)

### ReasoningLayer.generate_comprehensive_report.format_evidence(items: List[Dict])

- 类型: `nested_function`
- 位置: `src\analysis\layer3_reasoning.py:445`
- 内部调用(去重): join, lines.append, e.get

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:490 | format_evidence | {format_evidence(victim_leaks)}
- src\analysis\layer3_reasoning.py:492 | format_evidence | {format_evidence(suspect_assets)}

---

## 文件：`src\config.py`

**主要依赖（imports）**：
- os
- from pathlib import Path

（本文件未检测到可记录的类/函数定义，或仅包含常量/导入。）

---

## 文件：`src\linkage\__init__.py`

**文件概述**：Identity/linkage analyzers for multi-group tracing.

（本文件未检测到可记录的类/函数定义，或仅包含常量/导入。）

---

## 文件：`src\linkage\identity_resolver.py`

**主要依赖（imports）**：
- re
- from collections import defaultdict
- from dataclasses import dataclass
- from itertools import combinations
- from typing import Dict, Iterable, List, Set, Tuple

**定义清单**：
- `class` `_DSU(class)` (L9-L27)
- `nested_function` `_DSU.__init__(self)` (L12-L13)
- `nested_function` `_DSU.find(self, x: str)` (L15-L21)
- `nested_function` `_DSU.union(self, a: str, b: str)` (L23-L27)
- `class` `IdentityResolver(class)` (L30-L426) - Cross-group, cross-account identity linkage with conservative hard-merging:
- `method` `IdentityResolver._node_id(msg: Dict)` (L48-L49)
- `method` `IdentityResolver._normalize_username(u: str)` (L52-L53)
- `method` `IdentityResolver._is_noise_username(u: str)` (L56-L58)
- `method` `IdentityResolver._is_first_person(text: str)` (L61-L62)
- `method` `IdentityResolver._is_self_claim(self, text: str, ptype: str, token: str)` (L64-L89)
- `method` `IdentityResolver._iter_pii_tokens(self, msg: Dict)` (L91-L120)
- `method` `IdentityResolver._node_is_suspect_like(node_signal: Dict)` (L123-L124)
- `method` `IdentityResolver.resolve(self, messages: List[Dict])` (L126-L272)
- `method` `IdentityResolver.build_trace_events(self, messages: List[Dict], node_to_cluster: Dict[str, str])` (L274-L297)
- `method` `IdentityResolver.attach_cluster_labels(self, messages: List[Dict], node_to_cluster: Dict[str, str])` (L299-L301)
- `method` `IdentityResolver.build_clue_chains(self, messages: List[Dict], node_to_cluster: Dict[str, str], group_summary: Dict)` (L303-L412)
- `method` `IdentityResolver.summarize(clusters: List[Dict], events: List[Dict])` (L415-L426)

### _DSU(class)

- 类型: `class`
- 位置: `src\linkage\identity_resolver.py:9`

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:127 | _DSU | dsu = _DSU()

### _DSU.__init__(self)

- 类型: `nested_function`
- 位置: `src\linkage\identity_resolver.py:12`

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### _DSU.find(self, x: str)

- 类型: `nested_function`
- 位置: `src\linkage\identity_resolver.py:15`
- 内部调用(去重): self.find

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:20 | self.find | self.parent[x] = self.find(self.parent[x])
- src\linkage\identity_resolver.py:24 | self.find | pa = self.find(a)
- src\linkage\identity_resolver.py:25 | self.find | pb = self.find(b)
- src\linkage\identity_resolver.py:207 | dsu.find | clusters_nodes[dsu.find(node)].add(node)

### _DSU.union(self, a: str, b: str)

- 类型: `nested_function`
- 位置: `src\linkage\identity_resolver.py:23`
- 内部调用(去重): self.find

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:173 | dsu.union | dsu.union(anchor, n)
- src\linkage\identity_resolver.py:191 | dsu.union | dsu.union(anchor, n)
- src\linkage\identity_resolver.py:203 | dsu.union | dsu.union(anchor, n)

### IdentityResolver(class)

- 类型: `class`
- 位置: `src\linkage\identity_resolver.py:30`
- Docstring(首行): Cross-group, cross-account identity linkage with conservative hard-merging:

**在项目中被调用的位置（按名称匹配）**：
- main.py:101 | IdentityResolver | identity_resolver = IdentityResolver()

### IdentityResolver._node_id(msg: Dict)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:48`
- 内部调用(去重): msg.get

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:136 | self._node_id | node = self._node_id(msg)
- src\linkage\identity_resolver.py:277 | self._node_id | node = self._node_id(msg)
- src\linkage\identity_resolver.py:301 | self._node_id | msg["identity_cluster"] = node_to_cluster.get(self._node_id(msg), "")
- src\linkage\identity_resolver.py:334 | self._node_id | "cluster_id": node_to_cluster.get(self._node_id(msg), ""),

### IdentityResolver._normalize_username(u: str)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:52`
- 内部调用(去重): str.strip.lower, str.strip, str

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:153 | self._normalize_username | username_nodes[self._normalize_username(user)].append(node)
- src\linkage\identity_resolver.py:197 | self._normalize_username | by_user[self._normalize_username(m["username"])].append(m["node"])

### IdentityResolver._is_noise_username(u: str)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:56`
- 内部调用(去重): str.strip.lower, u.startswith, str.strip, str

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:152 | self._is_noise_username | if not self._is_noise_username(user):

### IdentityResolver._is_first_person(text: str)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:61`
- 内部调用(去重): any

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:252 | self._is_first_person | first_person = self._is_first_person(text)
- src\linkage\identity_resolver.py:66 | self._is_first_person | first = self._is_first_person(t)

### IdentityResolver._is_self_claim(self, text: str, ptype: str, token: str)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:64`
- 内部调用(去重): str, self._is_first_person, re.search, any

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:162 | self._is_self_claim | "self_claim": self._is_self_claim(text, ptype, pval),
- src\linkage\identity_resolver.py:332 | self._is_self_claim | "self_claim": self._is_self_claim(text, ptype, pval),

### IdentityResolver._iter_pii_tokens(self, msg: Dict)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:91`
- 内部调用(去重): msg.get, pii_details.items, isinstance, str.lower, str, str.strip, out.append

**在项目中被调用的位置（按名称匹配）**：
- src\linkage\identity_resolver.py:155 | self._iter_pii_tokens | for ptype, pval in self._iter_pii_tokens(msg):
- src\linkage\identity_resolver.py:313 | self._iter_pii_tokens | for ptype, pval in self._iter_pii_tokens(msg):

### IdentityResolver._node_is_suspect_like(node_signal: Dict)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:123`
- 内部调用(去重): node_signal.get

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### IdentityResolver.resolve(self, messages: List[Dict])

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:126`
- 内部调用(去重): _DSU, defaultdict, username_nodes.values, token_mentions.items, node_meta.keys, clusters_nodes.items, clusters.sort, self._node_id, msg.get, str, self._iter_pii_tokens, list, token.split, by_user.items, clusters_nodes.add, set, clusters.append, get, self._is_noise_username, username_nodes.append, token_mentions.append, dict.fromkeys, len, dsu.union, by_user.append, members.append, aliases.add, groups.add, shared_pii.add, sorted, self._is_self_claim, dsu.find, self._normalize_username, m.get, shared_pii.get, shared_pii.items

**在项目中被调用的位置（按名称匹配）**：
- main.py:152 | identity_resolver.resolve | identity_result = identity_resolver.resolve(processed_msgs)

### IdentityResolver.build_trace_events(self, messages: List[Dict], node_to_cluster: Dict[str, str])

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:274`
- 内部调用(去重): self._node_id, node_to_cluster.get, str, self.EVENT_KEYWORDS.items, any, msg.get, events.append, int

**在项目中被调用的位置（按名称匹配）**：
- main.py:154 | identity_resolver.build_trace_events | trace_events = identity_resolver.build_trace_events(processed_msgs, identity_result["node_to_cluster"])

### IdentityResolver.attach_cluster_labels(self, messages: List[Dict], node_to_cluster: Dict[str, str])

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:299`
- 内部调用(去重): node_to_cluster.get, self._node_id

**在项目中被调用的位置（按名称匹配）**：
- main.py:153 | identity_resolver.attach_cluster_labels | identity_resolver.attach_cluster_labels(processed_msgs, identity_result["node_to_cluster"])

### IdentityResolver.build_clue_chains(self, messages: List[Dict], node_to_cluster: Dict[str, str], group_summary: Dict)

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:303`
- 内部调用(去重): set, defaultdict, token_mentions.items, chains.sort, enumerate, str, self._iter_pii_tokens, sorted, token.split, chains.append, group_summary.get, msg.get, token_mentions.append, len, list, candidate_pairs.append, int, self._is_self_claim, node_to_cluster.get, m.get, combinations, conf_rank.get, x.get, self._node_id, get

**在项目中被调用的位置（按名称匹配）**：
- main.py:156 | identity_resolver.build_clue_chains | clue_chains = identity_resolver.build_clue_chains(processed_msgs, identity_result["node_to_cluster"], group_summary)

### IdentityResolver.summarize(clusters: List[Dict], events: List[Dict])

- 类型: `method`
- 位置: `src\linkage\identity_resolver.py:415`
- 内部调用(去重): defaultdict, len, dict, e.get, c.get

**在项目中被调用的位置（按名称匹配）**：
- main.py:155 | identity_resolver.summarize | linkage_summary = identity_resolver.summarize(identity_result["clusters"], trace_events)

---

## 文件：`src\loader.py`

**主要依赖（imports）**：
- json
- from pathlib import Path

**定义清单**：
- `module_function` `load_json_data(filepath)` (L6-L25) - 加载原始JSON数据

### load_json_data(filepath)

- 类型: `module_function`
- 位置: `src\loader.py:6`
- Docstring(首行): 加载原始JSON数据
- 内部调用(去重): Path, filepath.exists, FileNotFoundError, open, json.load, isinstance, ValueError

**在项目中被调用的位置（按名称匹配）**：
- main.py:116 | load_json_data | raw_data = load_json_data(raw_path)

---

## 文件：`src\models\embedding.py`

**主要依赖（imports）**：
- os
- torch
- from src.config import EMBEDDING_MODEL_NAME, HF_MIRROR_URL, MODEL_CACHE_DIR

**定义清单**：
- `class` `EmbeddingEngine(class)` (L15-L45)
- `method` `EmbeddingEngine.__init__(self)` (L16-L30)
- `method` `EmbeddingEngine.get_embedding(self, text)` (L32-L45)

### EmbeddingEngine(class)

- 类型: `class`
- 位置: `src\models\embedding.py:15`

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:35 | EmbeddingEngine | self.embedder = EmbeddingEngine()

### EmbeddingEngine.__init__(self)

- 类型: `method`
- 位置: `src\models\embedding.py:16`
- 内部调用(去重): torch.cuda.is_available, SentenceTransformer

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### EmbeddingEngine.get_embedding(self, text)

- 类型: `method`
- 位置: `src\models\embedding.py:32`
- 内部调用(去重): torch.zeros, enumerate, torch.norm, torch.no_grad, self.model.encode, str, norm.item, ord

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:371 | self.embedder.get_embedding | current_vec = self.embedder.get_embedding(text)
- src\analysis\layer3_reasoning.py:382 | self.embedder.get_embedding | current_vec = self.embedder.get_embedding(text)

---

## 文件：`src\models\llm_wrapper.py`

**主要依赖（imports）**：
- os
- re

**定义清单**：
- `class` `QwenWrapper(class)` (L10-L75)
- `method` `QwenWrapper.__init__(self)` (L11-L21)
- `method` `QwenWrapper._fallback_response(prompt: str)` (L24-L49)
- `method` `QwenWrapper.generate_response(self, prompt: str)` (L51-L75)

### QwenWrapper(class)

- 类型: `class`
- 位置: `src\models\llm_wrapper.py:10`

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:36 | QwenWrapper | self.llm = QwenWrapper()

### QwenWrapper.__init__(self)

- 类型: `method`
- 位置: `src\models\llm_wrapper.py:11`
- 内部调用(去重): os.getenv, openai.OpenAI

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### QwenWrapper._fallback_response(prompt: str)

- 类型: `method`
- 位置: `src\models\llm_wrapper.py:24`
- 内部调用(去重): str, text.lower, any

**在项目中被调用的位置（按名称匹配）**：
- src\models\llm_wrapper.py:53 | self._fallback_response | return self._fallback_response(prompt)
- src\models\llm_wrapper.py:73 | self._fallback_response | return (content or "").strip() or self._fallback_response(prompt)
- src\models\llm_wrapper.py:75 | self._fallback_response | return self._fallback_response(prompt)

### QwenWrapper.generate_response(self, prompt: str)

- 类型: `method`
- 位置: `src\models\llm_wrapper.py:51`
- 内部调用(去重): self._fallback_response, self.client.chat.completions.create, strip

**在项目中被调用的位置（按名称匹配）**：
- src\analysis\layer3_reasoning.py:405 | self.llm.generate_response | raw_result = self.llm.generate_response(prompt)

---

## 文件：`src\profiling\group_profile.py`

**主要依赖（imports）**：
- from __future__ import annotations
- re
- from collections import defaultdict
- from typing import Dict, List, Tuple
- from src.config import SUSPECT_GROUP_HINTS, VICTIM_GROUP_HINTS

**定义清单**：
- `class` `GroupProfiler(class)` (L10-L538) - Aggregate message-level decisions into group-level role lists and evidence.
- `method` `GroupProfiler.__init__(self)` (L20-L69)
- `method` `GroupProfiler._normalize_role(role: str)` (L72-L76)
- `method` `GroupProfiler._is_system_user(username: str)` (L79-L80)
- `method` `GroupProfiler._is_noise_user(username: str)` (L83-L90)
- `method` `GroupProfiler._has_long_number(text: str)` (L93-L94)
- `method` `GroupProfiler._is_first_person_text(text: str)` (L97-L98)
- `method` `GroupProfiler._contains_any(haystack: str, needles: List[str])` (L101-L102)
- `method` `GroupProfiler._group_bias_from_message(self, message: Dict)` (L104-L116)
- `method` `GroupProfiler._guess_pii_label(self, key: str, text: str, content: str)` (L118-L140)
- `method` `GroupProfiler._update_behavior_signals(self, username: str, text: str, group_bias: str)` (L142-L193)
- `method` `GroupProfiler.update(self, message: Dict)` (L195-L264)
- `method` `GroupProfiler._user_scores(self, username: str, network_stats: Dict)` (L266-L337)
- `method` `GroupProfiler._classify_users(self, network_stats: Dict)` (L339-L444)
- `method` `GroupProfiler._format_evidence(evidence_list: List[Dict])` (L447-L453)
- `method` `GroupProfiler.get_summary_context(self, network_stats, influence_threshold=0.1)` (L455-L538)

### GroupProfiler(class)

- 类型: `class`
- 位置: `src\profiling\group_profile.py:10`
- Docstring(首行): Aggregate message-level decisions into group-level role lists and evidence.

**在项目中被调用的位置（按名称匹配）**：
- main.py:100 | GroupProfiler | group_p = GroupProfiler()

### GroupProfiler.__init__(self)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:20`
- 内部调用(去重): set, defaultdict

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### GroupProfiler._normalize_role(role: str)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:72`
- 内部调用(去重): str.strip.lower, str.strip, str

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:208 | self._normalize_role | role = self._normalize_role(llm_res.get("role", "other"))

### GroupProfiler._is_system_user(username: str)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:79`
- 内部调用(去重): str.strip.lower, str.strip, str

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:201 | self._is_system_user | if not username or self._is_system_user(username):

### GroupProfiler._is_noise_user(username: str)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:83`
- 内部调用(去重): str.strip.lower, u.startswith, str.strip, str

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:380 | self._is_noise_user | if is_water or self._is_noise_user(username):

### GroupProfiler._has_long_number(text: str)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:93`
- 内部调用(去重): bool, re.search

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:164 | self._has_long_number | if self._has_long_number(t) and any(x in t for x in ["卡号", "户名", "银行", "下发", "新卡", "换卡"]):
- src\profiling\group_profile.py:186 | self._has_long_number | or (self._has_long_number(t) and any(x in t for x in ["\u5361\u53f7", "\u6237\u540d", "\u8eab\u4efd\u8bc1", "\u7535\u8bdd", "\u624b\u673a\u53f7", "\u4f4f\u5740", "\u5730\u5740"]))

### GroupProfiler._is_first_person_text(text: str)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:97`
- 内部调用(去重): any

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:157 | self._is_first_person_text | first_person = self._is_first_person_text(t)
- src\profiling\group_profile.py:218 | self._is_first_person_text | first_person = self._is_first_person_text(text)

### GroupProfiler._contains_any(haystack: str, needles: List[str])

- 类型: `method`
- 位置: `src\profiling\group_profile.py:101`
- 内部调用(去重): any

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:109 | self._contains_any | suspect_hit = self._contains_any(hay, [str(x).lower() for x in SUSPECT_GROUP_HINTS])
- src\profiling\group_profile.py:110 | self._contains_any | victim_hit = self._contains_any(hay, [str(x).lower() for x in VICTIM_GROUP_HINTS])

### GroupProfiler._group_bias_from_message(self, message: Dict)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:104`
- 内部调用(去重): str, lower, self._contains_any, message.get, str.lower

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:211 | self._group_bias_from_message | group_bias = self._group_bias_from_message(message)

### GroupProfiler._guess_pii_label(self, key: str, text: str, content: str)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:118`
- 内部调用(去重): str.lower, str, any, len

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:238 | self._guess_pii_label | label = self._guess_pii_label(key, text, content)

### GroupProfiler._update_behavior_signals(self, username: str, text: str, group_bias: str)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:142`
- 内部调用(去重): str.strip, self._is_first_person_text, any, self._has_long_number, str

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:212 | self._update_behavior_signals | self._update_behavior_signals(username, text, group_bias)

### GroupProfiler.update(self, message: Dict)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:195`
- 内部调用(去重): message.get, str, self._normalize_role, self._group_bias_from_message, self._update_behavior_signals, self._is_first_person_text, any, pii_details.items, self._is_system_user, isinstance, llm_res.get, rec.get, str.strip, self._guess_pii_label, self._seen_evidence.add, self.stats.append

**在项目中被调用的位置（按名称匹配）**：
- main.py:126 | msg.update | msg.update(l2_nlp.process(msg))
- main.py:134 | group_p.update | group_p.update(msg)
- src\analysis\layer1_regex.py:137 | message.update | message.update(
- src\analysis\layer3_reasoning.py:99 | candidate_idx.update | candidate_idx.update(range(start, len(self.memory_pool)))
- src\analysis\layer3_reasoning.py:103 | candidate_idx.update | candidate_idx.update(user_hist[-4:])
- src\analysis\layer3_reasoning.py:108 | candidate_idx.update | candidate_idx.update(linked[-6:])
- src\analysis\layer3_reasoning.py:121 | candidate_idx.update | candidate_idx.update(idx for idx, _ in sem_scores[: self.semantic_top_k])
- src\profiling\user_profile.py:63 | self.profiles.update | self.profiles[u]["keywords"].update(msg.get("nlp_keywords", []))

### GroupProfiler._user_scores(self, username: str, network_stats: Dict)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:266`
- 内部调用(去重): self.user_stats.get, rec.get, votes.get, max, float, network_stats.get.get, int, network_stats.get

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:348 | self._user_scores | victim_score, scammer_score, irrelevant_score = self._user_scores(username, network_stats)

### GroupProfiler._classify_users(self, network_stats: Dict)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:339`
- 内部调用(去重): set, self.user_stats.keys, self._user_scores, rec.get, max, cleaned_victims.add, cleaned_suspects.add, self._is_noise_user, irrelevant.add, suspects.add, int, victims.add

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:456 | self._classify_users | victims, suspects, irrelevant, victim_scores, scammer_scores = self._classify_users(network_stats)

### GroupProfiler._format_evidence(evidence_list: List[Dict])

- 类型: `method`
- 位置: `src\profiling\group_profile.py:447`
- 内部调用(去重): join, lines.append

**在项目中被调用的位置（按名称匹配）**：
- src\profiling\group_profile.py:533 | self._format_evidence | "suspect_assets_str": self._format_evidence(suspect_assets),
- src\profiling\group_profile.py:534 | self._format_evidence | "victim_leaks_str": self._format_evidence(victim_leaks),

### GroupProfiler.get_summary_context(self, network_stats, influence_threshold=0.1)

- 类型: `method`
- 位置: `src\profiling\group_profile.py:455`
- 内部调用(去重): self._classify_users, sorted, self.user_stats.get, list, float, self._format_evidence, suspect_assets.append, victim_leaks.append, final_victims.append, rec.get, network_stats.get.get, victim_scores.get, self.user_stats.get.get, scammer_scores.get, network_stats.get

**在项目中被调用的位置（按名称匹配）**：
- main.py:149 | group_p.get_summary_context | group_summary = group_p.get_summary_context(network_stats)

---

## 文件：`src\profiling\user_profile.py`

**主要依赖（imports）**：
- from collections import Counter

**定义清单**：
- `class` `UserProfiler(class)` (L42-L85)
- `method` `UserProfiler.__init__(self)` (L43-L44)
- `method` `UserProfiler.aggregate(self, msg_list)` (L46-L72)
- `method` `UserProfiler.finalize(self)` (L74-L85)

### UserProfiler(class)

- 类型: `class`
- 位置: `src\profiling\user_profile.py:42`

**在项目中被调用的位置（按名称匹配）**：
- main.py:99 | UserProfiler | user_p = UserProfiler()

### UserProfiler.__init__(self)

- 类型: `method`
- 位置: `src\profiling\user_profile.py:43`

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### UserProfiler.aggregate(self, msg_list)

- 类型: `method`
- 位置: `src\profiling\user_profile.py:46`
- 内部调用(去重): msg.get, self.profiles.update, isinstance, self.profiles.append, set, Counter, decision.get

**在项目中被调用的位置（按名称匹配）**：
- main.py:147 | user_p.aggregate | user_p.aggregate(processed_msgs)

### UserProfiler.finalize(self)

- 类型: `method`
- 位置: `src\profiling\user_profile.py:74`
- 内部调用(去重): sorted, self.profiles.items, summary.append, round, data.most_common

**在项目中被调用的位置（按名称匹配）**：
- main.py:148 | user_p.finalize | final_users = user_p.finalize()

---

## 文件：`src\storage\__init__.py`

**文件概述**：Storage adapters for local multi-database linkage.

（本文件未检测到可记录的类/函数定义，或仅包含常量/导入。）

---

## 文件：`src\storage\multi_db.py`

**主要依赖（imports）**：
- json
- re
- sqlite3
- from datetime import datetime
- from pathlib import Path
- from typing import Dict, Iterable, List, Tuple

**定义清单**：
- `class` `MultiDBManager(class)` (L9-L321) - Global + per-account sqlite storage.
- `method` `MultiDBManager.__init__(self, db_root: Path)` (L17-L28)
- `method` `MultiDBManager._init_global_schema(self)` (L30-L106)
- `method` `MultiDBManager._safe_filename(name: str)` (L109-L113)
- `method` `MultiDBManager._get_account_conn(self, username: str)` (L115-L157)
- `method` `MultiDBManager._iter_pii_items(message: Dict)` (L160-L173)
- `method` `MultiDBManager.store_message(self, message: Dict)` (L175-L254)
- `method` `MultiDBManager.store_identity_clusters(self, clusters: List[Dict])` (L256-L285)
- `method` `MultiDBManager.store_trace_events(self, events: List[Dict])` (L287-L306)
- `method` `MultiDBManager.close(self)` (L308-L321)

### MultiDBManager(class)

- 类型: `class`
- 位置: `src\storage\multi_db.py:9`
- Docstring(首行): Global + per-account sqlite storage.

**在项目中被调用的位置（按名称匹配）**：
- main.py:102 | MultiDBManager | db = MultiDBManager(DATA_DB_DIR)

### MultiDBManager.__init__(self, db_root: Path)

- 类型: `method`
- 位置: `src\storage\multi_db.py:17`
- 内部调用(去重): Path, self.db_root.mkdir, self.account_root.mkdir, sqlite3.connect, self._init_global_schema

**在项目中被调用的位置（按名称匹配）**：未发现（可能未被调用/仅运行时动态调用/或为同名遮蔽导致静态检索不到）

### MultiDBManager._init_global_schema(self)

- 类型: `method`
- 位置: `src\storage\multi_db.py:30`
- 内部调用(去重): self.global_conn.cursor, cur.execute, self.global_conn.commit

**在项目中被调用的位置（按名称匹配）**：
- src\storage\multi_db.py:28 | self._init_global_schema | self._init_global_schema()

### MultiDBManager._safe_filename(name: str)

- 类型: `method`
- 位置: `src\storage\multi_db.py:109`
- 内部调用(去重): str.strip, re.sub, str

**在项目中被调用的位置（按名称匹配）**：
- src\storage\multi_db.py:116 | self._safe_filename | key = self._safe_filename(username)

### MultiDBManager._get_account_conn(self, username: str)

- 类型: `method`
- 位置: `src\storage\multi_db.py:115`
- 内部调用(去重): self._safe_filename, sqlite3.connect, conn.cursor, cur.execute, conn.commit

**在项目中被调用的位置（按名称匹配）**：
- src\storage\multi_db.py:223 | self._get_account_conn | acc_conn = self._get_account_conn(message.get("username", "unknown"))

### MultiDBManager._iter_pii_items(message: Dict)

- 类型: `method`
- 位置: `src\storage\multi_db.py:160`
- 内部调用(去重): message.get, pii_details.items, isinstance, str.strip, rows.append, str

**在项目中被调用的位置（按名称匹配）**：
- src\storage\multi_db.py:205 | self._iter_pii_items | for pii_type, pii_value in self._iter_pii_items(message):
- src\storage\multi_db.py:247 | self._iter_pii_items | for pii_type, pii_value in self._iter_pii_items(message):

### MultiDBManager.store_message(self, message: Dict)

- 类型: `method`
- 位置: `src\storage\multi_db.py:175`
- 内部调用(去重): self.global_conn.cursor, cur.execute, int, self._iter_pii_items, self.global_conn.commit, self._get_account_conn, acc_conn.cursor, acc_cur.execute, acc_conn.commit, isinstance, message.get, float, llm.get, datetime.now.isoformat, datetime.now

**在项目中被调用的位置（按名称匹配）**：
- main.py:136 | db.store_message | db.store_message(msg)

### MultiDBManager.store_identity_clusters(self, clusters: List[Dict])

- 类型: `method`
- 位置: `src\storage\multi_db.py:256`
- 内部调用(去重): self.global_conn.cursor, cur.execute, datetime.now.isoformat, self.global_conn.commit, c.get, datetime.now, json.dumps, m.get

**在项目中被调用的位置（按名称匹配）**：
- main.py:158 | db.store_identity_clusters | db.store_identity_clusters(identity_result["clusters"])

### MultiDBManager.store_trace_events(self, events: List[Dict])

- 类型: `method`
- 位置: `src\storage\multi_db.py:287`
- 内部调用(去重): self.global_conn.cursor, cur.execute, self.global_conn.commit, e.get, int

**在项目中被调用的位置（按名称匹配）**：
- main.py:159 | db.store_trace_events | db.store_trace_events(trace_events)

### MultiDBManager.close(self)

- 类型: `method`
- 位置: `src\storage\multi_db.py:308`
- 内部调用(去重): self.account_conns.values, self.account_conns.clear, self.global_conn.commit, self.global_conn.close, conn.commit, conn.close

**在项目中被调用的位置（按名称匹配）**：
- main.py:107 | db.close | db.close()
- main.py:187 | db.close | db.close()
- src\storage\multi_db.py:311 | self.global_conn.close | self.global_conn.close()
- src\storage\multi_db.py:318 | conn.close | conn.close()

---

## 文件：`src\utils.py`

**主要依赖（imports）**：
- json
- logging
- from datetime import datetime
- from pathlib import Path

**定义清单**：
- `module_function` `setup_logger()` (L8-L15) - 配置日志显示
- `module_function` `save_json(data, filename, folder_path)` (L18-L27) - 保存处理后的数据到JSON

### setup_logger()

- 类型: `module_function`
- 位置: `src\utils.py:8`
- Docstring(首行): 配置日志显示
- 内部调用(去重): logging.basicConfig, logging.getLogger

**在项目中被调用的位置（按名称匹配）**：
- main.py:90 | setup_logger | logger = setup_logger()

### save_json(data, filename, folder_path)

- 类型: `module_function`
- 位置: `src\utils.py:18`
- Docstring(首行): 保存处理后的数据到JSON
- 内部调用(去重): datetime.now.strftime, Path, open, json.dump, datetime.now

**在项目中被调用的位置（按名称匹配）**：
- main.py:164 | save_json | processed_path = save_json(processed_msgs, Path(f"multi_groups_full_{timestamp}"), DATA_PROC_DIR)
- main.py:165 | save_json | users_path = save_json(network_stats, Path(f"multi_groups_users_{timestamp}"), DATA_PROC_DIR)
- main.py:166 | save_json | clusters_path = save_json(identity_result["clusters"], Path(f"identity_clusters_{timestamp}"), DATA_PROC_DIR)
- main.py:167 | save_json | traces_path = save_json(trace_events, Path(f"cross_group_traces_{timestamp}"), DATA_PROC_DIR)
- main.py:168 | save_json | linkage_path = save_json(linkage_summary, Path(f"linkage_summary_{timestamp}"), DATA_PROC_DIR)
- main.py:169 | save_json | clue_chain_path = save_json(clue_chains, Path(f"clue_chains_{timestamp}"), DATA_PROC_DIR)

---

## 文件：`test.py`

**主要依赖（imports）**：
- json

（本文件未检测到可记录的类/函数定义，或仅包含常量/导入。）

