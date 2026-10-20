| 证据编号 | 来源文件路径 | 引用位置/字段 | 原文/数值摘录 | 该证据支持的要点(一句话) | 因果链环节 |
|---|---|---|---|---|---|
| 1 | [Isoform分析结果] | 序列相似性 | 92.5% | FLM_wt 与 FLM_alt 具有高度序列相似性，符合同一基因剪接变体的典型特征 | 剪接差异 |
| 2 | [Isoform分析结果] | 总差异数 | 1 | 两序列仅存在1个氨基酸差异，强烈提示由选择性剪接导致的微小序列变化 | 剪接差异 |
| 3 | [FASTA 序列信息] | 序列比对（末尾） | FLM_wt: ...LDRSAVQAGETSMF<br>FLM_alt: ...LFLVL | 差异位于C端，FLM_wt 含较长保守尾部（"SAVQAGETSMF"），而 FLM_alt 截短为 "LFLVL"，暗示可能因外显子跳跃或提前终止导致 | 序列差异 |
| 4 | [Annotation Full JSON] | domains → locations (PF00319, SRF-TF) | FLM_wt domain end: 55; full length: 94<br>(推断：C端39aa在结构域外) | 差异区域位于关键DNA结合/二聚化结构域（SRF-TF）之外，但可能影响蛋白稳定性或互作界面 | 序列差异→结构影响 |
| 5 | [Annotation Full JSON] | goXRefs under PTHR48019 | GO:0006357 (regulation of transcription by RNA pol II)<br>GO:0000981 (DNA-binding TF activity) | 两个变体均注释到相同核心转录调控功能，但C端截短可能削弱转录激活能力（因激活域常位于C端） | 结构差异→功能影响 |
| 6 | [Isoform分析结果] | 长度差异 | 9 氨基酸 | FLM_alt 比 FLM_wt 短9个氨基酸，与FASTA观察一致，符合选择性剪接导致的阅读框内缺失 | 剪接差异 |