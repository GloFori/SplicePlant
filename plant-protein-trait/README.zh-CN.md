# 植物单蛋白功能 → 性状关联 Skill

这套项目同时支持 **AI Skill 调用**和**服务器命令行运行**。审稿人不需要使用 Codex，也可以运行全部脚本。

当前版本为 `0.1.0`。已提供程序测试与合成数据试运行，**没有把任何真实生物学案例写成已经复现成功**。拟南芥示例等你后续补充；包内不含此前棉花序列、群体关联信息或服务器私有路径。

## 先检查能不能运行

进入本目录，在 Python 3.10+ 环境执行：

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py --output runs/smoke
```

核心代码只用 Python 标准库，不再强制要求 Bio、requests、openai、sklearn。上面三步不联网、不调用模型、不下载数据库。`runs/smoke` 必须是新目录。

## 真实分析顺序

1. 准备**一条蛋白质 FASTA**，用 `check_sequence.py` 检查。CDS、转录本与 GFF3 可以作为独立来源核验材料，但不要当作蛋白输入。
2. 通过 `4_cotton.py` 收集 BLAST、InterPro/Pfam、GO 证据；这个历史文件名也适用于拟南芥等其他植物。
3. `new_llm.py prepare` 直接读取实际 `annotation_full.json`，冻结证据，生成四组提示词。
4. `new_llm.py run` 使用你配置的 Ollama、DashScope 或兼容接口执行四次独立请求。
5. `new_llm.py freeze` 只有在四组都成功后才能冻结结果。
6. `summarize.py` 输出 A/B 对照和证据表。独立研究背景只能在这一步之后加入。

具体命令见 [英文 README](README.md) 和 [执行指南](references/workflow.md)。原脚本的问题及本版改动见 [审计记录](docs/AUDIT.md)。这是一次明确记录的单蛋白重构，不是与旧 CLI 完全兼容的直接替换。

## A 与 B 的区别

- **Targeted**：已知关注某个性状，检查蛋白功能证据是否支持合理联系。
- **Blind**：不向模型提供目标性状或关联研究背景，观察证据自然指向哪些过程与性状。
- 每条分别执行 verify 与 explore；后者用于产生待验证的机制假设。
- A 支持而 B 未出现相关方向，只能写成 targeted functional compatibility/support，不能称为独立性状预测。
- 程序不会因为缺少证据而制造第二个蛋白、结构分数、通路或性状结论。

## 给审稿人什么

需要归档代码 commit/tag、FASTA、原始注释响应、数据库版本、完整提示词、实际模型 ID、四组原始输出和 SHA256 清单。只发布 Skill 或截图，不能替代这些运行产物。

若模型未配置，程序可以完成提示词准备，但不能声称已经完成 LLM 分析。若线上数据库或模型更新，重新运行不保证逐字相同；应以归档的证据和响应作为复核依据。

## 上传你的现有仓库

推荐把完整目录放在仓库的 `skills/plant-protein-trait/`，不要覆盖仓库已有 README。把包内 `.github/workflows/ci.yml` 另行复制到仓库根的 `.github/workflows/plant-protein-trait.yml` 才会自动运行 CI。详见 [上传说明](docs/GITHUB_HANDOFF.md)。

真实拟南芥示例后补到 `examples/arabidopsis/`，按 [示例资料要求](examples/README.md) 记录 accession、来源版本、校验值和原始结果。包没有替你选择基因，也没有填入假的拟南芥注释。

项目许可证沿用你最终仓库的实际授权决定，目前没有擅自替你新增 MIT 等许可证；发表前补齐真实作者、引用和授权信息。
