# myara-metac-bot

Metaculus FutureEval / MiniBench 预测 bot —— fork 自 [Metaculus/metac-bot-template](https://github.com/Metaculus/metac-bot-template)（commit `8c898aa`）。

完整文档在项目根目录（`RESEARCH.md` / `ARCHITECTURE.md` / `SETUP.md` / `TEST_REPORT.md` / `README.md`）。

## 与官方模板的差异

1. **默认 dry-run**：`python main.py` 不带参数 = 只跑 `bot-testing-area` 沙盒取题，不提交任何预测。提交必须加 `--publish`。
2. **正式赛双保险**：`--mode tournament --publish` 需同时满足环境变量 `ALLOW_PUBLISH=yes`（CI 里写在 workflow 文件中，等于"改文件=人工授权"）。
3. **模型零代码切换**：`DEFAULT_LLM_MODEL` / `PARSER_LLM_MODEL` / `RESEARCHER_MODE` 环境变量（如 `openrouter/openai/gpt-4o-mini`、`no_research`、`asknews/news-summaries`）。
4. **运行留痕**：每轮写 `../runs/run-*.jsonl`（模式/每题结果/耗时/用量/退出码），本地跑生效；CI 跑看 Actions 日志。
5. **一题一预测**：tournament/minibench 模式强制 `skip_previously_forecasted_questions=True`。

## 本地运行

```bash
cd bot
cp .env.template .env   # 填 METACULUS_TOKEN 与至少一个 LLM key
uv sync --locked

uv run python main.py                      # 沙盒 dry-run（默认）
uv run python main.py --mode test_questions --publish   # 沙盒真实提交（需你手动触发=授权）
uv run python offline_test.py              # 完全离线冒烟测试（无需任何 key）
```

## GitHub Actions（正式部署）

- `Test Bot` workflow：手动触发，向 bot-testing-area 提交。
- `Forecast on new AI tournament questions`：每 20 分钟；**启用并确认 workflow 内 `ALLOW_PUBLISH: "yes"` 后才会向正式赛提交**。暂停 = Disable workflow。

## 密钥

只放 GitHub Secrets 或本地 `.env`（已 gitignore）。绝不写入代码或日志。
