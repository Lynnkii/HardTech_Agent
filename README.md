# HardTech_Agent
这是一个基于 AI 的自动化投研助手，专门为**硬科技领域**（芯片、半导体、机器人、新能源等）的分析师和投资者设计。它可以自动检索全网情报，评估来源可靠性，并生成简评报告。

## 核心功能

- **双引擎深度搜索**：结合 Tavily 高级搜索 API，同时抓取行业新闻与券商研报维度信息。
- **来源可信度评分**：内置评分算法，自动识别政府官网、核心财经媒体（如财新、上海证券报）并提高权重。
- **大模型深度分析**：调用 DeepSeek (V3/R1) 模型，自动分析技术壁垒、TAM、国产替代潜力等关键投研指标。
- **多格式导出**：支持一键生成并下载 **Markdown**、**纯文本 (TXT)** 以及 **PDF** 格式的投研报告。
- **风险提示卡片**：自动提取研报中的核心风险点，以醒目的可视化卡片形式呈现。

## 技术栈

- **UI 框架**: [Streamlit](https://streamlit.io/)
- **LLM**: [DeepSeek API](https://platform.deepseek.com/)
- **搜索**: [Tavily AI](https://tavily.com/)
- **PDF 生成**: ReportLab

## 快速开始

### 1. 获取 API Keys
你需要准备以下两个 API Key：
- [DeepSeek API Key](https://platform.deepseek.com/)
- [Tavily API Key](https://tavily.com/)

### 2. 本地运行
1. 克隆仓库：
   ```bash
   git clone https://github.com/你的用户名/你的仓库名.git
   cd 你的仓库名
2. 安装依赖：建议使用 Python 3.9+ 环境：
   ```bash
   pip install -r requirements.txt
3. 运行应用：
   ```bash
   streamlit run app.py
