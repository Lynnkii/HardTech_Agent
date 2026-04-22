import io
import os
import socket
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from tavily import TavilyClient


st.set_page_config(
    page_title="硬科技投研分析 Agent",
    page_icon=":bar_chart:",
    layout="wide",
)


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f5f7fa 0%, #eef2f7 100%);
        }
        .block-container {
            max-width: 980px;
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }
        .main-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1rem 1rem 0.6rem 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.9rem;
        }
        .meta-note {
            color: #64748b;
            font-size: 0.86rem;
        }
        .risk-card {
            border-left: 6px solid #b91c1c;
            background: #fff1f2;
            color: #7f1d1d;
            border-radius: 8px;
            padding: 14px 16px;
            margin: 10px 0 16px 0;
        }
        .risk-card-title {
            font-weight: 700;
            margin-bottom: 6px;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def search_hardtech_updates(company_name: str, tavily_api_key: str) -> dict[str, Any]:
    client = TavilyClient(api_key=tavily_api_key)
    query_general = (
        f"{company_name} 最新融资 技术专利突破 核心高管变动 "
        "硬科技 芯片 机器人 新能源 半导体"
    )
    query_reports = (
        f"{company_name} 券商研报 行业研究 公司深度报告 "
        "投资逻辑 市场空间 TAM 国产替代"
    )
    general_result = client.search(
        query=query_general,
        search_depth="advanced",
        max_results=12,
        include_answer=True,
        include_raw_content=False,
    )
    report_result = client.search(
        query=query_reports,
        search_depth="advanced",
        max_results=12,
        include_answer=True,
        include_raw_content=False,
    )

    merged_results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in (general_result.get("results", []) or []) + (report_result.get("results", []) or []):
        key = f"{str(item.get('url', '')).strip()}|{str(item.get('title', '')).strip()}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_results.append(item)

    merged_answer = "；".join(
        [x for x in [general_result.get("answer", ""), report_result.get("answer", "")] if str(x).strip()]
    )
    return {"answer": merged_answer, "results": merged_results}


def _parse_published_date(value: Any) -> datetime | None:
    if not value:
        return None
    date_str = str(value).strip()
    if not date_str:
        return None
    if date_str.endswith("Z"):
        date_str = date_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _filter_results_by_window(results: list[dict[str, Any]], days: int | None) -> list[dict[str, Any]]:
    if days is None:
        return results
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered: list[dict[str, Any]] = []
    for item in results:
        published_dt = _parse_published_date(item.get("published_date"))
        if published_dt and published_dt >= cutoff:
            filtered.append(item)
    return filtered


def _score_source_credibility(url: str, content: str, published_date: Any) -> int:
    domain = (urlparse(url).netloc or "").lower()
    score = 45
    high_conf_domains = (
        "gov.cn",
        "cs.com.cn",
        "stcn.com",
        "cninfo.com.cn",
        "reuters.com",
        "bloomberg.com",
        "wsj.com",
    )
    medium_conf_domains = (
        "36kr.com",
        "caixin.com",
        "yicai.com",
        "thepaper.cn",
        "techcrunch.com",
    )
    if any(domain.endswith(x) for x in high_conf_domains):
        score += 35
    elif any(domain.endswith(x) for x in medium_conf_domains):
        score += 20
    else:
        score += 10

    if url.startswith("https://"):
        score += 5
    if len(content.strip()) > 120:
        score += 8

    published_dt = _parse_published_date(published_date)
    if published_dt:
        day_diff = (datetime.now(timezone.utc) - published_dt).days
        if day_diff <= 7:
            score += 7
        elif day_diff <= 30:
            score += 4
    return max(0, min(100, score))


def _format_search_context(company_name: str, search_result: dict[str, Any]) -> str:
    answer = str(search_result.get("answer", "")).strip()
    results = search_result.get("results", []) or []
    lines = [f"目标公司：{company_name}"]
    if answer:
        lines.append(f"Tavily 概要：{answer}")

    lines.append("检索证据：")
    for idx, item in enumerate(results[:8], start=1):
        title = str(item.get("title", "无标题")).strip()
        content = str(item.get("content", "")).strip()
        url = str(item.get("url", "")).strip()
        score = int(item.get("credibility_score", 0))
        published = str(item.get("published_date", "")).strip()
        lines.append(f"{idx}. 标题：{title}")
        lines.append(f"   来源可信度评分：{score}/100")
        if published:
            lines.append(f"   发布时间：{published}")
        if content:
            lines.append(f"   摘要：{content}")
        if url:
            lines.append(f"   链接：{url}")
    return "\n".join(lines)


def generate_investment_review(
    company_name: str, deepseek_api_key: str, context_text: str
) -> str:
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    prompt = f"""
你是“资深 CICC 硬科技分析师”，请输出机构风格的中文研究简报，严谨、克制、证据驱动。

输出格式必须是 Markdown，并且必须包含以下一级/二级标题：
# {company_name} 投资价值简评
## 一、技术壁垒
## 二、市场规模（TAM）
## 三、核心竞品
## 四、国产替代潜力
## 五、主要风险
## 六、综合判断（仅供参考）

写作要求：
1) 关键术语使用加粗，例如 **技术护城河**、**量产能力**、**验证周期**。
2) 每个维度给出 2-4 条要点，能量化则尽量量化。
3) 如证据不足，明确标注“公开信息有限”。
4) 风险部分请给出至少 3 条，并覆盖技术、商业化、政策/合规维度。
5) 结论使用“积极/中性/谨慎”三档之一，并给出一句核心理由。
6) 只输出最终报告正文，不要输出你的思考过程、推理过程、前置说明或“根据以上信息”等套话。

以下为检索到的公司动态信息：
{context_text}
"""
    candidate_models = ["deepseek-reasoner", "deepseek-chat"]
    last_error: Exception | None = None
    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text = response.choices[0].message.content if response.choices else ""
            if text.strip():
                return text.strip()
        except Exception as exc:  # pragma: no cover - runtime API guard
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("DeepSeek 未返回有效内容。")


def _build_markdown_report(company_name: str, review: str) -> str:
    lines = [
        f"# {company_name} 投资价值简评报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 投资价值简评",
        "",
        review,
        "",
        "---",
        "仅供研究演示，不构成投资建议。",
    ]
    return "\n".join(lines)


def _build_pdf_report(company_name: str, report_text: str) -> bytes:
    buffer = io.BytesIO()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle(f"{company_name} 投资价值简评报告")
    pdf.setFont("STSong-Light", 11)
    _, page_height = A4
    y = page_height - 46
    for raw_line in report_text.splitlines():
        line = raw_line if raw_line.strip() else " "
        for i in range(0, len(line), 40):
            piece = line[i : i + 40]
            pdf.drawString(40, y, piece)
            y -= 18
            if y <= 52:
                pdf.showPage()
                pdf.setFont("STSong-Light", 11)
                y = page_height - 46
    pdf.save()
    buffer.seek(0)
    return buffer.read()


def _extract_risk_points(review: str) -> list[str]:
    lines = review.splitlines()
    risk_points: list[str] = []
    in_risk_section = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if "主要风险" in line or "风险" in line and line.startswith("##"):
            in_risk_section = True
            continue
        if in_risk_section and line.startswith("##"):
            break
        if in_risk_section and (line.startswith("-") or line[0].isdigit()):
            cleaned = line.lstrip("-0123456789. ").strip()
            if cleaned:
                risk_points.append(cleaned)
        if len(risk_points) >= 4:
            break
    return risk_points


def _render_risk_card(review: str) -> None:
    risk_points = _extract_risk_points(review)
    if not risk_points:
        return
    card_content = "<br/>".join([f"- {point}" for point in risk_points])
    st.markdown(
        (
            '<div class="risk-card">'
            '<div class="risk-card-title">风险提示卡片</div>'
            f"{card_content}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _friendly_error_message(exc: Exception, provider: str) -> str:
    detail = str(exc).lower()
    if "401" in detail or "403" in detail or "invalid" in detail or "unauthorized" in detail:
        return f"{provider} API Key 可能无效，请检查后重试。"
    if "quota" in detail or "rate limit" in detail or "429" in detail:
        return f"{provider} 调用配额不足或请求过于频繁，请稍后重试。"
    if "not found" in detail or "model" in detail and "supported" in detail:
        return f"{provider} 模型暂不可用，请稍后重试或更换可用模型。"
    if "timeout" in detail or "timed out" in detail or isinstance(exc, TimeoutError):
        return f"{provider} 请求超时，请稍后重试。"
    if "connection" in detail or "network" in detail or isinstance(exc, (OSError, socket.timeout)):
        return f"无法连接 {provider} 服务，请检查网络或稍后再试。"
    return f"{provider} 服务暂时不可用，请稍后重试。"


def _example_report(company_name: str = "示例公司") -> str:
    return f"""# {company_name} 投资价值简评

## 一、技术壁垒
- **核心工艺能力**：具备多代产品迭代路径，研发投入强度较高。
- **量产可行性**：已完成若干场景验证，但规模化节奏仍需跟踪。

## 二、市场规模（TAM）
- **需求空间**：下游应用扩张明确，潜在 TAM 较大。
- **渗透进程**：成本与性能优化将决定渗透率提升速度。

## 三、核心竞品
- **国内竞品**：同赛道公司持续加码，价格与交付能力竞争加剧。
- **海外竞品**：技术成熟度较高，但本地化服务与合规响应存在差异。

## 四、国产替代潜力
- **自主可控**：关键环节具备替代逻辑，但部分上游环节仍受约束。
- **政策弹性**：政策与产业基金支持有望强化商业化推进。

## 五、主要风险
- 技术迭代慢于预期导致竞争优势削弱。
- 客户导入与订单兑现节奏存在不确定性。
- 政策、监管与地缘因素可能影响供应链稳定。

## 六、综合判断（仅供参考）
**中性**。公司具备一定技术积累和替代潜力，但短期兑现仍需验证。
"""


def main() -> None:
    _render_styles()

    st.sidebar.title("配置")
    deepseek_api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="请粘贴 DeepSeek API Key",
    )
    tavily_api_key = st.sidebar.text_input(
        "Tavily API Key",
        type="password",
        placeholder="请粘贴 Tavily API Key",
    )
    with st.sidebar.expander("可选：Wind 接口（预留）", expanded=False):
        wind_api_endpoint = st.text_input(
            "Wind API Endpoint（可选）",
            placeholder="例如：https://your-wind-gateway/api",
        )
        wind_api_token = st.text_input(
            "Wind Token（可选）",
            type="password",
            placeholder="可留空",
        )
        if wind_api_endpoint.strip() and wind_api_token.strip():
            st.caption("已填写 Wind 参数（当前为预留入口，后续可接真实数据）。")
    if deepseek_api_key.strip() and tavily_api_key.strip():
        st.sidebar.success("API Key 已就绪")
    else:
        st.sidebar.warning("请完整填写 DeepSeek 与 Tavily Key 后再搜索。")

    st.title("硬科技投研分析 Agent")
    st.markdown(
        '<p class="meta-note">建议优先输入公司全称或“公司名+业务关键词”以获得更稳定结果。</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    company_name = st.text_input(
        "请输入硬科技公司名",
        placeholder="例如：地平线芯片",
    )
    time_window_option = st.selectbox(
        "时间窗口筛选",
        options=["近7天", "近30天", "不限"],
        index=0,
    )
    search_enabled = bool(deepseek_api_key.strip() and tavily_api_key.strip() and company_name.strip())
    action_col_1, action_col_2 = st.columns([3, 1.2])
    with action_col_1:
        search_clicked = st.button(
            "搜索并生成投资价值简评",
            use_container_width=True,
            disabled=not search_enabled,
        )
    with action_col_2:
        show_example = st.button("查看示例报告", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if not search_enabled:
        st.info("填写公司名、DeepSeek Key 与 Tavily Key 后可启动分析。")

    if show_example:
        sample_report = _example_report(company_name.strip() or "示例公司")
        st.subheader("示例报告")
        st.markdown(sample_report)
        st.download_button(
            label="下载示例 Markdown",
            data=sample_report.encode("utf-8"),
            file_name=f"{(company_name.strip() or '示例公司')}_示例报告.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if search_clicked:
        with st.status("Agent 正在执行任务...", expanded=True) as status:
            status.write("正在检索全网硬科技情报...")
            try:
                search_result = search_hardtech_updates(company_name.strip(), tavily_api_key.strip())
            except Exception as exc:
                status.update(label="检索失败", state="error")
                st.error(_friendly_error_message(exc, "Tavily"))
                return
            status.write("正在加强研报维度检索（券商/行业报告）...")

            results = search_result.get("results", []) or []
            days_map = {"近7天": 7, "近30天": 30, "不限": None}
            filtered_results = _filter_results_by_window(results, days_map.get(time_window_option))
            for item in filtered_results:
                item["credibility_score"] = _score_source_credibility(
                    str(item.get("url", "")),
                    str(item.get("content", "")),
                    item.get("published_date"),
                )

            if not filtered_results:
                status.update(label="无可用结果", state="error")
                st.info("当前时间窗口下未检索到有效结果，建议切换到“近30天”或“不限”。")
                return

            st.subheader("最新动态（检索摘要）")
            for item in filtered_results[:6]:
                with st.container():
                    st.markdown(f"**{item.get('title', '无标题')}**")
                    st.write(item.get("content", ""))
                    published = str(item.get("published_date", "")).strip() or "未知"
                    score = int(item.get("credibility_score", 0))
                    st.caption(f"发布时间：{published} | 来源可信度评分：{score}/100")
                    if item.get("url"):
                        st.markdown(f"[来源链接]({item['url']})")
                    st.divider()

            search_result["results"] = filtered_results
            context_text = _format_search_context(company_name.strip(), search_result)

            status.write("正在利用 DeepSeek 分析技术护城河...")
            try:
                review = generate_investment_review(
                    company_name=company_name.strip(),
                    deepseek_api_key=deepseek_api_key.strip(),
                    context_text=context_text,
                )
            except Exception as exc:
                status.update(label="分析失败", state="error")
                st.error(_friendly_error_message(exc, "DeepSeek"))
                return

            status.write("正在生成投研报告...")
            report_md = _build_markdown_report(
                company_name=company_name.strip(),
                review=review,
            )
            report_txt = report_md.replace("#", "").replace("*", "")
            report_pdf = _build_pdf_report(company_name.strip(), report_txt)
            status.update(label="分析完成", state="complete")

        st.subheader("投资价值简评")
        st.markdown(review if review else "未生成内容，请稍后重试。")
        _render_risk_card(review)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="下载 Markdown 报告",
                data=report_md.encode("utf-8"),
                file_name=f"{company_name.strip()}_投资价值简评.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="下载 文本报告（TXT）",
                data=report_txt.encode("utf-8"),
                file_name=f"{company_name.strip()}_投资价值简评.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col3:
            st.download_button(
                label="下载 PDF 报告",
                data=report_pdf,
                file_name=f"{company_name.strip()}_投资价值简评.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.caption("免责声明：本页面内容仅用于研究演示，不构成任何投资建议。")


if __name__ == "__main__":
    main()
