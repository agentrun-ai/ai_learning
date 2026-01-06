"""
AI搜学助手 - 流式输出架构

核心设计原则：
1. 代码控制流程流转，不依赖 LLM 自主决策
2. 严格执行每个阶段的要求（如必须收集足够数据）
3. 真正的流式输出：搜索、分析、撰写都实时更新
4. 数据质量筛选：相关性、时效性、贡献度
5. 多平台、多角度搜索
6. 深度分析，符合商业标准
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from ag_ui.core import EventType, StateSnapshotEvent
from dataclasses import dataclass, field
from dotenv import load_dotenv
from datetime import datetime
from playwright.async_api import async_playwright

import os
import asyncio
import json
import re

load_dotenv()

from agentrun.integration.pydantic_ai import model
from agentrun.utils.config import Config
from agentrun.sandbox import TemplateType, Sandbox, BrowserSandbox


# =============================================================================
# 自定义 Deps 类（包含 state 和 run_id）
# =============================================================================

@dataclass
class StateDeps:
    """Agent 依赖，包含状态和运行 ID"""
    state: "OpinionState"
    run_id: str = ""


# =============================================================================
# 全局状态存储（用于独立的状态流式更新）
# =============================================================================

global_state_store: Dict[str, "OpinionState"] = {}


# =============================================================================
# 数据模型
# =============================================================================

class SearchResult(BaseModel):
    """搜索结果"""
    title: str
    url: str
    snippet: str
    source: str
    date: str
    platform: str = "baidu"
    relevance_score: float = 0.0  # 相关性得分
    detailed_content: str = ""  # 深入抓取的详细内容


class AnalysisResult(BaseModel):
    """分析结果"""
    keywords: List[str] = Field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_distribution: Dict[str, int] = Field(default_factory=dict)
    heat_trend: List[int] = Field(default_factory=list)
    summary: str = ""
    key_opinions: List[Dict[str, str]] = Field(default_factory=list)
    risk_assessment: Dict[str, str] = Field(default_factory=dict)


class SandboxInfo(BaseModel):
    """Sandbox 信息"""
    sandbox_id: str
    vnc_url: str
    livestream_url: str
    active: bool = True
    created_at: str = ""


class OpinionState(BaseModel):
    """系统状态"""
    keyword: str = ""
    status: str = "idle"
    logs: List[str] = Field(default_factory=list)
    max_results: int = 50

    raw_data: List[SearchResult] = Field(default_factory=list)
    collected_data_summary: List[Dict[str, str]] = Field(default_factory=list)
    
    analysis: Optional[AnalysisResult] = None
    analysis_progress: str = ""  # 分析进度文本（流式）
    
    report_text: str = ""
    final_html: str = ""
    
    collection_progress: int = 0
    current_phase: str = ""
    
    # Sandbox 管理
    sandboxes: List[SandboxInfo] = Field(default_factory=list)
    active_sandbox_id: str = ""


# =============================================================================
# 配置
# =============================================================================

agentrun_model_name = os.getenv("AGENTRUN_MODEL_NAME", "")
model_name = os.getenv("MODEL_NAME")
agentrun_browser_sandbox_name = os.getenv("AGENTRUN_BROWSER_SANDBOX_NAME", "")

if not agentrun_model_name:
    raise ValueError("AGENTRUN_MODEL_NAME is not set")

config = Config(timeout=180)
agentrun_model = model(agentrun_model_name, model=model_name, config=config)


# =============================================================================
# Browser Sandbox 管理 - 支持多 Sandbox
# =============================================================================

_sandboxes: Dict[str, BrowserSandbox] = {}
_sandbox_lock = asyncio.Lock()


class SandboxTemplateNotFoundError(Exception):
    """Sandbox 模板不存在错误"""
    pass


class SandboxCreationError(Exception):
    """Sandbox 创建错误"""
    pass


async def create_browser_sandbox() -> Optional[BrowserSandbox]:
    """创建新的 Browser Sandbox 实例
    
    Raises:
        SandboxTemplateNotFoundError: 模板不存在时抛出
        SandboxCreationError: 其他创建错误时抛出
    """
    if not agentrun_browser_sandbox_name:
        return None
    
    async with _sandbox_lock:
        print("🌐 正在创建新的 Browser Sandbox...")
        try:
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.BROWSER,
                template_name=agentrun_browser_sandbox_name,
            )
            _sandboxes[sandbox.sandbox_id] = sandbox
            print(f"✅ Browser Sandbox 创建成功: {sandbox.sandbox_id}")
            return sandbox
        except Exception as e:
            error_msg = str(e).lower()
            # 检测模板不存在的错误
            template_not_found_patterns = [
                "template not found",
                "template does not exist",
                "no such template",
                "template_not_found",
                "not found",
                "无法找到模板",
            ]
            if any(pattern in error_msg for pattern in template_not_found_patterns):
                print(f"❌ Sandbox 模板不存在: {agentrun_browser_sandbox_name}")
                raise SandboxTemplateNotFoundError(
                    f"Sandbox 模板 '{agentrun_browser_sandbox_name}' 不存在，请检查 AGENTRUN_BROWSER_SANDBOX_NAME 配置"
                )
            else:
                print(f"❌ 创建 Sandbox 失败: {e}")
                raise SandboxCreationError(f"创建 Sandbox 失败: {e}")


async def get_browser_sandbox(sandbox_id: str = None) -> Optional[BrowserSandbox]:
    """获取指定或任意可用的 Browser Sandbox
    
    Raises:
        SandboxTemplateNotFoundError: 模板不存在时抛出
        SandboxCreationError: 其他创建错误时抛出
    """
    async with _sandbox_lock:
        if sandbox_id and sandbox_id in _sandboxes:
            return _sandboxes[sandbox_id]
        
        for sid, sandbox in _sandboxes.items():
            return sandbox
        
        if agentrun_browser_sandbox_name:
            try:
                sandbox = await Sandbox.create_async(
                    template_type=TemplateType.BROWSER,
                    template_name=agentrun_browser_sandbox_name,
                )
                _sandboxes[sandbox.sandbox_id] = sandbox
                return sandbox
            except Exception as e:
                error_msg = str(e).lower()
                # 检测模板不存在的错误
                template_not_found_patterns = [
                    "template not found",
                    "template does not exist",
                    "no such template",
                    "template_not_found",
                    "not found",
                    "无法找到模板",
                ]
                if any(pattern in error_msg for pattern in template_not_found_patterns):
                    raise SandboxTemplateNotFoundError(
                        f"Sandbox 模板 '{agentrun_browser_sandbox_name}' 不存在，请检查 AGENTRUN_BROWSER_SANDBOX_NAME 配置"
                    )
                else:
                    raise SandboxCreationError(f"创建 Sandbox 失败: {e}")
        
        return None


async def remove_sandbox(sandbox_id: str) -> None:
    """从管理列表中移除指定的 Sandbox"""
    async with _sandbox_lock:
        if sandbox_id in _sandboxes:
            del _sandboxes[sandbox_id]
            print(f"🗑️ Sandbox 已从管理列表中移除: {sandbox_id[:8]}...")


async def recreate_sandbox_if_closed(sandbox_id: str, error_message: str) -> Optional[BrowserSandbox]:
    """检测 sandbox 是否已关闭，如果关闭则重新创建
    
    Args:
        sandbox_id: 当前使用的 sandbox ID
        error_message: 错误信息
    
    Returns:
        新创建的 sandbox 实例，如果不需要重建则返回 None
    """
    # 检测 sandbox 关闭相关的错误
    closed_error_patterns = [
        "Target page, context or browser has been closed",
        "Browser has been closed",
        "Target closed",
        "Connection closed",
        "Session closed",
        "Page closed",
        "Context closed",
    ]
    
    is_closed_error = any(pattern.lower() in error_message.lower() for pattern in closed_error_patterns)
    
    if is_closed_error:
        print(f"⚠️ 检测到 Sandbox 已关闭: {error_message[:100]}")
        print(f"🔄 正在重新创建 Sandbox...")
        
        # 从管理列表中移除旧的 sandbox
        await remove_sandbox(sandbox_id)
        
        # 创建新的 sandbox
        new_sandbox = await create_browser_sandbox()
        if new_sandbox:
            print(f"✅ 新 Sandbox 创建成功: {new_sandbox.sandbox_id[:8]}...")
            return new_sandbox
        else:
            print(f"❌ 创建新 Sandbox 失败")
            return None
    
    return None


async def get_all_sandboxes() -> List[Dict[str, Any]]:
    """获取所有 Sandbox 信息"""
    from urllib.parse import urlparse, parse_qs, urlencode
    
    result = []
    async with _sandbox_lock:
        for sandbox_id, sandbox in _sandboxes.items():
            try:
                vnc_url = sandbox.get_vnc_url()
                access_token = sandbox.data_api.access_token
                
                parsed = urlparse(vnc_url)
                query_dict = parse_qs(parsed.query)
                query_dict["recording"] = ["false"]
                if access_token:
                    query_dict["Authorization"] = [access_token]
                
                new_path = parsed.path.replace("/ws/liveview", "/ws/livestream")
                new_query = urlencode(query_dict, doseq=True)
                livestream_url = f"{parsed.scheme}://{parsed.netloc}{new_path}?{new_query}"
                
                result.append({
                    "sandbox_id": sandbox_id,
                    "vnc_url": vnc_url,
                    "livestream_url": livestream_url,
                    "active": True,
                })
            except Exception as e:
                print(f"⚠️ 获取 Sandbox {sandbox_id} 信息失败: {e}")
                result.append({
                    "sandbox_id": sandbox_id,
                    "vnc_url": "",
                    "livestream_url": "",
                    "active": False,
                })
    
    return result


# =============================================================================
# 数据质量筛选
# =============================================================================

async def evaluate_relevance(keyword: str, title: str, snippet: str) -> float:
    """
    评估搜索结果的相关性（严格版）
    
    返回 0-1 的得分：
    - 1.0: 高度相关（关键词完全出现）
    - 0.5: 中等相关
    - 0.0: 不相关
    
    核心原则：关键词必须在标题或摘要中出现，否则视为不相关
    """
    # 合并标题和摘要
    text = f"{title} {snippet}"
    text_lower = text.lower()
    
    # 检测关键词是否为中文
    has_chinese_keyword = any('\u4e00' <= char <= '\u9fff' for char in keyword)
    result_has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    # 核心规则：中文关键词必须在结果中有中文内容
    if has_chinese_keyword and not result_has_chinese:
        return 0.0  # 直接返回 0，不相关
    
    # 排除明显的无关网站（在计算分数前先检查）
    irrelevant_patterns = [
        "calculator", "deepseek", "chegg", "stackoverflow", "github.com", 
        "npmjs", "pypi", "pizza", "wordreference", "cambridge", "yahoo字典",
        "翻译", "dictionary", "词典"
    ]
    if any(pattern in text_lower for pattern in irrelevant_patterns):
        return 0.0  # 直接返回 0
    
    score = 0.0
    
    # 1. 关键词完全匹配（最重要，必须条件）
    keyword_in_text = keyword in text
    if keyword_in_text:
        score += 0.6  # 基础分
    else:
        # 关键词不在文本中，检查是否有部分匹配
        if has_chinese_keyword:
            keyword_chars = list(keyword)
            matched_chars = sum(1 for char in keyword_chars if char in text)
            char_match_ratio = matched_chars / len(keyword_chars) if keyword_chars else 0
            
            # 如果匹配率低于 50%，直接返回 0
            if char_match_ratio < 0.5:
                return 0.0
            
            score += 0.4 * char_match_ratio
        else:
            # 英文关键词，检查单词匹配
            keyword_words = keyword.lower().split()
            matched_words = sum(1 for word in keyword_words if word in text_lower)
            word_match_ratio = matched_words / len(keyword_words) if keyword_words else 0
            
            if word_match_ratio < 0.5:
                return 0.0
            
            score += 0.4 * word_match_ratio
    
    # 2. 时效性加分
    time_keywords = ["最新", "今日", "近日", "昨日", "本周", "2024", "2025", "刚刚", "最近", "12月", "11月", "10月"]
    if any(tk in text for tk in time_keywords):
        score += 0.1
    
    # 3. 内容相关性加分
    opinion_keywords = ["评价", "评论", "看法", "观点", "讨论", "热议", "争议", "反响", "解析", "如何理解", "怎么学", "讲解"]
    if any(ok in text for ok in opinion_keywords):
        score += 0.1
    
    # 4. 平台来源加分（知乎、微博等）
    platform_keywords = ["知乎", "微博", "豆瓣", "B站", "bilibili", "抖音", "小红书", "新浪", "网易", "搜狐", "腾讯"]
    if any(pk in text for pk in platform_keywords):
        score += 0.1
    
    # 5. 排除广告和无关内容
    ad_keywords = ["广告", "推广", "优惠", "折扣", "促销", "点击立即", "免费下载", "立即购买", "官方旗舰店", "购物"]
    if any(ak in text for ak in ad_keywords):
        score -= 0.3
    
    return max(0.0, min(1.0, score))


def is_valid_result(result: SearchResult, keyword: str) -> bool:
    """判断搜索结果是否有效"""
    # 基本检查
    if not result.title or not result.url:
        return False
    
    # URL 有效性
    if not result.url.startswith("http"):
        return False
    
    # 排除明显的广告或无关页面
    exclude_domains = ["ad.", "ads.", "click.", "track."]
    if any(ed in result.url.lower() for ed in exclude_domains):
        return False
    
    return True


# =============================================================================
# 搜索查询生成器
# =============================================================================

def generate_search_queries(keyword: str) -> List[Dict[str, str]]:
    """生成多平台学习资料搜索查询
    
    使用百度搜索，通过关键词 + 平台名称的方式搜索
    """
    queries = []
    
    # 基础定义搜索（最高优先级）
    queries.extend([
        {"query": f"{keyword} 是什么", "category": "definition"},
        {"query": f"{keyword} 定义", "category": "definition"},
        {"query": f"{keyword} 概念", "category": "definition"},
        {"query": f"{keyword} 百科", "category": "definition"},
    ])
    
    # 原理和详解搜索
    queries.extend([
        {"query": f"{keyword} 原理", "category": "principle"},
        {"query": f"{keyword} 工作原理", "category": "principle"},
        {"query": f"{keyword} 详解", "category": "principle"},
        {"query": f"{keyword} 通俗解释", "category": "principle"},
    ])
    
    # 教程和入门搜索
    queries.extend([
        {"query": f"{keyword} 教程", "category": "tutorial"},
        {"query": f"{keyword} 入门", "category": "tutorial"},
        {"query": f"{keyword} 入门教程", "category": "tutorial"},
        {"query": f"{keyword} 学习", "category": "tutorial"},
    ])
    
    # 知乎科普（知乎内容质量较高）
    queries.extend([
        {"query": f"{keyword} 知乎", "category": "zhihu"},
        {"query": f"{keyword} 知乎 科普", "category": "zhihu"},
        {"query": f"{keyword} 知乎 入门", "category": "zhihu"},
        {"query": f"{keyword} 如何理解 知乎", "category": "zhihu"},
    ])
    
    # B站教程（视频学习资源）
    queries.extend([
        {"query": f"{keyword} B站 教程", "category": "bilibili"},
        {"query": f"{keyword} B站 科普", "category": "bilibili"},
        {"query": f"{keyword} 哔哩哔哩 讲解", "category": "bilibili"},
    ])
    
    # 技术博客和文档
    queries.extend([
        {"query": f"{keyword} CSDN", "category": "blog"},
        {"query": f"{keyword} 博客园", "category": "blog"},
        {"query": f"{keyword} 技术博客", "category": "blog"},
    ])
    
    # 实例和案例
    queries.extend([
        {"query": f"{keyword} 例子", "category": "example"},
        {"query": f"{keyword} 案例", "category": "example"},
        {"query": f"{keyword} 实例", "category": "example"},
        {"query": f"{keyword} 应用", "category": "example"},
    ])
    
    # 进阶学习
    queries.extend([
        {"query": f"{keyword} 进阶", "category": "advanced"},
        {"query": f"{keyword} 深入理解", "category": "advanced"},
        {"query": f"{keyword} 高级", "category": "advanced"},
    ])
    
    return queries


# =============================================================================
# 主 Agent
# =============================================================================

opinion_agent = Agent(
    agentrun_model,
    deps_type=StateDeps,
    system_prompt="""你是 AI 搜学助手的执行者。

你的任务是按照以下严格流程帮助用户学习和理解知识：

【流程】
1. 收到关键词后，调用 collect_data 工具收集学习资料
2. 数据收集完成后，调用 analyze_data 工具分析和提取知识点
3. 分析完成后，调用 write_report 工具撰写知识讲解文档
4. 报告完成后，调用 render_html 工具生成 HTML

【重要规则】
- 必须按顺序调用工具
- 每个工具只调用一次
- 不要跳过任何步骤
- 不要编造数据，所有内容基于搜索到的资料

当用户输入要学习的概念、名词或事物时，立即开始执行流程。
""",
    retries=3,
)


# =============================================================================
# 工具：数据收集（流式输出）
# =============================================================================

async def push_state_event(run_id: str, state: OpinionState):
    """推送状态更新事件到事件队列
    
    Args:
        run_id: 运行 ID，用于找到对应的事件队列
        state: 当前状态
    """
    import time
    from event_queue import event_manager
    
    # 创建 STATE_SNAPSHOT 事件，确保 timestamp 是数字
    event = StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=state.model_dump(),
        timestamp=int(time.time() * 1000)  # 毫秒时间戳
    )
    
    # 推送到队列
    await event_manager.push_event(run_id, event)


async def llm_decide_exploration(
    keyword: str,
    page_url: str,
    page_content: str,
    source: str,
    available_actions: List[Dict[str, str]]
) -> Dict:
    """让 LLM 决定是否需要进一步探索页面
    
    Args:
        keyword: 搜索关键词
        page_url: 当前页面 URL
        page_content: 当前页面已提取的内容
        source: 来源平台
        available_actions: 可用的操作列表，如 [{"action": "click_comments", "description": "点击查看评论区"}]
    
    Returns:
        {"should_explore": bool, "action": str, "reason": str}
    """
    if not available_actions:
        return {"should_explore": False, "action": None, "reason": "没有可用的操作"}
    
    prompt = f"""你是学习资料收集助手。请根据以下信息决定是否需要进一步探索页面获取更多学习资料。

【搜索关键词】{keyword}

【当前页面】{page_url}

【来源平台】{source}

【已获取内容预览】（前500字）
{page_content[:500]}

【可用操作】
{json.dumps(available_actions, ensure_ascii=False, indent=2)}

【决策标准】
1. 如果当前内容已经足够丰富（超过300字有效内容），可能不需要进一步探索
2. 如果是知乎/B站等平台，评论区通常包含重要的学习讨论，值得探索
3. 如果页面需要登录才能查看更多内容，则不探索
4. 如果相关推荐可能包含更多相关学习资料，可以考虑探索
5. 权衡时间成本，每个页面最多探索1-2个操作

请返回 JSON 格式（必须是有效 JSON）：
{{
    "should_explore": true/false,
    "action": "操作名称（如果 should_explore 为 true）",
    "reason": "决策原因（简短说明）"
}}
"""
    
    try:
        explorer = Agent(
            agentrun_model,
            system_prompt="你是学习资料收集助手，帮助决定是否需要深入探索页面。只返回有效的 JSON。",
            retries=2,
        )
        
        result = await explorer.run(prompt)
        response_text = result.output if hasattr(result, 'output') else str(result.data)
        
        # 解析 JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            decision = json.loads(json_match.group())
            return decision
    except Exception as e:
        print(f"   ⚠️ LLM 探索决策失败: {str(e)[:50]}")
    
    return {"should_explore": False, "action": None, "reason": "决策失败，跳过探索"}


async def explore_page_with_llm(
    page,
    keyword: str,
    url: str,
    source: str,
    initial_content: str
) -> str:
    """使用 LLM 控制的页面深入探索
    
    Args:
        page: Playwright 页面对象
        keyword: 搜索关键词
        url: 页面 URL
        source: 来源平台
        initial_content: 初始提取的内容
    
    Returns:
        探索后获取的额外内容
    """
    extra_content = ""
    
    # 根据平台定义可用操作
    available_actions = []
    
    if "weibo.com" in url:
        # 微博可用操作
        available_actions = [
            {"action": "view_comments", "description": "查看评论区内容", "selector": ".WB_feed_expand, [class*='comment'], .comment-list"},
            {"action": "view_retweets", "description": "查看转发内容", "selector": ".WB_feed_expand, [class*='repost']"},
        ]
    elif "zhihu.com" in url:
        # 知乎可用操作
        available_actions = [
            {"action": "view_more_answers", "description": "查看更多回答", "selector": ".AnswerItem, .List-item"},
            {"action": "view_comments", "description": "查看评论", "selector": ".Comments-container, .CommentItem"},
        ]
    elif "bilibili.com" in url:
        # B站可用操作
        available_actions = [
            {"action": "view_comments", "description": "查看评论区热门评论", "selector": ".reply-item, .root-reply"},
            {"action": "view_related", "description": "查看相关推荐视频", "selector": ".video-page-card, .recommend-list"},
        ]
    elif any(x in url for x in ["tieba.baidu.com"]):
        # 贴吧可用操作
        available_actions = [
            {"action": "view_replies", "description": "查看楼中楼回复", "selector": ".lzl_content, .j_lzl_c"},
        ]
    
    if not available_actions:
        return extra_content
    
    # 让 LLM 决定是否探索
    decision = await llm_decide_exploration(
        keyword=keyword,
        page_url=url,
        page_content=initial_content,
        source=source,
        available_actions=available_actions
    )
    
    if not decision.get("should_explore", False):
        print(f"   ℹ️ LLM 决定不探索: {decision.get('reason', '未知原因')}")
        return extra_content
    
    action = decision.get("action")
    print(f"   🔍 LLM 决定探索: {action} - {decision.get('reason', '')}")
    
    # 执行探索操作
    try:
        for action_def in available_actions:
            if action_def["action"] == action:
                selector = action_def["selector"]
                
                # 尝试滚动到评论区或相关区域
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(1)
                
                # 尝试点击展开按钮（如果有）
                expand_selectors = [
                    "button:has-text('展开')",
                    "a:has-text('展开')",
                    "span:has-text('展开')",
                    "button:has-text('查看更多')",
                    "a:has-text('查看更多')",
                    ".expand-btn",
                    ".more-btn",
                ]
                for exp_sel in expand_selectors:
                    try:
                        expand_btn = await page.query_selector(exp_sel)
                        if expand_btn:
                            await expand_btn.click()
                            await asyncio.sleep(1)
                            print(f"   ✅ 点击展开按钮")
                            break
                    except:
                        pass
                
                # 提取内容
                for sel in selector.split(", "):
                    try:
                        elems = await page.query_selector_all(sel.strip())
                        for elem in elems[:10]:  # 最多取10个元素
                            text = await elem.inner_text()
                            if text and len(text) > 10:
                                extra_content += text[:300] + "\n---\n"
                                if len(extra_content) > 2000:
                                    break
                        if len(extra_content) > 500:
                            break
                    except:
                        pass
                
                if extra_content:
                    print(f"   ✅ 探索获取到 {len(extra_content)} 字额外内容")
                break
                
    except Exception as e:
        print(f"   ⚠️ 探索操作失败: {str(e)[:50]}")
    
    return extra_content
    print(f"📡 状态已推送: {state.status} - {state.current_phase}")


@opinion_agent.tool
async def collect_data(
    ctx: RunContext[StateDeps],
    keyword: str,
) -> str:
    """
    收集学习资料 - 流式输出，每条数据实时更新
    
    Args:
        keyword: 要分析的关键词
    
    Returns:
        收集结果描述
    """
    state = ctx.deps.state
    run_id = ctx.deps.run_id  # 从 deps 获取运行 ID
    
    state.keyword = keyword
    state.status = "collecting"
    state.current_phase = "资料收集"
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 开始收集「{keyword}」的学习资料...")
    state.raw_data = []
    state.collected_data_summary = []
    
    # 推送初始状态
    await push_state_event(run_id, state)
    
    target_count = state.max_results
    collected = []
    seen_urls = set()
    
    # 获取搜索查询
    queries = generate_search_queries(keyword)
    
    # 创建新的 Sandbox
    try:
        sandbox = await create_browser_sandbox()
        if not sandbox:
            state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Browser Sandbox 未配置")
            state.status = "error"
            await push_state_event(run_id, state)
            return "Browser Sandbox 未配置，请设置 AGENTRUN_BROWSER_SANDBOX_NAME 环境变量"
    except SandboxTemplateNotFoundError as e:
        # 模板不存在 - 明确报错并结束任务
        error_msg = str(e)
        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {error_msg}")
        state.status = "error"
        state.current_phase = "错误"
        await push_state_event(run_id, state)
        raise RuntimeError(f"无法启动数据收集: {error_msg}")
    except SandboxCreationError as e:
        # 其他创建错误
        error_msg = str(e)
        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {error_msg}")
        state.status = "error"
        state.current_phase = "错误"
        await push_state_event(run_id, state)
        raise RuntimeError(f"无法启动数据收集: {error_msg}")
    
    # 更新 Sandbox 信息
    sandbox_info = await get_all_sandboxes()
    state.sandboxes = [SandboxInfo(**s) for s in sandbox_info]
    state.active_sandbox_id = sandbox.sandbox_id
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 浏览器已就绪: {sandbox.sandbox_id[:8]}...")
    
    # 推送 Sandbox 就绪状态
    await push_state_event(run_id, state)
    
    # 跟踪每个搜索类别的连续低相关性次数
    category_low_relevance_count: Dict[str, int] = {}
    max_low_relevance_per_category = 2  # 连续 2 次低相关则跳过该类别
    skipped_categories: set = set()
    
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.connect_over_cdp(sandbox.get_cdp_url())
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()
            
            query_index = 0
            max_retries = 5
            retry_count = 0
            sandbox_retry_count = 0  # Sandbox 重建次数
            max_sandbox_retries = 3  # 最多重建 3 次
            
            while len(collected) < target_count:
                if query_index >= len(queries):
                    if retry_count >= max_retries:
                        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 已达到最大重试次数，当前收集 {len(collected)} 条")
                        break
                    
                    retry_count += 1
                    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 数据不足 ({len(collected)}/{target_count})，第 {retry_count} 次补充搜索...")
                    
                    extra_queries = [
                        {"query": f"{keyword} 第{retry_count}页", "category": "extra"},
                        {"query": f"{keyword} 相关", "category": "extra"},
                        {"query": f"{keyword} 资讯", "category": "extra"},
                    ]
                    queries.extend(extra_queries)
                
                query_info = queries[query_index]
                query_index += 1
                
                query = query_info["query"]
                category = query_info["category"]
                
                # 检查该类别是否已被跳过
                if category in skipped_categories:
                    print(f"⏭️ 跳过低效类别: {category}")
                    continue
                
                state.current_phase = f"数据收集 ({len(collected)}/{target_count})"
                state.collection_progress = int(len(collected) / target_count * 100)
                
                try:
                    # URL 编码查询参数，使用 quote_plus 将空格编码为 + 而非 %20
                    from urllib.parse import quote_plus
                    encoded_query = quote_plus(query)
                    # 使用百度搜索
                    search_url = f"https://www.baidu.com/s?wd={encoded_query}"
                    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔎 搜索 [{category}]: {query[:30]}...")
                    print(f"🔎 搜索 URL: {search_url}")
                    
                    # 每次搜索前推送状态更新
                    await push_state_event(run_id, state)
                    
                    await page.goto(search_url, timeout=30000)
                    await page.wait_for_load_state("domcontentloaded")
                    await asyncio.sleep(2)
                    
                    # 检查当前 URL 是否仍然是搜索页面
                    current_url = page.url
                    print(f"📍 当前页面 URL: {current_url}")
                    
                    # 如果被重定向到非搜索页面，尝试重新搜索
                    if "baidu.com/s" not in current_url and "baidu.com/baidu" not in current_url:
                        print(f"⚠️ 页面被重定向，尝试重新导航...")
                        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 页面重定向，重试...")
                        await page.goto(search_url, timeout=30000)
                        await page.wait_for_load_state("domcontentloaded")
                        await asyncio.sleep(2)
                    
                    # 只使用精确的选择器，避免选中无关元素
                    # 百度搜索结果通常在 #content_left 下的 .result 或 .c-container 元素中
                    result_elements = await page.query_selector_all("#content_left .result, #content_left .c-container")
                    
                    # 打印调试信息
                    print(f"🔢 找到 {len(result_elements) if result_elements else 0} 个搜索结果元素")
                    
                    # 如果没有结果，尝试其他选择器
                    if not result_elements:
                        # 尝试备用选择器
                        alt_selectors = [
                            "#content_left .result-op",
                            ".result.c-container",
                            "#content_left > div[id^='1'], #content_left > div[id^='2'], #content_left > div[id^='3'], #content_left > div[id^='4'], #content_left > div[id^='5']",
                        ]
                        for alt_sel in alt_selectors:
                            result_elements = await page.query_selector_all(alt_sel)
                            if result_elements:
                                print(f"✅ 使用备用选择器 {alt_sel} 找到 {len(result_elements)} 个结果")
                                break
                    
                    # 如果仍然没有结果，记录日志并继续
                    if not result_elements:
                        print(f"⚠️ 搜索 [{category}] 未找到结果: {query[:30]}...")
                        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 未找到结果: {query[:30]}...")
                        # 打印页面 HTML 片段用于调试
                        try:
                            page_title = await page.title()
                            print(f"📄 页面标题: {page_title}")
                        except:
                            pass
                        continue
                    
                    new_results_in_query = 0
                    low_relevance_in_query = 0
                    
                    for elem in result_elements:
                        if len(collected) >= target_count:
                            break
                        
                        try:
                            # 百度搜索结果的标题在 h3 > a 或 .t > a
                            title_elem = await elem.query_selector("h3 a, .t a, a[href*='baidu.com/link']")
                            # 百度搜索结果的摘要在 .c-abstract 或 .c-span-last
                            snippet_elem = await elem.query_selector(".c-abstract, .c-span-last, .c-row .c-span9, span.content-right_8Zs40")
                            
                            if title_elem:
                                title = await title_elem.inner_text()
                                baidu_link_url = await title_elem.get_attribute("href") or ""
                                
                                # 跳过无效链接或百度跳转链接已访问过的
                                if not baidu_link_url or baidu_link_url in seen_urls:
                                    continue
                                
                                snippet = ""
                                if snippet_elem:
                                    snippet = await snippet_elem.inner_text()
                                
                                # 标记百度跳转链接已处理
                                seen_urls.add(baidu_link_url)
                                
                                # ========== 核心修改：先深入抓取获取真实URL和内容，再评估相关性 ==========
                                detailed_content = ""
                                real_url = baidu_link_url  # 默认使用百度链接
                                detail_page = None
                                
                                try:
                                    print(f"🔗 深入抓取: {title[:40]}...")
                                    
                                    # 打开新页面，跟踪百度跳转获取真实URL
                                    detail_page = await context.new_page()
                                    await detail_page.goto(baidu_link_url, timeout=15000)
                                    await detail_page.wait_for_load_state("domcontentloaded")
                                    await asyncio.sleep(1)
                                    
                                    # 获取跳转后的真实URL
                                    real_url = detail_page.url
                                    print(f"   📍 真实URL: {real_url[:60]}...")
                                    
                                    # 如果真实URL已经访问过，跳过
                                    if real_url in seen_urls:
                                        print(f"   ⏭️ 已访问过该URL，跳过")
                                        await detail_page.close()
                                        continue
                                    seen_urls.add(real_url)
                                    
                                    # 识别来源（基于真实URL）
                                    source = "网络"
                                    if "weibo.com" in real_url:
                                        source = "微博"
                                    elif "zhihu.com" in real_url:
                                        source = "知乎"
                                    elif "tieba.baidu.com" in real_url:
                                        source = "贴吧"
                                    elif any(x in real_url for x in ["news", "sina", "sohu", "163", "qq.com", "people", "cctv", "xinhua"]):
                                        source = "新闻"
                                    elif "bilibili.com" in real_url:
                                        source = "B站"
                                    elif "douyin.com" in real_url or "tiktok.com" in real_url:
                                        source = "抖音"
                                    elif "csdn.net" in real_url:
                                        source = "CSDN"
                                    elif "jianshu.com" in real_url:
                                        source = "简书"
                                    elif "cnblogs.com" in real_url:
                                        source = "博客园"
                                    elif "baike.baidu.com" in real_url:
                                        source = "百度百科"
                                    elif "wikipedia.org" in real_url:
                                        source = "维基百科"
                                    
                                    # 根据不同平台提取内容
                                    if "zhihu.com" in real_url:
                                        # 知乎：提取问题描述和回答
                                        content_selectors = [
                                            ".QuestionRichText",  # 问题描述
                                            ".RichContent-inner",  # 回答内容
                                            ".Post-RichText",  # 文章内容
                                        ]
                                        for sel in content_selectors:
                                            content_elems = await detail_page.query_selector_all(sel)
                                            for content_elem in content_elems[:3]:  # 最多取3个
                                                text = await content_elem.inner_text()
                                                if text and len(text) > 50:
                                                    detailed_content += text[:1000] + "\n\n"
                                                    if len(detailed_content) > 2000:
                                                        break
                                            if len(detailed_content) > 500:
                                                break
                                                
                                    elif "weibo.com" in real_url:
                                        # 微博：提取微博正文
                                        content_selectors = [
                                            ".WB_text",
                                            "[class*='detail_wbtext']",
                                            ".weibo-text",
                                        ]
                                        for sel in content_selectors:
                                            content_elems = await detail_page.query_selector_all(sel)
                                            for content_elem in content_elems[:5]:
                                                text = await content_elem.inner_text()
                                                if text and len(text) > 20:
                                                    detailed_content += text[:500] + "\n\n"
                                            if detailed_content:
                                                break
                                    
                                    elif "bilibili.com" in real_url:
                                        # B站：提取视频描述、评论等
                                        content_selectors = [
                                            ".video-desc",  # 视频描述
                                            ".desc-info-text",  # 新版描述
                                            ".basic-desc-info",  # 基本描述
                                            ".reply-content",  # 评论内容
                                            ".root-reply-content",  # 根评论
                                            ".article-content",  # 专栏文章
                                            ".opus-module-content",  # 动态内容
                                        ]
                                        for sel in content_selectors:
                                            content_elems = await detail_page.query_selector_all(sel)
                                            for content_elem in content_elems[:8]:  # B站评论较多，多取一些
                                                text = await content_elem.inner_text()
                                                if text and len(text) > 15:
                                                    detailed_content += text[:400] + "\n\n"
                                                    if len(detailed_content) > 2500:
                                                        break
                                            if len(detailed_content) > 800:
                                                break
                                    
                                    elif "baike.baidu.com" in real_url:
                                        # 百度百科：提取词条内容
                                        content_selectors = [
                                            ".lemma-summary",  # 摘要
                                            ".para",  # 正文段落
                                            ".basic-info",  # 基本信息
                                        ]
                                        for sel in content_selectors:
                                            content_elems = await detail_page.query_selector_all(sel)
                                            for content_elem in content_elems[:10]:
                                                text = await content_elem.inner_text()
                                                if text and len(text) > 30:
                                                    detailed_content += text[:800] + "\n\n"
                                                    if len(detailed_content) > 3000:
                                                        break
                                            if len(detailed_content) > 1000:
                                                break
                                    
                                    elif "csdn.net" in real_url:
                                        # CSDN：提取博客内容
                                        content_selectors = [
                                            "#content_views",  # 文章内容
                                            ".article_content",
                                            ".markdown_views",
                                        ]
                                        for sel in content_selectors:
                                            content_elems = await detail_page.query_selector_all(sel)
                                            for content_elem in content_elems[:3]:
                                                text = await content_elem.inner_text()
                                                if text and len(text) > 100:
                                                    detailed_content += text[:2000] + "\n\n"
                                                    if len(detailed_content) > 3000:
                                                        break
                                            if len(detailed_content) > 500:
                                                break
                                                
                                    else:
                                        # 通用：提取文章正文
                                        content_selectors = [
                                            "article",
                                            ".article-content",
                                            ".post-content",
                                            ".content",
                                            "main p",
                                            ".entry-content",
                                            "#article",
                                            ".text",
                                        ]
                                        for sel in content_selectors:
                                            content_elems = await detail_page.query_selector_all(sel)
                                            for content_elem in content_elems[:3]:
                                                text = await content_elem.inner_text()
                                                if text and len(text) > 100:
                                                    detailed_content += text[:1500] + "\n\n"
                                                    if len(detailed_content) > 2000:
                                                        break
                                            if len(detailed_content) > 500:
                                                break
                                    
                                    if detailed_content:
                                        print(f"   ✅ 获取到 {len(detailed_content)} 字详细内容")
                                        
                                        # LLM 控制的深入探索（评论区、相关推荐等）
                                        try:
                                            extra_content = await explore_page_with_llm(
                                                page=detail_page,
                                                keyword=keyword,
                                                url=real_url,
                                                source=source,
                                                initial_content=detailed_content
                                            )
                                            if extra_content:
                                                detailed_content += "\n\n【深入探索内容】\n" + extra_content
                                                print(f"   ✅ 深入探索后总计 {len(detailed_content)} 字")
                                        except Exception as e:
                                            print(f"   ⚠️ 深入探索失败: {str(e)[:50]}")
                                    else:
                                        print(f"   ⚠️ 未能提取详细内容")
                                        
                                except Exception as e:
                                    print(f"   ⚠️ 深入抓取失败: {str(e)[:50]}")
                                finally:
                                    if detail_page:
                                        await detail_page.close()
                                
                                # ========== 基于详情页内容评估相关性 ==========
                                # 使用详细内容（如果有）或摘要来评估相关性
                                content_for_relevance = detailed_content if detailed_content else snippet
                                relevance = await evaluate_relevance(keyword, title, content_for_relevance)
                                
                                # 只保留相关性 >= 0.3 的结果
                                if relevance < 0.3:
                                    print(f"   ⏭️ 跳过低相关性结果: {title[:30]}... (得分: {relevance:.2f})")
                                    low_relevance_in_query += 1
                                    continue
                                
                                print(f"   ✅ 相关性评分: {relevance:.2f}")
                                
                                # 使用真实URL而不是百度跳转链接
                                url = real_url
                                
                                # 合并摘要和详细内容
                                full_content = snippet.strip()
                                if detailed_content:
                                    full_content = detailed_content.strip()[:3000]
                                
                                result = SearchResult(
                                    title=title.strip(),
                                    url=url,
                                    snippet=full_content[:500],  # 摘要保留500字
                                    source=source,
                                    date=datetime.now().strftime("%Y-%m-%d"),
                                    platform="baidu",
                                    relevance_score=relevance,
                                    detailed_content=detailed_content[:3000] if detailed_content else ""  # 详细内容
                                )
                                
                                # 验证结果有效性
                                if not is_valid_result(result, keyword):
                                    continue
                                
                                collected.append(result)
                                state.raw_data.append(result)
                                state.collected_data_summary.append({
                                    "title": result.title[:50],
                                    "url": result.url,
                                    "source": result.source,
                                    "relevance": f"{relevance:.0%}",
                                })
                                state.collection_progress = int(len(collected) / target_count * 100)
                                new_results_in_query += 1
                                
                                # 流式输出：每收集一条就打印并推送状态
                                print(f"💾 [{len(collected)}/{target_count}] [{relevance:.0%}] {source}: {title[:40]}...")
                                
                                # 每收集 1 条数据就发送状态更新（实时）
                                await push_state_event(run_id, state)
                                
                        except Exception as e:
                            print(f"⚠️ 解析结果失败: {e}")
                            continue
                    
                    # 更新类别的低相关性计数
                    if new_results_in_query == 0 and low_relevance_in_query > 3:
                        category_low_relevance_count[category] = category_low_relevance_count.get(category, 0) + 1
                        if category_low_relevance_count[category] >= max_low_relevance_per_category:
                            skipped_categories.add(category)
                            state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⏭️ 跳过低效链路: {category}")
                            print(f"⏭️ 类别 {category} 连续 {max_low_relevance_per_category} 次低相关，已跳过")
                    elif new_results_in_query > 0:
                        # 重置计数
                        category_low_relevance_count[category] = 0
                    
                    if new_results_in_query > 0:
                        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 本次获得 {new_results_in_query} 条有效结果")
                        # 有新结果时发送状态更新
                        await push_state_event(run_id, state)
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️ 搜索失败: {error_msg}")
                    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 搜索失败: {error_msg[:50]}")
                    
                    # 检测是否是 sandbox 关闭错误
                    if sandbox_retry_count < max_sandbox_retries:
                        new_sandbox = await recreate_sandbox_if_closed(sandbox.sandbox_id, error_msg)
                        if new_sandbox:
                            sandbox_retry_count += 1
                            sandbox = new_sandbox
                            state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Sandbox 已重建 ({sandbox_retry_count}/{max_sandbox_retries})")
                            
                            # 更新 Sandbox 信息到状态
                            sandbox_info = await get_all_sandboxes()
                            state.sandboxes = [SandboxInfo(**s) for s in sandbox_info]
                            state.active_sandbox_id = sandbox.sandbox_id
                            state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 新浏览器已就绪: {sandbox.sandbox_id[:8]}...")
                            
                            # 推送新的 sandbox 状态到前端
                            await push_state_event(run_id, state)
                            
                            # 重新连接浏览器
                            try:
                                browser = await playwright.chromium.connect_over_cdp(sandbox.get_cdp_url())
                                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                                page = context.pages[0] if context.pages else await context.new_page()
                                print(f"✅ 已重新连接到新 Sandbox")
                                continue  # 继续搜索循环
                            except Exception as reconnect_error:
                                print(f"❌ 重新连接 Sandbox 失败: {reconnect_error}")
                                state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 重新连接失败: {str(reconnect_error)[:50]}")
                
                await asyncio.sleep(1)
    
    except Exception as e:
        error_msg = str(e)
        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 数据收集出错: {error_msg[:100]}")
        print(f"❌ 数据收集出错: {e}")
        
        # 在顶层异常处理中也检测 sandbox 关闭错误
        new_sandbox = await recreate_sandbox_if_closed(state.active_sandbox_id, error_msg)
        if new_sandbox:
            # 更新状态中的 sandbox 信息
            sandbox_info = await get_all_sandboxes()
            state.sandboxes = [SandboxInfo(**s) for s in sandbox_info]
            state.active_sandbox_id = new_sandbox.sandbox_id
            state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Sandbox 已重建: {new_sandbox.sandbox_id[:8]}...")
            await push_state_event(run_id, state)
    
    # 按相关性排序
    state.raw_data.sort(key=lambda x: x.relevance_score, reverse=True)
    
    state.status = "collected"
    state.collection_progress = 100
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 资料收集完成: {len(collected)} 条学习资料")
    state.current_phase = f"已收集 {len(collected)} 条资料"
    
    # 发送最终状态
    await push_state_event(run_id, state)
    
    return f"资料收集完成，共 {len(collected)} 条学习资料"


# =============================================================================
# 工具：数据分析（流式输出）
# =============================================================================

@opinion_agent.tool
async def analyze_data(
    ctx: RunContext[StateDeps],
) -> str:
    """
    分析收集到的数据 - 流式输出分析过程
    
    Returns:
        分析结果描述
    """
    state = ctx.deps.state
    run_id = ctx.deps.run_id  # 从 deps 获取运行 ID
    
    state.status = "analyzing"
    state.current_phase = "知识分析"
    state.analysis_progress = ""
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 开始知识点提取和分析...")
    
    # 发送初始状态
    await push_state_event(run_id, state)
    
    # 统计来源分布
    source_stats = {}
    for item in state.raw_data:
        source_stats[item.source] = source_stats.get(item.source, 0) + 1
    
    # 流式输出：分析进度
    state.analysis_progress = "正在统计数据来源分布...\n"
    state.analysis_progress += f"数据来源: {json.dumps(source_stats, ensure_ascii=False)}\n\n"
    await push_state_event(run_id, state)
    
    # 准备数据 - 包含详细内容
    data_for_analysis = []
    detailed_contents = []
    for item in state.raw_data[:30]:  # 取前30条
        data_for_analysis.append({
            "title": item.title,
            "snippet": item.snippet[:300],
            "source": item.source,
            "relevance": item.relevance_score,
        })
        # 收集有详细内容的数据
        if item.detailed_content:
            detailed_contents.append({
                "title": item.title,
                "source": item.source,
                "content": item.detailed_content[:1500],  # 截取详细内容
            })
    
    state.analysis_progress += "正在提取核心概念和知识点...\n"
    state.analysis_progress += f"已获取 {len(detailed_contents)} 条详细内容用于知识提取\n"
    await push_state_event(run_id, state)
    
    # 构建详细内容部分
    detailed_section = ""
    if detailed_contents:
        detailed_section = f"""

【详细内容摘录】（深入抓取的原文内容）：
{json.dumps(detailed_contents[:10], ensure_ascii=False, indent=2)}
"""
    
    # 使用 LLM 分析
    analysis_prompt = f"""
请对以下关于「{state.keyword}」的 {len(state.raw_data)} 条学习资料进行知识提取和分析。

═══════════════════════════════════════════════════════════════
【资料概览】
═══════════════════════════════════════════════════════════════
- 学习主题: {state.keyword}
- 资料总量: {len(state.raw_data)} 条
- 资料来源分布: {json.dumps(source_stats, ensure_ascii=False)}
- 收集时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

═══════════════════════════════════════════════════════════════
【资料样本】（按相关性排序的代表性资料）
═══════════════════════════════════════════════════════════════
{json.dumps(data_for_analysis, ensure_ascii=False, indent=2)}

{detailed_section}

═══════════════════════════════════════════════════════════════
【分析要求】
═══════════════════════════════════════════════════════════════

请基于以上资料，进行以下维度的知识提取和分析：

1. **核心概念提取**: 
   - 提取 15-20 个核心概念和术语
   - 按重要性排序，包含定义词、原理词、应用词、相关概念

2. **难度评估**:
   - 计算综合难度得分 (-1 到 1，-1表示入门级，0表示中级，1表示高级)
   - 统计内容类型分布：定义类、原理类、应用类的百分比
   - 分析学习难度的主要因素

3. **学习热度**:
   - 预估该知识点的学习热度趋势 (0-100)
   - 识别学习重点和难点

4. **关键知识点提炼**:
   - 从资料中提炼 5-8 个最重要的知识点
   - 标注知识点来源和类型（定义/原理/应用）
   - 引用原文中的关键解释

5. **学习建议**:
   - 前置知识：学习该概念需要哪些前置知识
   - 学习路径：建议的学习顺序和方法
   - 进阶方向：掌握基础后可以深入的方向

请返回 JSON 格式（确保是有效的 JSON，不要有多余内容）：
{{
    "keywords": ["核心概念1", "核心概念2", "核心概念3", ...],
    "sentiment_score": 0.0,
    "sentiment_distribution": {{"定义": 40, "原理": 35, "应用": 25}},
    "heat_trend": [30, 45, 60, 80, 70, 55, 40],
    "summary": "300-500字的知识概述，包含：1)概念定义 2)核心原理 3)主要应用 4)学习价值",
    "key_opinions": [
        {{"viewpoint": "具体知识点内容，引用原文解释", "source": "来源平台", "sentiment": "定义/原理/应用", "influence": "核心/重要/补充"}}
    ],
    "risk_assessment": {{
        "spread_risk": "高/中/低",
        "spread_reason": "前置知识要求",
        "reputation_risk": "高/中/低",
        "reputation_reason": "学习难度说明",
        "trend": "上升/平稳/下降",
        "trend_reason": "学习路径建议"
    }}
}}
"""
    
    try:
        analyzer = Agent(
            agentrun_model,
            system_prompt="""你是资深知识分析专家和教育顾问，擅长从海量学习资料中提炼核心知识点。

【分析原则】
1. 资料驱动：所有结论必须基于收集的资料，不能凭空臆断
2. 深度提炼：不停留在表面，挖掘概念背后的深层原理
3. 原文引用：提炼知识点时要引用原文中的关键解释
4. 结构化分析：将知识点按定义、原理、应用进行分类
5. 学习导向：关注学习路径和难度评估

【输出要求】
- 必须返回有效的 JSON 格式
- summary 要详细，300-500 字，包含概念的完整解释
- key_opinions 要具体，引用原文的知识点解释
- 学习建议要有具体的前置知识和学习路径""",
            retries=3,
        )
        
        state.analysis_progress += "正在调用 AI 进行知识提取...\n"
        await push_state_event(run_id, state)
        
        # 使用 run_stream 实现流式输出分析过程
        response_text = ""
        last_event_length = 0
        last_event_time = asyncio.get_event_loop().time()
        
        async with analyzer.run_stream(analysis_prompt) as result:
            async for text in result.stream_text():
                response_text = text  # 累积获取完整响应
                
                # 实时显示分析进度
                current_time = asyncio.get_event_loop().time()
                content_delta = len(response_text) - last_event_length
                time_delta = current_time - last_event_time
                
                # 每 200 字符或每 0.5 秒更新一次
                if content_delta >= 200 or (content_delta > 0 and time_delta >= 0.5):
                    # 显示正在分析的进度
                    state.analysis_progress = f"正在分析中... ({len(response_text)} 字)\n"
                    await push_state_event(run_id, state)
                    last_event_length = len(response_text)
                    last_event_time = current_time
        
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            analysis_data = json.loads(json_match.group())
            
            state.analysis = AnalysisResult(
                keywords=analysis_data.get("keywords", [state.keyword]),
                sentiment_score=float(analysis_data.get("sentiment_score", 0)),
                sentiment_distribution=analysis_data.get("sentiment_distribution", {"正面": 33, "中性": 34, "负面": 33}),
                heat_trend=analysis_data.get("heat_trend", [50]*7),
                summary=analysis_data.get("summary", f"关于「{state.keyword}」的知识分析"),
                key_opinions=analysis_data.get("key_opinions", []),
                risk_assessment=analysis_data.get("risk_assessment", {}),
            )
            
            # 流式输出：分析结果
            state.analysis_progress += f"\n✅ 知识提取完成！\n"
            state.analysis_progress += f"- 难度评分: {state.analysis.sentiment_score:.2f}\n"
            state.analysis_progress += f"- 核心概念: {', '.join(state.analysis.keywords[:5])}\n"
            state.analysis_progress += f"- 内容分布: 定义 {state.analysis.sentiment_distribution.get('定义', 0)}%, "
            state.analysis_progress += f"原理 {state.analysis.sentiment_distribution.get('原理', 0)}%, "
            state.analysis_progress += f"应用 {state.analysis.sentiment_distribution.get('应用', 0)}%\n"
            
            state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 知识提取完成: 难度评分 {state.analysis.sentiment_score:.2f}")
        else:
            raise ValueError("无法解析 JSON")
            
    except Exception as e:
        print(f"⚠️ 分析出错: {e}")
        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 分析出错，使用默认值")
        state.analysis_progress += f"\n⚠️ 分析出错: {str(e)[:50]}\n"
        
        state.analysis = AnalysisResult(
            keywords=[state.keyword],
            sentiment_score=0,
            sentiment_distribution={"正面": 33, "中性": 34, "负面": 33},
            heat_trend=[50, 50, 50, 50, 50, 50, 50],
            summary=f"关于「{state.keyword}」的知识分析",
        )
    
    state.status = "analyzed"
    state.current_phase = "知识提取完成"
    
    # 发送最终状态
    await push_state_event(run_id, state)
    
    return "知识分析完成"


# =============================================================================
# 工具：撰写报告（流式输出）
# =============================================================================

@opinion_agent.tool
async def write_report(
    ctx: RunContext[StateDeps],
) -> str:
    """
    撰写知识讲解文档 - 流式输出文档内容
    
    Returns:
        文档撰写结果描述
    """
    state = ctx.deps.state
    run_id = ctx.deps.run_id  # 从 deps 获取运行 ID
    
    state.status = "writing"
    state.current_phase = "文档撰写"
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 开始撰写知识讲解文档...")
    state.report_text = ""
    
    # 发送初始状态
    await push_state_event(run_id, state)
    
    analysis = state.analysis or AnalysisResult()
    keyword = state.keyword
    data_count = len(state.raw_data)
    
    # 来源统计
    source_stats = {}
    for item in state.raw_data:
        source_stats[item.source] = source_stats.get(item.source, 0) + 1
    
    # 流式输出：先输出文档框架
    state.report_text = f"# {keyword} 知识讲解\n\n"
    state.report_text += f"> 正在生成知识讲解文档，请稍候...\n\n"
    state.report_text += f"**资料来源**: {data_count} 条学习资料，来源: {', '.join(source_stats.keys())}\n\n"
    await push_state_event(run_id, state)
    
    # 准备引用数据（包含完整链接）
    references_data = []
    for i, item in enumerate(state.raw_data[:20]):
        ref = {
            "id": i + 1,
            "title": item.title,
            "source": item.source,
            "url": item.url,  # 完整链接
            "snippet": item.snippet[:200],
            "relevance": f"{item.relevance_score:.0%}",
        }
        if item.detailed_content:
            ref["content"] = item.detailed_content[:600]
        references_data.append(ref)
    
    # 生成引用列表（带链接）供报告附录使用
    references_list_md = "\n".join([
        f"[{ref['id']}] [{ref['title'][:50]}...]({ref['url']}) - {ref['source']}"
        for ref in references_data
    ])
    
    # 使用 LLM 生成知识讲解文档
    report_prompt = f"""
请为「{keyword}」撰写一份通俗易懂的知识讲解文档。

═══════════════════════════════════════════════════════════════
【资料基础】
═══════════════════════════════════════════════════════════════
- 整理时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- 资料量: {data_count} 条学习资料
- 资料来源: {json.dumps(source_stats, ensure_ascii=False)}
- 内容分布: 定义 {analysis.sentiment_distribution.get('定义', 0)}%、原理 {analysis.sentiment_distribution.get('原理', 0)}%、应用 {analysis.sentiment_distribution.get('应用', 0)}%
- 难度评分: {analysis.sentiment_score:.2f} (范围 -1 到 1，-1为入门，0为中级，1为高级)
- 核心概念: {', '.join(analysis.keywords[:15])}
- 学习建议: 前置知识 {analysis.risk_assessment.get('spread_risk', '中')}、难度 {analysis.risk_assessment.get('reputation_risk', '中')}、学习路径 {analysis.risk_assessment.get('trend', '平稳')}

═══════════════════════════════════════════════════════════════
【知识概述】
═══════════════════════════════════════════════════════════════
{analysis.summary}

═══════════════════════════════════════════════════════════════
【关键知识点】
═══════════════════════════════════════════════════════════════
{json.dumps(analysis.key_opinions, ensure_ascii=False, indent=2)}

═══════════════════════════════════════════════════════════════
【参考资料】（编号 [1]-[{len(references_data)}]，文档中必须引用）
═══════════════════════════════════════════════════════════════
{json.dumps(references_data, ensure_ascii=False, indent=2)}

═══════════════════════════════════════════════════════════════
【文档撰写要求】
═══════════════════════════════════════════════════════════════

**核心要求 - 必须包含以下六个维度的知识讲解：**

1. **概念定义与简介**
   - 用通俗易懂的语言解释「{keyword}」是什么
   - 给出标准的学术/专业定义
   - 解释这个概念的起源和发展历史
   - 说明为什么这个概念重要

2. **核心原理与机制**
   - 详细解释「{keyword}」的工作原理
   - 分解核心组成部分和关键要素
   - 使用类比和比喻帮助理解
   - 配合图示说明（用文字描述）

3. **详细解释与深入理解**
   - 展开讲解各个重要知识点
   - 引用权威资料的解释（使用 [编号] 格式引用上述资料）
   - 解答常见疑问和误区
   - 提供多角度的理解方式

4. **实际应用与案例分析**
   - 列举「{keyword}」的实际应用场景
   - 提供具体的案例说明
   - 分析应用中的注意事项
   - 展示成功应用的实例

5. **关联知识与进阶学习**
   - 列出学习「{keyword}」需要的前置知识
   - 介绍与之相关的其他概念
   - 提供进阶学习的方向和资源
   - 建议学习路径和时间规划

6. **总结与学习建议**
   - 总结核心要点
   - 提供记忆技巧和理解窍门
   - 给出具体的学习行动建议
   - 推荐后续学习资源

**文档结构要求：**
- 篇幅: 3000-5000 字，内容详实
- 格式: Markdown，层次清晰
- 必须包含的章节:
  1. 一、概念简介
  2. 二、核心原理
  3. 三、详细讲解
  4. 四、应用案例
  5. 五、关联知识与进阶
  6. 六、总结与学习建议
  7. 附录：参考资料

**引用规范（重要！）：**
- 在文档正文中，使用 Markdown 链接格式引用参考资料
- 格式示例：根据[百度百科](URL)的定义...
- 或者：[这篇教程](URL)详细解释了...
- 每个章节至少引用 2-3 个资料来源
- 引用要自然融入行文，增强可信度
- 附录中列出所有参考资料的完整链接列表

**写作风格：**
- 通俗易懂、循序渐进
- 多用类比和实例帮助理解
- 保持教育性和趣味性
- 适合零基础读者阅读

**附录格式要求：**
在文档末尾的"附录：参考资料"章节中，按以下格式列出所有引用：
{references_list_md}

请直接输出 Markdown 格式的知识讲解文档，不要包含其他说明文字。
"""
    
    try:
        writer = Agent(
            agentrun_model,
            system_prompt="""你是顶级知识讲解专家和科普作家，拥有 15 年以上教育和科普写作经验。
你曾为多家知名科普平台撰稿，擅长将复杂的专业知识转化为通俗易懂的讲解内容。
你的文档需要达到可直接用于教学和自学的专业标准。

═══════════════════════════════════════════════════════════════
【核心能力】
═══════════════════════════════════════════════════════════════
1. **知识提炼**: 从海量资料中提取核心概念，梳理知识结构
2. **深入浅出**: 用通俗语言和生动类比解释复杂概念
3. **循序渐进**: 按照认知规律组织内容，由浅入深
4. **实例驱动**: 用丰富的案例和应用场景加深理解

═══════════════════════════════════════════════════════════════
【文档撰写原则】
═══════════════════════════════════════════════════════════════

**1. 资料驱动，有据可依**
- 每个知识点必须有资料支撑
- 使用 [编号] 格式引用具体资料来源
- 引用要自然融入行文，例如：
  - "根据[百度百科](URL)的定义..."
  - "[这篇教程](URL)详细解释了..."
  - "综合多个资料来源[2][7][9]..."

**2. 通俗易懂，深入浅出**
- 避免一上来就使用专业术语
- 先用生活化语言解释，再引入专业定义
- 多用类比、比喻帮助理解
- 配合实例说明抽象概念

**3. 结构清晰，逻辑递进**
- 使用清晰的标题层级（#、##、###）
- 从定义到原理到应用，循序渐进
- 善用列表、表格、引用块增强可读性
- 每个章节有明确的学习目标

**4. 准确严谨，概念清晰**
- 核心概念要给出准确定义
- 区分相似概念，避免混淆
- 指出常见误区和理解偏差
- 保证知识的准确性和权威性

**5. 学习导向，实用性强**
- 站在学习者角度组织内容
- 提供具体的学习建议和路径
- 推荐进一步学习的资源
- 设计思考题帮助巩固理解

**6. 趣味性与专业性兼顾**
- 适当使用有趣的例子和故事
- 保持内容的专业性和深度
- 让学习过程更加愉快

═══════════════════════════════════════════════════════════════
【质量标准】
═══════════════════════════════════════════════════════════════
- 篇幅: 3000-5000 字，内容详实不空洞
- 引用: 每个章节至少引用 2-3 个资料来源
- 深度: 概念解释透彻，原理讲解清楚
- 实用: 提供可操作的学习建议和行动指南""",
            retries=3,
        )
        
        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 正在生成知识讲解内容...")
        await push_state_event(run_id, state)
        
        # 使用 run_stream 实现 token-by-token 流式输出
        report_content = ""
        last_event_length = 0
        last_event_time = asyncio.get_event_loop().time()
        
        async with writer.run_stream(report_prompt) as result:
            async for text in result.stream_text():
                report_content = text  # 累积获取完整响应
                state.report_text = report_content
                
                current_time = asyncio.get_event_loop().time()
                content_delta = len(report_content) - last_event_length
                time_delta = current_time - last_event_time
                
                # 更细粒度的流式输出：每 100 字符或每 0.3 秒发送一次
                if content_delta >= 100 or (content_delta > 0 and time_delta >= 0.3):
                    await push_state_event(run_id, state)
                    last_event_length = len(report_content)
                    last_event_time = current_time
        
        # 清理可能的代码块标记
        report_content = re.sub(r'^```\w*\n?', '', report_content)
        report_content = re.sub(r'\n?```$', '', report_content)
        
        state.report_text = report_content
        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 知识讲解文档完成: {len(report_content)} 字")
        
    except Exception as e:
        print(f"⚠️ 文档撰写出错: {e}")
        state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 文档撰写出错，使用模板")
        
        state.report_text = generate_template_report(state)
    
    state.status = "written"
    state.current_phase = "文档完成"
    
    # 发送最终状态
    await push_state_event(run_id, state)
    
    return f"知识讲解文档完成，共 {len(state.report_text)} 字"


def generate_template_report(state: OpinionState) -> str:
    """生成模板文档 - 知识讲解模板"""
    analysis = state.analysis or AnalysisResult()
    keyword = state.keyword
    data_count = len(state.raw_data)
    
    source_stats = {}
    for item in state.raw_data:
        source_stats[item.source] = source_stats.get(item.source, 0) + 1
    
    return f"""# {keyword} 知识讲解

> **内容概述**: {analysis.summary or f'基于 {data_count} 条学习资料整理的知识讲解'}

---

## 一、概念简介

### 1.1 资料来源
- **整理时间**: {datetime.now().strftime('%Y-%m-%d')}
- **资料来源**: {', '.join(source_stats.keys()) if source_stats else '网络'}
- **资料量**: 共 {data_count} 条学习资料
- **来源分布**:
{chr(10).join([f'  - {k}: {v} 条 ({v/data_count*100:.1f}%)' for k, v in source_stats.items()]) if source_stats and data_count > 0 else '  - 暂无数据'}

### 1.2 什么是「{keyword}」

「{keyword}」是一个重要的概念/事物。根据收集的资料，它涉及以下核心知识点：

{chr(10).join([f'- **{kw}**' for kw in (analysis.keywords[:6] if analysis.keywords else [keyword])])}

### 1.3 学习概览

| 维度 | 说明 |
|------|------|
| **难度等级** | {'入门级' if analysis.sentiment_score < -0.3 else '中级' if analysis.sentiment_score < 0.3 else '高级'} |
| **内容分布** | 定义 {analysis.sentiment_distribution.get('定义', 33)}%、原理 {analysis.sentiment_distribution.get('原理', 34)}%、应用 {analysis.sentiment_distribution.get('应用', 33)}% |
| **学习热度** | {'较高' if analysis.heat_trend and analysis.heat_trend[-1] > 50 else '中等'} |

---

## 二、核心原理

### 2.1 基本定义

「{keyword}」的标准定义和核心含义。这个概念的本质是什么，它解决什么问题，有什么特点。

### 2.2 工作原理

详细解释「{keyword}」的工作原理和机制：

1. **基础原理**: 概念的基本工作方式
2. **核心机制**: 关键的运作机制
3. **组成要素**: 主要的组成部分

### 2.3 关键概念

| 概念 | 说明 |
|------|------|
{chr(10).join([f'| **{kw}** | 相关概念的解释说明 |' for kw in (analysis.keywords[:5] if analysis.keywords else [keyword])])}

---

## 三、详细讲解

### 3.1 深入理解

{chr(10).join([f'**知识点{i+1}**: {op.get("viewpoint", "")[:200]}... [来源: {op.get("source", "网络")}]' for i, op in enumerate(analysis.key_opinions[:3] if analysis.key_opinions else [{"viewpoint": "这是一个重要的知识点", "source": "综合"}])])}

### 3.2 常见问题

- **问题1**: 关于「{keyword}」的常见疑问
- **问题2**: 容易混淆的概念辨析
- **问题3**: 学习中的常见误区

### 3.3 重点难点

| 类型 | 内容 | 建议 |
|------|------|------|
| **重点** | 核心概念和原理 | 反复理解，多做练习 |
| **难点** | 抽象概念的理解 | 结合实例，循序渐进 |
| **易错点** | 概念混淆 | 对比学习，区分异同 |

---

## 四、应用案例

### 4.1 实际应用场景

「{keyword}」在实际中的主要应用领域：

1. **场景一**: 第一个典型应用场景
2. **场景二**: 第二个典型应用场景
3. **场景三**: 第三个典型应用场景

### 4.2 案例分析

通过具体案例来理解「{keyword}」的实际应用。

### 4.3 应用建议

| 场景 | 适用情况 | 注意事项 |
|------|----------|----------|
| 基础应用 | 入门学习和简单场景 | 掌握基础概念 |
| 进阶应用 | 复杂场景和专业领域 | 深入理解原理 |
| 高级应用 | 创新和前沿领域 | 持续学习更新 |

---

## 五、关联知识与进阶

### 5.1 前置知识

学习「{keyword}」之前，建议先了解：

- 基础知识点 1
- 基础知识点 2
- 基础知识点 3

### 5.2 关联概念

与「{keyword}」相关的其他重要概念：

{chr(10).join([f'- **{kw}**: 与主题相关的概念' for kw in (analysis.keywords[6:10] if len(analysis.keywords) > 6 else [])])}

### 5.3 进阶学习路径

| 阶段 | 学习内容 | 建议时长 |
|------|----------|----------|
| **入门** | 基础概念和定义 | 1-2周 |
| **进阶** | 深入原理和应用 | 2-4周 |
| **精通** | 高级应用和创新 | 持续学习 |

---

## 六、总结与学习建议

### 6.1 核心要点

1. **概念定义**: 「{keyword}」的核心含义
2. **关键原理**: 最重要的工作原理
3. **主要应用**: 典型的应用场景
4. **学习难度**: {'入门级，适合初学者' if analysis.sentiment_score < -0.3 else '中级，需要一定基础' if analysis.sentiment_score < 0.3 else '高级，需要扎实基础'}

### 6.2 学习建议

| 阶段 | 建议 | 目标 |
|------|------|------|
| **第一步** | 理解基本概念和定义 | 知道是什么 |
| **第二步** | 学习核心原理和机制 | 理解为什么 |
| **第三步** | 实践应用和案例分析 | 掌握怎么用 |

### 6.3 推荐资源

- 继续深入学习相关知识
- 查阅更多专业资料
- 动手实践加深理解

---

**文档生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**资料来源数**: {data_count} 条
**整理方法**: 多平台资料采集 + AI 智能整理

---

### 附录：参考资料

以下为本次整理的部分参考资料：

{chr(10).join([f'{i+1}. [{item.title[:50]}...]({item.url}) - {item.source} (相关性: {item.relevance_score:.0%})' for i, item in enumerate(state.raw_data[:10])])}

---
*本文档由 AI 搜学助手自动生成，仅供参考。*
*⚠️ 内容由AI生成，仅供参考，您据此所作判断及操作均由您自行承担责任。*
"""


# =============================================================================
# 工具：渲染 HTML
# =============================================================================

@opinion_agent.tool
async def render_html(
    ctx: RunContext[StateDeps],
) -> str:
    """
    将 Markdown 文档渲染为 HTML，包含可视化图表
    
    Returns:
        渲染结果描述
    """
    state = ctx.deps.state
    run_id = ctx.deps.run_id  # 从 deps 获取运行 ID

    state.status = "rendering"
    state.current_phase = "HTML 渲染"
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎨 渲染知识讲解文档...")

    # 推送初始状态
    await push_state_event(run_id, state)

    try:
        import markdown
        report_html = markdown.markdown(
            state.report_text,
            extensions=["extra", "tables", "toc"]
        )
    except ImportError:
        report_html = state.report_text.replace("\n\n", "</p><p>").replace("\n", "<br>")
        report_html = f"<p>{report_html}</p>"

    # 准备图表数据
    analysis = state.analysis or AnalysisResult()

    # 内容分布数据（原情感分布）
    sentiment_data = analysis.sentiment_distribution or {"定义": 33, "原理": 34, "应用": 33}
    sentiment_colors = {
        "定义": "#52c41a",
        "原理": "#1890ff", 
        "应用": "#ff4d4f"
    }

    # 热度趋势数据
    heat_trend = analysis.heat_trend or [50, 50, 50, 50, 50, 50, 50]
    heat_labels = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    # 关键词数据（用于词云）
    keywords = analysis.keywords or [state.keyword]
    keyword_weights = []
    for i, kw in enumerate(keywords[:20]):
        weight = 100 - i * 5  # 权重递减
        keyword_weights.append({"name": kw, "value": max(weight, 20)})

    # 来源分布数据
    source_stats = {}
    for item in state.raw_data:
        source_stats[item.source] = source_stats.get(item.source, 0) + 1

    state.final_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{state.keyword} - AI 搜学助手</title>
    <!-- 所有链接在新窗口打开 -->
    <base target="_blank">
    <script>
        // 动态加载 ECharts（支持 iframe 嵌入和独立打开）
        (function() {{
            var baseUrl = '';
            try {{
                // 如果在 iframe 中，使用父页面的 origin
                if (window.parent && window.parent.location && window.parent !== window) {{
                    baseUrl = window.parent.location.origin;
                }}
            }} catch(e) {{
                // 跨域情况下使用当前页面的 origin
                baseUrl = window.location.origin || '';
            }}
            
            function loadScript(url) {{
                return new Promise(function(resolve, reject) {{
                    var script = document.createElement('script');
                    script.src = baseUrl + url;
                    script.onload = resolve;
                    script.onerror = function() {{
                        // 如果本地加载失败，尝试 CDN
                        var fallbackUrl = url.includes('wordcloud') 
                            ? 'https://cdn.jsdelivr.net/npm/echarts-wordcloud@2.1.0/dist/echarts-wordcloud.min.js'
                            : 'https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js';
                        var fallbackScript = document.createElement('script');
                        fallbackScript.src = fallbackUrl;
                        fallbackScript.onload = resolve;
                        fallbackScript.onerror = reject;
                        document.head.appendChild(fallbackScript);
                    }};
                    document.head.appendChild(script);
                }});
            }}
            
            // 按顺序加载 ECharts
            loadScript('/echarts/echarts.min.js').then(function() {{
                return loadScript('/echarts/echarts-wordcloud.min.js');
            }}).then(function() {{
                window.dispatchEvent(new CustomEvent('echarts-ready'));
            }}).catch(function(e) {{
                console.error('Failed to load ECharts:', e);
            }});
        }})();
    </script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 40px 20px;
            line-height: 1.8;
        }}
        .container {{ 
            max-width: 1100px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.98); 
            padding: 50px; 
            border-radius: 16px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .disclaimer {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 24px;
            font-size: 14px;
            color: #856404;
        }}
        h1 {{ 
            color: #1a202c; 
            border-bottom: 3px solid #667eea; 
            padding-bottom: 15px; 
            margin-bottom: 30px;
            font-size: 2em;
        }}
        h2 {{
            color: #2d3748;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        h3 {{
            color: #4a5568;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        p {{ margin-bottom: 16px; color: #4a5568; }}
        ul, ol {{ margin-left: 24px; margin-bottom: 16px; color: #4a5568; }}
        li {{ margin-bottom: 8px; }}
        strong {{ color: #2d3748; }}
        blockquote {{
            background: #f7fafc;
            border-left: 4px solid #667eea;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #f7fafc;
            font-weight: 600;
        }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 30px 0;
        }}
        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        /* 图表区域样式 */
        .charts-section {{
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 12px;
            padding: 30px;
            margin: 30px 0;
        }}
        .charts-title {{
            text-align: center;
            color: #2d3748;
            font-size: 1.5em;
            margin-bottom: 25px;
            font-weight: 600;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .chart-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}
        .chart-card-title {{
            text-align: center;
            color: #4a5568;
            font-size: 1em;
            margin-bottom: 15px;
            font-weight: 500;
        }}
        .chart-container {{
            width: 100%;
            height: 280px;
        }}
        .wordcloud-container {{
            grid-column: span 2;
        }}
        .wordcloud-chart {{
            height: 320px;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .wordcloud-container {{
                grid-column: span 1;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="disclaimer">
            ⚠️ <strong>免责声明</strong>：内容由AI生成，仅供参考，您据此所作判断及操作均由您自行承担责任。
        </div>
        
        <!-- 文档正文 -->
        {report_html}
        
        <!-- 可视化图表区域（放在文章后面） -->
        <div class="charts-section">
            <div class="charts-title">📊 知识结构可视化</div>
            <div class="charts-grid">
                <!-- 内容分布饼图 -->
                <div class="chart-card">
                    <div class="chart-card-title">内容类型分布</div>
                    <div id="sentimentChart" class="chart-container"></div>
                </div>
                
                <!-- 学习热度折线图 -->
                <div class="chart-card">
                    <div class="chart-card-title">学习热度趋势</div>
                    <div id="heatChart" class="chart-container"></div>
                </div>
                
                <!-- 来源分布柱状图 -->
                <div class="chart-card">
                    <div class="chart-card-title">资料来源分布</div>
                    <div id="sourceChart" class="chart-container"></div>
                </div>
                
                <!-- 难度评估仪表盘 -->
                <div class="chart-card">
                    <div class="chart-card-title">学习难度评估</div>
                    <div id="riskChart" class="chart-container"></div>
                </div>
                
                <!-- 核心概念词云 -->
                <div class="chart-card wordcloud-container">
                    <div class="chart-card-title">核心概念词云</div>
                    <div id="wordcloudChart" class="chart-container wordcloud-chart"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 等待 ECharts 加载完成后再初始化图表
        function initCharts() {{
            if (typeof echarts === 'undefined') {{
                // ECharts 还未加载，等待后重试
                setTimeout(initCharts, 100);
                return;
            }}
            
            // 内容分布饼图
            var sentimentChart = echarts.init(document.getElementById('sentimentChart'));
        sentimentChart.setOption({{
            tooltip: {{ trigger: 'item', formatter: '{{b}}: {{c}}% ({{d}}%)' }},
            legend: {{ bottom: '5%', left: 'center' }},
            series: [{{
                type: 'pie',
                radius: ['40%', '70%'],
                avoidLabelOverlap: false,
                itemStyle: {{ borderRadius: 10, borderColor: '#fff', borderWidth: 2 }},
                label: {{ show: false, position: 'center' }},
                emphasis: {{
                    label: {{ show: true, fontSize: 20, fontWeight: 'bold' }}
                }},
                labelLine: {{ show: false }},
                data: [
                    {{ value: {sentiment_data.get('定义', 33)}, name: '定义', itemStyle: {{ color: '#52c41a' }} }},
                    {{ value: {sentiment_data.get('原理', 34)}, name: '原理', itemStyle: {{ color: '#1890ff' }} }},
                    {{ value: {sentiment_data.get('应用', 33)}, name: '应用', itemStyle: {{ color: '#ff4d4f' }} }}
                ]
            }}]
        }});
        
        // 热度趋势折线图
        var heatChart = echarts.init(document.getElementById('heatChart'));
        heatChart.setOption({{
            tooltip: {{ trigger: 'axis' }},
            xAxis: {{ type: 'category', data: {json.dumps(heat_labels)}, boundaryGap: false }},
            yAxis: {{ type: 'value', min: 0, max: 100 }},
            series: [{{
                type: 'line',
                smooth: true,
                data: {json.dumps(heat_trend)},
                areaStyle: {{
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: 'rgba(102, 126, 234, 0.5)' }},
                        {{ offset: 1, color: 'rgba(102, 126, 234, 0.05)' }}
                    ])
                }},
                lineStyle: {{ color: '#667eea', width: 3 }},
                itemStyle: {{ color: '#667eea' }}
            }}]
        }});
        
        // 来源分布柱状图
        var sourceChart = echarts.init(document.getElementById('sourceChart'));
        sourceChart.setOption({{
            tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
            xAxis: {{ type: 'category', data: {json.dumps(list(source_stats.keys()))} }},
            yAxis: {{ type: 'value' }},
            series: [{{
                type: 'bar',
                data: {json.dumps(list(source_stats.values()))},
                itemStyle: {{
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: '#667eea' }},
                        {{ offset: 1, color: '#764ba2' }}
                    ]),
                    borderRadius: [5, 5, 0, 0]
                }}
            }}]
        }});
        
        // 难度评估仪表盘
        var riskChart = echarts.init(document.getElementById('riskChart'));
        var difficultyScore = {analysis.sentiment_score * 50 + 50};  // 转换为 0-100，-1为入门(0)，0为中级(50)，1为高级(100)
        var difficultyColor = difficultyScore < 40 ? '#52c41a' : (difficultyScore < 60 ? '#faad14' : '#ff4d4f');
        riskChart.setOption({{
            series: [{{
                type: 'gauge',
                startAngle: 180,
                endAngle: 0,
                min: 0,
                max: 100,
                splitNumber: 5,
                itemStyle: {{ color: difficultyColor }},
                progress: {{ show: true, width: 20 }},
                pointer: {{ show: false }},
                axisLine: {{ lineStyle: {{ width: 20 }} }},
                axisTick: {{ show: false }},
                splitLine: {{ show: false }},
                axisLabel: {{ show: false }},
                title: {{ show: false }},
                detail: {{
                    valueAnimation: true,
                    fontSize: 28,
                    offsetCenter: [0, '0%'],
                    formatter: function(value) {{
                        if (value < 40) return '入门级';
                        if (value < 60) return '中级';
                        return '高级';
                    }},
                    color: difficultyColor
                }},
                data: [{{ value: difficultyScore }}]
            }}]
        }});
        
        // 关键词词云
        var wordcloudChart = echarts.init(document.getElementById('wordcloudChart'));
        wordcloudChart.setOption({{
            series: [{{
                type: 'wordCloud',
                shape: 'circle',
                left: 'center',
                top: 'center',
                width: '90%',
                height: '90%',
                sizeRange: [14, 60],
                rotationRange: [-45, 45],
                gridSize: 8,
                drawOutOfBound: false,
                textStyle: {{
                    fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif',
                    fontWeight: 'bold',
                    color: function() {{
                        var colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'];
                        return colors[Math.floor(Math.random() * colors.length)];
                    }}
                }},
                data: {json.dumps(keyword_weights)}
            }}]
        }});
        
            // 响应式调整
            window.addEventListener('resize', function() {{
                sentimentChart.resize();
                heatChart.resize();
                sourceChart.resize();
                riskChart.resize();
                wordcloudChart.resize();
            }});
        }}
        
        // 页面加载完成后初始化图表
        // 监听 echarts-ready 事件（优先）
        window.addEventListener('echarts-ready', initCharts);
        
        // 同时检查 ECharts 是否已加载（用于独立打开 HTML 的情况）
        if (document.readyState === 'complete') {{
            setTimeout(initCharts, 500);
        }} else {{
            window.addEventListener('load', function() {{
                setTimeout(initCharts, 500);
            }});
        }}
    </script>
</body>
</html>"""

    state.status = "complete"
    state.current_phase = "整理完成"
    state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 知识整理完成")

    # 发送最终状态
    await push_state_event(run_id, state)

    return "HTML 渲染完成"
