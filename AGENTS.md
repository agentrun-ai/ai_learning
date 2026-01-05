# AI 搜学助手 - 架构文档

## 📋 目录
1. [系统概述](#系统概述)
2. [核心架构](#核心架构)
3. [数据模型](#数据模型)
4. [流程设计](#流程设计)
5. [技术实现](#技术实现)
6. [配置说明](#配置说明)

---

## 系统概述

### 功能定位
AI 搜学助手系统，提供：
- **资料收集**: 多平台学习资料抓取（百科、知乎、B站教程、技术博客等）
- **知识分析**: 核心概念提取、难度评估、知识结构梳理
- **文档生成**: 通俗易懂的知识讲解文档
- **可视化**: 图文并茂的 HTML 文档

### 质量标准
- **通俗易懂**: 循序渐进，适合零基础读者
- **可靠性**: 基于真实资料，不编造内容
- **实时性**: 前端实时显示收集和分析进度

---

## 核心架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                   用户输入学习概念                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Opinion Agent                             │
│                   代码控制流程流转                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段 1: 资料收集 (collect_data)                            │
│  ┌──────────────────────────────────────┐                  │
│  │ - 多平台搜索（百科、知乎、B站等）      │                  │
│  │ - 严格保证资料量                      │                  │
│  │ - 实时更新前端                        │                  │
│  │ - VNC 浏览器预览                      │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  阶段 2: 知识分析 (analyze_data)                            │
│  ┌──────────────────────────────────────┐                  │
│  │ - 核心概念提取                        │                  │
│  │ - 难度评估                            │                  │
│  │ - 学习建议                            │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  阶段 3: 文档撰写 (write_report)                            │
│  ┌──────────────────────────────────────┐                  │
│  │ - 3000-5000 字知识讲解                │                  │
│  │ - 6 部分结构化内容                    │                  │
│  │ - 通俗易懂的教育标准                  │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  阶段 4: HTML 渲染 (render_html)                            │
│  ┌──────────────────────────────────────┐                  │
│  │ - Markdown → HTML                    │                  │
│  │ - 精美样式                            │                  │
│  │ - 免责声明                            │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   前端实时显示         │
            │  - 资料列表 (逐条)     │
            │  - 进度条 (实时)       │
            │  - VNC 预览           │
            │  - 最终 HTML 文档      │
            └───────────────────────┘
```

### 设计原则

1. **代码控制流程**: 不依赖 LLM 自主决策，通过代码严格控制每个阶段
2. **严格资料收集**: 必须收集到目标数量，不足时自动补充搜索
3. **实时状态同步**: 每个工具返回 StateSnapshotEvent，前端立即更新
4. **多 Sandbox 支持**: 支持多个浏览器沙箱并行工作

---

## 数据模型

### SearchResult
```python
class SearchResult(BaseModel):
    title: str          # 标题
    url: str            # 链接
    snippet: str        # 摘要
    source: str         # 来源（百科/知乎/B站等）
    date: str           # 日期
    platform: str       # 搜索平台
```

### AnalysisResult
```python
class AnalysisResult(BaseModel):
    keywords: List[str]                    # 核心概念列表
    sentiment_score: float                 # 难度得分 (-1到1)
    sentiment_distribution: Dict[str, int] # 内容分布（定义/原理/应用）
    heat_trend: List[int]                  # 学习热度趋势
    summary: str                           # 知识概述
    key_opinions: List[Dict[str, str]]     # 关键知识点
    risk_assessment: Dict[str, str]        # 学习建议
```

### OpinionState
```python
class OpinionState(BaseModel):
    keyword: str = ""                      # 学习概念
    status: str = "idle"                   # 当前状态
    logs: List[str] = []                   # 日志列表
    max_results: int = 20                  # 最大收集数量
    
    raw_data: List[SearchResult] = []      # 原始资料
    collected_data_summary: List[Dict] = [] # 前端摘要
    analysis: Optional[AnalysisResult]     # 分析结果
    report_text: str = ""                  # Markdown 文档
    final_html: str = ""                   # 最终 HTML
    
    collection_progress: int = 0           # 收集进度
    current_phase: str = ""                # 当前阶段
    sandboxes: List[SandboxInfo] = []      # Sandbox 列表
    active_sandbox_id: str = ""            # 当前活动 Sandbox
```

---

## 流程设计

### 资料收集策略

多平台搜索查询模板：
```python
SEARCH_TEMPLATES = {
    "definition": ["{keyword} 是什么", "{keyword} 定义"],
    "principle": ["{keyword} 原理", "{keyword} 工作原理"],
    "tutorial": ["{keyword} 教程", "{keyword} 入门"],
    "zhihu": ["{keyword} 知乎科普"],
    "bilibili": ["{keyword} B站教程"],
    "blog": ["{keyword} CSDN", "{keyword} 技术博客"],
    "example": ["{keyword} 案例", "{keyword} 应用"],
}
```

### 资料收集保证

```python
# 持续搜索直到达到目标数量
while len(collected) < target_count:
    # 执行搜索...
    
    # 如果用完预定义查询，生成补充查询
    if query_index >= len(queries):
        if retry_count >= max_retries:
            break
        retry_count += 1
        # 生成补充查询...
```

### 文档结构

6 部分知识讲解文档：

**核心要点**：
1. 概念定义与简介
2. 核心原理与机制
3. 详细解释与深入理解
4. 实际应用与案例分析

**文档章节**：
1. **概念简介**: 资料来源 + 基本定义 + 学习概览
2. **核心原理**: 基本定义 + 工作原理 + 关键概念
3. **详细讲解**: 深入理解 + 常见问题 + 重点难点
4. **应用案例**: 应用场景 + 案例分析 + 应用建议
5. **关联知识与进阶**: 前置知识 + 关联概念 + 学习路径
6. **总结与学习建议**: 核心要点 + 学习建议 + 推荐资源

---

## 技术实现

### 后端 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/agent` | POST | AG-UI 主端点 |
| `/api/browser/vnc` | GET | 获取 VNC URL |
| `/api/browser/sandboxes` | GET | 获取所有 Sandbox |
| `/api/browser/screenshot` | GET | 获取浏览器截图 |

### 状态同步机制

每个工具返回 `StateSnapshotEvent`，确保前端实时更新：

```python
@opinion_agent.tool
async def collect_data(ctx, keyword):
    # ... 收集逻辑 ...
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT, 
        snapshot=state
    )
```

### 前端状态管理

使用 `useAgentState` Hook 管理状态：

```typescript
const { state, running, sendMessage } = useAgentState<AgentState>({
    name: 'opinion_agent',
    agentUrl: 'http://localhost:8000/api/agent',
    initialState: { ... }
});
```

---

## 配置说明

### 环境变量

```bash
# 必需
AGENTRUN_MODEL_NAME=your-model-name
AGENTRUN_BROWSER_SANDBOX_NAME=your-sandbox-name
```

### 前端配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 最大资料数量 | 20 | 范围 5-100 |

### 超时设置

```python
config = Config(timeout=180)  # 3分钟超时
```

---

## 免责声明

**内容由AI生成，仅供参考，您据此所作判断及操作均由您自行承担责任。**

---

**文档版本**: 2.0  
**最后更新**: 2026-01-05
