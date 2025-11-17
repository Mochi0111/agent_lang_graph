# 参考URL
# https://arpable.com/artificial-intelligence/agent/langgraph-ai-agent/

from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

# =========================
# 各オブジェクト初期化
# =========================
query_extractor_llm = None
vector_db = None
llm = None
search_necessity_llm = None # 軽量モデルが推奨

# =========================
# RAGシステムの状態定義
# =========================
class RAGState(TypedDict):
    messages: List[BaseMessage]
    query: str
    search_results: List[Document]
    context: str

# =========================
# RAGの各処理ノード(関数として定義)
def extract_query(state: RAGState, config: RunnableConfig) -> Dict:
    # 自身で実装
    return {
        "messages": [],
        "query": "",
        "search_results": [],
        "context": "",
    }

def perform_search(state: RAGState, config: RunnableConfig) -> Dict:
    # 自身で実装
    pass

def prepare_context(state: RAGState, config: RunnableConfig) -> Dict:
    # 自身で実装
    pass

def generate_answer(state: RAGState, config: RunnableConfig) -> Dict:
    #自身で実装
    pass

def direct_answer(state: RAGState, config: RunnableConfig) -> Dict:
    # LLMの知識だけで回答するノード
    pass

# ================================
# 検索の必要性を判断するルーター関数
# ================================
def search_router(state: RAGState) -> str:
    last_message = state["messages"][-1].content
    needs_search_response = search_necessity_llm.invoke(
        f"この質問に答えるために外部情報の検索が必要ですか？ はい/いいえ で答えてください：{last_message}"
    )

    # LLMの応答の揺らぎを考慮
    if "はい" in needs_search_response.content or "必要" in needs_search_response.content:
        print(">>> 判断: 検索が必要")
        return "extract_query"
    else:
        print(">>> 判断: 検索は不要")
        return "direct_answer"
    

# ===============================
# 