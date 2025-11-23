# 参考URL
# https://arpable.com/artificial-intelligence/agent/langgraph-ai-agent/

import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS

# =========================
# 環境変数設定
# =========================
load_dotenv("/lg_agent/.env", override=True)

# =========================
# 各オブジェクト初期化
# =========================
query_extractor_llm = None
vector_db = None
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")
search_necessity_llm = ChatOpenAI(model="gpt-5-nano-2025-08-07") # 軽量モデルが推奨
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = FAISS.load_local(
    folder_path="/lg_agent/app/faiss_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# =========================
# RAGシステムの状態定義
# =========================
class RAGState(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    query: str
    search_results: List[Document]
    context: str

# =========================
# RAGの各処理ノード(関数として定義)
# =========================
def extract_query(state: RAGState, config: RunnableConfig) -> Dict:
    """ユーザー入力(query)を取得"""
    print("質問してください。")
    user_input = str(input())
    state['messages'].append(HumanMessage(user_input))
    state['query'] = user_input

def perform_search(state: RAGState, config: RunnableConfig) -> Dict:
    """検索"""
    last_message = state["messages"][-1].content
    search_result = retriever.invoke(last_message)
    state["search_results"].append(search_result)

def prepare_context(state: RAGState, config: RunnableConfig) -> Dict:
    """検索結果を加えた指示prompt作成"""
    prompt_template = """
    ユーザー質問に対して検索したcontext情報を参照して回答してください。
    ユーザー質問: {user_input}
    Context: {context}
    """
    params = {"user_input": state['query'], "context": state['search_results']}
    prompt = prompt_template.format(**params)
    state['messages'].append(SystemMessage(prompt))

def generate_answer(state: RAGState, config: RunnableConfig) -> Dict:
    """検索結果を踏まえた回答生成"""
    last_message = state['messages'][-1].content
    response = llm.invoke(last_message)
    print(response.content)

def direct_answer(state: RAGState, config: RunnableConfig) -> Dict:
    """LLMの知識だけで回答するノード"""
    last_message = state["messages"][-1].content
    response = llm.invoke(f"次の内容について回答してください。: {last_message}")
    print(response.content)

# =========================
# グラフビルダー初期化・ノードの追加
# =========================
graph_builder = StateGraph(RAGState)
graph_builder.add_node("extract_query_node", extract_query)
graph_builder.add_node("perform_search_node", perform_search)
graph_builder.add_node("prepare_context_node", prepare_context)
graph_builder.add_node("generate_answer_node", generate_answer)
graph_builder.add_node("direct_answer_node", direct_answer)

graph_builder.set_entry_point("extract_query_node")

# =========================
# 検索の必要性を判断するルーター関数
# =========================
def search_router(state: RAGState) -> str:
    last_message = state["messages"][-1].content
    needs_search_response = search_necessity_llm.invoke(
        f"この質問に答えるために外部情報の検索が必要ですか？ はい/いいえ で答えてください：{last_message}"
    )

    # LLMの応答の揺らぎを考慮
    if "はい" in needs_search_response.content or "必要" in needs_search_response.content:
        print(">>> 判断: 検索が必要")
        state['messages'].append(AIMessage(">>> 判断: 検索が必要"))
        return "perform_search"
    else:
        print(">>> 判断: 検索は不要")
        state['messages'].append(AIMessage(">>> 判断: 検索は不要"))
        return "direct_answer"

# =========================
# 条件付きエッジの追加
# =========================
graph_builder.add_conditional_edges(
    "extract_query_node",
    search_router,
    {
        "direct_answer": "direct_answer_node",
        "perform_search": "perform_search_node"
    }
)

graph_builder.add_edge("direct_answer_node", END)
graph_builder.add_edge("perform_search_node", "prepare_context_node")
graph_builder.add_edge("prepare_context_node", "generate_answer_node")
graph_builder.add_edge("generate_answer_node", END)

# =========================
# グラフのコンパイル・実行
# =========================
graph = graph_builder.compile()

graph.invoke(
    {
        "messages": [],
        "query": "",
        "search_results": [],
        "context": ""
    }, 
    debug=True
)

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png(output_file_path="/lg_agent/lang_graph_practice/lang_graph_rag.png")))