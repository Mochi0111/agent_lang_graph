from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

# Stateを宣言
class State(TypedDict):
    value: str

# Nodeを宣言
def node(state: State, config: RunnableConfig):
    # 更新するStateの値を返す
    return {"value": "1"}

def node2(state: State, config: RunnableConfig):
    return {"value": "2"}

# Graphの作成
graph_builder = StateGraph(State)

# Nodeの追加
graph_builder.add_node("node", node)
graph_builder.add_node("node2", node2)

# Nodeをedgeに追加
graph_builder.add_edge("node", "node2")

# Graphの視点を宣言
graph_builder.set_entry_point("node")

# Graphの終点を宣言
graph_builder.set_finish_point("node2")

# Graphをコンパイル
graph = graph_builder.compile()

# Graphの実行(引数にはStateの初期値を渡す)
graph.invoke({"value": ""}, debug=True)