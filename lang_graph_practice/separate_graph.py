from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
import random

class State(TypedDict):
    value: str

def start_node(state: State, config: RunnableConfig):
    return {"value": "1"}

def node2(state: State, config: RunnableConfig):
    return {"value": "2"}

def node3(state: State, config: RunnableConfig):
    return {"value": "3"}


graph_builder = StateGraph(State)
graph_builder.add_node("start_node", start_node)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("end_node", lambda state: {"value": state["value"]})

graph_builder.set_entry_point("start_node")

def routing(state: State, config: RunnableConfig) -> Literal["node2", "node3"]:
    random_num = random.randint(0, 1)
    if random_num == 0:
        return "node2"
    else:
        return "node3"
    
# 第一引数には、一つ前のNodeを指定する
# 第二引数には、分岐を決定する関数を指定する
graph_builder.add_conditional_edges(
    "start_node",
    routing,
)

graph_builder.add_edge("node2", "end_node")
graph_builder.add_edge("node3", "end_node")

graph_builder.set_finish_point("end_node")

# gRAPHをコンパイル
graph = graph_builder.compile()

# Graphの実行(引数にはStateの初期値を渡す)
graph.invoke({"value": ""}, debug=True)


# Graphの可視化
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png(output_file_path="/lg_agent/lang_graph_practice/separate_graph.png")))