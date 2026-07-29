from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from decouple import config
load_dotenv()
import os 


from langchain_huggingface import HuggingFaceEndpoint
llm = ChatOpenAI(
    model="cohere/north-mini-code:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPEN_ROUTER_API_KEY")
)

# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#     temperature=0.7,
#     max_new_tokens=512,
# )
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)