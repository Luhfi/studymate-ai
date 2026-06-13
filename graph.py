from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import chains


# STATE — Data yang mengalir di seluruh graph
class StudyMateState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # history chat
    intent: str            # hasil klasifikasi router
    search_results: str    # hasil pencarian web (jika ada)
    final_response: str    # respons final untuk ditampilkan


# NODE: Router — Tentukan intent dari input user
def router_node(state: StudyMateState) -> dict:
    last_message = state["messages"][-1]
    user_input = last_message.content

    router_chain = chains.get_router_chain()
    intent = router_chain.invoke({"input": user_input}).strip().lower()

    # Validasi intent
    valid_intents = ["research", "analysis", "writing", "flashcard", "qa"]
    if intent not in valid_intents:
        intent = "qa"  # fallback default

    print(f"[Router] Intent: '{intent}'")
    return {"intent": intent}


# NODE: Research — Cari di web, lalu sintesis
def research_node(state: StudyMateState) -> dict:
    user_input = state["messages"][-1].content

    # Coba web search
    try:
        search_tool = chains.get_search_tool()
        results = search_tool.invoke(user_input)
        search_text = "\n\n".join([
            f"[Sumber {i+1}]: {r.get('content', '')[:500]}"
            for i, r in enumerate(results[:3])
        ])
    except Exception as e:
        search_text = f"(Web search tidak tersedia: {e})"

    # Generate response
    research_chain = chains.get_research_chain()
    response = research_chain.invoke({
        "input": user_input,
        "search_results": search_text
    })

    return {
        "search_results": search_text,
        "final_response": response,
        "messages": [AIMessage(content=response)]
    }


# NODE: Analysis — Analisis teks
def analysis_node(state: StudyMateState) -> dict:
    user_input = state["messages"][-1].content
    analysis_chain = chains.get_analysis_chain()
    response = analysis_chain.invoke({"input": user_input})
    return {
        "final_response": response,
        "messages": [AIMessage(content=response)]
    }


# NODE: Writing — Buat konten
def writing_node(state: StudyMateState) -> dict:
    user_input = state["messages"][-1].content
    writing_chain = chains.get_writing_chain()
    response = writing_chain.invoke({"input": user_input})
    return {
        "final_response": response,
        "messages": [AIMessage(content=response)]
    }


# NODE: Flashcard — Buat kartu belajar
def flashcard_node(state: StudyMateState) -> dict:
    user_input = state["messages"][-1].content
    flashcard_chain = chains.get_flashcard_chain()
    response = flashcard_chain.invoke({"input": user_input})
    return {
        "final_response": response,
        "messages": [AIMessage(content=response)]
    }


# NODE: Q&A — Jawab pertanyaan langsung
def qa_node(state: StudyMateState) -> dict:
    user_input = state["messages"][-1].content
    qa_chain = chains.get_qa_chain()
    response = qa_chain.invoke({"input": user_input})
    return {
        "final_response": response,
        "messages": [AIMessage(content=response)]
    }


# CONDITIONAL EDGE — Tentukan node berikutnya
def route_to_agent(state: StudyMateState) -> str:
    return state["intent"]


# BUILD WORKFLOW — Rakit semua nodes dan edges
def build_workflow():
    graph = StateGraph(StudyMateState)

    # Daftarkan semua node
    graph.add_node("router",    router_node)
    graph.add_node("research",  research_node)
    graph.add_node("analysis",  analysis_node)
    graph.add_node("writing",   writing_node)
    graph.add_node("flashcard", flashcard_node)
    graph.add_node("qa",        qa_node)

    # Entry point: START → router
    graph.add_edge(START, "router")

    # Routing kondisional dari router ke agent yang tepat
    graph.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "research":  "research",
            "analysis":  "analysis",
            "writing":   "writing",
            "flashcard": "flashcard",
            "qa":        "qa",
        }
    )

    # Semua agent langsung ke END
    for node in ["research", "analysis", "writing", "flashcard", "qa"]:
        graph.add_edge(node, END)

    return graph.compile()


# Compile graph (dipakai di app.py)
studymate_graph = build_workflow()


# Visualisasi graph dalam format Mermaid (untuk debug/demo)
def get_mermaid_diagram() -> str:
    try:
        return studymate_graph.get_graph().draw_mermaid()
    except Exception:
        return ""