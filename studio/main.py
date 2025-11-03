from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from . import schemas, nodes
from .utils import check_env


def verify_env_vars():
    check_env("OPENAI_API_KEY")
    check_env("LANGSMITH_API_KEY")
    check_env("TAVILY_API_KEY")


def get_interview_builder():
    """Sub graph to do an interview"""
    interview_builder = StateGraph(schemas.InterviewState)
    interview_builder.add_node("ask_question", nodes.generate_question)
    interview_builder.add_node("search_web", nodes.search_web)
    interview_builder.add_node("search_wikipedia", nodes.search_wikipedia)
    interview_builder.add_node("answer_question", nodes.generate_answer)
    interview_builder.add_node("save_interview", nodes.save_interview)
    interview_builder.add_node("write_section", nodes.write_section)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_conditional_edges(
        "ask_question",
        nodes.route_messages,
        path_map=["search_web", "search_wikipedia", "save_interview"],
    )
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("answer_question", "ask_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    return interview_builder


def build_graph():
    builder = StateGraph(schemas.WholeGraphState)
    interview_builder = get_interview_builder()

    builder.add_node("create_analysts", nodes.create_analysts)
    builder.add_node("human_feedback", nodes.get_human_feedback)
    builder.add_node("conduct_interview", interview_builder.compile())
    builder.add_node("write_report", nodes.write_report)
    builder.add_node("write_introduction", nodes.write_introduction)
    builder.add_node("write_conclusion", nodes.write_conclusion)
    builder.add_node("finalize_report", nodes.finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback",
        nodes.initiate_all_interviews,
        ["create_analysts", "conduct_interview"],
    )
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(
        ["write_introduction", "write_conclusion", "write_report"], "finalize_report"
    )
    builder.add_edge("finalize_report", END)

    # Compile
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
    return graph


def initate_state(
    topic: str | None = None, max_analysts: int = 2, max_sources: int = 3
):
    graph_state = schemas.WholeGraphState(
        max_analysts=max_analysts,
        topic=topic or "The adoption of AI in SMEs in Africa, with a focus on Kenya",
        human_analyst_feedback=None,
        analysts=[],
        max_num_of_sources=max_sources,
    )
    return graph_state


def run_analysts(
    graph, graph_state: schemas.WholeGraphState | None, thread: RunnableConfig
):
    for event in graph.stream(graph_state, config=thread, stream_mode="values"):
        analysts = event.get("analysts", "")
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)


def update_human_feedback(graph, human_feedback: str | None, thread: RunnableConfig):
    graph.update_state(
        thread,
        {"human_analyst_feedback": human_feedback},
        as_node="human_feedback",
    )


def run_rest_of_graph(graph, thread: RunnableConfig):
    for event in graph.stream(None, thread, stream_mode="updates"):
        print("--Node--")
        node_name = next(iter(event.keys()))
        print(node_name)


graph = build_graph()


if __name__ == "__main__":
    graph = build_graph()
    thread = RunnableConfig(configurable={"thread_id": "1"})
    topic = input("\nInput a topic for research::")
    max_analysts = int(input("\nInput the number of analysts you want::"))
    max_sources = int(
        input(
            "\nInput the max number of sources you want to reference per analyst(recommend 1-3)::"
        )
    )

    graph_state = initate_state(
        topic=topic, max_analysts=max_analysts, max_sources=max_sources
    )
    run_analysts(graph, graph_state, thread)
    human_feedback = input(
        "\n\n\nInput any feedback you want to carry out. Type None if all good to proceed::"
    )

    while human_feedback.strip().lower() != "none":
        update_human_feedback(
            graph=graph,
            human_feedback=human_feedback,
            thread=thread,
        )
        run_analysts(graph, None, thread)
        human_feedback = input(
            "\n\n\nAny feedback you want to carry out. Type None if all good to proceed::"
        )

    update_human_feedback(graph, None, thread)
    run_rest_of_graph(graph, thread)
