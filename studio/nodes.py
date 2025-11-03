from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    get_buffer_string,
)
from dotenv import load_dotenv


import instructions
import schemas
from utils import save_markdown

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def create_analysts(state: schemas.WholeGraphState) -> dict:
    topic = state.topic
    max_analysts = state.max_analysts
    human_analyst_feedback = state.human_analyst_feedback or ""

    structured_llm = llm.with_structured_output(schemas.AnalystPersonas)
    system_message = instructions.analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    personas = structured_llm.invoke(
        [SystemMessage(system_message)]
        + [HumanMessage(content="Generate the analyst personas")]
    )

    return {"analysts": personas.analysts}


def get_human_feedback(state: schemas.WholeGraphState):
    pass


def generate_question(state: schemas.InterviewState) -> dict:

    analyst = state.analyst
    messages = state.messages

    system_message = instructions.question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}


def route_messages(state: schemas.InterviewState, name: str = "expert"):
    """Route between question and answer"""

    messages = state.messages
    max_num_turns = state.max_num_of_turns

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question
    # If the question content contains thank you so much for your help, it skips to save the interview
    last_question = messages[-1]
    if "Thank you so much for your help" in last_question.content:
        return "save_interview"

    return [Send("search_web", state), Send("search_wikipedia", state)]


def search_web(state: schemas.InterviewState) -> dict:
    """Search web using Tavily"""
    structured_llm = llm.with_structured_output(schemas.SearchQuery)
    output = structured_llm.invoke(
        [SystemMessage(content=instructions.search_instructions)] + state.messages
    )

    tavily_search = TavilySearch(max_results=state.max_num_of_sources)
    output = tavily_search.invoke({"query": output.search_query})
    search_results = output.get("results", output)

    formatted_search_results = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_results
        ]
    )

    return {"context": [formatted_search_results]}


def search_wikipedia(state: schemas.InterviewState) -> dict:
    """Search Wikipidea"""
    structured_llm = llm.with_structured_output(schemas.SearchQuery)
    output = structured_llm.invoke(
        [SystemMessage(content=instructions.search_instructions)] + state.messages
    )

    search_results = WikipediaLoader(
        query=output.search_query, load_max_docs=state.max_num_of_sources
    ).load()

    formatted_search_results = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_results
        ]
    )

    return {"context": [formatted_search_results]}


def generate_answer(state: schemas.InterviewState):
    """Answer a question"""
    # Get state
    analyst = state.analyst
    messages = state.messages
    context = state.context

    system_message = instructions.answer_instructions.format(
        goals=analyst.persona, context=context
    )
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    return {"messages": [answer]}


def save_interview(state: schemas.InterviewState):
    """Save interviews"""

    interview = get_buffer_string(state.messages)

    return {"interview": interview}


def write_section(state: schemas.InterviewState):
    """Node to answer a question"""

    interview = state.interview
    context = state.context
    analyst = state.analyst

    # Write section using either the both source docs and interview
    system_message = instructions.section_writer_instructions.format(
        focus=analyst.description
    )
    section = llm.invoke(
        [SystemMessage(content=system_message)]
        + [
            HumanMessage(
                content=f"Use this source to write your section: {context} {interview}"
            )
        ]
    )

    return {"sections": [section.content]}


def initiate_all_interviews(state: schemas.WholeGraphState):
    """Run each interview using send in parallel"""
    if state.human_analyst_feedback or not state.analysts:
        return "create_analysts"

    else:
        topic = state.topic
        return [
            Send(
                "conduct_interview",
                schemas.InterviewState(
                    analyst=analyst,
                    max_num_of_sources=state.max_num_of_sources,
                    messages=[
                        HumanMessage(
                            content=f"So you said you were writing an article on {topic}?"
                        )
                    ],
                ),
            )
            for analyst in state.analysts
        ]


def write_report(state: schemas.WholeGraphState):
    # Full set of sections
    sections = state.sections
    topic = state.topic

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = instructions.report_writer_instructions.format(
        topic=topic, context=formatted_str_sections
    )
    report = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Write a report based upon these memos.")]
    )
    return {"content": report.content}


def write_introduction(state: schemas.WholeGraphState):
    # Full set of sections
    sections = state.sections
    topic = state.topic

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    intro_instructions = instructions.intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke(
        [intro_instructions] + [HumanMessage(content=f"Write the report introduction")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: schemas.WholeGraphState):
    # Full set of sections
    sections = state.sections
    topic = state.topic

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    conc_instructions = instructions.intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke(
        [conc_instructions] + [HumanMessage(content=f"Write the report conclusion")]
    )
    return {"conclusion": conclusion.content}


import re
from pathlib import Path


def finalize_report(state: schemas.WholeGraphState):
    """The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""
    # Save full final report
    content = state.content
    if content and state.introduction and state.conclusion:
        if content.startswith("## Insights"):
            content = content.strip("## Insights")
        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except:
                sources = None
        else:
            sources = None

        final_report = (
            state.introduction
            + "\n\n---\n\n"
            + content
            + "\n\n---\n\n"
            + state.conclusion
        )
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources

        save_markdown(state.topic, final_report)
        print(final_report)

        return {"final_report": final_report}
    raise ValueError(f"Content, intro, or conclusion missing")
