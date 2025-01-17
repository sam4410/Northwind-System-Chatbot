import os
from chains.northwind_cypher_chain import northwind_cypher_chain
from chains.northwind_review_chain import reviews_vector_chain
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI

NORTHWIND_AGENT_MODEL = os.getenv("NORTHWIND_AGENT_MODEL")

northwind_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about customer experiences, feelings, or any other qualitative
        question that could be answered about a customer using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are customers satisfied with their purchased products and staff services?", the input should be
        "Are customers satisfied with their purchased products and staff services?".
        """,
    ),
    Tool(
        name="Graph",
        func=northwind_cypher_chain.invoke,
        description="""Useful for answering questions about customers,
        products, suppliers, product categories, customer review
        statistics, and order details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many orders
        have there been in year 2012?", the input should be "How many orders have
        there been in year 2012?".
        """,
    ),
]

chat_model = ChatOpenAI(
    model=NORTHWIND_AGENT_MODEL,
    temperature=0,
)

northwind_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=northwind_agent_prompt,
    tools=tools,
)

northwind_rag_agent_executor = AgentExecutor(
    agent=northwind_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
