from agents.northwind_rag_agent import northwind_rag_agent_executor
from fastapi import FastAPI
from models.northwind_rag_query import NorthwindQueryInput, NorthwindQueryOutput
from utils.async_utils import async_retry
import uvicorn
import asyncio

app = FastAPI(
    title="Northwind Chatbot",
    description="Endpoints for a northwind system graph RAG chatbot",
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await northwind_rag_agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/northwind-rag-agent")
async def query_northwind_agent(
    query: NorthwindQueryInput,
) -> NorthwindQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response

async def main():
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())