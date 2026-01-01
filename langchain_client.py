import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    base_url="https://inference-api.nvidia.com/v1/",
    api_key=os.environ.get("NVIDIA_API_KEY"),
    model="openai/openai/gpt-5-nano",
)


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "/colossus/mcps/udemy-mcp-crash-course/.venv/bin/python",
                "args": [
                    "/colossus/mcps/udemy-mcp-crash-course/servers/math_server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    )

    # In newer versions, we access tools via client.get_tools() but not as a context manager
    tools = await client.get_tools()
    agent = create_agent(llm, tools)

    # We need to ensure we close sessions properly if the client supports it,
    # but based on the error, direct context manager is not supported.
    # The client might manage sessions internally or we just use get_tools().

    try:
        res = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="what is the weather in Tokyo? using the weather tool."
                    )
                ]
            }
        )
        print(f"Final Output: {res['messages'][-1].content}")
    finally:
        # Check if there is a close or cleanup method if needed
        pass


if __name__ == "__main__":
    asyncio.run(main())
