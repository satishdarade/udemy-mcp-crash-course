import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Configure the LLM to use NVIDIA's inference API
# You need to set NVIDIA_API_KEY in your .env file or environment
# Update below variables in your .env file or environment
# NVIDIA_API_KEY
# LANGCHAIN_API_KEY
# LANGCHAIN_TRACING_V2
# LANGCHAIN_ENDPOINT
# LANGCHAIN_PROJECT
llm = ChatOpenAI(
    base_url="https://inference-api.nvidia.com/v1/",
    api_key=os.environ.get("NVIDIA_API_KEY"),
    model="openai/openai/gpt-5-nano",
)

stdio_server_param = StdioServerParameters(
    command="python",
    args=["/colossus/mcps/udemy-mcp-crash-course/servers/math_server.py"],
)


async def main():
    async with stdio_client(stdio_server_param) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("Session initialized")
            tools = await load_mcp_tools(session)
            agent = create_agent(llm, tools)
            res = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="what is 1 + 1 * 54? using the math tool.")
                    ]
                }
            )
            print(f"Final Output: {res['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
