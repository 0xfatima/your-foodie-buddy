# Restaurant agent: connects to the MCP server, lets the LLM pick tools,
# and loops until it has a final answer (ReAct loop). No UI yet.
import os
import asyncio
from pathlib import Path
from fastmcp.client import Client, PythonStdioTransport
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

SERVER_SCRIPT = str(Path(__file__).parent / "server.py")

SYSTEM_PROMPT = """You are Connoisseur Companion, a friendly AI guide to California's restaurant scene.
You have access to a database of California restaurants through these tools:
- get_restaurant_info: look up a specific restaurant by name (cuisine, rating, price range, signature dish).
- recommend_by_vibe: find restaurants that match a mood or atmosphere, such as "moody", "romantic" or "zen".
- get_review: retrieve the detailed review of a restaurant.

Use the tools whenever a question is about specific restaurants, recommendations or reviews.
Base your answers only on what the tools return; if nothing is found, say so and suggest another search.
Keep answers short, clear and conversational."""

project_id = os.environ.get("WATSONX_AI_PROJECT_ID") or os.environ.get("WATSONX_PROJECT_ID")


def make_model():
    return ChatWatsonx(
        model_id="ibm/granite-4-h-small",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params={"temperature": 0.7},
    )


async def chat_with_agent(user_message: str, history: list) -> str:
    """Discover the server's tools, then call them until the LLM gives a text reply."""
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        # 1. Discover tools and convert them to the format the LLM expects
        mcp_tools = await client.list_tools()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]
        model = make_model().bind_tools(openai_tools)

        # 2. Rebuild the conversation
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        # 3. ReAct loop
        for _ in range(10):
            response = await model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
                    )
                return str(raw)

            for tool_call in response.tool_calls:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = " ".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in result.content
                ) if result.content else "(no result)"
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return "I wasn't able to complete that request. Please try again."


# Simple terminal chat until we add the UI
async def main():
    history = []
    print("Connoisseur Companion (type 'exit' to quit)")
    while True:
        user_message = input("\nYou: ").strip()
        if user_message.lower() in ("exit", "quit"):
            break
        if not user_message:
            continue
        reply = await chat_with_agent(user_message, history)
        print(f"\nAgent: {reply}")
        history += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]


if __name__ == "__main__":
    asyncio.run(main())
