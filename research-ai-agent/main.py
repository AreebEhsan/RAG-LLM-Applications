from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import search_tool, wiki_tool, save_tool
import re
import json

load_dotenv()

class ResearchAgent(BaseModel):
    topic: str
    summary: str
    sources: str
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = PydanticOutputParser(pydantic_object=ResearchAgent)

prompt = ChatPromptTemplate.from_messages(
    [
        (
    "system",
    """
You are a research assistant that will help generate a research paper.

You may use tools like "search", "wikipedia", and "save_text_to_file".
If you use "save_text_to_file", make sure to pass a well-formatted string that includes:
- Topic
- Summary
- Sources
- Tools Used

Wrap the final output in this format and provide no other text:
{format_instructions}

Make sure:
- You include all tools you used in the 'tools_used' field.
- You provide a valid, complete JSON response with no missing or truncated values.
"""

        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool,save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})
print("\n📦 Raw response:\n", raw_response)

try:
    output = raw_response.get("output")

    # Handle wrapped output if needed
    if isinstance(output, list) and isinstance(output[0], dict) and "text" in output[0]:
        output = output[0]["text"]

    # Remove triple backticks or json tags if any
    output = re.sub(r"^```json|```$", "", output.strip(), flags=re.MULTILINE).strip()

    structured_response = parser.parse(output)

    # ✅ Pretty-print using Pydantic model's JSON dump
    print("\n✅ Structured JSON Output:\n")
    print(structured_response.model_dump_json(indent=2))

    # ✅ Optional: Also show in a clean human-readable way
    print("\n🧠 Research Summary:\n")
    print(f"📝 Topic: {structured_response.topic}")
    print(f"🔎 Summary:\n{structured_response.summary}")
    print(f"🌐 Sources: {structured_response.sources}")
    print(f"🛠️ Tools Used: {', '.join(structured_response.tools_used)}")

except Exception as e:
    print("❌ Error parsing response:", e)
    print("Raw Response:\n", raw_response)
# ✅ Automatically save full structured output to file
from tools import save_to_txt  # if not already imported

save_data = (
    f"📝 Topic: {structured_response.topic}\n"
    f"🔎 Summary: {structured_response.summary}\n"
    f"🌐 Sources: {structured_response.sources}\n"
    f"🛠️ Tools Used: {', '.join(structured_response.tools_used)}"
)

print("\n💾 Saving to file...")
print(save_to_txt(save_data))
