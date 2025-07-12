from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data:str, filename: str= "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_txt = f"----Research Output----\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename,"a", encoding="utf-8") as f:
        f.write(formatted_txt)
    return f"Data successfully saved to {filename}"
save_tool = Tool(
    name = "save_text_to_file",
    func = save_to_txt,
    description="Saves structured research data to a text file."
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "search",
    func = search.run,
    description="Search Web for the Information",
)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki_tool = Tool(
    name="wikipedia",
    func=WikipediaQueryRun(api_wrapper=wiki_wrapper).run,
    description="Use Wikipedia to answer factual or historical queries."
)