
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('fifa news on which team qualifies for the quater-finals')

print(results)
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)

print( "below \n")
print(search_tool.args_schema.model_json_schema())