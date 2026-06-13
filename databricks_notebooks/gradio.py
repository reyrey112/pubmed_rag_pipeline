import sys
sys.path.append("/Workspace/Users/reydencdavies@gmail.com/rag_pipeline/databricks_notebooks")
import gradio as gr
from rag_query import rag_query 

def chat_fn(query, history=None, *args, **kwargs):
    result = rag_query(query)
    answer = result["answer"]
    sources = ", ".join(result["sources"])
    return f"{answer}\n\n**Sources:** {sources}"

# result = chat_fn("test question")
# print(result)

demo = gr.ChatInterface(fn=chat_fn, title="Pharma RAG")
demo.launch(share=True)  # share=True gives a public URL