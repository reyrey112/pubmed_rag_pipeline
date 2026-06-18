# Databricks notebook source
# MAGIC %md
# MAGIC # Pharma RAG Pipeline Notebook
# MAGIC This notebook sets up the workspace path, imports the core query logic, and launches a Gradio chat interface.

# COMMAND ----------
%load_ext autoreload
%autoreload 2
# Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# To disable autoreload; run %autoreload 0

# COMMAND ----------
# Cell 0: Environment Setup and Imports
%pip install -U typing_extensions pydantic pydantic-core gradio transformers torch databricks-vectorsearch sentence-transformers
dbutils.library.restartPython()

# COMMAND ----------
# Cell 1: Environment Setup and Imports
import sys
sys.path.append("/Workspace/Users/reydencdavies@gmail.com/rag_pipeline/databricks_notebooks")

import gradio as gr
from rag_query import rag_query 

# COMMAND ----------
# Cell 2: Define Core Chat Function
def chat_fn(query, history=None, *args, **kwargs):
    result = rag_query(query)
    answer = result["answer"]
    sources = ", ".join(result["sources"])
    return f"{answer}\n\n**Sources:** {sources}"

# Optional Debugging:
# result = chat_fn("test question")
# print(result)

# COMMAND ----------
# Cell 3: Launch Gradio Interface
demo = gr.ChatInterface(fn=chat_fn, title="Pharma RAG")
demo.launch(share=True)  # share=True gives a public URL