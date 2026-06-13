import sys
sys.path.append("/Workspace/Users/reydencdavies@gmail.com/rag_pipeline")

from abstracts_to_chunks import create_chunks

create_chunks("rag_pipeline.bronze.abstracts", "rag_pipeline.silver.chunks")