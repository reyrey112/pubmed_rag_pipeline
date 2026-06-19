import pandas as pd
import json
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# generate quesion chunk pairs for retrival evaluation
def generate_eval_set(
    chunks_table: str, eval_table: str, sample_size: int = 50, use_gemini=False
):
    chunks_pd = (
        spark.table(chunks_table)
        .select("chunk_id", "pmid", "chunk")
        .orderBy("chunk_id")
        .limit(sample_size * 3)
        .toPandas()
    )

    # filter out short chunks that are hard for quesitons
    chunks_pd = chunks_pd[chunks_pd["chunk"].str.len() > 200]
    chunks_pd = chunks_pd.sample(n=min(sample_size, len(chunks_pd)), random_state=42)

    if use_gemini:
        from google import genai

        api_key = dbutils.secrets.get(scope="rag_pipeline", key="GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)

        generate_fn = lambda text: generate_question_gemini(text, client)

    else:
        import os, getpass

        cache_dir = f"/tmp/hf_cache_{getpass.getuser()}"

        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"

        from transformers import pipeline

        generator = pipeline("text-generation", model="gpt2")
        generate_fn = lambda text: generate_question_cheap(text, generator)

    eval_pairs = []

    for _, row in chunks_pd.iterrows():
        question = generate_fn(row["chunk"])
        if question:
            eval_pairs.append(
                {
                    "question": question,
                    "chunk_id": row["chunk_id"],
                    "pmid": row["pmid"],
                    "source_chunk": row["chunk"],
                }
            )

    df_eval = spark.createDataFrame(pd.DataFrame(eval_pairs))
    df_eval.write.format("delta").mode("overwrite").saveAsTable(eval_table)

    print(f"Generated {len(eval_pairs)} eval pairs")
    return eval_pairs


def generate_question_cheap(chunk_text: str, generator) -> str:
    prompt = (
        f"Generate one specific question that this text answers: {chunk_text[:300]}"
    )
    result = generator(
        prompt,
        max_new_tokens=50,  # use max_new_tokens instead of max_length
        do_sample=False,
        pad_token_id=50256,  # needed for gpt2 to avoid warning
    )
    # text-generation returns the full text including prompt — strip the prompt
    full_text = result[0]["generated_text"]
    question = full_text.replace(prompt, "").strip()
    return question


def generate_question_gemini(chunk_text: str, client, max_retries: int = 3, delay_seconds: int = 10) -> str:
    from google.genai import types
    import time
    prompt = f"""Read this excerpt from a research paper and write ONE specific factual question that this excerpt directly answers. The question should be answerable using ONLY the information in this excerpt.

    Excerpt:
    {chunk_text}

    Respond with ONLY the question, nothing else."""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    system_instruction="You are a strict evaluation dataset generator. Output only the requested text.",
                ),
            )
            return response.text.strip()

        except Exception as e:
            print(f"Attempt {attempt} of {max_retries} failed with error: {e}")
            if attempt < max_retries:
                print(f"Waiting {delay_seconds} seconds before retrying...")
                time.sleep(delay_seconds)
            else:
                print("Max retries reached. Failing gracefully.")

    return ""


if __name__ == "__main__":
    # generate_eval_set(
    #     chunks_table="rag_pipeline.silver.chunks",
    #     eval_table="rag_pipeline.silver.test_questions",
    #     sample_size=50,
    #     use_gemini=False
    # )

    generate_eval_set(
        chunks_table="rag_pipeline.silver.chunks",
        eval_table="rag_pipeline.silver.eval_questions",
        sample_size=100,
        use_gemini=True,
    )
