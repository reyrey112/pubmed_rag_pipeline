import pandas as pd
import json
import os
import argparse
import mlflow
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/hf_cache"

EVAL_TABLE = "rag_pipeline.silver.eval_questions"
MLFLOW_EXPERIMENT = "/Users/reydencdavies@gmail.com/generation_model_evaluation"

GEN_MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
]


def load_eval_data(sample_size=20):
    df = spark.table(EVAL_TABLE).select("question", "source_chunk").toPandas()
    return df.sample(n=min(sample_size, len(df)), random_state=42)


def generate_answer(question, context, generator):
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}
Answer:"""

    result = generator(prompt, max_new_tokens=150)
    full_text = result[0]["generated_text"]
    # text-generation models echo the prompt - strip it
    return full_text.replace(prompt, "").strip()


def judge_answer(question, context, answer, client):
    from google.genai import types

    prompt = f"""Rate this answer on a scale of 1-5 for each criterion. Respond with ONLY a JSON object, no other text.

Question: {question}
Context: {context}
Answer: {answer}

Criteria:
- faithfulness (1-5): Is the answer grounded in the context, with no hallucinated facts?
- relevance (1-5): Does it directly answer the question asked?
- conciseness (1-5): Is it appropriately brief without unnecessary repetition?

JSON format: {{"faithfulness": X, "relevance": X, "conciseness": X}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # strip markdown code fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=150,
                # Force the model to think and behave purely as a JSON engine
                system_instruction="You are a strict LLM evaluation judge. Respond with ONLY a valid raw JSON object, no markdown code fences, and no other text.",
            ),
        )

        text = response.text.strip()
        # strip markdown code fences just in case
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Could not parse judge response or error occurred. Error: {e}")
        return {"faithfulness": None, "relevance": None, "conciseness": None}


def run_evaluation(sample_size=20, gen_models=None):
    from transformers import pipeline
    from google import genai

    gen_models = gen_models or GEN_MODELS

    api_key = dbutils.secrets.get(scope="rag_pipeline", key="GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    df_eval = load_eval_data(sample_size)
    print(f"Evaluating on {len(df_eval)} questions")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    all_results = []

    for model_name in gen_models:
        print(f"\nEvaluating: {model_name}")

        with mlflow.start_run(run_name=model_name.split("/")[-1]):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("sample_size", len(df_eval))

            generator = pipeline("text2text-generation", model=model_name)

            scores = {"faithfulness": [], "relevance": [], "conciseness": []}

            for i, row in df_eval.iterrows():
                question = row["question"]
                context = row["source_chunk"]

                answer = generate_answer(question, context, generator)
                judgment = judge_answer(question, context, answer, client)

                for key in scores:
                    if judgment[key] is not None:
                        scores[key].append(judgment[key])

                print(
                    f"  [{i}] faithfulness={judgment['faithfulness']} "
                    f"relevance={judgment['relevance']} "
                    f"conciseness={judgment['conciseness']}"
                )

            avg_scores = {
                f"avg_{key}": sum(vals) / len(vals) if vals else None
                for key, vals in scores.items()
            }
            mlflow.log_metrics({k: v for k, v in avg_scores.items() if v is not None})

            all_results.append({"model": model_name, **avg_scores})
            print(f"  Averages: {avg_scores}")

    df_results = pd.DataFrame(all_results)
    print("\n=== RESULTS SUMMARY ===")
    print(df_results.to_string(index=False))

    from pyspark.sql.functions import current_timestamp

    spark.createDataFrame(df_results).withColumn(
        "evaluated_at", current_timestamp()
    ).write.format("delta").mode("append").saveAsTable(
        "rag_pipeline.silver.generation_eval_results"
    )

    return df_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=GEN_MODELS)
    args = parser.parse_args()

    run_evaluation(sample_size=args.sample_size, gen_models=args.models)


if __name__ == "__main__":
    main()
