# Databricks notebook source
# MAGIC %md
# MAGIC ## Libraries and `config`

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks]>=3.1 langgraph==0.3.4 databricks-langchain databricks-agents uv datasets ragas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

mlflow.langchain.autolog() 

# COMMAND ----------

mlflow.set_experiment("/Users/kyra.wulffert@databricks.com/RAGAS_mlflow_eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the registered agent

# COMMAND ----------

catalog = "kyra_wulffert"
schema = "default"

# COMMAND ----------

model_name = f"{catalog}.{schema}.ai_billing_agent"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}" 
agent = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

agent.predict({"messages": [{"role": "user", "content": "Based on my usage in the last six months and my current contract, would you recommend keeping this plan or changing to another? My customer id is 4401"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a synthetic dataset

# COMMAND ----------

faq_table = (f"{catalog}.{schema}.billing_faq_dataset")

# COMMAND ----------

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df

faq_table = (f"{catalog}.{schema}.billing_faq_dataset")

# "Ghost text" for agent description and question guidelines - feel free to modify as you see fit.
agent_description = f"""
The agent is an AI assistant that answers questions about billing. Questions unrelated to billing are irrelevant. Include questions that are irrelevant or ask for sensitive data too to the test that the agent ignores them.  You can use for some example questions the customer id 4401
"""
question_guidelines = f"""
# User personas
- Customer of a telco provider
- Customer support agent

# Example questions
- How can I set up autopay for my bill?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

docs_df = (
    spark.table(faq_table)
    .withColumnRenamed("faq", "content")  
)
pandas_docs_df = docs_df.toPandas()
pandas_docs_df["doc_uri"] = pandas_docs_df["index"].astype(str)
evals = generate_evals_df(
    docs=pandas_docs_df,  # Pass your docs. They should be in a Pandas or Spark DataFrame with columns `content STRING` and `doc_uri STRING`.
    num_evals=10,  # How many synthetic evaluations to generate
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)
display(evals)

# COMMAND ----------

print(f"Synthetic evals: {len(evals)} rows – columns: {list(evals.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate agent with mlflow.genai.evaluate()

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness, Safety, RetrievalSufficiency

eval_results = mlflow.genai.evaluate(
    data=evals,
    predict_fn=lambda messages: agent.predict({"messages": messages}),
    scorers=[RelevanceToQuery(), Safety(), Correctness()], # add more scorers here if they're applicable
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Adapt the synthetic dataset to RAGAS

# COMMAND ----------

import pandas as pd
from tqdm.auto import tqdm

evals["messages"] = evals["inputs"].apply(lambda d: d["messages"])

# COMMAND ----------

# Returns the model’s final assistant answer string.
def ask_one(chat_msgs):
    one_row = pd.DataFrame({"messages": [chat_msgs]})
    out     = agent.predict(one_row)                
    transcript = out["messages"]          
    for turn in reversed(transcript):
        if turn["role"] == "assistant":
            return turn["content"]
    return ""

# iterate over the 10 rows in evals
answers = [
    ask_one(msgs) for msgs in tqdm(evals["messages"], desc="agent.predict")
]
evals["answer"] = answers


# COMMAND ----------

faq_pd = (
    docs_df.toPandas()                 # keep only two columns if you like
           .set_index("index")["content"]
           .to_dict()
)

# COMMAND ----------

evals["contexts"] = evals["source_id"].apply(lambda sid: [faq_pd.get(int(sid),"")])

# COMMAND ----------

evals

# COMMAND ----------

# Turn the expectations cell into a single reference string.
# • If it’s the dict structure generate_evals_df emits, join all facts.
# • Otherwise just cast to str().

def expectations_to_str(obj) -> str:
    if isinstance(obj, dict) and "expected_facts" in obj:
        facts = obj["expected_facts"]
        if isinstance(facts, list):
            return " ".join(facts).strip()
    return str(obj)           # fallback for any other shape


# COMMAND ----------

from datasets import Dataset

ragas_ds = Dataset.from_dict(
    {
        "question":     evals["inputs"].apply(lambda d: d["messages"][0]["content"]),
        "answer":       evals["answer"],
        "contexts":     evals["contexts"],
        "ground_truth": evals["expectations"].apply(expectations_to_str),
    }
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run RAGAS

# COMMAND ----------

from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

chat_llm = LangchainLLMWrapper(
    ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct",
                   temperature=0.0,
                   max_tokens=512,
                   timeout=300)            
)
emb_llm  = LangchainEmbeddingsWrapper(
    DatabricksEmbeddings(endpoint="databricks-bge-large-en")
)

# COMMAND ----------

from ragas import evaluate
from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    answer_relevancy,
    faithfulness,
)
from ragas.run_config import RunConfig


run_cfg = RunConfig(max_workers=1, timeout=300, max_retries=3)

ragas_res = evaluate(
    ragas_ds,
    metrics=[
        LLMContextPrecisionWithoutReference(),
        answer_relevancy,
        faithfulness,
    ],
    llm=chat_llm,
    embeddings=emb_llm,
    run_config=run_cfg,
)

# COMMAND ----------

ragas_res.to_pandas()

# COMMAND ----------

import uuid, json, mlflow, pandas as pd, numpy as np
from pandas.api.types import is_scalar

ragas_scores = ragas_res.scores                    # list-of-dicts
eval_tbl     = eval_results.tables["eval_results"] # DF with 'assessments'
run_id       = eval_results.run_id                # evaluation run to enrich

# shape (rows, ragas_metrics)
ragas_df = pd.DataFrame(ragas_scores)              

enriched_df = eval_tbl.copy(deep=True).reset_index(drop=True)

def ragas_to_assessments(ragas_dict: dict[str, float]) -> list[dict]:
    out = []
    for metric, score in ragas_dict.items():
        if pd.isna(score):
            continue
        out.append(
            {
                "assessment_id": f"ragas-{uuid.uuid4().hex[:8]}",
                "name":          f"ragas_{metric}",
                "value":         float(score),
                "assessor":      "RAGAS",
                "version":       "0.2.15",
            }
        )
    return out

# merge row-by-row
for i, ragas_dict in enumerate(ragas_scores):
    cell = enriched_df.at[i, "assessments"]
    if isinstance(cell, str):
        cell = json.loads(cell)
    if not isinstance(cell, list):
        cell = [cell]
    cell.extend(ragas_to_assessments(ragas_dict))
    enriched_df.at[i, "assessments"] = cell

# JSON-safe DataFrame (lists/dicts → strings)
def _jsonify(val):
    if is_scalar(val) or val is None:
        return val
    try:
        return json.dumps(val, default=str)
    except TypeError:
        return str(val)

safe_df = enriched_df.applymap(_jsonify)

# aggregate metrics  (mean & p90) and log everything
means = ragas_df.mean(numeric_only=True).round(4)
p90s  = ragas_df.quantile(0.90, numeric_only=True).round(4)

agg_metrics = (
    {f"ragas_mean_{k}": v for k, v in means.items()}
    | {f"ragas_p90_{k}":  v for k, v in p90s.items()}
)

with mlflow.start_run(run_id=run_id):
    mlflow.log_table(safe_df, "eval_results_with_ragas.json")
    mlflow.log_metrics(agg_metrics)

print(f"Logged per-row table + aggregate metrics to run {run_id}")
