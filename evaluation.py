import os
import sys
from unittest.mock import MagicMock

from dotenv import load_dotenv
load_dotenv()

# fix ragas crash on vertexai import
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
import langchain_community.llms
langchain_community.llms.VertexAI = MagicMock()

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings


judge_llm = init_chat_model(model="command-r-plus-08-2024", temperature=0.0)

# wrap for ragas + fix cohere compatibility
ragas_judge_llm = LangchainLLMWrapper(judge_llm)
ragas_judge_llm.is_finished = lambda response: True

judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# replace this with real Q/A pairs from your PDF
data = {
    "question": ["What is the topic of the PDF?"],
    "answer": ["The topic of the PDF is investment, specifically discussing different investment plans..."],
    "contexts": [
        ["Investment plans and strategies for various scenarios...", "Systematic Investment Plans (SIPs) benefits..."]
    ],
    "ground_truth": ["The document is about investment strategies, SIPs, and risk factors."]
}
dataset = Dataset.from_dict(data)

print("Running the LLM Judge... (This takes a moment)")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_judge_llm,
    embeddings=judge_embeddings,
    raise_exceptions=True
)

print("\n--- Final Scores ---")
print(results)
