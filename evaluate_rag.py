from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# A sample evaluation dataset structure based on your domain
data = {
    "question": ["What is the main philosophical theme of The Stranger?"],
    "answer": ["The text explores the absurdity of life and the indifference of the universe."],
    "contexts": [["In 'The Stranger', Camus explores the absurdity of life and existential themes..."]],
    "ground_truth": ["The main theme is the absurdity of human existence and existentialism."]
}

dataset = Dataset.from_dict(data)

# Run the evaluation metrics
results = evaluate(
    dataset, 
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print("\n--- RAGAS Evaluation Results ---")
print(results.to_pandas())