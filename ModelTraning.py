import torch
import pandas as pd
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    InputExample,
    losses,
)
from transformers import EarlyStoppingCallback
from peft import LoraConfig, TaskType

DATASET_NAME = "mrinjera/testing"
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_MODEL_PATH = "./sbert-function-retrieval"
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 1000
EVALUATION_STEPS = 5000


def load_and_process_dataset(
    dataset_name: str, token: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dataset and process it for training."""
    print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, token=token)

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # convert train_df to Dataset

    print(f"Train dataset loaded with {len(train_df)} examples")
    print(f"Test dataset loaded with {len(test_df)} examples")
    print(f"Dataset columns: {train_df.columns.tolist()}")

    train_df = train_df.dropna(subset=["docstring", "question"])
    test_df = test_df.dropna(subset=["docstring", "question"])

    print(f"  Train: {len(train_df)} examples")
    print(f"  Test: {len(test_df)} examples")

    return train_df, test_df


# NOTE: This is not being used in the current implementation
def create_training_examples(df: pd.DataFrame) -> List[InputExample]:
    """Create InputExample objects for SBERT training."""
    examples = []

    for idx, row in df.iterrows():
        # input example with question as query and func_name + docstring as positive document
        combined_text = f"{row['function_name']} {row['docstring']}"
        example = InputExample(texts=[str(row["question"]), combined_text], label=1.0)
        examples.append(example)

    print(f"Created {len(examples)} training examples")
    return examples


def create_evaluation_data(
    eval_df: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]:
    """Create evaluation data for Information Retrieval evaluation."""
    # Split data for evaluation
    # eval_df = df.sample(n=min(1000, len(df) // 10), random_state=42)

    queries = {}
    corpus = {}
    relevant_docs = {}

    for idx, row in eval_df.iterrows():
        query_id = f"q_{idx}"
        doc_id = f"d_{idx}"

        queries[query_id] = row["question"]
        corpus[doc_id] = f"{row['function_name']} {row['docstring']}"
        relevant_docs[query_id] = doc_id

    print(
        f"Created evaluation data with {len(queries)} queries and {len(corpus)} documents"
    )
    return queries, corpus, relevant_docs


def create_validation_split(
    train_df: pd.DataFrame, validation_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a validation split from training data for monitoring during training."""
    train_split, val_split = train_test_split(
        train_df, test_size=validation_size, random_state=42, stratify=None
    )

    print("Created validation split:")
    print(f"  Training: {len(train_split)} examples")
    print(f"  Validation: {len(val_split)} examples")

    return train_split, val_split


def evaluate_on_test_set(
    model: SentenceTransformer,
    test_queries: Dict[str, str],
    test_corpus: Dict[str, str],
    test_relevant_docs: Dict[str, set],
) -> Dict[str, float]:
    """Evaluate the trained model on the test set."""
    print("Evaluating model on test set...")

    # Create test evaluator
    test_evaluator = InformationRetrievalEvaluator(
        queries=test_queries,
        corpus=test_corpus,
        relevant_docs=test_relevant_docs,
        name="test-eval",
    )

    # Evaluate
    test_score = test_evaluator(model, output_path=OUTPUT_MODEL_PATH + "/test_results")

    print(f"Test evaluation completed. Score: {test_score}")
    return test_score


def initialize_model(model_name: str) -> SentenceTransformer:
    transformer = models.Transformer(MODEL_NAME)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(), pooling_mode="max"
    )
    model = SentenceTransformer(
        modules=[transformer, pooling],
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{model_name} adapter finetuned on CodeSearchNet question-docstring pairs",
        ),
    )
    return model


def train_model(
    model: SentenceTransformer,
    train_dataset: Dataset | pd.DataFrame,
    val_queries: Dict[str, str],
    val_corpus: Dict[str, str],
    val_relevant_docs: Dict[str, set],
    device: torch.device,
) -> SentenceTransformer:
    """Train the SBERT model with Multiple Negatives Ranking Loss."""
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
    )

    if isinstance(train_dataset, pd.DataFrame):
        train_dataset = Dataset.from_pandas(train_dataset)

    model.add_adapter(peft_config, adapter_name="lora_adapter")

    loss = losses.MultipleNegativesRankingLoss(model)

    # TODO: consider including epochs, warmup steps, optimizer
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_MODEL_PATH + "/training_checkpoints/",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=25,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine",
        # run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )

    evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        relevant_docs=val_relevant_docs,
        name="CodeSearchNet retrieval evaluator",
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.01
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[early_stopping],
    )

    trainer.train()

    model.save_pretrained(OUTPUT_MODEL_PATH)

    return model
