import json
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Tuple, Dict
from datasets import Dataset, DatasetDict
from huggingface_hub.errors import RepositoryNotFoundError
from datasets import load_dataset
from QGPipeline import QGPipeline


def dataset_exists(dataset_name, token=None):
    try:
        load_dataset(dataset_name, token=token, streaming=True)
        return True
    except (RepositoryNotFoundError, OSError, ValueError):
        return False


class DocstringDatasetProcessor:
    def __init__(
        self,
        hf_dataset_name: str,
        batch_size: int = 1000,
        token: str = "",
        local_cache_dir: str = "./cache",
        private_repo: bool = False,
    ):

        self.hf_dataset_name = hf_dataset_name
        self.batch_size = batch_size
        self.private_repo = private_repo
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(exist_ok=True)

        self.processed_count = 0
        self.failed_count = 0

        self.token = token

        self.mounted_dataset_dict = DatasetDict()

    def process_batch(
        self, batch_data: List[Tuple[str, str]], pipeline: QGPipeline, batch_id: int, split: str = ""
    ):
        """Process a batch of (func_name, docstring) tuples. Results are saved locally; uploading is handled separately after all batches are processed."""
        batch_results = []
        batch_success_count = 0
        batch_failure_count = 0

        try:
            generator = pipeline(batch_data)
            for result in generator:
                if isinstance(result, dict):
                    if result["success"]:
                        #TODO: consider reducing memory overhead as batch docstring exists in two places
                        batch_results.append(
                            {
                                "function_name": result["function_name"],
                                "docstring": result["docstring"],
                                "question": result["model_output"]["question"],
                            }
                        )
                        batch_success_count += 1
                    else:
                        print(
                            f"Failed to process {result['function_name']}: {result['error']}"
                        )
                        batch_failure_count += 1

        except Exception as e:
            # entire batch failed
            print(f"Catastrophic batch failure {batch_id}: {e}")
            batch_failure_count = len(batch_data)
            batch_success_count = 0

        self.processed_count += batch_success_count
        self.failed_count += batch_failure_count

        print(
            f"Batch {batch_id}: {batch_success_count} successful, "
            f"{batch_failure_count} failed out of {len(batch_data)} items"
        )

        if batch_results:
            self._save_batch_locally(batch_results, batch_id, split=split)

    def _save_batch_locally(self, batch_results: List[Dict], batch_id: int, split: str = ""):
        if split:
            split_dir = self.local_cache_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            batch_file = split_dir / f"batch_{batch_id}.jsonl"
        else:
            batch_file = self.local_cache_dir / f"batch_{batch_id}.jsonl"
        with open(batch_file, "w") as f:
            for item in batch_results:
                json.dump(item, f)
                f.write("\n")

    def process_full_dataset(self, dataset, pipeline: QGPipeline, start_idx: int = 0, split: str = ""):
        """Process the entire data set and store all processed batches locally."""

        if not self.token:
            print("Hugging face token not provided. Terminating")
            return

        print(f"Starting processing of {len(dataset)} items from index {start_idx}")

        start_time = time.time()

        for batch_start in tqdm(
            range(start_idx, len(dataset), self.batch_size), desc="Processing batches"
        ):
            batch_end = min(batch_start + self.batch_size, len(dataset))
            batch_data = dataset[batch_start:batch_end]
            batch_id = batch_start // self.batch_size

            self.process_batch(batch_data, pipeline, batch_id, split=split)

        total_time = time.time() - start_time
        print(
            f"Processed {self.processed_count} items in {total_time:.2f} seconds. Rate: {self.processed_count / total_time:.2f} items/sec"
        )

        print("All batches processed and stored locally.")


    def _can_upload(self):
        namespace = self.hf_dataset_name.split('/')[0]
        try:
            if dataset_exists(self.hf_dataset_name):
                return True
            else:
                from huggingface_hub import whoami
                user_info = whoami(token=self.token)
                if namespace == user_info["name"]:
                    return True
                else:
                    return False
        except Exception as e:
            print(e)
            return False

    def mount_for_upload(self, directory_path: str, *, split: str):
        """Perepares locally processed data to upload"""
        import glob
        import os

        local_data = []
        batch_files = sorted(glob.glob(os.path.join(directory_path, "batch_*.jsonl")))
        for batch_file in batch_files:
            with open(batch_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    local_data.append(item)

        local_dataset = Dataset.from_list(local_data)
        self.mounted_dataset_dict[split] = local_dataset

    def upload_local_to_hf(self, hf_dataset_name, private_repo=False):

        dataset_dict = self.mounted_dataset_dict
        token = self.token
        print("Uploading dataset to Hugging Face...")

        dataset_dict.push_to_hub(
            hf_dataset_name,
            token=token,
            private=private_repo,
            commit_message=f"Added {str(dataset_dict.num_rows)}"
        )

        print(
            f"Successfully uploaded dataset to https://huggingface.co/datasets/{hf_dataset_name}"
        )

    def load_from_hf(self):
        """Load the dataset from Hugging Face"""
        try:
            dataset = load_dataset(self.hf_dataset_name, token=self.token)
            print(f"Successfully loaded dataset from {self.hf_dataset_name}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            raise

    # TODO: resume processing from local cache file
    # TODO: upload from colab cache to permanent file location (local or drive)
