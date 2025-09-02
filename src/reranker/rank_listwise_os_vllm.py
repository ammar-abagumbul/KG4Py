# This file is adapted from the SweRank repository.
# Original repository: https://github.com/SalesforceAIResearch/SweRank/
# Authors: Salesforce AI Research
# License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# License details: https://creativecommons.org/licenses/by-nc/4.0/
#
# Modifications have been made to integrate the code into the KG4Py project.
# List of major modifications:
# - Removed support for non-coding features
# - Added support for system messages

import os
import json
import random
from typing import Optional, Tuple, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
import numpy as np
from ftfy import fix_text
from vllm import LLM, SamplingParams, RequestOutput
from rankllm import Prompt

from .rankllm import PromptMode, RankLLM
from .result import Result

ALPH_START_IDX = ord('A') - 1

class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: Optional[str] = None,
        batched: bool = False,
        rerank_type: str = "text",
        code_prompt_type: str = "docstring",
    ) -> None:
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available on this device"
            self.world_size = torch.cuda.device_count()
            print(f"WORLD SIZE: {self.world_size}")
            if self.world_size > 1:
                os.environ['NCCL_P2P_DISABLE']='1'
                os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. Only RANK_GPT is supported."
            )

        self._llm = LLM(model=model, max_logprobs=30, enforce_eager=True, gpu_memory_utilization=0.9, max_model_len=32768, trust_remote_code=True, enable_chunked_prefill=True, tensor_parallel_size=1)
        self._tokenizer = self._llm.get_tokenizer()
        self.system_message_supported = "system" in self._tokenizer.chat_template
        self._batched = batched
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        self._rerank_type = rerank_type
        self._code_prompt_type = code_prompt_type

        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]

    def run_llm(
        self, prompt: Prompt, current_window_size: Optional[int] = None
    ) -> Tuple[str, int]:
        """Run the language model with appropriate restrictions for code vs text reranking"""
        temp = 0.
        if current_window_size is None:
            current_window_size = self._window_size
        params = SamplingParams(
            temperature=temp,
            max_tokens=self.get_total_output_tokens(current_window_size),
        )
        output = self._llm.generate([prompt], sampling_params=params, use_tqdm=True)[0]
        output_text = output.outputs[0].text.replace(self._tokenizer.eos_token, "")
        self._history.append({
            "prompt": prompt,
            "response": output_text,
            "second_run": {}
        })

        return output_text, len(output_text)

    def run_llm_batched(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Run batched inference with appropriate restrictions for code vs text reranking"""

        temp = 0.
        if current_window_size is None:
            current_window_size = self._window_size
        max_new_tokens = self.get_total_output_tokens(current_window_size)
        min_new_tokens = self.get_total_output_tokens(current_window_size)
        params = SamplingParams(
            temperature=temp,
            max_tokens=max_new_tokens,
            min_tokens=min_new_tokens,
        )
        outputs = self._llm.generate(prompts, sampling_params=params, use_tqdm=True)
        return [
            (output.outputs[0].text, len(output.outputs[0].token_ids))
            for output in outputs
        ]

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size

        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        token_str = " > ".join([f"[{chr(ALPH_START_IDX+i+1)}]" for i in range(current_window_size)])

        _output_token_estimate = len(self._tokenizer.encode(token_str)) + 2

        if self._window_size == current_window_size:
            self._output_token_estimate = _output_token_estimate
        return _output_token_estimate

    def get_total_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        """Get total number of output tokens"""
        base_tokens = self.num_output_tokens(current_window_size)
        return base_tokens

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        if self._code_prompt_type == "docstring":
            return self._add_prefix_prompt_doc_string(query, num)
        else:
            raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")

    def _add_post_prompt(self, query: str, num: int) -> str:
        if self._code_prompt_type == "docstring":
            return self._add_post_prompt_doc_string(query, num)
        else:
            raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")

    def _add_prefix_prompt_doc_string(self, query: str, num: int) -> str:
        return f"I will provide you with {num} code snippets, each indicated by a numerical identifier []. Rank the code snippets based on their relevance to the functionality described by the following doc string: {query}.\n"

    def _add_post_prompt_doc_string(self, query: str, num: int) -> str:
        example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"Doc String: {query}.\nRank the {num} code snippets above based on their relevance to the functionality described by the doc string. All the code snippets should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."

    def _add_prefix_prompt_github_issue(self, query: str, num: int) -> str:
        prefix_prompt = f"I will provide you with {num} code functions, each indicated by a numerical identifier []."
        prefix_prompt += f" Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}.\n"
        return prefix_prompt

    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv

    def _add_few_shot_examples_messages(self, messages):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
        return messages

    def create_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[str, int]:
        query = result.query
        max_query_len = self.get_num_tokens(query)
        num = len(result.hits[rank_start:rank_end])
        max_doc_length = 1024 if (self._rerank_type == "code") else 300
        min_doc_length = 300
        while True:
            messages = list()
            if self._system_message and self.system_message_supported:
                messages.append({"role": "system", "content": self._system_message})
            messages = self._add_few_shot_examples_messages(messages)
            query_tokens = self._tokenizer.tokenize(query)[:int(max_query_len)]
            truncated_query = self._tokenizer.convert_tokens_to_string(query_tokens)
            prefix = self._add_prefix_prompt(truncated_query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                if self._rerank_type == "code":
                    content = hit["content"]
                    content = content.replace("Title: Content: ", "")
                    tokenized_content = self._tokenizer.tokenize(content)
                    content_tokens = tokenized_content[:int(max_doc_length)]
                    truncated_content = self._tokenizer.convert_tokens_to_string(content_tokens)
                    identifier = str(rank)
                    input_context += f"[{identifier}] {self._replace_number(truncated_content)}\n"
                else:
                    content = hit["content"].replace("Title: Content: ", "").strip()
                    content = " ".join(content.split()[:max_doc_length])
                    identifier = str(rank)
                    input_context += f"[{identifier}] {self._replace_number(content)}\n"
            input_context += self._add_post_prompt(truncated_query, num)
            messages.append({"role": "user", "content": input_context})
            if self._system_message and not self.system_message_supported:
                messages[0]["content"] = self._system_message + "\n " + messages[0]["content"]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.get_total_output_tokens(rank_end - rank_start):
                break
            else:
                prefix_len = len(self._tokenizer.encode(prefix))
                if (len(query_tokens) + prefix_len) > (self.max_tokens() - min_doc_length *(rank_end - rank_start) - self.get_total_output_tokens(rank_end - rank_start)):
                    # Query truncation to ensure min doc length for each candidate document/code
                    offset = num_tokens - (self.max_tokens() - self.get_total_output_tokens(rank_end - rank_start))
                    max_query_len -= (offset//2 + 1)
                else:
                    # Document truncation
                    max_doc_length -= max(
                        1,
                        (
                            num_tokens - self.max_tokens() + self.get_total_output_tokens(rank_end - rank_start)
                        ) // ((rank_end - rank_start) * 4),
                    )
        return prompt, num_tokens

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[Prompt, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in chunks(results, batch_size):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
