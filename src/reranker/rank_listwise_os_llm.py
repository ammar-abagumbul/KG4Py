# This file is adapted from the SweRank repository.
# Original repository: https://github.com/SalesforceAIResearch/SweRank/
# Authors: Salesforce AI Research
# License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# License details: https://creativecommons.org/licenses/by-nc/4.0/
#
# Modifications have been made to integrate the code into the KG4Py project.

import json
import random
from typing import Optional, Tuple, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from ftfy import fix_text
from transformers import AutoTokenizer, AutoModelForCausalLM

from .ranakllm import PromptMode, RankLLM
from .result import Result

ALPH_START_IDX = ord("A") - 1


class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 0,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        batched: bool = False,
        rerank_type: str = "text",
        code_prompt_type: str = "docstring",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float32,
        device_map: Optional[Union[str, Dict[str, str]]] = "auto",
    ) -> None:
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. Only RANK_GPT is supported."
            )

        self._device = device
        self._batched = batched
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        self._rerank_type = rerank_type
        self._code_prompt_type = code_prompt_type
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=True, trust_remote_code=True
        )

        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            else:
                self._tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})

        dtype = torch_dtype
        if dtype is None:
            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                dtype = torch.float32

        model_load_kwargs = dict(
            trust_remote_code=True,
        )
        if device_map is not None:
            model_load_kwargs["device_map"] = device_map
        if load_in_8bit:
            model_load_kwargs["load_in_8bit"] = True
        if load_in_4bit:
            model_load_kwargs["load_in_4bit"] = True
        if not load_in_8bit and not load_in_4bit:
            model_load_kwargs["torch_dtype"] = dtype

        self._model = AutoModelForCausalLM.from_pretrained(model, **model_load_kwargs)

        if getattr(self._model, "get_input_embeddings", None) and (
            self._model.get_input_embeddings().weight.size(0) != len(self._tokenizer)
        ):
            self._model.resize_token_embeddings(len(self._tokenizer))

        self.system_message_supported = hasattr(self._tokenizer, "chat_template") and (
            self._tokenizer.chat_template is not None
            and "system" in self._tokenizer.chat_template
        )

        # TODO: if you decide to use few_shot_examples, implement a logic to load them

    def _evaluate_logits_from_next_token_distribution(
        self, logprobs: torch.Tensor, total: Tuple[int, int]
    ) -> Tuple[str, Dict[int, float]]:
        """
        Given logprobs over vocab of the next token, compute the ranking string and map.
        """
        evaluations: Dict[int, float] = {}

        start, end = total
        for n in range(start, end + 1):
            s = str(n)
            ids = self._tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                evaluations[n] = float(logprobs[ids[0]].item())
        sorted_evals = sorted(evaluations.itemms(), key=lambda x: -x[1])
        result_string = ">".join([f"[{k}]" for k, _ in sorted_evals])

        return result_string, evaluations

    def _get_next_token_logprobs(
        self, prompt: str, total: Tuple[int, int] = (1, 9)
    ) -> Tuple[str, Dict[int, float]]:
        """
        Compute next-token distribution after appending "[" to the prompt.
        """
        full = prompt + "["
        inputs = self._tokenizer(full, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        logits = out.logits[:, -1, :].squeeze(0)
        logprobs = F.log_softmax(logits, dim=-1)
        return self._evaluate_logits_from_next_token_distribution(logprobs, total)

    # ------ Generation ------

    def run_llm(
        self, prompt: str, current_window_size: Optional[int] = None
    ) -> Tuple[str, int]:
        """Run the LM"""
        if current_window_size is None:
            current_window_size = self._window_size

        if current_window_size is None:
            current_window_size = self._window_size

        max_new_tokens = self.get_total_output_tokens(current_window_size)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic sampling
            temperature=0.0,  # no temperature
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        output_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_text = output_text.replace(self._tokenizer.eos_token or "", "")

        self._history.append(
            {"prompt": prompt, "response": output_text, "second_run": {}}
        )
        return output_text, len(output_text)

    def run_llm_batched(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Run the LM in a batched manner."""
        if current_window_size is None:
            current_window_size = self._window_size

        max_new_tokens = self.get_total_output_tokens(current_window_size)
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic sampling
            temperature=0.0,  # no temperature
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        results: List[Tuple[str, int]] = []
        # outputs.sequences shape: (batch, seq_len_out)
        for i in range(len(prompts)):
            input_len = (inputs["attention_mask"][i] > 0).sum().item()
            gen_ids = outputs[i][input_len:]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append((text, len(gen_ids)))
        return results

    # ------ Token counting methods ------

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt, add_special_tokens=False))

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])
        _output_toke_estimate = len(self._tokenizer.encode(token_str)) + 2

        if self._window_size == current_window_size:
            self._output_token_estimate = _output_toke_estimate
        return _output_toke_estimate

    def get_total_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        return self.num_output_tokens(current_window_size)

    # ------ Prompt construction methods ------

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        if self._code_prompt_type == "docstring":
            return self._add_prefix_prompt_doc_string(query, num)
        elif self._code_prompt_type == "search":
            # NOTE: very similar to the docstring prompt, but perhaps a different wording could perform better
            return self._add_prefix_prompt_search(query, num)
        else:
            raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")

    def _add_post_prompt(self, query: str, num: int) -> str:
        if self._code_prompt_type == "docstring":
            return self._add_post_prompt_doc_string(query, num)
        elif self._code_prompt_type == "search":
            return self._add_post_prompt_search(query, num)
        else:
            raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")

    def _add_prefix_prompt_doc_string(self, query: str, num: int) -> str:
        return f"I will provide you with {num} code snippets, each indicated by a numerical identifier []. Rank the code snippets based on their relevance to the functionality described by the following doc string: {query}.\n"

    def _add_prefix_prompt_search(self, query: str, num: int) -> str:
        return f"I will provide you with {num} code snippets, each indicated by a numerical identifier []. Rank the code snippets based on their relevance to the following search query: {query}.\n"

    def _add_post_prompt_doc_string(self, query: str, num: int) -> str:
        example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        return f"Doc String: {query}.\nRank the {num} code snippets above based on their relevance to the functionality described by the doc string. All the code snippets should be included and listed using identifiers, in descending order of relevance. Example: {example_ordering}\nYour answer should strictly follow the format: [X] > [Y] > [Z] ... with no additional text."

    def _add_post_prompt_search(self, query: str, num: int) -> str:
        example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        return f"Doc String: {query}.\nRank the {num} code snippets above based on their relevance to the query provided. All the code snippets should be included and listed using identifiers, in descending order of relevance. Example: {example_ordering}\nYour answer should strictly follow the format: [X] > [Y] > [Z] ... with no additional text."

    def _add_few_shot_examples(self, conv):
        raise NotImplementedError("Few-shot not implemented yet")

    def _add_few_shot_examples_messages(self, messages):
        raise NotImplementedError("Few-shot not implemented yet")

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        """
        Create a prompt for the LLM based on the result and the rank range.
        """
        query = result.query
        max_query_len = self.get_num_tokens(query)
        num = len(result.hits[rank_start:rank_end])
        max_doc_length = 1024
        min_doc_length = 300
        while True:
            messages = []
            if self._system_message and self.system_message_supported:
                messages.append({"role": "system", "content": self._system_message})
            # messages = self._add_few_shot_examples_messages(messages)
            query_tokens = self._tokenizer.tokenize(query)[: int(max_query_len)]
            truncated_query = self._tokenizer.convert_tokens_to_string(query_tokens)
            prefix = self._add_prefix_prompt(truncated_query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                tokenized_content = self._tokenizer.tokenize(content)
                content_tokens = tokenized_content[: int(max_doc_length)]
                truncated_content = self._tokenizer.convert_tokens_to_string(
                    content_tokens
                )
                identifier = str(rank)
                input_context += (
                    f"[{identifier}] {self._replace_number(truncated_content)}\n"
                )
            input_context += self._add_post_prompt(query, num)
            messages.append({"role": "user", "content": input_context})
            if self._system_message and not self.system_message_supported:
                messages[0]["content"] = (
                    self._system_message + "\n" + messages[0]["content"]
                )
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.get_total_output_tokens(
                rank_end - rank_start
            ):
                break
            else:
                prefix_len = len(self._tokenizer.encode(prefix))
                if (len(query_tokens) + prefix_len) > (
                    self.max_tokens()
                    - min_doc_length * (rank_end - rank_start)
                    - self.get_total_output_tokens(rank_end - rank_start)
                ):
                    offset = num_tokens - (
                        self.max_tokens()
                        - self.get_total_output_tokens(rank_end - rank_start)
                    )
                    max_query_len -= offset // 2 + 1
                else:
                    max_doc_length -= max(
                        1,
                        (
                            num_tokens
                            - (
                                self.max_tokens()
                                - self.get_total_output_tokens(rank_end - rank_start)
                            )
                        )
                        // ((rank_end - rank_start) * 4),
                    )
        return prompt, num_tokens

    def create_prompt_batched(
        self,
        results: List[Result],
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []
        with ThreadPoolExecutor() as executor:
            for batch in chunks(results, batch_size):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(
                            result, use_alpha, rank_start, rank_end
                        ),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts
