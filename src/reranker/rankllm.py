# This file is adapted from the SweRank repository.
# Original repository: https://github.com/SalesforceAIResearch/SweRank/
# Authors: Salesforce AI Research
# License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# License details: https://creativecommons.org/licenses/by-nc/4.0/
#
# Modifications have been made to integrate the code into the KG4Py project.

import re
import copy
from enum import Enum

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Tuple
from .result import Result, RankingExecInfo

Prompt = Union[str, List[Dict[str, str]]]


class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    LRL = "LRL"

    def __str__(self):
        return self.value


class RankLLM(ABC):
    """
    Abstract base class for the language model to be used for reranking
    """

    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        num_few_shot_examples: int = 0,
    ):
        self.model = model
        self._context_size = context_size
        self._prompt_mode = prompt_mode
        self._num_few_shot_examples = num_few_shot_examples
        self._history = []
        self._rerank_type = "code_reasoning"

    def max_tokens(self):
        """
        Returns the maximum number of tokens that can be processed by the model
        """
        return self._context_size

    @abstractmethod
    def run_llm(self, prompt: Prompt, current_window_size: int) -> Tuple[str, int]:
        """
        Abstract method to run the target language model with a passed in prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt to be processed by the model.

        Returns:
            Tuple[str, int]: A tuple object containing the text response and the number of tokens in the response.
        """
        pass

    @abstractmethod
    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int,
    ) -> List[Tuple[Prompt, int]]:
        """
        Abstract method to create prompts for reranking in batches.

        Args:
            results (List[Result]): The results to be reranked.
            rank_start (int): The starting index for ranking.
            rank_end (int): The ending index for ranking.
            batch_size (int): The size of each batch.

        Returns:
            Tuple[List[Prompt], List[int]]: A tuple containing a list of prompts and a list of indices.
        """
        pass

    @abstractmethod
    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[Prompt, int]:
        """
        Abstract method to create a prompt based on the result and given ranking range.

        Args:
            result (Result): The result object containing data for prompt generation.
            rank_start (int): The starting rank for prompt generation.
            rank_end (int): The ending rank for prompt generation.

        Returns:
            Tuple[Union[str, List[Dict[str, str]]], int]: A tuple object containing the generated prompt and the number of tokens in the generated prompt.
        """
        pass

    def permutation_pipeline(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) ->  Result:
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range.

        Args:
            result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The processed result object after applying permutation.
        """
        prompt, in_token_count = self.create_prompt(result, rank_start, rank_end)
        if logging:
            print(f"prompt: {prompt}")

        permutation, out_token_count = self.run_llm(
            prompt, current_window_size=rank_end - rank_start
        )
        if logging:
            print(f"output: {permutation}")

        ranking_exec_info = RankingExecInfo(
            prompt, permutation, in_token_count, out_token_count
        )
        if result.ranking_exec_summary is None:
            result.ranking_exec_summary = []

        result.ranking_exec_summary.append(ranking_exec_info)

        result = self.receive_permutation(result, permutation, rank_start, rank_end)
        return result

    def permutation_pipeline_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> List[Result]:
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range for a batch of results.
        Args:
            results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The processed list of result objects after applying permutation.
        """
        prompts = []
        prompts = self.create_prompt_batched(
            results, rank_start, rank_end, batch_size=32
        )
        batched_results = self.run_llm_batched(
            [prompt for prompt, _ in prompts], current_window_size=rank_end - rank_start
        )
        results = []
        for index, (result, (prompt, in_token_count)) in enumerate(
            zip(results, prompts)
        ):
            permutation, out_token_count = batched_results[index]
            if logging:
                print(f"output: {permutation}")
            ranking_exec_info = RankingExecInfo(
                prompt, permutation, in_token_count, out_token_count
            )
            if result.ranking_exec_summary is None:
                result.ranking_exec_summary = []
            result.ranking_exec_summary.append(ranking_exec_info)
            result = self.receive_permutation(result, permutation, rank_start, rank_end)
            results.append(result)
        return results

    def sliding_window(
        self,
        retrieved_result: Result,
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        logging: bool = False,
    ):
        """
        Applies the sliding window algorithm to the reranking process.

        Args:
            retrieved_result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The result object after applying the sliding window technique.
        """
        rerank_result = copy.deepcopy(retrieved_result)
        end_pos = rank_end
        start_pos = rank_end - window_size
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_result = self.permutation_pipeline(
                rerank_result, start_pos, end_pos, logging=logging
            )
            end_pos -= step
            start_pos -= step
        return rerank_result

    def sliding_windows_batched(
        self,
        retrieved_results: List[Result],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        logging: bool = False,
    ) -> List[Result]:
        """
        Applies the sliding window algorithm to the reranking process for a batch of result objects.
        Args:
            retrieved_results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The list of result objects after applying the sliding window technique.
        """
        rerank_results = [copy.deepcopy(result) for result in retrieved_results]
        end_pos = rank_end
        start_pos = rank_end - window_size
        permutated_results = rerank_results
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            permutated_results = self.permutation_pipeline_batched(
                rerank_results, start_pos, end_pos, logging=logging
            )
            end_pos -= step
            start_pos -= step
        return permutated_results

    def receive_permutation(
        self,
        result: Result,   # original result before reranking
        permutation: str, # permutation string suggested by the model
        rank_start: int,
        rank_end: int,
    ) -> Result:
        """
        Processes and applies a permutation to the ranking results.

        This function takes a permutation string, representing the new order of items,
        and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
        'result' object based on this permutation.

        Args:
            result (Result): The result object containing the initial ranking results.
            permutation (str): A string representing the new order of items.
                            Each item in the string should correspond to a rank in the results.
            rank_start (int): The starting index of the range in the results to which the permutation is applied.
            rank_end (int): The ending index of the range in the results to which the permutation is applied.

        Returns:
            Result: The updated result object with the new ranking order applied.

        Note:
            This function assumes that the permutation string is a sequence of integers separated by spaces.
            Each integer in the permutation string corresponds to a 1-based index in the ranking results.
            The function first normalizes these to 0-based indices, removes duplicates, and then reorders
            the items in the specified range of the 'result.hits' list according to the permutation.
            Items not mentioned in the permutation string remain in their original sequence but are moved after
            the permuted items.
        """
        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt in response]
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
            if "rank" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["score"] = cut_range[j]["score"]
        # ERROR: double check this method, permutation string not applied correctly
        return result

    def parse_reasoning_permutation(self, response: str) -> Tuple[str, bool]:
        ranked_list_pattern = r"\s*(\[\d+\](?:\s*>\s*\[\d+\])*)\s*"
        end_of_reasoning_tag = "</think>"
        start_of_answer_tag = "<answer>"
        end_of_answer_tag = "</answer>"
        matched_ranked_list = None
        if end_of_answer_tag in response and end_of_reasoning_tag in response:
            parsed_answer = (
                response[
                    response.index(end_of_reasoning_tag) : response.index(
                        end_of_answer_tag
                    )
                ]
                .replace(start_of_answer_tag, "")
                .strip()
            )
            match = re.findall(ranked_list_pattern, parsed_answer)
            if match:
                print(len(match))
                matched_ranked_list = match[0].strip()
        if matched_ranked_list:
            print(f"re matched output: {matched_ranked_list}")
            return matched_ranked_list, True
        else:
            match = re.findall(ranked_list_pattern, response, re.DOTALL | re.MULTILINE)
            first_correct_match = None
            for cand in match:
                if ">" not in cand:
                    continue
                else:
                    first_correct_match = cand
                    break

            if first_correct_match:
                print(f"re matched output: {first_correct_match}")
                return first_correct_match, True
            else:
                print(f"re match FAILED: {response}")
                return response, False

    def run_llm_batched(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        ...

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        seen = set()
        unique_response = []
        for item in response:
            if item not in seen:
                seen.add(item)
                unique_response.append(item)
        return unique_response

    def _clean_response(self, response: str) -> str:
        # Todo: this does not appear ncessary
        if self._rerank_type == "code_reasoning":
            response, _ = self.parse_reasoning_permutation(response)

        new_response = ""
        for char in response:
            if not char.isdigit():
                new_response += " "
            else:
                new_response += char
            new_response = new_response.strip()

        return new_response

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)
