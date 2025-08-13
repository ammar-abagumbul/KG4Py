import re
import copy

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
from .result import Result, RankingExecInfo

Prompt = Union[str, List[Dict[str, str]]]


def RankLLM(ABC):
    """
    Abstract base class for the language model to be used for reranking
    """

    def __init__(
        self,
        model: str,
        context_size: int,
    ):
        self.model = model
        self._context_size = context_size
        self._history = []

    def max_tokens(self):
        """
        Returns the maximum number of tokens that can be processed by the model
        """
        return self._context_size

    @abstractmethod
    def run_llm(self, prompt: Prompt) -> Tuple[str, int]:
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
        self, results: List[Result], rank_start: int, rank_end: int
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
    ):
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

        result = self.recieve_permutation(result, permutation, rank_start, rank_end)
        return result

    def receive_permutation(
        self,
        result: Result,
        permutation: str,
        rank_start: int,
        rank_end: int,
        use_alpha: bool,
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
        response = self._clean_response(permutation, use_alpha)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
            if "rank" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["score"] = cut_range[j]["score"]
        return result

    def parse_reasoning_permutation(self, response: str) -> str:
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

    def _clean_response(self, response: str) -> str:
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
