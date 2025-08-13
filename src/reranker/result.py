from typing import List, Dict, Any


class RankingExecInfo:
    def __init__(
        self, prompt, response: str, input_token_count: int, output_token_count: int
    ):
        self.prompt = prompt
        self.response = response
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count

    def __repr__(self):
        return str(self.__dict__)


class Result:
    def __init__(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        ranking_exec_summary: List[RankingExecInfo] = None,
    ):
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)
