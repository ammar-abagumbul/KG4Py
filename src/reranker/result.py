# This file is adapted from the SweRank repository.
# Original repository: https://github.com/SalesforceAIResearch/SweRank/
# Authors: Salesforce AI Research
# License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# License details: https://creativecommons.org/licenses/by-nc/4.0/
#
# Modifications have been made to integrate the code into the KG4Py project.

from typing import Optional, List, Dict, Any


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
        ranking_exec_summary: Optional[List[RankingExecInfo]] = None,
    ):
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)
