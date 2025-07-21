import torch

from typing import (
    Tuple,
    Dict,
    Literal,
    List,
    Any,
    Generator,
    Union
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class QGPipeline:
    def __init__(
        self,
        model: str,
        qg_format: Literal["highlight"] = "highlight",
        exclude_after: List[str] = [],
        use_cuda: bool = False
    ):

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.qg_format = qg_format

        assert self.model.__class__.__name__ == "T5ForConditionalGeneration"

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.use_cuda = use_cuda
        self._exclude_after = exclude_after
        print(f"Using {self.device}")

    def __call__(self, input: Union[Tuple[str, str], List[Tuple[str, str]]]):
        if isinstance(input, tuple):
            # Handle single input
            func_name, docstring = input
            questions = self._generate_questions(func_name, docstring)
            output = [{'answer': func_name, 'question': que} for que in questions]
            if output:
                 return output[0]
            else:
                 return {}

        elif isinstance(input, list) and all(isinstance(item, tuple) for item in input):
            # Handle batch input
            return self._process_batch_generator(input)
        else:
            raise TypeError("Invalid input type. Expected a tuple (func_name, docstring) or a list of such tuples.")


    def _process_batch_generator(self, batch_input: List[Tuple[str, str]]) -> Generator[Dict[str, Any], None, None]:
        """
        Process batch input and yield results with error handling per item
        """
        for i, (func_name, docstring) in enumerate(batch_input):
            try:
                questions = self._generate_questions(func_name, docstring)
                output = [{'answer': func_name, 'question': que} for que in questions]

                if output:
                    yield {
                        'success': True,
                        'index': i,
                        'function_name': func_name,
                        'docstring': docstring,
                        'model_output': output[0],
                        'error': None
                    }
                else:
                    yield {
                        'success': False,
                        'index': i,
                        'function_name': func_name,
                        'docstring': docstring,
                        'model_output': {},
                        'error': 'No questions generated'
                    }

            except Exception as e:
                yield {
                    'success': False,
                    'index': i,
                    'function_name': func_name,
                    'docstring': docstring,
                    'model_output': {},
                    'error': str(e)
                }

    def _generate_questions(self, func_name, docstring):
        #TODO: This can be re-written in a more forceful way for the llm
        inputs = self._prepare_inputs_for_question_extraction(func_name, docstring)

        inputs = self._tokenize(inputs, padding=True, truncation=True)

        with torch.no_grad():
            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                num_beams=4,

                max_length=32
            )

        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        return questions

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        tokenized_inputs = self.tokenizer(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            return_tensors="pt"
        )
        return tokenized_inputs

    def _prepare_inputs_for_question_extraction(self, func_name, docstring):
        #NOTE: experimental, consider removing :params and :return values
        #manual observation suggests the model struggles to understand the pupose of the function in their presense
        for string in self._exclude_after:
            param_idx = docstring.find(string)
            if param_idx != -1:
                docstring = docstring[:param_idx]
            docstring = docstring.strip()
        input = f"answer: <hl>The function is {func_name}<hl>. Context: {docstring} </s>"

        return [input]

    @property
    def exclude_after(self):
        return self._exclude_after

    @exclude_after.setter
    def exclude_after(self, value):
        self._exclude_after = value
