import importlib
from typing import Optional
from evals.api import CompletionFn, CompletionResult
import baseten

from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling
import os
import requests
import json

key = os.environ["MY_BASETEN_API_KEY"]
header = {
    'Content-Type': 'application/json', 
    'Authorization': f'Api-Key {key}'
}

def get_prompt(prompt: str) -> str:
    model_inputs = {
        "prompt": prompt,
        "max_new_tokens": 300,
        "do_sample": True, 
        "temperature": 0.5,
        "return_full_text": False
    } 
    return json.dumps(model_inputs)

def falcon(json_input: str):
    return requests.post("https://app.baseten.co/models/pqvlAR0/predict", headers=header, data=json_input).json()['model_output']['data']['generated_text']

def wizard(json_input: str):
    return requests.post("https://app.baseten.co/models/DBODNA0/predict", headers=header, data=json_input).json()['model_output']

def run_model(model_name: str, prompt: str):
    import evals.completion_fns.baseten_llm as baseten_llm
    model_function = getattr(baseten_llm, model_name)
    json_input = get_prompt(prompt)
    output = model_function(json_input)
    return output

if __name__ == "__main__":
    prompt = "Give me a list of best selling candy in the US"
    output = run_model(prompt, 'falcon')



class BasetenCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]

class BasetenCompletionFn(CompletionFn):
    def __init__(self, llm: str, llm_kwargs: Optional[dict] = {}, **kwargs) -> None:
       self.model = baseten.deployed_model_version_id(llm)
 

    def __call__(self, prompt, **kwargs) -> BasetenCompletionResult:
        prompt = CompletionPrompt(prompt).to_formatted_prompt()
	    #TODO: investigate output of each type
        response = run_model("wizard", prompt)
        record_sampling(prompt=prompt, sampled=response)
        return BasetenCompletionResult(response)
