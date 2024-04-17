from transformers import AutoTokenizer, AutoModel
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/workspace/chatglm2-6b", trust_remote_code=True).cuda()
model = model.eval()

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "chatglm"
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        global model, tokenizer
        response, _ = model.chat(tokenizer, prompt, history=[])
        return response

if __name__ == "__main__":
    llm = CustomLLM()
    print(llm("你是谁"))