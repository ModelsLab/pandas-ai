from outlines.models import transformers
from pandasai.pipelines.pipeline_context import PipelineContext
from pandasai.prompts.base import BasePrompt
from .base import LLM
from typing import Any
from outlines import generate

class Outlines(LLM):
    def __init__(self,model:Any,tokenizer : Any) -> None:
        self.outline_model = outline_model = transformers(model=model,tokenizer=tokenizer)

    def call(self, instruction: BasePrompt, context: PipelineContext = None) -> str:
        prompt = instruction.to_string()
        memory = context.memory if context else None
        prompt = self.prepend_system_prompt(prompt, memory)
        generator = generate.text(self.outline_model)
        output = generator(prompt)
        return output

    @property
    def type(self) -> str:
        return "huggingface-text-generation"