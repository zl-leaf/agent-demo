from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain import PromptTemplate, LLMChain
from langchain.agents import BaseSingleActionAgent, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from typing import List, Tuple, Any, Union, Optional, Type
from custom_llm import CustomLLM
from custom_api import DeepCall
import re

agent_template = """
你现在是一个聪明的AI，你可以使用下面这些工具：
名称：Logout，作用：为用户进行登出操作
名称：Hello，作用：跟陌生人打招呼

回答时需要遵循以下用---括起来的格式：
---
Question: 用户需要执行什么操作
Thought: 根据用户需要执行的操作，从\"Logout\",\"Hello\"工具中选择一个满足要求的工具
Action: 选择工具的名称
Finish: 输出工具执行的结果
---

现在用户需要进行{input}操作，按照指定格式一步一步进行推理，并且按照以下格式回答问题
{result}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        if len(intermediate_steps) == 0:
            result = "Tool: 选择了什么工具，只需要回答工具名称"
        else:
            action, observation = intermediate_steps[0]
            result = "回答用户执行了什么工具，执行结果："+f"{observation}\n"
        kwargs["result"] = result
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        match = re.match(r'.*?Tool: (Logout)', llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        else:
            action = match.group(1).strip()
            return AgentAction(tool=action, tool_input="", log=llm_output)


if __name__ == "__main__":
    from custom_llm import CustomLLM

    llm = CustomLLM()
    tools = [
                Tool.from_function(
                    func=DeepCall.do,
                    name="Logout",
                    description="为用户进行登出操作"
                )
            ]
    output_parser = CustomOutputParser()
    prompt = CustomPromptTemplate(template=agent_template,
                                  tools=tools,
                                  input_variables=["input", "intermediate_steps", "result"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=["Logout"]
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    print(agent_executor.run(input="为用户执行登出操作", result="无执行任何工具"))
