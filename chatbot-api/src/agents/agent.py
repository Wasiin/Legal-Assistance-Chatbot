import os
import sys
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from chains.cypher_chain import judgement_cypher_chain
from chains.judgement_vector_chain import judgment_vector_chain
from chains.law_vector_chain import law_vector_chain
from chains.qa_vector_chain import qa_vector_chain
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_json_chat_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool, StructuredTool
# from tools.wait_times import (
#     get_current_wait_times,
#     get_most_available_hospital,
# )
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

from langchain_core.prompts import ChatPromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema


class RetrievalQAInput(BaseModel):
    action_input: Union[str, Dict[str, str]]  # รองรับทั้งข้อความและ dictionary
    query: Optional[str] = None
    context: Optional[str] = None

JUDGEMENT_AGENT_MODEL = os.getenv("JUDGEMENT_AGENT_MODEL")
#set_llm_cache(InMemoryCache())
llm_model=AzureAIChatCompletionsModel()

#judgment_agent_prompt = hub.pull("hwchase17/react-chat-json")

judgment_agent_prompt = ChatPromptTemplate([
    ("system", """Answer the following questions as best you can. You have access to the following tools:

        {tools}

        The way you use the tools is by specifying a json blob.
        Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

        The only values that should be in the "action" field are: {tool_names}

        The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

        ```
        {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}
        ```

        ALWAYS use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: the result of the action
        ... (this Thought/Action/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin! Reminder to always use the exact characters `Final Answer` when responding."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


tools = [
    # Tool(
    #     name="Graph",
    #     func=judgement_cypher_chain.invoke,
    #     description="""
    #         ฐานข้อมูล ด้วยการค้นหาด้วยคำสั่ง Cypher ในฐานข้อมูลทั้งหมด เพื่อวิเคราะห์ประเด็นข้อพิพาท, กฎหมายที่ใช้อ้างอิง,
    #         และผลกระทบของคำพิพากษาต่อคู่กรณี ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
    #     """,
    # ),
    StructuredTool(
        name="RetrievalQA",
        args_schema=RetrievalQAInput,
        func=lambda input: judgment_vector_chain.invoke(
            {"query": input} if isinstance(input, str) else {"query": input.query, "context": input.context}
        ),
        #func=judgment_vector_chain.invoke,
        description="""
            อธิบายช่วยเหลือ การให้ข้อมูล ความรู้ แนะนำ โดนใช้ข้อมูลเกี่ยวข้องกับคำพิพากษาหรือกรณีศึกษาในหัวข้อเดียวกันหรือมีความคล้ายคลึงกับ ระบุหัวข้อ เช่น การเช่าซื้อรถยนต์ การทำสัญญา
            โดยเน้นการระบุข้อมูลสำคัญ เช่น ประเด็นข้อพิพาท (Issue), กฎหมายที่เกี่ยวข้อง (R_Law), วิธีการดำเนินคดี (Operation)
            และผลคำพิพากษา (Penalty) เพื่อนำข้อมูลที่ได้มาวิเคราะห์ เปรียบเทียบ และสร้างคำตอบที่ครอบคลุมและสอดคล้องกับประเด็นที่ต้องการศึกษา
            ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
        """,
    ),
    # Tool(
    #     name="Law",
    #     func=law_vector_chain.ainvoke(),
    #     description="""
    #         อธิบายช่วยเหลือ การให้ข้อมูล ความรู้ แนะนำ เกี่ยวข้องกับข้อมูลทางกฎหมาย และสร้างคำตอบที่ครอบคลุมและสอดคล้องกับประเด็นที่ต้องการศึกษา
    #         ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
    #     """,
    # ),
    # Tool(
    #     name="QA",
    #     func=law_vector_chain.ainvoke(),
    #     description="""
    #         อธิบายช่วยเหลือ การให้ข้อมูล ความรู้ แนะนำ โดนใช้ข้อมูลเกี่ยวข้องกับคำพิพากษาหรือกรณีศึกษาในหัวข้อเดียวกันหรือมีความคล้ายคลึงกับ ระบุหัวข้อ เช่น การเช่าซื้อรถยนต์ การทำสัญญา
    #         โดยเน้นการระบุข้อมูลสำคัญ เช่น ประเด็นข้อพิพาท (Issue), กฎหมายที่เกี่ยวข้อง (R_Law), วิธีการดำเนินคดี (Operation)
    #         และผลคำพิพากษา (Penalty) เพื่อนำข้อมูลที่ได้มาวิเคราะห์ เปรียบเทียบ และสร้างคำตอบที่ครอบคลุมและสอดคล้องกับประเด็นที่ต้องการศึกษา
    #         ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
    #     """,
    # ),
]

# chat_model = ChatOllama(
#     model=JUDGEMENT_AGENT_MODEL,
#     temperature=0,
# )
chat_model=llm_model

judgment_rag_agent = create_json_chat_agent(
    llm=chat_model,
    tools=tools,
    prompt=judgment_agent_prompt,
)

response_schema = ResponseSchema(
    name="output",  # ชื่อของ schema
    description="The output of the structured parser.",  # คำอธิบายของ schema
    type="string"  # กำหนดประเภทของข้อมูล (default คือ "string")
)

parser = StructuredOutputParser.from_response_schemas([response_schema])

judgment_rag_agent_executor = AgentExecutor(
    agent=judgment_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
