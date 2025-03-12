import os
import sys
from pathlib import Path

# Adjust system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports (Custom chains)
from chains.judgement_vector_chain import judgment_vector_chain
from chains.law_vector_chain import law_vector_chain

# Third-party libraries
from pydantic import BaseModel

from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool, Tool
class ToolInput(BaseModel):
    query: str

import dotenv
dotenv.load_dotenv()

if os.getenv("MODE") == "cloud":
    llm_model = AzureAIChatCompletionsModel(temperature=0.2)
else: 
    llm_model = ChatOllama(model=os.getenv("OLLAMA_MODEL"),temperature=0.2)


tools = [
    StructuredTool(
        name="judgment",
        func=judgment_vector_chain.invoke,
        args_schema=ToolInput,
        description="""
                        ช่วยค้นหาและสรุปข้อมูลจากคำพิพากษาที่เกี่ยวข้องกับการเช่าซื้อรถยนต์และสัญญา โดยใช้เลขที่คำพิพากษาหรือคดีอ้างอิง ซึ่งอยู่ในรูปแบบตัวเลข
                        เช่น 0496/2546 แล้วเรียบเรียงข้อมูลให้ออกมาเป็น string เดียวที่ครอบคลุมทุกประเด็น เช่น เลขที่คำพิพากษา ข้อพิพาท กฎหมายที่เกี่ยวข้อง 
                        วิธีดำเนินคดี และผลคำพิพากษา โดยเขียนเป็นข้อความต่อเนื่อง ไม่มีการแยกหัวข้อหรือใช้ตัวแปรใด ๆ ให้อธิบายเป็นภาษาง่าย ๆ แบบที่ชาวบ้านคุยกัน
        """,
    ),
    StructuredTool(
        name="law",
        func=law_vector_chain.invoke,
        args_schema=ToolInput,
        description="""
                        อธิบายช่วยเหลือ การให้ข้อมูล ความรู้ แนะนำ เกี่ยวข้องกับข้อมูลทางกฎหมาย และสร้างคำตอบที่ครอบคลุมและสอดคล้องกับประเด็นที่ต้องการศึกษา
                        ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
        """,
    ),
]

system = '''Assistant is a large language model trained by Meta.

Assistant is designed to be able to assist with a wide range of tasks, from answering             simple questions to providing in-depth explanations and discussions on a wide range of             topics. As a language model, Assistant is able to generate human-like text based on             the input it receives, allowing it to engage in natural-sounding conversations and             provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly             evolving. It is able to process and understand large amounts of text, and can use this             knowledge to provide accurate and informative responses to a wide range of questions.             Additionally, Assistant is able to generate its own text based on the input it             receives, allowing it to engage in discussions and provide explanations and             descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks             and provide valuable insights and information on a wide range of topics. Whether             you need help with a specific question or just want to have a conversation about             a particular topic, Assistant is here to assist.'''

human = '''TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in             answering the users original question. The tools the human can use are:

{tools}

RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{
    "action": string, \ The action to take. Must be one of {tool_names}
    "action_input": string \ The input to the action
}}
```

**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted             in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string \ You should put what you want to return to use here
}}
```

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json             blob with a single action, and NOTHING else):

{input}'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_json_chat_agent(llm_model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)

agent_executor.invoke({"input": "สวัสดี"})

