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
    llm_model = AzureAIChatCompletionsModel()
else: 
    llm_model = ChatOllama(model=os.getenv("OLLAMA_MODEL"),temperature=0)


tools = [
    StructuredTool(
        name="Graph",
        func=judgment_vector_chain.invoke,
        args_schema=ToolInput,
        description="""
                        อธิบายช่วยเหลือ การให้ข้อมูล ความรู้ แนะนำ โดนใช้ข้อมูลเกี่ยวข้องกับคำพิพากษาหรือกรณีศึกษาในหัวข้อเดียวกันหรือมีความคล้ายคลึงกับ ระบุหัวข้อ เช่น การเช่าซื้อรถยนต์ การทำสัญญา
                        โดยเน้นการระบุข้อมูลสำคัญ เช่น ประเด็นข้อพิพาท (Issue), กฎหมายที่เกี่ยวข้อง (R_Law), วิธีการดำเนินคดี (Operation)
                        และผลคำพิพากษา (Penalty) เพื่อนำข้อมูลที่ได้มาวิเคราะห์ เปรียบเทียบ และสร้างคำตอบที่ครอบคลุมและสอดคล้องกับประเด็นที่ต้องการศึกษา
                        ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
        """,
    ),
    StructuredTool(
        name="Law",
        func=law_vector_chain.invoke,
        args_schema=ToolInput,
        description="""
                        อธิบายช่วยเหลือ การให้ข้อมูล ความรู้ แนะนำ เกี่ยวข้องกับข้อมูลทางกฎหมาย และสร้างคำตอบที่ครอบคลุมและสอดคล้องกับประเด็นที่ต้องการศึกษา
                        ตอบกลับเป็นภาษาไทยแบบชาวบ้านใช้คุยกันเท่านั้น
        """,
    ),
]

system = """Assistant เป็นระบบให้คำปรึกษาทางกฎหมายเกี่ยวกับการเช่าซื้อรถยนต์ โดยใช้โมเดลภาษาขั้นสูง LLaMA 3.2:3B ที่ถูกพัฒนาโดย Meta  

Assistant ถูกออกแบบมาเพื่อช่วยตอบคำถามและให้คำแนะนำเกี่ยวกับ **การเช่าซื้อรถยนต์** เท่านั้น ไม่ว่าจะเป็นเรื่องสัญญาเช่าซื้อ สิทธิและหน้าที่ของผู้ซื้อและผู้ให้เช่าซื้อ การผิดสัญญา การทวงถาม หรือแม้แต่แนวทางการดำเนินการตามกฎหมายเมื่อเกิดปัญหา  

Assistant สามารถอธิบายกฎหมายและคำพิพากษาที่เกี่ยวข้องในรูปแบบที่เข้าใจง่าย ไม่จำเป็นต้องมีความรู้ด้านกฎหมายมาก่อนก็สามารถใช้ได้ อย่างไรก็ตาม Assistant เป็นเพียงเครื่องมือให้ข้อมูลเบื้องต้น ไม่สามารถใช้แทนที่ทนายความหรือคำแนะนำจากหน่วยงานกฎหมายได้  

เป้าหมายของ Assistant คือช่วยให้คุณเข้าใจสิทธิและทางเลือกของตัวเองได้ง่ายขึ้น เมื่อเจอปัญหาเกี่ยวกับการเช่าซื้อรถยนต์ คุณสามารถใช้ Assistant เป็นแหล่งข้อมูลเพื่อช่วยให้คุณตัดสินใจได้อย่างมั่นใจและถูกต้อง  """

human = """TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

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
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string \ You should put what you want to return to use here
}}
```

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{input}"""

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

#agent_executor.invoke({"input": "สวัสดี"})

