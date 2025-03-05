import os

# Load environment variables
from dotenv import load_dotenv; load_dotenv()

# Third-party libraries
import httpx
import uvicorn  # type: ignore
from fastapi import FastAPI, HTTPException, Request  # type: ignore
from pydantic import BaseModel

# Local imports
from models.model import ReceiveLine



app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/get")
async def chat():
    from agents.agent2 import agent_executor
    query_res = await agent_executor.ainvoke({"input": "สวัสดี"})
    query_res["intermediate_steps"] = [
        str(s) for s in query_res["intermediate_steps"]
    ]
    print(query_res["output"])
    return query_res

@app.post("/test")
async def test(request: Request):
    from agents.agent import agent_executor
    # query_res = await judgment_rag_agent_executor.ainvoke({"input": "สวัสดี"})
    # query_res["intermediate_steps"] = [
    #     str(s) for s in query_res["intermediate_steps"]
    # ]
    # print(query_res["output"])
    json_data = await request.json()
    #json_data.get("input")
    text = json_data.get("text")
    print(text)
    chat = await agent_executor.ainvoke({"input": text})
    return chat

@app.post("/req_query")
async def query(line: ReceiveLine): #(query_input: JudgmentQueryInput) -> JudgmentQueryOutput:
    # query_res = await judgment_rag_agent_executor({input: query_input})
    # query_res["intermediate_steps"] = [
    #     str(s) for s in query_res["intermediate_steps"]
    # ]
    # return query_res
    #print(line)
    # print(line.events[0].replyToken)
    #print(type(line.events[0].message.text))
    from agents.agent import agent_executor
    try:
        chat = await agent_executor.ainvoke({"input": line.events[0].message.text})
        chat["intermediate_steps"] = [
            str(s) for s in chat["intermediate_steps"]
        ]
        print(chat["output"])
        payload = {
                "replyToken": line.events[0].replyToken,
                "messages": [
                    {
                        "type": "text",
                        "text": chat["output"]
                    },

                ]
            }
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['CHANNEL_ACCESS_TOKEN']}",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(os.environ["LINE_API_URL"], headers=headers, json=payload)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=(response.json()),
                )
        
        return {"message": "Webhook handled successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
    return {"status": 200}

if __name__ == "__main__":
    # for p in sys.path:
    #     print( p )
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    