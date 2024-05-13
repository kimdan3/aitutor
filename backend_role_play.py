from typing import List
from fastapi import FastAPI, UploadFile, File
from openai.resources.beta.threads import messages
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import langchain_core.pydantic_v1 as pyd1
import pydantic as pyd2
import os
os.environ["OPENAI_API_KEY"] = "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"
from openai import OpenAI


client = OpenAI()
model = ChatOpenAI(model="gpt-4-1106-preview")
app = FastAPI()


class Turn(pyd2.BaseModel):
    role: str = pyd2.Field(description="role")
    content: str = pyd2.Field(description="content")

class Messages(pyd2.BaseModel):
    messages: List[Turn] = pyd2.Field(description="message", default=[])


class Goal(pyd1.BaseModel):
    goal: str = pyd1.Field(description="goal.")
    goal_number: int = pyd1.Field(description="goal number.")
    accomplished: bool = pyd1.Field(description="true if goal is accomplished else false.")


class Goals(pyd1.BaseModel):
    goal_list: List[Goal] = pyd1.Field(default=[])


parser = JsonOutputParser(pydantic_object=Goals)
format_instruction = parser.get_format_instructions()


def detect_goal_completion(messages, roleplay):
    global parser, format_instruction

    # Conversation
    conversation ="\n".join([ f"{msg['role']}: {msg['content']}" for msg in messages]) 

    # Goals
    goal_list = roleplay_to_goal_map[roleplay]
    goals = "\n".join([f"- Goal Number {i}: {goal} " for i, goal in enumerate(goal_list)])

    
    prompt_template = """
# 대화
{conversation}
---
# 유저의 목표
{goals}
---
위 대화를 보고 유저가 goal들을 달성했는지 확인해서 아래 포맷으로 응답해.
{format_instruction}
"""

    chat_prompt_template = ChatPromptTemplate.from_messages([("user", prompt_template)])
    goal_check_chain = chat_prompt_template | model | parser

    outputs = goal_check_chain.invoke({"conversation": conversation,
                                       "goals": goals,
                                       "format_instruction": format_instruction})
    return outputs



type_to_msg_class_map = {
        "system":  SystemMessage,
        "user":  HumanMessage,
        "assistant":  AIMessage,
        }

def chat(messages):
    messages_lc = []
    for msg in messages:
        msg_class = type_to_msg_class_map[msg["role"]]
        msg_lc = msg_class(content=msg["content"])

        messages_lc.append(msg_lc)
        
    resp = model.invoke(messages_lc)
    return {"role": "assistant", "content": resp.content}


roleplay_to_system_prompt_map = {
        "hamburger": """\
- 너는 햄버거 가게의 직원이다.
- 아래의 단계로 질문을 한다.
1. 주문 할 메뉴 묻기
2. 더 주문 할 것이 없는지 묻기
3. 여기서 먹을지 가져가서 먹을지 질문한다.
4. 카드로 계산할지 현금으로 계산할지 질문한다.
5. 주문이 완료되면 인사를 하고 [END] 라고 이야기한다.
- 너는 영어로 응답한다.\
""",
        "immigration": """\
- 너는 출입국 사무소의 직원이다.
- 아래의 단계로 질문을 한다.
1. 이름 묻기
2. 여행의 목적 묻기
3. 며칠간 체류하는지 묻기
4. 어떤 호텔에서 체류하는지 묻기
5. 모든 질문에 답이 끝났으면 [END] 라고 이야기한다.
- 너는 영어로 응답한다.\
""",
        "restaurant":"""
- 너는 레스토랑 직원이다.
- 아래의 단계로 질문을 한다.
1. 총 몇 명인지 묻기.
2. 주문 할 메뉴 묻기.
3. 손님이 주문할 메뉴를 고르지 못할 경우, 오늘의 추천 메뉴로 아무 종류의 스테이크 하나 그리고 아무 종류의 파스타 하나 그리고 아무 종류의 피자 하나를 손님에게 권한다. 
4. 손님이 스테이크를 주문할 경우, 스테이크를 얼마나 구워줄 지 묻는다.
5. 더 주문 할 것이 없는지 묻기
6. 음식 가져다 주고 맛있게 먹으라고 말하기
7. 손님에게 음식이 입맛에 맞았는지, 더 필요한 것이 없는지 묻기
8. 모든 질문에 답이 끝났으면 [END]라고 이야기한다.\
- 너는 영어로 응답한다.
""",
        "museum":"""
- 너는 대영박물관 안내 센터 직원이다.
- 아래의 단계로 질문을 한다.
- 간단하게 필요한 정보만 제공한다. 너무 길게 말하지 않는다.
1. 궁금한 점이 있는지 묻는다.
2. 모든 질문에 답이 끝났으면 [END]라고 이야기한다.\
""",
        "accommodation":"""
- 너는 에어비앤비 숙박시설 주인이다.
- 아래의 단계로 질문을 한다.
1. 숙박시설을 예약해주어 감사하다고 말한 후, 궁금한 점이 있으면 물어달라고 한다.
2. 손님이 체크인 시간과 체크아웃 시간을 묻거든, 평균적인 체크인 시간과 체크아웃 시간을 말해준 후, 시간을 잘 지켜달라고 부탁한다.
3. 손님이 공항에서 숙소까지 가는 방법을 묻거든, 가상으로 꾸며내어 버스나 기차 번호와 숙소까지 오는 방법을 알려준다.
4. 손님이 공항에서 숙소까지 가는데 걸리는 시간을 묻거든, 가상으로 꾸며내어 응답한다.
5. 더 궁금한 사항이 없는지 묻기.
6. 모든 질문에 답이 끝났으면 [END]라고 말한다.\
"""
        }

roleplay_to_goal_map = {
        "hamburger": ["치즈버거 주문하기",
                      "콜라 주문하기"],
        "immigration": ["축구 경기 보러왔다고 말하기",
                        "5일 체류할 것이라 말하기"],
        "restaurant": ["2명이 앉을 자리가 있는지 물어보기",
                       "가장 인기 있는 요리가 무엇인지 물어보기",
                       "스테이크 미디엄으로 주문하기",
                       "남은 음식 포장해달라고 하기"],
        "museum": ["가방을 맡길 수 있는지 묻기",
                   "박물관 지도를 받을 수 있는지 묻기",
                   "한국어로 된 팸플릿이 있는지 묻기",
                   "사진을 촬영해도 되는지 묻기",
                   "가이드 투어가 있는지 묻기",
                   "박물관 투어를 신청하고 싶다고 말하기"],
        "accommodation": ["체크인 시간 묻기",
                          "공항에서 숙소까지 어떻게 가는지 묻기",
                          "버스로 가면 시간이 얼마나 걸리는지 묻기",
                          "무료 와이파이가 있는지 묻기",
                          "숙소가 안전한 지역에 있는지 묻기",
                          "지켜야 할 특별한 규칙이 있는지 묻기"]               
        }


@app.post("/chat", response_model=Turn)
def post_chat(messages: Messages):
    messages_dict = messages.model_dump()
    print(messages_dict)
    resp = chat(messages=messages_dict['messages'])

    return resp



@app.post("/chat/{roleplay}", response_model=Turn)
def post_chat_role_play(messages: Messages, roleplay: str):
    messages_dict = messages.model_dump()

    # 해당 롤플레이를 위한 system prompt 가져오기
    system_prompt = roleplay_to_system_prompt_map[roleplay]
    msgs = messages_dict['messages']

    msgs = [{"role": "system", "content": system_prompt}] + msgs 
    resp = chat(messages=msgs)

    return resp


@app.get("/{roleplay}/goals")
def get_roleplay_goals(roleplay: str):
    return roleplay_to_goal_map[roleplay]


@app.post("/{roleplay}/check_goals")
def post_roleplay_check_goal(messages: Messages, roleplay: str):
    messages_dict = messages.model_dump()
    messages = messages_dict['messages']
    goal_comp = detect_goal_completion(messages, roleplay)
    return goal_comp


@app.post("/transcribe")
def transcribe_audio(audio_file: UploadFile = File(...)):
    try:

        file_name = "tmp_audio_file.wav"
        with open(file_name, "wb") as f:
            f.write(audio_file.file.read())
        
        with open(file_name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                )
        
        text = transcript.text
    except Exception as e:
        print(e)
        text = f"voice recognition failed.. {e}"
        return {"status": "fail", "text": text}
    print(f"input: {text}")

    return {"status": "ok", "text": text}




