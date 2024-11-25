from typing import List
import base64
import pandas as pd
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = ""

# 페이지 설정
st.set_page_config(layout="wide")

# OpenAI 클라이언트 초기화
client = OpenAI()

# 세션 상태 초기화
if "curr_page" not in st.session_state:
    st.session_state["curr_page"] = "home"
    st.session_state["curr_topic"] = "home"

if "exam_context" not in st.session_state:
    st.session_state.exam_context = {}

# 주제 정보 정의
speaking_topic_to_topic_info_map = {
    'speaking_listen_and_answer': {'display_name': 'Listen and answer questions', 'emoji': '💭'},
    'speaking_express_an_opinion': {'display_name': 'Express your opinion', 'emoji': '🗣️'},
    'speaking_debate': {'display_name': 'Discuss controversial topics', 'emoji': '👩‍'},
    'speaking_describe_img': {'display_name': 'Describe a photo in detail', 'emoji': '🏞️'},
    'speaking_describe_charts': {'display_name': 'View and analyze charts', 'emoji': '📊'},
}

writing_topic_to_topic_info_map = {
    'writing_dictation': {'display_name': 'Dictation', 'emoji': '✏️'},
    'writing_responding_to_an_email': {'display_name': 'Reply to email', 'emoji': '✉️'},
    'writing_summarization': {'display_name': 'Summarize', 'emoji': '✍️'},
    'writing_writing_opinion': {'display_name': 'Express your opinion', 'emoji': '📝'},
}

# 오디오 자동 재생
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# 음성 인식
def recognize_speech():
    audio_bytes = audio_recorder("talk", pause_threshold=3.0)

    if audio_bytes:
        with open("./tmp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        with open("./tmp_audio.wav", "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"
            )
            return transcript.text
    return None

# 특정 주제로 이동
def go_to_topic(topic):
    st.session_state["curr_page"] = topic
    st.session_state["curr_topic"] = topic

# 결과를 반환하는 체인
def get_speaking_listen_and_answer_result(answer_text):
    model = ChatOpenAI(model="gpt-4-1106-preview")

    class Score(BaseModel):
        reason: str = Field(description="Infer whether Your Answer is appropriate for the Question. In English.")
        score: int = Field(description="Give a score between 0 and 10 for whether Your Answer is appropriate for the question.")

    parser = JsonOutputParser(pydantic_object=Score)
    format_instruction = parser.get_format_instructions()

    human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
        "{input}\n---\nInfer whether Your Answer is appropriate for the Question and give a score between 0 and 10. Please respond in the following format.: {format_instruction}",
        partial_variables={"format_instruction": format_instruction}
    )

    prompt_template = ChatPromptTemplate.from_messages([human_msg_prompt_template])

    chain = prompt_template | model | parser
    return chain.invoke({"input": answer_text, "format_instruction": format_instruction})

def get_speaking_express_an_opinion_result(answer_text):
    model = ChatOpenAI(model="gpt-4-1106-preview")

    class Score(BaseModel):
        reason: str = Field(description="This is a test where you have to express your opinion about a question. Infer whether comments were responded to appropriately and structurally. In English.")
        score: int = Field(description="Give a score between 0 and 10 for whether Your Answer expresses your opinion logically enough.")

    parser = JsonOutputParser(pydantic_object=Score)
    format_instruction = parser.get_format_instructions()

    human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
        "{input}\n---\nGive a score between 0 and 10 for whether Your Answer expresses your opinion logically enough. Please respond in the following format.: {format_instruction}",
        partial_variables={"format_instruction": format_instruction}
    )

    prompt_template = ChatPromptTemplate.from_messages([human_msg_prompt_template])

    chain = prompt_template | model | parser
    return chain.invoke({"input": answer_text, "format_instruction": format_instruction})

# Streamlit UI 및 이벤트 처리
if st.session_state["curr_page"] == "home":
    st.title("Speaking & Writing Test")

    tab1, tab2 = st.tabs(["Speaking Test", "Writing Test"])

    with tab1:
        cols = st.columns(2)

        for i, (topic, topic_info) in enumerate(speaking_topic_to_topic_info_map.items()):
            with cols[i % 2]:
                st.write(f"{topic_info['emoji']} **{topic_info['display_name']}**")
                st.button(
                    "Start",
                    key=f"start_{topic}_{i}",
                    on_click=lambda t=topic: go_to_topic(t)  # 특정 주제로 이동
                )
    with tab2:
        cols = st.columns(2)

        for i, (topic, topic_info) in enumerate(writing_topic_to_topic_info_map.items()):
            with cols[i % 2]:
                st.write(f"{topic_info['emoji']} **{topic_info['display_name']}**")
                st.button(
                    "Start",
                    key=f"start_{topic}_{i}",
                    on_click=lambda t=topic: go_to_topic(t)  # 특정 주제로 이동
                )
#####################################################################################

elif st.session_state["curr_page"] == "speaking_listen_and_answer":
    # 주제 제목 표시
    topic_info = speaking_topic_to_topic_info_map.get(st.session_state["curr_topic"], {})
    st.title(topic_info.get("display_name", "Unknown Topic"))  # 주제 제목 표시

    # 데이터 로드
    @st.cache_data
    def load_listen_and_answer_data():
        df = pd.read_csv("data/speaking_listen_and_answer/question_and_audio.csv")
        return df

    df = load_listen_and_answer_data()

    if "exam_start" not in st.session_state.exam_context:
        st.session_state.exam_context["exam_start"] = False

    if "question" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]
        st.session_state.exam_context["question"] = sample["question"]
        st.session_state.exam_context["audio_file_path"] = sample["audio_file_path"]

    # "Start" 버튼을 눌렀을 때 오디오 자동 재생
    if not st.session_state.exam_context["exam_start"]:
        if st.button("Start"):
            st.session_state.exam_context["exam_start"] = True
            autoplay_audio(st.session_state.exam_context["audio_file_path"])

    # 음성 인식 및 결과 표시
    if st.session_state.exam_context.get("exam_start", False):
        recognized_text = recognize_speech()

        if recognized_text:  # 음성 인식 후 결과 표시
            st.session_state.exam_context["user_answer"] = recognized_text
            
            st.markdown(f"- Question: {st.session_state.exam_context['question']}")
            st.markdown(f"- Your Answer: {st.session_state.exam_context['user_answer']}")

            with st.spinner("Calculating your test grade..."):
                result = get_speaking_listen_and_answer_result(recognized_text)
                st.markdown(f"{result['reason']}")
                st.markdown(f"#### Overall Score: {result['score']}")


#####################################################################################

elif st.session_state["curr_page"] == "speaking_express_an_opinion":
    # 주제 제목 표시
    topic_info = speaking_topic_to_topic_info_map.get(st.session_state["curr_topic"], {})
    st.title(topic_info.get("display_name", "Unknown Topic"))  # 주제 제목 표시

    # 데이터 로드
    @st.cache_data
    def load_speaking_express_an_opinion_data():
        df = pd.read_csv("data/speaking_express_an_opinion/question_and_audio.csv")
        return df

    df = load_speaking_express_an_opinion_data()

    if "exam_start" not in st.session_state.exam_context:
        st.session_state.exam_context["exam_start"] = False

    if "question" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]
        st.session_state.exam_context["question"] = sample["question"]
        st.session_state.exam_context["audio_file_path"] = sample["audio_file_path"]

    # "Start" 버튼을 눌렀을 때 오디오 자동 재생
    if not st.session_state.exam_context["exam_start"]:
        if st.button("Start"):
            st.session_state.exam_context["exam_start"] = True
            autoplay_audio(st.session_state.exam_context["audio_file_path"])

    # 음성 인식 및 결과 표시
    if st.session_state.exam_context.get("exam_start", False):
        recognized_text = recognize_speech()

        if recognized_text:  # 음성 인식 후 결과 표시
            st.session_state.exam_context["user_answer"] = recognized_text
            
            st.markdown(f"- Question: {st.session_state.exam_context['question']}")
            st.markdown(f"- Your Answer: {st.session_state.exam_context['user_answer']}")

            with st.spinner("Calculating your test grade..."):
                result = get_speaking_express_an_opinion_result(recognized_text)
                st.markdown(f"{result['reason']}")
                st.markdown(f"#### Overall Score: {result['score']}")

#####################################################################################

elif st.session_state["curr_page"] == "speaking_debate":

    st.title("Discuss controversial topics")  # 주제 제목 표시

    con1 = st.container()
    con2 = st.container()

    user_input = ""

    if "model" not in st.session_state.exam_context:
        st.session_state.exam_context["model"] = ChatOpenAI(model="gpt-3.5-turbo")

    if "messages" not in st.session_state.exam_context:
        system_prompt = """\
- You are the AI language test proctor.
- To improve users' English skills, discuss and ask questions about any topic.
- When you receive a response from the user twice, summarize the conversation and ask no more questions.
"""

        model = st.session_state.exam_context["model"]
        question = model.invoke("Create a controversial question for me.").content

        st.session_state.exam_context["messages"] = [
            SystemMessage(content=system_prompt),  # SystemMessage를 정의하여 초기화
            AIMessage(content=question),
        ]

        speech_file_path = "tmp_speak.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=question
        )
        response.stream_to_file(speech_file_path)
        autoplay_audio(speech_file_path)

    # 메시지 표시 및 음성 인식
    with con1:
        for message in st.session_state.exam_context['messages']:
            if isinstance(message, SystemMessage):
                continue
            role = 'user' if isinstance(message, HumanMessage) else 'assistant'
            with st.chat_message(role):
                st.markdown(message.content)

    with con2:
        user_input = recognize_speech()

    # 사용자 입력 처리 및 추가 응답
    if user_input:
        st.session_state.exam_context['messages'].append(HumanMessage(content=user_input))

        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            model = st.session_state.exam_context["model"]

            for chunk in model.stream(st.session_state.exam_context['messages']):
                full_response += (chunk.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            speech_file_path = "tmp_speak.mp3"
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=full_response
            )
            response.stream_to_file(speech_file_path)

            autoplay_audio(speech_file_path)

        st.session_state.exam_context['messages'].append(AIMessage(content=full_response))

    # 최대 턴을 넘었을 때 평가
    turn_len = len(st.session_state.exam_context['messages'])
    max_turn_len = 6

    if turn_len >= max_turn_len:

        def get_speaking_debate_result(conversation):
            model = ChatOpenAI(model="gpt-4-1106-preview")

            class Score(BaseModel):
                reason: str = Field(description="Infer how logically and fluently the user responded in English to a given conversation.")
                score: int = Field(description="Give a score between 0 and 10 based on fluency and logic.")

            parser = JsonOutputParser(pydantic_object=Score)
            format_instruction = parser.get_format_instructions()

            human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
                "{input}\n---\nIn a given conversation, give a score between 0 and 10 based on fluency and logic. Please respond in the following format.: {format_instruction}",
                partial_variables={"format_instruction": format_instruction}
            )

            prompt_template = ChatPromptTemplate.from_messages([human_msg_prompt_template])
            
            chain = prompt_template | model | parser
            return chain.invoke({"input": conversation, "format_instruction": format_instruction})  # 명시적으로 전달

        with st.container(border=True):
            st.subheader("Evaluation Result")

            with st.spinner("Evaluating..."):

                conversation = ""
                for msg in st.session_state.exam_context["messages"]:
                    role = 'User' if isinstance(msg, HumanMessage) else 'AI'
                    conversation += f"{role}: {msg.content}\n"

                result = get_speaking_debate_result(conversation)

            grade = ""

            if result['score'] >= 8:
                grade = "Advanced"
            elif 4 < result['score'] < 8:
                grade = "Intermediate"
            elif result['score'] <= 4:
                grade = "Beginner"

            grade_text = f"{grade}, {result['score']}"

            st.markdown(f"{result['reason']}")
            st.markdown(f"#### Grade: {grade_text}")

#####################################################################################

elif st.session_state["curr_page"] == "speaking_describe_img":
    topic_info = speaking_topic_to_topic_info_map.get(st.session_state["curr_topic"], {})
    st.title(topic_info.get("display_name", "Unknown Topic"))

    # 데이터 로드
    @st.cache_data
    def load_speaking_describe_img():
        df = pd.read_csv("data/speaking_describe_img/desc_img.csv")
        return df

    df = load_speaking_describe_img()

    if "img_path" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        img_path = sample["img_path"]
        desc = sample["desc"]

        st.session_state.exam_context["img_path"] = img_path
        st.session_state.exam_context["desc"] = desc
        st.session_state.exam_context["recognized_text"] = ""

    st.image(st.session_state.exam_context["img_path"])

    with st.container():
        recognized_text = recognize_speech()  # 음성 인식
        if recognized_text:
            st.session_state.exam_context["recognized_text"] = recognized_text
        st.write(st.session_state.exam_context["recognized_text"])

    submit = st.button("Submit")

    if submit:
        # 평가 체인 함수
        def get_speaking_describe_img(user_input, ref):
            model = ChatOpenAI(model="gpt-4-1106-preview")

            class Evaluation(BaseModel):
                score: int = Field(description="Describe a photo expression score (0~10).")
                feedback: str = Field(description="Detailed feedback for describing photos.")

            parser = JsonOutputParser(pydantic_object=Evaluation)
            format_instructions = parser.get_format_instructions()  # 포맷 지침 가져오기

            human_prompt_template = HumanMessagePromptTemplate.from_template(
                "This is a language speaking test about describing pictures. Evaluate the user's response by comparing it with the Reference.\nUser: {input}\nReference: {ref}\n{format_instructions}",
                partial_variables={"format_instructions": format_instructions}
            )

            prompt = ChatPromptTemplate.from_messages([human_prompt_template])

            # 체인 생성 및 호출 (format_instructions 추가)
            eval_chain = prompt | model | parser
            result = eval_chain.invoke({
                "input": user_input,
                "ref": ref,
                "format_instructions": format_instructions,  # 누락된 부분 추가
            })
            return result

        st.title("Results and Feedback")

        with st.spinner("Generating results and feedback..."):
            result = get_speaking_describe_img(
                user_input=st.session_state.exam_context["recognized_text"],
                ref=st.session_state.exam_context["desc"],
            )

        grade = ""

        if result['score'] >= 8:
            grade = "Advanced"
        elif 4 < result['score'] < 8:
            grade = "Intermediate"
        elif result['score'] <= 4:
            grade = "Beginner"

        grade_text = f"{grade}, ({result['score']}/10)"

        st.markdown(f"{result['feedback']}")
        st.markdown(f"#### Grade: {grade_text}")


#####################################################################################


elif st.session_state["curr_page"] == "speaking_describe_charts":
    topic_info = speaking_topic_to_topic_info_map.get(st.session_state["curr_topic"], {})
    st.title(topic_info.get("display_name", "Unknown Topic"))

    # 데이터 로드
    @st.cache_data
    def load_speaking_describe_charts():
        df = pd.read_csv("data/speaking_describe_charts/desc_charts.csv")
        return df

    df = load_speaking_describe_charts()

    if "img_path" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        img_path = sample["img_path"]
        desc = sample["desc"]

        st.session_state.exam_context["img_path"] = img_path
        st.session_state.exam_context["desc"] = desc
        st.session_state.exam_context["recognized_text"] = ""

    st.image(st.session_state.exam_context["img_path"])

    with st.container():
        recognized_text = recognize_speech()  # 음성 인식
        if recognized_text:
            st.session_state.exam_context["recognized_text"] = recognized_text
        st.write(st.session_state.exam_context["recognized_text"])

    submit = st.button("Submit")

    if submit:
        # 평가 체인 함수
        def get_speaking_describe_img(user_input, ref):
            model = ChatOpenAI(model="gpt-4-1106-preview")

            class Evaluation(BaseModel):
                score: int = Field(description="Score for reporting and presenting charts. 0~10 points.")
                feedback: str = Field(description="Detailed feedback for presenting charts. Markdown format. in English.")

            parser = JsonOutputParser(pydantic_object=Evaluation)
            format_instructions = parser.get_format_instructions()  # 포맷 지침 가져오기

            human_prompt_template = HumanMessagePromptTemplate.from_template(
                "This is a language speaking test based on diagrams or charts and presentations. Evaluate the user's response by comparing it with the Reference.\nUser: {input}\nReference: {ref}\n{format_instructions}",
                partial_variables={"format_instructions": format_instructions}
            )

            prompt = ChatPromptTemplate.from_messages([human_prompt_template])

            # 체인 생성 및 호출 (format_instructions 추가)
            eval_chain = prompt | model | parser
            result = eval_chain.invoke({
                "input": user_input,
                "ref": ref,
                "format_instructions": format_instructions,  # 누락된 부분 추가
            })
            return result

        st.title("Results and Feedback")

        with st.spinner("Generating results and feedback..."):
            result = get_speaking_describe_img(
                user_input=st.session_state.exam_context["recognized_text"],
                ref=st.session_state.exam_context["desc"],
            )

        grade = ""

        if result['score'] >= 8:
            grade = "Advanced"
        elif 4 < result['score'] < 8:
            grade = "Intermediate"
        elif result['score'] <= 4:
            grade = "Beginner"

        grade_text = f"{grade}, ({result['score']}/10)"

        st.markdown(f"{result['feedback']}")
        st.markdown(f"#### Grade: {grade_text}")

#####################################################################################

elif st.session_state["curr_page"] == "writing_dictation":

    topic_info = writing_topic_to_topic_info_map[st.session_state.curr_topic]
    st.title(topic_info.get('display_name', 'Unknown Topic'))

    # 데이터 로드
    @st.cache_data
    def load_writing_dictation():
        df = pd.read_csv("./data/writing_dictation/sent_and_audio.csv")
        return df

    df = load_writing_dictation()

    if "sentence" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        sentence = sample["sentence"]
        audio_file_path = sample["audio_file_path"]

        st.session_state.exam_context["sample"] = sample
        st.session_state.exam_context["sentence"] = sentence
        st.session_state.exam_context["audio_file_path"] = audio_file_path

    # 시작 버튼을 눌렀을 때 오디오 재생 및 초기화
    if st.button("Start"):
        st.session_state.exam_context["exam_start"] = True
        st.session_state.exam_context["do_speech"] = True

    # 오디오 재생 및 사용자 입력 처리
    if st.session_state.exam_context.get("exam_start", False):
        if st.session_state.exam_context["do_speech"]:
            autoplay_audio(st.session_state.exam_context["audio_file_path"])
            st.session_state.exam_context["do_speech"] = False

        # 사용자 답변 입력
        user_answer = st.text_input("Enter your answer")
        if user_answer:
            st.session_state.exam_context["user_answer"] = user_answer
        
        if st.session_state.exam_context.get("user_answer"):
            with st.container():
                answer_text = f"""
                - Original sentence: {st.session_state.exam_context["sentence"]}
                - Your Answer: {st.session_state.exam_context["user_answer"]}
                """
                st.markdown(answer_text)
            
            # 체인에서 결과를 가져오기
            def get_writing_dictation_result(answer_text, ref):
                model = ChatOpenAI(model="gpt-4-1106-preview")

                class Evaluation(BaseModel):
                    reason: str = Field(description="Inference for dictation assessment")
                    score: int = Field(description="Dictation score, 0~10 points")

                parser = JsonOutputParser(pydantic_object=Evaluation)
                format_instructions = parser.get_format_instructions()  # 포맷 지침 가져오기

                human_prompt_template = HumanMessagePromptTemplate.from_template(
                    "This is a dictation test. Evaluate the user's response by comparing it with the reference.\nUser: {input}\nReference: {ref}\n{format_instructions}",
                    partial_variables={"format_instructions": format_instructions}  # 올바르게 전달
                )

                prompt_template = ChatPromptTemplate.from_messages([human_prompt_template])

                # 체인에서 평가 결과 가져오기
                chain = prompt_template | model | parser
                return chain.invoke({
                    "input": answer_text,
                    "ref": ref,
                    "format_instructions": format_instructions,  # 명시적으로 전달
                })

            # 결과 표시
            with st.container():
                st.subheader("Evaluation Result")

                with st.spinner("Calculating results..."):
                    model_result = get_writing_dictation_result(answer_text, st.session_state.exam_context["sentence"])

                model_score = model_result['score']

                st.markdown(f"{model_result['reason']}")
                st.markdown(f"#### Score: {model_score}")

