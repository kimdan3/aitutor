# Correct imports
from typing import List
import streamlit as st
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate  # Correct import for ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import langchain_core.pydantic_v1 as pyd1

st.set_page_config(page_title="AI English Proofreader ", layout="wide")

# Define models for parsing output
class Grammar(pyd1.BaseModel):
    reason_list: List[str] = pyd1.Field(description="Reasons for grammatical errors.")

class EnglishProficiencyScore(pyd1.BaseModel):
    vocabulary: int = pyd1.Field(description="Score for vocabulary, between 0 and 10.")
    coherence: int = pyd1.Field(description="Score for coherence, between 0 and 10.")
    clarity: int = pyd1.Field(description="Score for clarity, between 0 and 10.")
    score: int = pyd1.Field(description="Overall score between 0 and 10.")

class Correction(pyd1.BaseModel):
    reason: str = pyd1.Field(description="Reason the original English sentence is awkward or wrong.")
    correct_sentence: str = pyd1.Field(description="Corrected sentence.")

# Chain builders
def build_grammar_analysis_chain(model):
    parser = JsonOutputParser(pydantic_object=Grammar)
    format_instruction = parser.get_format_instructions()

    human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
        "{input}\n---\nIdentify grammatical errors and list them. Follow the format: {format_instruction}",
        partial_variables={"format_instruction": format_instruction}
    )

    prompt_template = ChatPromptTemplate.from_messages([human_msg_prompt_template])
    chain = prompt_template | model | parser
    return chain

def build_proficiency_scoring_chain(model):
    parser = JsonOutputParser(pydantic_object=EnglishProficiencyScore)
    format_instruction = parser.get_format_instructions()

    human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
        "{input}\n---\nRate the English proficiency of the text. Follow the format: {format_instruction}",
        partial_variables={"format_instruction": format_instruction}
    )

    prompt_template = ChatPromptTemplate.from_messages([human_msg_prompt_template])
    chain = prompt_template | model | parser
    return chain

def build_correction_chain(model):
    parser = JsonOutputParser(pydantic_object=Correction)
    format_instruction = parser.get_format_instructions()

    human_msg_prompt_template = HumanMessagePromptTemplate.from_template(  # Fixed syntax
        "{input}\n---\nProvide grammatical corrections. Follow the format: {format_instruction}",
        partial_variables={"format_instruction": format_instruction}
    )

    prompt_template = ChatPromptTemplate.from_messages([human_msg_prompt_template])
    chain = prompt_template | model | parser
    return chain

# Initialize model and chains if not already set
if "model" not in st.session_state:
    model = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key="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")
    st.session_state.model = model

if "grammar_analysis_chain" not in st.session_state:
    st.session_state.grammar_analysis_chain = build_grammar_analysis_chain(st.session_state.model)

if "proficiency_scoring_chain" not in st.session_state:
    st.session_state.proficiency_analysis_chain = build_proficiency_scoring_chain(st.session_state.model)

if "correction_chain" not in st.session_state:
    st.session_state.correction_chain = build_correction_chain(st.session_state.model)

# Explicitly retrieve 'format_instruction'
grammar_format_instruction = JsonOutputParser(pydantic_object=Grammar).get_format_instructions()
correction_format_instruction = JsonOutputParser(pydantic_object=Correction).get_format_instructions()
proficiency_format_instruction = JsonOutputParser(pydantic_object=EnglishProficiencyScore).get_format_instructions()

# User interface and chain invocations
st.title("AI Proofreading Service")

user_input = st.text_area("Enter your text:")

# Grammar Analysis and Correction with proper 'format_instruction'
if st.button("Analyze"):
    st.subheader("Grammar")
    with st.spinner("Analyzing..."):
        grammar_analysis = st.session_state.grammar_analysis_chain.invoke({
            "input": user_input,
            "format_instruction": grammar_format_instruction
        })

    st.markdown("\n".join([f"- {item}" for item in grammar_analysis["reason_list"]]))

    st.subheader("Proofread")
    with st.spinner("Proofreading..."):
        correction_result = st.session_state.correction_chain.invoke({
            "input": user_input,
            "format_instruction": correction_format_instruction
        })

    st.markdown("Reason: " + correction_result["reason"])
    st.markdown("Corrected sentence: " + correction_result["correct_sentence"])

    # Proficiency Analysis
    with st.sidebar:
        st.title("Proficiency Analysis")
        with st.spinner("Analyzing..."):
            proficiency_result = st.session_state.proficiency_analysis_chain.invoke({
                "input": user_input,
                "format_instruction": proficiency_format_instruction
            })

        st.markdown(f"Coherence: {proficiency_result['coherence']}/10")
        st.markdown(f"Clarity: {proficiency_result['clarity']}/10")
        st.markdown(f"Vocabulary: {proficiency_result['vocabulary']}/10")
        st.markdown(f"Overall score: {proficiency_result['score']}/10")
