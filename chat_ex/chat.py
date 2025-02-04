import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from collections import deque

# 모델 로드
st.title("LLaMA 3 Chatbot 🤖")
st.sidebar.header("Settings")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=300, step=50)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    model = torch.compile(model)
    return tokenizer, model

tokenizer, model = load_model()

# 문장이 끝날 때까지 응답 유지
class StopOnSentenceEnd(StoppingCriteria):
    def __init__(self, min_length):
        self.min_length = min_length

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.min_length:
            return False  # 최소 길이 도달 전에는 중단하지 않음
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return text.strip().endswith(('.', '!', '?'))  # 문장이 완전히 끝날 때만 중단

def generate_response(user_input, max_tokens=300):
    """
    모델을 실행하여 응답을 생성하는 함수
    """
    prompt = f"User: {user_input}"  # Bot을 미리 추가하지 않고, 사용자 입력만 사용

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    stopping_criteria = StoppingCriteriaList([StopOnSentenceEnd(min_length=max_tokens // 2)])

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=int(max_tokens * 1.2),
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=False,
            stopping_criteria=stopping_criteria
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # "Bot:"이 자동 포함되면 제거
    if response.startswith("Bot:"):
        response = response[len("Bot:"):].strip()

    return response


def generate_full_response(user_input, max_tokens=300, max_turns=1):
    """
    모델이 중간에 잘리는 경우, 추가적으로 응답을 생성하는 함수
    """
    response_list = []  # 여러 응답을 저장할 리스트

    for _ in range(max_turns):
        new_text = generate_response(user_input, max_tokens)
        
        # 응답이 너무 짧으면 추가 생성
        if len(new_text.split()) < max_tokens // 3:
            user_input = new_text  # 새 응답을 다음 입력으로 설정
            continue
        
        response_list.append(new_text)
        if new_text.strip().endswith(('.', '!', '?')):  # 문장이 완전히 끝났다면 중단
            break

    return " ".join(response_list).strip()  # 모든 응답을 합쳐 최종 결과 반환



def chat_with_memory(history, user_input, max_tokens=150):
    """
    사용자의 입력을 받아 챗봇 응답을 생성하고, 대화 이력을 관리하는 함수
    """
    # 챗봇 응답 생성
    response = generate_response(user_input, max_tokens)

    # 중복된 응답을 방지하고 정리하여 추가
    if not history or history[-1] != f"User: {user_input}":
        history.append(f"User: {user_input}")
    
    if not history or history[-1] != f"Bot: {response}":
        history.append(f"Bot: {response}")

    return response, history, max_tokens


# Streamlit 인터페이스
st.subheader("Ask Something To Bot! ")

# 대화 기록을 저장할 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=10)  # 최근 10개의 대화만 유지

# 사용자 입력
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        response, updated_history, max_tokens = chat_with_memory(st.session_state.history, user_input, max_tokens)

st.subheader("Chat History")
for message in st.session_state.history:
    st.write(message)
