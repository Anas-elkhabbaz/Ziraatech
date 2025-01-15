import wave
import numpy as np
import streamlit as st
from transformers import SeamlessM4Tv2Model, AutoProcessor
from backend.auth import authenticate_user, register_user
from backend.farm import add_farm_data, get_all_farms_by_user
import sounddevice as sd
import queue
from pinecone import Pinecone
from huggingface_hub import InferenceClient
import json
from sentence_transformers import SentenceTransformer
import requests
import torch

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "current_farm" not in st.session_state:
    st.session_state["current_farm"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "login"
if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = None
if "chat_response" not in st.session_state:
    st.session_state["chat_response"] = None
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Define page navigation function
def change_page(page_name):
    st.session_state["page"] = page_name

# Resample function without using librosa
def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    resample_ratio = target_sr / orig_sr
    new_length = int(len(audio) * resample_ratio)
    resampled_audio = np.interp(
        np.linspace(0, len(audio), new_length, endpoint=False),
        np.arange(len(audio)),
        audio
    )
    return resampled_audio

# Function to record audio live
def record_audio(duration, sample_rate=16000):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    audio_data = []
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        st.write("Recording...")
        sd.sleep(int(duration * 1000))
        while not q.empty():
            audio_data.append(q.get())
    
    return np.concatenate(audio_data, axis=0).flatten()

# Pinecone and LLM setup
HF_TOKEN = "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"  # Replace with your Hugging Face token
repo_id = "HuggingFaceH4/zephyr-7b-beta"

# Create an Inference Client with the Hugging Face model
llm_client = InferenceClient(
    model=repo_id,
    token=HF_TOKEN,
    timeout=50000  # Adjust timeout as needed
)

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone client
pc = Pinecone(
    api_key="pcsk_6fTkHk_SRAYHnuioZQXcschyhbz5ctoaLYvQnQ9VsSB6eEFSEn2gVNiYtfkNhqiE69MkJY"
)

# Define Pinecone index details
index_name = "anas1"

# Check if the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of embeddings
        metric="cosine",  # Use cosine similarity
        delete_protection="enabled"
    )

# Connect to the index
index = pc.Index(name=index_name)

# Function to retrieve relevant data from Pinecone
def retrieve_relevant_data(question):
    vectorized_question = sentence_model.encode(question).tolist()
    results = index.query(
        vector=vectorized_question,
        top_k=1,
        include_metadata=True
    )
    relevant_data = [
        match["metadata"].get("content", "No content available") for match in results["matches"]
    ]
    return relevant_data

# Function to call the LLM with the retrieved context
def call_llm_with_context(inference_client, question, context):
    prompt = (
        f"You are an intelligent assistant. Use the provided context below as a helpful resource to answer the question as accurately and concisely as possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"If the context is helpful, incorporate it into your answer. Otherwise, use your own knowledge to provide the best possible answer."
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 3000,
            "temperature": 0.5,
            "top_p": 0.9
        }
    }

    try:
        response = inference_client.post(json=payload)
        response_data = response.decode('utf-8')
        parsed_response = json.loads(response_data)

        if isinstance(parsed_response, list) and len(parsed_response) > 0:
            generated_text = parsed_response[0].get("generated_text", "")
            answer_start = generated_text.find("\n\nAnswer:\n")
            if answer_start != -1:
                return generated_text[answer_start + len("\n\nAnswer:\n"):].strip()
            else:
                return "The response does not contain an answer section."
        else:
            return "The response structure is unexpected or empty."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to generate a chatbot response
def chatbot_response(question, farm_data, conversation_history):
    relevant_data = retrieve_relevant_data(question)
    context = " ".join(relevant_data[:1]) if relevant_data else "No relevant context available."

    preprompt = (
        f"The farm's name is {farm_data.get('farm_name', 'unknown')}. "
        f"It is located in {farm_data.get('location', 'unknown')}. "
        f"The following context may help answer the question: {context}"
    )

    response = call_llm_with_context(llm_client, question, preprompt)

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model1 = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    final_translations = []
    for line in response.split("."):
        if line.strip():
            text_inputs = processor(text=line.strip(), src_lang="eng", return_tensors="pt")
            output_tokens = model1.generate(**text_inputs, tgt_lang="ary", generate_speech=False)
            translated_line = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
            final_translations.append(translated_line)

    return ". ".join(final_translations) + "."

# Function to fetch weather data
def get_weather(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "weather": data["weather"][0]["description"],
        }
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Main Chatbot Logic
if st.session_state["page"] == "chatbot" and st.session_state["authenticated"]:
    st.title("Chatbot")
    if st.session_state["current_farm"]:
        st.write(f"Chatting about: {st.session_state['current_farm']['farm_name']} located at {st.session_state['current_farm']['location']}")

        # Option to ask a new question
        new_question = st.text_input("Type your question here:")
        if st.button("Submit Question"):
            if new_question.strip():
                # Add user question to conversation
                st.session_state["conversation"].append({"user": new_question})

                # Generate response from chatbot
                st.session_state["chat_response"] = chatbot_response(
                    new_question,
                    st.session_state["current_farm"],
                    st.session_state["conversation"]
                )

                # Add bot response to conversation
                st.session_state["conversation"][-1]["bot"] = st.session_state["chat_response"]

        # Display conversation history
        if st.session_state["conversation"]:
            st.write("### Conversation History")
            for turn in st.session_state["conversation"]:
                st.write(f"**User**: {turn['user']}")
                st.write(f"**Bot**: {turn.get('bot', '...')}")
    else:
        st.warning("No farm selected. Please go back and select a farm.")
        if st.button("Back to Farm Selection"):
            change_page("farm_form")
