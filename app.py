import wave
import numpy as np
import streamlit as st
from transformers import SeamlessM4Tv2Model, AutoProcessor
from backend.auth import authenticate_user, register_user
from backend.farm import add_farm_data, get_all_farms_by_user
from backend.chatbot1 import chatbot_response
import sounddevice as sd
import queue

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

# Page 1: Authentication
if st.session_state["page"] == "login":
    st.title("Authentication")
    auth_option = st.selectbox("Choose an option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_option == "Register":
        email = st.text_input("Email")
        if st.button("Register"):
            if register_user(username, password, email):
                st.success("User registered successfully! Please log in.")
            else:
                st.error("User already exists.")
    else:  # Login
        if st.button("Login"):
            if authenticate_user(username, password):
                st.success("Login successful!")
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                change_page("farm_form")
            else:
                st.error("Invalid credentials.")

# Page 2: Farm Form and Selection
elif st.session_state["page"] == "farm_form" and st.session_state["authenticated"]:
    st.title("Farm Information")
    st.subheader(f"Welcome, {st.session_state['username']}!")

    st.subheader("Add a New Farm")
    farm_name = st.text_input("Farm Name")
    location = st.text_input("Location")
    if st.button("Add Farm"):
        add_farm_data(farm_name, location, st.session_state["username"])
        st.success("Farm added successfully!")

    st.subheader("Select a Farm")
    farms = get_all_farms_by_user(st.session_state["username"])
    if farms:
        selected_farm = st.selectbox("Choose a farm", [f"{farm['farm_name']} - {farm['location']}" for farm in farms])
        if st.button("Proceed to Chatbot"):
            st.session_state["current_farm"] = next(
                farm for farm in farms if f"{farm['farm_name']} - {farm['location']}" == selected_farm
            )
            change_page("chatbot")
    else:
        st.warning("No farms available. Please add a farm.")

# Page 3: Chatbot
elif st.session_state["page"] == "chatbot" and st.session_state["authenticated"]:
    st.title("Chatbot")
    if st.session_state["current_farm"]:
        st.write(f"Chatting about: {st.session_state['current_farm']['farm_name']} located at {st.session_state['current_farm']['location']}")

        duration = st.number_input("Recording duration (seconds):", min_value=1, max_value=60, value=5)
        if st.button("Record Audio"):
            audio_sample = record_audio(duration)

            # Resample to 16000 Hz if necessary
            sample_rate = 16000

            # Process the audio input
            processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            audio_inputs = processor(audios=audio_sample, sampling_rate=sample_rate, return_tensors="pt")

            # Generate translation from the audio input
            model1 = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
            output_tokens = model1.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)

            # Decode the output tokens and store in session state
            st.session_state["translated_text"] = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

        if st.session_state["translated_text"]:
            st.write(f"Translation from audio: {st.session_state['translated_text']}")

        if st.button("Get Response"):
            # Use the translated text stored in session state
            if st.session_state["translated_text"]:
                st.session_state["chat_response"] = chatbot_response(
                    st.session_state["translated_text"],
                    st.session_state["current_farm"],
                    st.session_state["conversation"]
                )
                # Append to conversation
                st.session_state["conversation"].append({
                    "user": st.session_state["translated_text"],
                    "bot": st.session_state["chat_response"]
                })

        # Display conversation history
        if st.session_state["conversation"]:
            st.write("### Conversation History")
            for turn in st.session_state["conversation"]:
                st.write(f"**User**: {turn['user']}")
                st.write(f"**Bot**: {turn['bot']}")
    else:
        st.warning("No farm selected. Please go back and select a farm.")
        if st.button("Back to Farm Selection"):
            change_page("farm_form")

# Logout Option
if st.session_state["authenticated"]:
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["current_farm"] = None
        st.session_state["conversation"] = []
        change_page("login")
