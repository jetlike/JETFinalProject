from vision.hand_tracker import get_pointing_target
from audio.listener import wait_for_trigger_and_record
from audio.transcriber import transcribe_audio
from llm.query_engine import get_llm_response

def main():
    print("System ready. Say 'Hey Bot' to start.")

    # 1. Wait for audio input
    audio_path = wait_for_trigger_and_record()

    # 2. Transcribe speech to text
    user_text = transcribe_audio(audio_path)

    # 3. Get object from vision based on pointing gesture
    object_info = get_pointing_target()

    # 4. Send to LLM for response
    response = get_llm_response(user_text, object_info)

    # 5. Output result
    print("Bot:", response)

if __name__ == "__main__":
    main()
