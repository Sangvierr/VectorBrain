import ollama

def main():
    # Ollama LLM 호출 (deepseek-r1:8b)
    try:
        response = ollama.chat(
            model='deepseek-r1:8b',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the Fibonacci sequence?"}
            ]
        )
        print("Answer:", response)

    except Exception as e:
        print("Error:", e)
        print("Make sure the model is downloaded via 'ollama pull deepseek-r1:8b'")

if __name__ == "__main__":
    main()