import ollama

def main():
    # Ollama LLM 호출 (gemma3:1b)
    try:
        response = ollama.chat(
            model='gemma3:1b',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the Fibonacci sequence?"}
            ]
        )
        print("Answer:", response['message'].get('content'))

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()