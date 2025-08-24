# test_qwen_rag_ko.py
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def main():
    # 1. 문서
    corpus = [
        "피보나치 수열은 각 숫자가 앞의 두 숫자의 합인 수의 나열입니다.",
        "프랑스의 수도는 파리입니다.",
        "파이썬은 인공지능과 데이터 과학 분야에서 인기 있는 프로그래밍 언어입니다."
    ]

    # 2. 벡터화
    model_embed = SentenceTransformer('jhgan/ko-sroberta-multitask')
    embeddings = model_embed.encode(corpus)
    d = embeddings.shape[1]

    # 3. FAISS 인덱스 생성 및 벡터 추가
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))

    # 4. 사용자 질문 (한국어)
    query = "피보나치 수열이 무엇인가요?"
    query_vec = model_embed.encode([query])

    # 5. 가장 유사한 문서 검색
    k = 1
    _, I = index.search(np.array(query_vec), k)
    context = corpus[I[0][0]]

    # 6. Ollama Qwen3 4B 모델 호출
    try:
        response = ollama.chat(
            model='qwen3:4b',
            messages=[
                {"role": "system", "content": "당신은 친절하고 정확한 한국어 조언을 주는 AI 어시스턴트입니다."},
                {"role": "user", "content": f"{query}\n참고 문서: {context}"}
            ]
        )
        print("답변:", response['message']['content'])

    except Exception as e:
        print("[e] Error:", e)

if __name__ == "__main__":
    main()
