def rag_answer(query: str) -> str:
    _init_rag()
    if _faiss_index is None:
        return f"RAG system failed to load. Details: {_init_error}"

    try:
        from huggingface_hub import InferenceClient
        import time

        # 1. Retrieve context from FAISS instantly
        query_vector = _embed_model.encode([query], convert_to_numpy=True)
        _, indices = _faiss_index.search(query_vector, 3)
        
        retrieved_context = ""
        for i, idx in enumerate(indices[0]):
            doc_text = _subset_df.iloc[idx]['combined_doc']
            retrieved_context += f"--- Reference Case {i+1} ---\n{doc_text}\n\n"

        system_prompt = (
            "You are an empathetic and knowledgeable medical AI assistant. "
            "Answer the patient's question accurately using ONLY the provided reference cases from medical history below. "
            "If the answer cannot be found in the references, state so cautiously."
        )
        user_content = f"### Reference Cases:\n{retrieved_context}\n### Patient Query:\n{query}\n### Doctor's Professional Response:"

        token = _get_token()
        
        # Using a widely supported router model with the :cheapest tag for serverless access
        client = InferenceClient(token=token)
        
        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct:cheapest",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                break
            except Exception as api_err:
                if "503" in str(api_err) or "loading" in str(api_err).lower():
                    print(f"Model is waking up (cold start), retrying in 5 seconds... (Attempt {attempt+1}/3)")
                    time.sleep(5)
                else:
                    raise api_err

        if response is None:
            return "The AI model container is taking too long to wake up from sleep mode. Please try asking your question again in a moment."
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"
