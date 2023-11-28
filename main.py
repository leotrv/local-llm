from llama_cpp import Llama


llm = Llama(model_path="/home/leotraven/Development/llms/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf", chat_format="llama-2")
result = llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
              "content": "Describe this image in detail please."
          }
      ],
      stream=True
)
print(result["choices"][0]["message"]["content"])