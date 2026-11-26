from llm_client import get_response

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant"
    }
]

print('AI assistant started')
print('Type \'exit\' to quit')

while True:
    user_input = input("\nYou: ")
    if(user_input.lower() == 'exit'):
        break
    messages.append({
        'role': 'user',
        'content': user_input
    })
    answer = get_response(messages)
    print(f"\nAI: {answer}")
    messages.append({
        "role": "assistant",
        "content": answer
    })