
from ai_brain import get_ai_brain
ai = get_ai_brain('llama3.2:latest')
ai.initialize('AndyL', '活泼可爱', '口语化')
while True:
    text = input('你: ')
    if text.lower() in ['quit', 'exit']: break
    response = ai.process_text(text)
    print('AndyL:', response)
