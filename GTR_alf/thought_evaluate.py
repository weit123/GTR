import numpy as np
import json
import torch.nn as nn
import torch
import traceback
import copy

class SingleLineArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        result = super().encode(obj)
        result = result.replace('\"\\\"', '').replace('\\\"\"', '')
        return result

def extract_thought(text_action):
    check_keys = ['thoughts', 'action']
    try:
        if text_action.startswith('<s>') and text_action.endswith('</s>'):
            raw = text_action[3:-4].strip()
            data = json.loads(raw)
            if set(check_keys).issubset(set(data.keys())):
                action = data['action']
                del data['action']
                return json.dumps(data), True, action
        return "", False, None
    except Exception as e:
        return "", False, None

def thought_eval_and_correct(model, corrector, tokenizer, text, img, task, admissible_actions, action_history):
    thought, valid, action = extract_thought(text)
    print(f"THOUGHT: {thought}, {valid}")
    cont = True
    corrected = False
    format_reward = 5
    
    if not valid:
        labels = torch.tensor([[-100]])
        correct_tok = []
        format_reward = 0
        thought_token_length = None
    else:
        ok = False
        while not ok:
            thought_token_length = len(tokenizer.encode(thought))
            eval_answer = corrector.generate_correction(img, thought, task, action_history, admissible_actions, temperature=0.4)
            try:
                result = json.loads(eval_answer)
                print("RESULT:", result)
                evaluation = result["evaluation"].lower()

                if 'yes' in evaluation:
                    labels = torch.tensor([[-100]])
                    correct_tok = []
                else:
                    corrected = True
                    correction = result["correction"]
                    print("CORRECTION:", correction)
                    correction['action'] = action
                    correct_res = json.dumps(correction, cls=SingleLineArrayEncoder, indent=2)
                    action_res = f'"action": "{action}"'
                    correct_tok = tokenizer.encode(correct_res)[1:]
                    labels = copy.deepcopy(correct_tok)
                    action_tok = tokenizer.encode(action_res)[1:]
                    len_a_t = len(action_tok)
                    for i in range(len(labels) - len_a_t, 0, -1):
                        if labels[i:i + len_a_t] == action_tok:
                            labels[i:i + len_a_t] = [-100] * len_a_t
                            break
                    labels = torch.tensor(labels).unsqueeze(0)
                ok = True
            except:
                print("EVAL ERROR")
                print("EVAL ANSWER:", eval_answer)
                traceback.print_exc()

    return torch.tensor(correct_tok).unsqueeze(0), labels, format_reward, cont, valid, corrected, thought_token_length