import os
import json
from openai import AzureOpenAI
import base64
from io import BytesIO
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import numpy as np
import random
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
import traceback

class GPT4oAgent():

    def __init__(self):
        api_type: str = "azure"
        api_key = "YOUR_API_KEY"
        api_base = "YOUR_API_BASE"
        api_version = "2024-12-01-preview"
        self.model = "gpt-4o"

        # we use the azure api of gpt models
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            base_url=f"{api_base}/openai/deployments/{self.model}",
        )

        self.solver = Solver()

        # register solver tool
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "find_all_correct_formulas",
                    "description": "Use this function to find all correct formulas for 24 points card game. Input should be four card ranks. Only use it when the target formula is NOT DETERMINED!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "card1": {
                                "type": "string",
                                "description": "The first card by its rank",
                            },
                            "card2": {
                                "type": "string",
                                "description": "The second card by its rank",
                            },
                            "card3": {
                                "type": "string",
                                "description": "The third card by its rank",
                            },
                            "card4": {
                                "type": "string",
                                "description": "The fourth card by its rank",
                            },
                        },
                        "required": ["card1", "card2", "card3", "card4"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                    "parallel_tool_calls": False,
                }
            }
        ]

    # wrapper for consistent openai api access
    @retry(wait=wait_random_exponential(min=1, max=15))
    def generate_correction(self, env_name, img, thoughts, infos, target_formula, temperature=0):
        base64_image = self.image_to_base64(img)
        system_prompt, qs = self.get_correction_prompt(env_name, thoughts, infos, target_formula)

        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": qs},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=temperature,
            tools=self.tools,
        )
        response_message = response.choices[0].message 
        messages.append(response_message)

        tool_calls = response_message.tool_calls
        if tool_calls:
            all_calls_args = [json.loads(calls.function.arguments) for calls in tool_calls]
            assert self.check_args(all_calls_args) 
            for call in tool_calls:
                tool_call_id = call.id
                tool_function_name = call.function.name
                tool_args = json.loads(call.function.arguments)
                if not set(['card1', 'card2', 'card3', 'card4']).issubset(set(tool_args.keys())):
                    print(f"tool args error! {tool_args}")
                    raise RuntimeError
                card1, card2, card3, card4 = tool_args['card1'], tool_args['card2'], tool_args['card3'], tool_args['card4']

                results = self.points24_tool(card1, card2, card3, card4)
                if len(results) == 0:
                    results = "No possible solution"

                messages.append({
                    "role":"tool", 
                    "tool_call_id":tool_call_id, 
                    "name": tool_function_name, 
                    "content": str(results),
                })

            model_response_with_function_call = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=temperature,
            )

            return model_response_with_function_call.choices[0].message.content
        else: 
            return response_message.content
    
    def check_args(self, tool_args):
        for i in range(len(tool_args) - 1):
            if self.card2num(tool_args[i]['card1']) != self.card2num(tool_args[i+1]['card1']) or \
               self.card2num(tool_args[i]['card2']) != self.card2num(tool_args[i+1]['card2']) or \
               self.card2num(tool_args[i]['card3']) != self.card2num(tool_args[i+1]['card3']) or \
               self.card2num(tool_args[i]['card4']) != self.card2num(tool_args[i+1]['card4']) :
               return False
        return True

    def image_to_base64(self, data):
        img = Image.fromarray(data)
        byte_data = BytesIO()
        img.save(byte_data, format="JPEG")
        base64_image = base64.b64encode(byte_data.getvalue()).decode("utf-8")
        return base64_image

    def get_correction_prompt(self, env_name, thought, infos, target_formula=None):
        if env_name == 'gym_cards/NumberLine-v0':
            system_prompt = "You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
            qs = "You will be given the thought of a player playing this game. The player's thought may be wrong, please evaluate its correctness by the following aspects: \n"
            qs = qs + "(1) What are the current and target number in the image? Does the number in the thought match your observation?\n"
            qs = qs + "(2) Does the player choose the correct action to reach the target number? \n"
            qs = qs + "Please briefly answer the above questions, then give your final evaluation. If the thought is incorrect, use all available information for thought correction: determine the next proper action and finally provide the correct thought. \n"

            qs = qs + "Your response should be a valid json file in the following format: \{\n"
            qs = qs + " \"answer1\": {Text, answer to the first quesion}, \n"
            qs = qs + " \"answer2\": {Text, answer to the second quesion}, \n"
            qs = qs + " \"evaluation\": {YES or NO}, \n"
            qs = qs + " \"correction\": {Json object, the correct thought. None if the thought is correct} \n"
            qs = qs + "\} \n\n"

            qs = qs + f"[Thought] {thought} \n\n"

        elif env_name == 'gym_cards/Blackjack-v0':
            system_prompt = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "

            qs = "You will be given the thought of a player playing this game. The player's thought may be wrong, please evaluate its correctness by the following aspects: \n"
            qs = qs + "(1) What are the cards in the image? What are the total points of the two players? Does the points in the thought match your observation?\n"
            qs = qs + "(2) In your opinion, does the player choose the correct action to win the game? \n"
            qs = qs + "Please briefly answer the above questions, then give your final evaluation. If the thought is incorrect, use all available information for thought correction: determine the next proper action and finally provide the correct thought. \n"

            qs = qs + "Your response should be a valid json file in the following format: \{\n"
            qs = qs + " \"answer1\": {Text, answer to the first quesion}, \n"
            qs = qs + " \"answer2\": {Text, answer to the second quesion}, \n"
            qs = qs + " \"evaluation\": {YES or NO}, \n"
            qs = qs + " \"correction\": {Json object, the correct thought. None if the thought is correct} \n"
            qs = qs + "\} \n\n"

            qs = qs + f"[Thought] {thought} \n\n"

        elif env_name == 'gym_cards/EZPoints-v0':
            try:
                text_formula = ''.join(str(element) for element in infos[0]['Formula'])
            except:
                text_formula = ''
                
            system_prompt = f"You are an expert card game player. You are observing two cards in the image and the current formula. The goal is to output a formula that evaluates to 12, and each number can only be used once. "
            system_prompt = system_prompt + "The number or operator include ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '='], and the chosen number or operator will be appended to the current formula to reach the correct target formula that evaluates to 12. "
            system_prompt = system_prompt + "Note that 'J', 'Q', and 'K' count as '10'. \n"

            qs = "You will be given the current formula and the thought of a player playing this game. The player's thought may be wrong, please evaluate its correctness by the following aspects: \n"
            qs = qs + "(1) What are the two cards in the image? What are the recognized card ranks in the thought? According to the rules, does the ranks in the thought match your observation, regardless of the order?\n"
            qs = qs + "(2) What is the proposed formula the player is trying to reach in the thought? Does the proposed formula evaluates to 12? \n"
            qs = qs + "(3) Does the player choose the correct action to reach the proposed formula or choose '=' if the current formula is complete? \n"
            qs = qs + "Please briefly answer the above questions, then give your final evaluation. If the thought is incorrect, use all available information for thought correction: determine the next **single number or character** to append to the current formula and finally provide the correct thought. \n"

            qs = qs + "Your response should be a valid json file in the following format: \{\n"
            qs = qs + " \"answer1\": {Text, answer to the first quesion}, \n"
            qs = qs + " \"answer2\": {Text, answer to the second quesion}, \n"
            qs = qs + " \"answer3\": {Text, answer to the third quesion}, \n" 
            qs = qs + " \"evaluation\": {YES or NO}, \n"
            qs = qs + " \"correction\": {Json object, the correct thought. None if the thought is correct} \n"
            qs = qs + "\} \n\n"

            qs = qs + f"[Current Formula] '{text_formula}' \n"
            qs = qs + f"[Thought] {thought} \n\n"

        elif env_name == 'gym_cards/Points24-v0':
            try:
                text_formula = ''.join(str(element) for element in infos[0]['Formula'])
            except:
                text_formula = ''

            system_prompt = f"You are an expert 24 points card game player. You are observing four cards in the image and the current formula. The goal is to output a formula that evaluates to 24, and each number can only be used once."
            system_prompt = system_prompt + "The number or operator include ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '='], and the chosen number or operator will be appended to the current formula to reach the correct target formula that evaluates to 24. "
            system_prompt = system_prompt + "Note that 'J', 'Q', and 'K' count as '10'. \n"
            
            qs = "You will be given the current formula, the thought of a player playing this game, and a target formula. The player's thought may be wrong, please evaluate its correctness by the following aspects: \n"
            qs = qs + "(1) What are the four cards in the image? **If the target formula is 'NOT DETERMINED'**, use the 'find_all_correct_formulas' tool function to find all possible correct formulas by the four cards in the image. Remember the correct formulas, and do not output the result.\n"
            qs = qs + "(2) What are the recognized card ranks in the thought? According to the rules, does the ranks in the thought match your observation in question (1), regardless of the order?\n"
            qs = qs + "(3) What is the proposed formula the player is trying to reach in the thought? Does the proposed formula match the target formula or, if the target formula is 'NOT DETERMINED', one of the possible correct formulas in question (1)? \n"
            qs = qs + "(4) Does the player choose the correct action to reach the proposed formula or choose '=' if the current formula is complete? \n"
            qs = qs + "Please briefly answer the above questions, then give your final evaluation. If the thought is incorrect, use all available information for thought correction: determine the next **single number or character** to append to the current formula and finally provide the correct thought. \n"

            qs = qs + "Your response should be a valid json file in the following format: \{\n"
            qs = qs + " \"answer1\": {Text, answer to the first quesion}, \n"
            qs = qs + " \"answer2\": {Text, answer to the second quesion}, \n"
            qs = qs + " \"answer3\": {Text, answer to the third quesion}, \n" 
            qs = qs + " \"answer4\": {Text, answer to the third quesion}, \n" 
            qs = qs + " \"evaluation\": {YES or NO}, \n"
            qs = qs + " \"possible_solution\": {YES or NO, indicating Whether there is a possible solution. None if the thought is correct}, \n"
            qs = qs + " \"target_formula\": {The given target formula if it is not None. The proposed formula in the thought if the thought is correct. Otherwise, choose an appropriate target formula from all possible correct formulas obtained from the tool function for the player to reach. }, \n"
            qs = qs + " \"correction\": {Json object, the correct thought. None if the thought is correct} \n"
            qs = qs + "\} \n\n"

            qs = qs + f"[Current Formula] '{text_formula}' \n"
            qs = qs + f"[Thought] {thought} \n"
            qs = qs + f"[Target Formula] '{target_formula}' \n\n"

        return system_prompt, qs
        
    def card2num(self, card):
        tr = ['J', 'Q', 'K', '11', '12', '13']
        if card in tr:
            return '10'
        elif card == 'A':
            return '1'
        else:
            return card

    def points24_tool(self, a, b, c, d):
        return self.solver.get_formula([self.card2num(a), self.card2num(b), self.card2num(c), self.card2num(d)])

class Solver():
    def __init__(self):
        pass
    
    def get_formula(self, digits):
        """
            Code obtained from here: https://rosettacode.org/wiki/24_game/Solve#Python
            This function takes a list of 4 digits and returns
            True if a solution exists, False otherwise.
            If true, we also save the solution.
        """
        digilen = len(digits)
        # length of an exp without brackets
        exprlen = 2 * digilen - 1
        # permute all the digits
        # added shuffle to avoid always the same solution
        digiperm = sorted(set(permutations(digits)))
        random.shuffle(digiperm)
        # All the possible operator combinations
        opcomb = list(product('+-*/', repeat=digilen-1))
        # All the bracket insertion points:
        brackets = ([()] + [(x, y)
                            for x in range(0, exprlen, 2)
                            for y in range(x+4, exprlen+2, 2)
                            if (x, y) != (0, exprlen+1)]
                    + [(0, 3+1, 4+2, 7+3)])  # double brackets case
        solution = []
        for d in digiperm:
            for ops in opcomb:
                if '/' in ops:
                    d2 = [('F(%s)' % i) for i in d]  # Use Fractions for accuracy
                else:
                    d2 = d
                ex = list(chain.from_iterable(zip_longest(d2, ops, fillvalue='')))
                for b in brackets:
                    exp = ex[::]
                    for insertpoint, bracket in zip(b, '()'*(len(b)//2)):
                        exp.insert(insertpoint, bracket)
                    txt = ''.join(str(i) for i in exp)
                    try:
                        num = eval(txt)
                    except ZeroDivisionError:
                        continue
                    if num == 24:
                        if '/' in ops:
                            exp = [(term if not term.startswith('F(') else term[2:-1])
                                for term in exp]
                        ans = ''.join(str(i) for i in exp).rstrip()
                        solution.append(ans)
        return solution