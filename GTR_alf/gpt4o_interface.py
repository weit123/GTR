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

    # wrapper for consistent openai api access
    @retry(wait=wait_random_exponential(min=1, max=15))
    def generate_correction(self, img, thoughts, task, action_history, admissible_actions, obs_text, temperature=0):
        base64_image = self.image_to_base64(img)
        system_prompt, qs = self.get_correction_prompt(thoughts, task, action_history, admissible_actions, obs_text)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": qs},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def image_to_base64(self, data):
        img = Image.fromarray(data)
        byte_data = BytesIO()
        img.save(byte_data, format="JPEG")
        base64_image = base64.b64encode(byte_data.getvalue()).decode("utf-8")
        return base64_image

    def get_correction_prompt(self, thought, task, action_history, admissible_actions):
        refomratted_admissible_actions = ", ".join(f"'{s}'" for s in admissible_actions)
        system_prompt = f"You are an expert in the ALFRED Embodied Environment. The environment requires the player to navigate, take certain objects, interact with objects if necessary, and finally put objects in the designated place to complete the task.\n"
        qs = f"You will be given the visual observation and thought of a player in this environment. The task is to {task}."
        qs = qs + f"You are also given the previous actions the player has taken: {action_history}. "
        qs = qs + f"All admissible actions of the current situation are: [{refomratted_admissible_actions}]. \n"
        qs = qs + "Please evaluate if the reasoning is correct by the following aspects:\n"
        qs = qs + "(1) What objects are in your sight and whether you are holding a certain object? Does the thought correctly identify the image?\n"
        qs = qs + "(2) Based on the task description and the action history, what should be the player's next sub-goal (notice that the tasks require the player to first pick up certain objects, interact with receptacles if the task is cooling, heating, cleaning or looking in light, and finally placing the object)? Does the thought align with the sub-goal?\n"
        qs = qs + "(3) Based on the task description and the action history, does the player choose one of the admissible actions to reach the sub-goal? Does the action take effect? If the target object is not in sight, go to an unexplored location; if there is a required object, take it; if the task requires cooling, heating, cleaning, or looking in light, navigate and interact with the receptacles.\n"
        
        qs = qs + "Please briefly answer the above questions, then give your final evaluation. If the thought is incorrect, include all available information for thought correction: choose one correct step from the admissible actions for the player to finish the task, and finally provide the correct thought. \n"

        qs = qs + "Your response should be a valid json file in the following format: \{\n"
        qs = qs + " \"answer1\": {Text, answer to the first quesion}, \n"
        qs = qs + " \"answer2\": {Text, answer to the second quesion}, \n"
        qs = qs + " \"answer3\": {Text, answer to the third quesion}, \n" 
        qs = qs + " \"evaluation\": {YES or NO}, \n"
        qs = qs + " \"correction\": {Json object, the correct thought. None if the thought is correct} \n"
        qs = qs + "\} \n\n"

        qs = qs + f"[The player's thought] {thought}"

        return system_prompt, qs 