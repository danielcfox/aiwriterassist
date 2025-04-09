#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:07:22 2025

@author: dfox
"""

from dotenv import load_dotenv
import json
import openai
import os
import pickle
import time

from llm_handler import LLMHandler

load_dotenv()

class LLMOpenAIAPIHandler(LLMHandler):
    """
    Class to handle OpenAI API interactions for LLMs.
    This class provides methods to interact with OpenAI's API, including
    sending prompts, receiving responses, and managing fine-tuning jobs.
    """
    def __init__(self, model='gpt-4o-mini'):
        """
        Initialize the LLMOpenAIAPIHandler with a specific model and optional fine-tuned model.
        :param type str, model: The name of the model to use.
        :param type str, use_fine_tuned_model: Optional path to a fine-tuned model file.
        """
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.details_training_file = None
        self.details_fine_tuned_model = None
        self.train_file = None
        
        if os.path.exists(model):
            print(f"Loading fine-tuned model from file {model}...")
            self.load_details_fine_tuned(model)
            # above sets self.model_name to the fine-tuned model name
        else:
            self.model_name = model

        # self.model_name = None
        # self.use_fine_tuned_model = None

        # for key, value in kwargs.items():
        #     if key == 'model_name':
        #         self.model_name = value
        #     elif key == 'model_filename':
        #         self.use_fine_tuned_model = value
        
        # if self.use_fine_tuned_model is not None:
        #      if os.path.exists(self.use_fine_tuned_model):
        #         print(f"Loading fine-tuned model from file {self.use_fine_tuned_model}...")
        #         self.load_details_fine_tuned(self.use_fine_tuned_model)
        #         self.model_name = self.details_fine_tuned_model.fine_tuned_model

        # if self.model_name is None:
        #     self.model_name = 'gpt-4o-mini'

        super().__init__(self.model_name)

    def get_details_fine_tuned_model(self):
        """
        Retrieve the details of the fine-tuned model.
        :return: type openai.openai_object.OpenAIObject, The details of the fine-tuned model.
        """
        if self.details_fine_tuned_model is not None:
        #     self.details_fine_tuned_model = openai.fine_tuning.jobs.retrieve("ftjob-o6QngDQg79l8lZS51hmtsTCj")
        #     print(self.details_fine_tuned_model)
        # else:
            self.details_fine_tuned_model = openai.fine_tuning.jobs.retrieve(self.details_fine_tuned_model.id)
        return self.details_fine_tuned_model

    def get_fine_tuned_model_pending(self):
        """
        Check if the fine-tuned model is still pending or running.
        :return: type bool, True if the fine-tuned model is pending or running, False otherwise.
        """
        if self.details_fine_tuned_model is None:
            return False
        self.details_fine_tuned_model = openai.fine_tuning.jobs.retrieve(self.details_fine_tuned_model.id)
        if self.details_fine_tuned_model.status == "pending" or self.details_fine_tuned_model.status == "running":
            return True
        return False

    def get_prompt_response(self, prompt, max_output_tokens, temperature):
        """
        Get a response from the LLM for a given prompt.
        :param type str, prompt: The input prompt for the LLM.
        :param type int, max_tokens: The maximum number of tokens to generate.
        :param type float, temperature: The sampling temperature to use.
        :return: The response from the LLM.
        """
        # print(f"Prompt:\n\n{prompt}\n\n")
        messages = [
            {"role": "system",
             "content": "You are an assistant to the author of the work given. Always follow instructions exactly. Be precise."},
            {"role": "user", "content": prompt}
        ]

        # if self.details_fine_tuned_model is not None:
        #     model = self.details_fine_tuned_model.fine_tuned_model
        # else:
        #     model = self.model
        # print(f"model: {model}")

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature
        )

        response = completion.choices[0].message.content.strip()
        cleaned_response = response.lstrip("```json").rstrip("```").strip()
        resp_wordlen = len(cleaned_response.split())
        # print(f"Prompt Response (length {resp_wordlen} words):\n\n{cleaned_response}\n\n")
        return cleaned_response

    def get_write_scene_prompt_response(self, prompt, author_name, max_output_tokens, temperature):
        """
        Get a response from the LLM for a given prompt to write a scene.
        :param type str, prompt: The input prompt for the LLM.
        :param type int, max_tokens: The maximum number of tokens to generate.
        :param type float, temperature: The sampling temperature to use.
        :return: The response from the LLM.
        """
        # print(f"Prompt:\n\n{prompt}\n\n")
        messages = [{"role": "system", "content": f"You are fiction writer '{author_name}'"}, 
                    {"role": "user", "content": f"Write a fiction scene in the style of the writer named '{author_name}'"},
                    {"role": "assistant", "content": prompt}
                    ]

        # if self.details_fine_tuned_model is not None:
        #     model = self.details_fine_tuned_model.fine_tuned_model
        # else:
        #     model = self.model
        # print(f"model: {model}")

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature
        )

        response = completion.choices[0].message.content.strip()
        cleaned_response = response.lstrip("```json").rstrip("```").strip()
        resp_wordlen = len(cleaned_response.split())
        # print(f"Prompt Response (length {resp_wordlen} words):\n\n{cleaned_response}\n\n")
        return cleaned_response

    def fine_tune_submit(self, train_filename_list, author_name):
        """Fine-tune GPT-4o-mini on writing style using OpenAI API.
        :param type list, train_filename_list: List of training file names.
        :param type str, author_name: Name of the author whose style is being fine-tuned.
        :param type int, maxwait: Maximum seconds to wait for the fine-tuning job to complete.
        :return: type openai.openai_object.OpenAIObject, The details of the fine-tuned model.
        """
        # self.fine_tuned_model_id = None
        self.fine_tuned_waiting = False
        self.status_fine_tuned_model = None # 'fine_tuned_model' contains the id of the finetuned_model

        training_samples = []

        for train_filename in train_filename_list:

            with open(train_filename, "r", encoding="utf-8") as fp:
                dataset = json.load(fp)

            training_samples.extend(
                [
                    {"messages": [{"role": "system", "content": f"You are fiction writer '{author_name}'"},
                                  {"role": "user", 
                                   "content": f"Write a fiction scene in the style of the writer named '{author_name}'"},
                                  {"role": "assistant", "content": scene["body"]}
                                 ]
                    }
                    for scene in dataset['scene_list']
                ]       
            )
                # {"prompt": "Write a fiction scene in the style of the writer named 'D.C.P. Fox'",
                #  "completion": scene["body"]}

        # print(f"training samples ({len(training_samples)}):")
        # print(json.dumps(training_samples, indent=4))

        with open("training_data.jsonl", "w", encoding="utf-8") as f:
            for entry in training_samples:
                f.write(json.dumps(entry) + "\n")

        try:
            with open("training_data.jsonl", "rb") as f:
                self.train_file = openai.files.create(file=f, purpose="fine-tune")
                print(f"Training file uploaded successfully! File ID: {self.train_file.id}")

        except openai.APIError as e:
            print(f"An error occurred: {e}")
            return None

        except FileNotFoundError:
            print(f"Error: File not found at training_data.jsonl")
            return None

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        self.details_fine_tuned_model = self.client.fine_tuning.jobs.create(
            training_file=self.train_file.id,
            model="gpt-4o-mini-2024-07-18"
            # method={
            #     "type": "supervised",
            #     "supervised": {
            #         "hyperparameters": {"n_epochs": 4,
            #                             "batch_size": 8,
            #                             "learning_rate_multiplier": 0.1}
            #         }
            #     }
        )

        print(f"Fine-tuning job submitted: {self.details_fine_tuned_model}")
        return self.details_fine_tuned_model

    def wait_fine_tuning_model(self, details_filename, max_seconds_wait):
        """
        Wait for the fine-tuning job to complete.
        :param type int, max_seconds_wait: Maximum seconds to wait for the fine-tuning job to complete.
        :return: type str, The ID of the fine-tuned model.
        """

        # Wait for fine-tuning job to complete (polling)
        seconds_wait = 0
        while True:
            self.get_details_fine_tuned_model()
            print(f"status of fine-tuning:\n\n{self.details_fine_tuned_model}")
            if self.details_fine_tuned_model is None:
                return None
            if self.details_fine_tuned_model.status != 'succeeded':
                if seconds_wait >= max_seconds_wait:
                    return None
                time.sleep(min(max_seconds_wait - seconds_wait, 60))  # Check every 60 seconds
                seconds_wait += 60
                continue
            self.dump_details_fine_tuned(details_filename)
            self.model_name = self.details_fine_tuned_model.fine_tuned_model
            print(f"Fine-tuning job completed successfully! Fine-tuned model ID: {self.model_name}")
            return self.model_name
        
    def dump_details_fine_tuned(self, filename):
        """
        Save the details of the fine-tuned model to a file.
        :param type str, filename: The name of the file to save the details.
        """
        if self.details_fine_tuned_model is not None:
            print(self.details_fine_tuned_model)
            with open(filename, 'wb') as fp:
                pickle.dump(self.details_fine_tuned_model, fp)

    def load_details_fine_tuned(self, filename):
        """
        Load the details of the fine-tuned model from a file.
        :param type str, filename: The name of the file to load the details from.
        """
        with open(filename, 'rb') as fp:
            self.details_fine_tuned_model = pickle.load(fp)
            model_name = self.details_fine_tuned_model.fine_tuned_model
            if model_name.startswith("ft:"):
                self.model_name = model_name
            else:
                raise ValueError(f"File {filename} does not contain a valid fine-tuned model.")



