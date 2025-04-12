#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:07:22 2025

@author: dfox
"""

from dotenv import load_dotenv
import json
from llm_handler import LLMHandler
import openai
import os
import pickle
import time
from typing import List, Optional, Union

# openai.openai_object.object
load_dotenv()

class LLMOpenAIAPIHandler(LLMHandler):
    """
    Class to handle OpenAI API interactions for LLMs.
    This class provides methods to interact with OpenAI's API, including:
    - Sending prompts and receiving responses.
    - Managing fine-tuning jobs.
    - Retrieving model details and available models.

    Attributes:
        client (openai.OpenAI): The OpenAI API client object.
        details_model (Optional[object]): Details of the fine-tuned model.
        model_name (str): The name of the model being used.
        verbose (bool): Flag to enable verbose logging.
    """

    def __init__(self, model_spec: str, model: str, **kwargs) -> None:
        """
        Initialize the LLMOpenAIAPIHandler with a specific model or fine-tuned model details.

        Args:
            model_spec (str): Specifies the type of the model argument.
                Options are:
                - 'openai_details_filename': Indicates that the model argument is a filename
                  containing the object details of a fine-tuned model.
                - 'model_name': Indicates that the model argument is the name of the model
                  available through the GPT-4o API.
            model (str): The model argument, which can either be:
                - A filename (if model_spec is 'openai_details_filename').
                - The name of the model (if model_spec is 'model_name').
            **kwargs: Additional keyword arguments for customization.
        """
        super().__init__(model_spec, model, **kwargs)

    def _open(self) -> Optional[object]:
        """
        Open a connection to the OpenAI API.
        :return: The OpenAI API client object.
        """
         # we assume only use OPENAI_API_KEY environment variable
         # and this makes this class object a singleton
         # You can instantiate this object more than once, but you must specify a different api key
         # which means probably passing in another argument that is the api key environment variable
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before using this class.")
        return openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_models_available(self) -> List[str]:
        """
        Retrieve the list of available models from OpenAI API.
        :return: The list of available models.
        """
        return []
        # disabled for now
        try:
            models_available = openai.api_resources.Model.list()
            # models_available = openai.Model.list()
            model_names = [model_obj.id for model_obj in models_available.data]
            return model_names
        except Exception as e:
            print(f"Error retrieving models: {e}")
            return []
        
    def _get_details_fine_tuned_model(self) -> None:
        """
        Retrieve the details of the fine-tuned model.

        :raises ValueError: If `details_model` is not set before calling this method.
        """
        if self.details_model is None:
            raise ValueError("details_model is None. Please set it before using this method.")
        self.details_model = openai.fine_tuning.jobs.retrieve(self.details_model.id)

    def get_fine_tuned_model_pending(self, details_model) -> bool:
        """
        Check if the fine-tuned model is still pending or running.
        :param details_model: The details of the fine-tuned model.

        :rtype: object 
        :rtype: bool
        :return:
            1. The details of the fine-tuned model.
            2. True if the fine-tuned model is pending or running.
            3. False otherwise.
        """
        self.details_model = openai.fine_tuning.jobs.retrieve(details_model.id)
        return self.details_model.status in {"pending", "running"}

    def _get_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Get a response from the LLM for a given prompt.

        :param prompt: The input prompt for the LLM.
        :param kwargs: Additional arguments, including:
            - max_tokens (int): The maximum number of tokens to generate.
            - temperature (float): The sampling temperature to use (0.0 to 1.0).
        :return: The response from the LLM as a string.
        :raises ValueError: If required arguments are missing or invalid.
        """
        for key in kwargs:
            if key not in ["max_tokens", "temperature"]:
                raise ValueError(f"Invalid argument '{key}'. Only 'max_tokens' and 'temperature' are allowed.")
        if "max_tokens" not in kwargs:
            raise ValueError("max_tokens is required in kwargs.")
        max_tokens = kwargs.get("max_tokens")
        if self.verbose:
            print(f"max_tokens: {max_tokens}")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if "temperature" not in kwargs:
            raise ValueError("temperature is required in kwargs.")
        temperature = kwargs.get("temperature")
        if self.verbose:
            print(f"temperature: {temperature}")
        if not isinstance(temperature, (float)) or temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be a float between 0.0 and 1.0.")
        temperature = kwargs.get("temperature", 0.7)

        messages = [
            {"role": "system", "content": "You are an assistant to the author of the work given. Always follow instructions exactly. Be precise."},
            {"role": "user", "content": prompt}
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response = completion.choices[0].message.content.strip()
        return response.lstrip("```json").rstrip("```").strip()

    def fine_tune_submit(self, train_filename_list: List[str], author_name: str) -> Optional[object]:
        """
        Fine-tune the model on writing style using OpenAI API.
        :param train_filename_list: List of training file names.
        :param author_name: Name of the author whose style is being fine-tuned.
        :return: The details of the fine-tuned model.
        """
        training_samples = []
        train_file: Optional[object] = None

        for train_filename in train_filename_list:
            with open(train_filename, "r", encoding="utf-8") as fp:
                dataset = json.load(fp)

            training_samples.extend(
                [
                    {"messages": [{"role": "system", "content": f"You are fiction writer '{author_name}'"},
                                  {"role": "user", "content": f"Write a fiction scene in the style of the writer named '{author_name}'"},
                                  {"role": "assistant", "content": scene["body"]}]
                    }
                    for scene in dataset['scene_list'] if len(scene["body"]) > 0
                ]
            )

        with open("training_data.jsonl", "w", encoding="utf-8") as f:
            for entry in training_samples:
                f.write(json.dumps(entry) + "\n")

        try:
            with open("training_data.jsonl", "rb") as f:
                train_file = openai.files.create(file=f, purpose="fine-tune")
                if self.verbose:
                    print(f"Training file uploaded successfully! File ID: {train_file.id}")
        except openai.APIError as e:
            print(f"An error occurred: {e}")
            return None
        except FileNotFoundError:
            print(f"Error: File not found at training_data.jsonl")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        self.details_model = self.client.fine_tuning.jobs.create(
            training_file=train_file.id,
            model="gpt-4o-mini-2024-07-18"
        )

        print(f"Fine-tuning job submitted: {self.details_model}")
        return self.details_model

    def wait_fine_tuning_model(self, details_filename: str, max_seconds_wait: int) -> Optional[str]:
        """
        Wait for the fine-tuning job to complete.
        :param details_filename: The filename to save the fine-tuned model details.
        :param max_seconds_wait: Maximum seconds to wait for the fine-tuning job to complete.
        :return: The name of the fine-tuned model.
        """
        seconds_wait = 0
        while True:
            self._get_details_fine_tuned_model()
            self.dump_details_fine_tuned(details_filename)
            print(f"Status of fine-tuning:\n\n{self.details_model}")
            if self.details_model is None:
                return None
            if self.details_model.status != 'succeeded':
                if seconds_wait >= max_seconds_wait:
                    return None
                time.sleep(min(max_seconds_wait - seconds_wait, 60))  # Check every 60 seconds
                seconds_wait += 60
                continue
            self.model_name = self.details_model.fine_tuned_model
            print(f"Fine-tuning job completed successfully! Fine-tuned model ID: {self.model_name}")
            return self.model_name

    def dump_details_fine_tuned(self, filename: str) -> None:
        """
        Save the details of the fine-tuned model to a file.
        :param filename: The name of the file to save the details.
        """
        if self.details_model is not None:
            with open(filename, 'wb') as fp:
                pickle.dump(self.details_model, fp)

    def _load_details_model_from_uri(self, filename: str) -> None:
        """
        Load the details of the fine-tuned model from a file.
        Only local filename of fine-tuned model is supported at this time
        :param filename: The name of the file to load the details from.
        """
        if os.path.exists(filename):
            print(f"Loading model details from file {filename}...")
            with open(filename, 'rb') as fp:
                self.details_model = pickle.load(fp)
                self.model_name = self.details_model.fine_tuned_model
                if not self.model_name.startswith("ft:"):
                    raise ValueError(f"File {filename} does not contain a valid fine-tuned model.")
        else:
            raise FileNotFoundError(f"The specified uri '{filename}' does not exist.")
    
    def get_write_scene_prompt_response(self, prompt: str, author_name: str, **kwargs) -> str:
        """
        Get a response from the LLM for a given prompt to write a scene.

        :param prompt: The input prompt for the LLM.
        :param author_name: The name of the author whose style the LLM should emulate.
        :param kwargs: Additional arguments, including:
            - max_tokens (int): The maximum number of tokens to generate.
            - temperature (float): The sampling temperature to use (0.0 to 1.0).
        :return: The response from the LLM as a string.
        :raises ValueError: If required arguments are missing or invalid.
        """
        for key in kwargs:
            if key not in ["max_tokens", "temperature"]:
                raise ValueError(f"Invalid argument '{key}'. Only 'max_tokens' and 'temperature' are allowed.")
        max_tokens = kwargs.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        temperature = kwargs.get("temperature", 0.7)
        if not isinstance(temperature, (float)) or temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be a float between 0.0 and 1.0.")

        if self.verbose:
            print(f"Prompt:\n\n{prompt}\n\n")
        messages = [{"role": "system", "content": f"You are fiction writer '{author_name}'"}, 
                    {"role": "user", "content": f"Write a fiction scene in the style of the writer named '{author_name}'"},
                    {"role": "assistant", "content": prompt}
                    ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        response = completion.choices[0].message.content.strip()
        cleaned_response = response.lstrip("```json").rstrip("```").strip()
        resp_wordlen = len(cleaned_response.split())
        if self.verbose:
            print(f"Prompt Response (length {resp_wordlen} words):\n\n{cleaned_response}\n\n")
        return cleaned_response

