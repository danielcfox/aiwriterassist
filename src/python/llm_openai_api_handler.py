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

    # def __init__(self, model_spec: str, model: str, **kwargs) -> None:
    def __init__(self, **kwargs) -> None:
        """
        Initialize the LLMOpenAIAPIHandler with a specific model or fine-tuned model details.

        Args:
            model_spec (str): Specifies the type of the model argument.
                Options are:
                - 'openai_details_uri': Indicates that the model argument is a filename
                  containing the object details of a fine-tuned model.
                - 'model_name': Indicates that the model argument is the name of the model
                  available through the GPT-4o API.
            model (str): The model argument, which can either be:
                - A filename (if model_spec is 'openai_details_uri').
                - The name of the model (if model_spec is 'model_name').
            **kwargs: Additional keyword arguments for customization.
        """
        super().__init__(**kwargs)

    def _open(self) -> Optional[object]:
        """
        Open a connection to the OpenAI API.
        :return: The OpenAI API client object.
        :raises ValueError: If the OPENAI_API_KEY environment variable is not set.
        """
         # we assume only use OPENAI_API_KEY environment variable
         # and this makes this class object a singleton
         # You can instantiate this object more than once, but you must specify a different api key
         # which means probably passing in another argument that is the api key environment variable
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before using this class.")
        return openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _get_details_fine_tuned_model(self) -> None:
        """
        Retrieve the details of the fine-tuned model.

        :raises Value error: If `self.details_model` is not set before calling this method.
            'self.details_model' is originally set in fine_tune_submit method
        :return: None
        """
        if self.details_model is None:
            self._load_details_model()
        if self.details_model is not None and self.details_model.id is not None and len(self.details_model.id) > 0:
            self.details_model = openai.fine_tuning.jobs.retrieve(self.details_model.id)

    def _get_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Get a response from the LLM for a given prompt.

        :param prompt: The input prompt for the LLM.
        :param kwargs: Additional arguments, including:
            - max_tokens (int): The maximum number of tokens to generate.
            - temperature (float): The sampling temperature to use (0.0 to 1.0).
        :rtype: str
        :return: The response from the LLM
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

    # def _load_details_model_from_uri(self) -> None:
    #     """
    #     Load the details of the fine-tuned model from a file.
    #     Only local filename of fine-tuned model is supported at this time
    #     :param filename: The name of the file to load the details from.
    #     """
    #     if os.path.exists(self.details_uri):
    #         print(f"Loading model details from file {self.details_uri}...")
    #         with open(self.details_uri, 'rb') as fp:
    #             self.details_model = pickle.load(fp)
    #             print(f"self.details_model: {self.details_model}")
    #             self.model_name = self.details_model.fine_tuned_model
    #             if self.model_name is None:
    #                 self._get_details_fine_tuned_model()
    #                 self.model_name = self.details_model.fine_tuned_model
    #             # if self.model_name is None or not self.model_name.startswith("ft:"):
    #             #     raise ValueError(f"File {filename} does not contain a valid fine-tuned model.")
    #     else:
    #         raise FileNotFoundError(f"The specified uri '{self.details_uri}' does not exist.")

    def _load_details_model(self) -> None:
        """
        Load the details of the fine-tuned model from a file.
        """
        if self.details_uri is None or len(self.details_uri) == 0 or not os.path.exists(self.details_uri):
            # if self.model_name is None:
            #     raise ValueError("The model name is not set and the fine-tuned model has not been created yet.")
            print(f"Fine-tuned model has not been created yet. Using base model {self.model_name}...")
            return
        # if os.path.exists(self.details_uri):
        print(f"Loading model details from file {self.details_uri}...")
        with open(self.details_uri, 'rb') as fp:
            self.details_model = pickle.load(fp)
            if self.verbose:
                print(f"self.details_model: {self.details_model}")
            if self.details_model.fine_tuned_model is None:
                self._get_details_fine_tuned_model()
            # self.details_model_name = self.details_model.fine_tuned_model
            # if self.model_name is None or not self.model_name.startswith("ft:"):
            #     raise ValueError(f"File {filename} does not contain a valid fine-tuned model.")
        # else:
            # raise FileNotFoundError(f"The specified uri '{self.details_uri}' does not exist.")

    def get_models_available(self) -> List[str]:
        """
        Retrieve the list of available models from OpenAI API.
        :return: The list of available models.
        """
        try:
            models_available = openai.models.list()
            model_names = [model_obj.id for model_obj in models_available.data]
            return model_names
        except Exception as e:
            print(f"Error retrieving models: {e}")
            return []

    def get_fine_tuned_model_pending(self) -> bool:
        """
        Check if the fine-tuned model is still pending or running.
        :param details_model: The details of the fine-tuned model.
            'details_model' is originally set in fine_tune_submit method
        :rtype: bool
        :return:
            1. True if the fine-tuned model is pending or running.
            2. False otherwise.
        """
        status = self.get_fine_tuned_model_status()
        return status in {'validating_files', 'pending', 'running', 'not_started', 'in_progress', 'incomplete', 'paused'}

    def get_fine_tuned_model_status(self) -> str:
        """
        Get the status of the fine-tuned model.
        :return: The status of the fine-tuned model.
        """
        self._get_details_fine_tuned_model()
        if self.details_model is None or self.details_model.id is None:
            return "unknown"
            # raise ValueError("details_model is None. Please set it before using this method.")
        self.details_model = openai.fine_tuning.jobs.retrieve(self.details_model.id)
        # self.details_model_name = self.details_model.fine_tuned_model
        self.dump_details_fine_tuned()
        return self.details_model.status
        # show me all of the return values for self.details_model.status
        # return self.details_model.status
        # return self.details_model.status == "succeeded"
        # return self.details_model.status == "failed"
        # return self.details_model.status == "pending"
        # return self.details_model.status == "running"
        # return self.details_model.status == "cancelled"
        # return self.details_model.status == "paused"
        # return self.details_model.status == "stopped"
        # return self.details_model.status == "completed"
        # return self.details_model.status == "deleted"
        # return self.details_model.status == "not_started"
        # return self.details_model.status == "unknown"
        # return self.details_model.status == "in_progress"
        # return self.details_model.status == "incomplete"

    def fine_tune_submit(self, train_filename_list: List[str], **kwargs) -> Optional[object]:
        """
        Fine-tune the model on writing style using OpenAI API.
        :param train_filename_list: List of training file names.
        :param author_name: Name of the author whose style is being fine-tuned.
        :return: The details of the fine-tuned model.
        """
        author_name = kwargs.get("author_name", "Unknown Author")
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

            print(f"From {train_filename}: Number of training samples: {len(training_samples)}")

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
            model=self.model_name
        )

        print(f"Fine-tuning job submitted: {self.details_model}")
        self.dump_details_fine_tuned()
        return self.details_model

    def wait_fine_tuning_model(self, max_seconds_wait: int) -> Optional[str]:
        """
        Wait for the fine-tuning job to complete.
        :param details_uri: The filename to save the fine-tuned model details.
        :param max_seconds_wait: Maximum seconds to wait for the fine-tuning job to complete.
        :return: The name of the fine-tuned model.
        """
        seconds_wait = 0
        self._load_details_model()
        if self.details_model is None:
            return None
        # self.details_model.id = 'ftjob-dbM0L56YYCX9Ihg3CabpvdW1'
        # self.dump_details_fine_tuned(details_uri)
        # self._load_details_model_from_uri(details_uri)
        while self.details_model.fine_tuned_model is None:
            # if self._get_details_fine_tuned_model() is None:
            #     print("Fine-tuning job failure status: unknown")
            #     return None
            # self.dump_details_fine_tuned(details_uri)
            # print(f"Status of fine-tuning:\n\n{self.details_model}")
            # if self.details_model is None:
            #     return None
            if self.get_fine_tuned_model_pending():
                if seconds_wait >= max_seconds_wait:
                    return None
                time.sleep(min(max_seconds_wait - seconds_wait, 60))  # Check every 60 seconds
                seconds_wait += 60
                continue
            if self.details_model.status == "succeeded":
                print(f"Fine-tuning job completed successfully! Fine-tuned model ID: {self.details_model.id}, "
                      + f"Name: {self.details_model.fine_tuned_model}")
                return self.details_model.fine_tuned_model
            # if self.details_model.status == "failed":
            print(f"Fine-tuning job status: {self.details_model.status}, error: {self.details_model.error}")
            return None
        if self.details_model.status == "succeeded":
            print(f"Fine-tuning job completed successfully! Fine-tuned model ID: {self.details_model.id}, "
                    + f"Name: {self.details_model.fine_tuned_model}")
            return self.details_model.fine_tuned_model
        print(f"Fine-tuning job did not complete. Status: {self.details_model.status}, "
              + f"Fine-tuned model ID: {self.details_model.id}, "
              + f"Name: {self.details_model.fine_tuned_model}")
        return self.details_model.fine_tuned_model

    def dump_details_fine_tuned(self) -> None:
        """
        Save the details of the fine-tuned model to a file.
        :param filename: The name of the file to save the details.
        """
        if self.details_model is not None:
            with open(self.details_uri, 'wb') as fp:
                pickle.dump(self.details_model, fp)

    def get_write_scene_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Get a response from the LLM for a given prompt.
        :param prompt: The input prompt for the LLM."
        :params kwargs: Additional keyword arguments for the LLM.
                        - author_name: The name of the author to include in the prompt.
                        - max_tokens: The maximum number of tokens to generate.
                        - temperature: The sampling temperature to use.
                        - verbose: If True, print verbose output.
        :return: The response from the LLM.
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
        messages = [{"role": "system", "content": f"You are fiction writer '{self.author_name}'"},
                    {"role": "user", "content": f"Write a fiction scene in the style of the writer named '{self.author_name}'"},
                    {"role": "assistant", "content": prompt}
                    ]

        completion = self.client.chat.completions.create(
            model=self._get_model_name(),
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

    def _get_model_name(self):
        if (self.details_model is not None and self.details_model.fine_tuned_model is not None
            and self.details_model.status == 'succeeded' and len(self.details_model.fine_tuned_model) > 0):
            return self.details_model.fine_tuned_model
        self._get_details_fine_tuned_model()
        if (self.details_model is not None and self.details_model.fine_tuned_model is not None
            and self.details_model.status == 'succeeded' and len(self.details_model.fine_tuned_model) > 0):
            return self.details_model.fine_tuned_model

        return self.model_name
