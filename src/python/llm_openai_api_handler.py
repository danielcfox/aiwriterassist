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

load_dotenv()

class LLMOpenAIAPIHandler(LLMHandler):
    """
    Handler for OpenAI API interactions with support for base models and fine-tuning.

    This class provides a comprehensive interface for working with OpenAI's API,
    including sending prompts to models, managing fine-tuning jobs, and storing
    model state information. It supports both base OpenAI models and custom
    fine-tuned models that can be trained on specific writing styles.

    The handler manages:
    1. API authentication via environment variables
    2. Prompt creation and response parsing
    3. Fine-tuning job submission and monitoring
    4. Model state persistence and loading
    5. Writing-specific prompt optimization

    Attributes:
        client (openai.OpenAI): The OpenAI API client instance.
        details_model (openai.openai_object.OpenAIObject): Details of the fine-tuned model.
        details_uri (str): Path to file storing fine-tuned model details.
        model_name (str): Name of the model (base or fine-tuned).
        author_name (str): Author name for stylistic prompts.
        verbose (bool): Whether to output detailed logs.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the OpenAI API handler with model configuration.

        Sets up the handler with either a base model or fine-tuned model
        by loading model details and establishing API connection.

        Parameters:
            **kwargs: Configuration parameters including:
                - model_name (str, required): Base model identifier (e.g., "gpt-4o").
                - details_uri (str, optional): Path to saved fine-tuned model details.
                - author_name (str, optional): Author whose style to emulate.
                - verbose (bool, optional): Enable verbose logging.

        Returns:
            None

        Raises:
            ValueError: If required parameters are missing.
            FileNotFoundError: If specified details_uri doesn't exist.
        """
        self.generate_prompt_only = kwargs.get("generate_prompt_only", False)
        self.fine_tune_submit_filename = kwargs.get("submit_filename", "fine_tune_submit.jsonl")

        super().__init__(**kwargs)

    def _open(self) -> Optional[object]:
        """
        Establish connection to the OpenAI API.

        Creates and returns an OpenAI client using the API key
        from environment variables.

        Returns:
            Optional[object]: The OpenAI API client object if successful.

        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.
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
        Update the current fine-tuned model details from the API.

        Retrieves the most recent state of the fine-tuning job
        from OpenAI's API and updates the local model details.

        Returns:
            None

        Raises:
            ValueError: If self.details_model is not initialized.
        """
        if self.details_model is None:
            self._load_details_model()
        if self.details_model is not None and self.details_model.id is not None and len(self.details_model.id) > 0:
            self.details_model = openai.fine_tuning.jobs.retrieve(self.details_model.id)

    def _get_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the model and get a completion response.

        This method handles validation of parameters, constructs the API request
        with appropriate system and user messages, and processes the response.

        Parameters:
            prompt (str): The input text prompt to send to the model.
            **kwargs: Additional parameters:
                - max_tokens (int, required): Maximum length of response.
                - temperature (float, required): Sampling temperature (0.0-1.0).

        Returns:
            str: Processed text response from the model.

        Raises:
            ValueError: If required arguments are missing or invalid.
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

    def _load_details_model(self) -> None:
        """
        Load fine-tuned model details from a file.

        Attempts to load previously saved model details from the specified
        details_uri, which contains serialized information about a fine-tuning job.

        Returns:
            None

        Note:
            If details_uri is invalid or doesn't exist, this method will fall back
            to using the base model specified in model_name.
        """
        if self.details_uri is None or len(self.details_uri) == 0 or not os.path.exists(self.details_uri):
            print(f"Fine-tuned model has not been created yet. Using base model {self.model_name}...")
            return
        print(f"Loading model details from file {self.details_uri}...")
        with open(self.details_uri, 'rb') as fp:
            self.details_model = pickle.load(fp)
            if self.verbose:
                print(f"self.details_model: {self.details_model}")
            if self.details_model.fine_tuned_model is None:
                self._get_details_fine_tuned_model()

    def get_models_available(self) -> List[str]:
        """
        Retrieve a list of all available models from OpenAI.

        Queries the OpenAI API to get all models the user has access to,
        including base models and fine-tuned models.

        Returns:
            List[str]: List of model identifiers available through the API.
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
        Check if a fine-tuning job is still in progress.

        Determines whether the fine-tuning job associated with the current
        model details is still running or pending completion.

        Returns:
            bool: True if the fine-tuning is still in progress, False otherwise.
        """
        status = self.get_fine_tuned_model_status()
        return status in {'validating_files', 'pending', 'running', 'not_started', 'in_progress', 'incomplete', 'paused'}

    def get_fine_tuned_model_status(self) -> str:
        """
        Get the current status of the fine-tuning job.

        Retrieves and returns the current status string from the API,
        and also updates the local model details and saves them to disk.

        Returns:
            str: Status of the fine-tuning job (e.g., "succeeded", "running", etc.).
        """
        self._get_details_fine_tuned_model()
        if self.details_model is None or self.details_model.id is None:
            return "unknown"
            # raise ValueError("details_model is None. Please set it before using this method.")
        self.details_model = openai.fine_tuning.jobs.retrieve(self.details_model.id)
        # self.details_model_name = self.details_model.fine_tuned_model
        self.dump_details_fine_tuned()
        return self.details_model.status
        # all of the return values for self.details_model.status
        # "succeeded"
        # "failed"
        # "pending"
        # "running"
        # "cancelled"
        # "paused"
        # "stopped"
        # "completed"
        # "deleted"
        # "not_started"
        # "unknown"
        # "in_progress"
        # "incomplete"

    def fine_tune_submit(self, corpus_train_prompt_list: List[tuple[str, str]], **kwargs) -> Optional[object]:
        """
        Submit a fine-tuning job based on training files.

        Creates a new fine-tuning job using the provided training files,
        which should contain scene examples in the author's writing style.

        Parameters:
            train_filename_list (List[str]): Paths to JSON files with training scenes.
            **kwargs: Additional parameters:
                - author_name (str, optional): Name of author whose style to emulate.

        Returns:
            Optional[object]: Fine-tuning job details object if successful, None if failed.

        Notes:
            This method creates a JSONL training file from the input JSON files,
            uploads it to OpenAI, and initiates the fine-tuning process.
        """
        # author_name = kwargs.get("author_name", "Unknown Author")
        # generate_prompt_only = kwargs.get("generate_prompt_only", False)

        training_samples = []
        train_file: Optional[object] = None

        for prompt, response in corpus_train_prompt_list:
            # prompt, response = prompt_and_response
            training_samples.append(
                {"messages": [{"role": "system",
                               "content": f"You are a fiction writer, writing in the style of '{self.author_name}'"},
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response}]
                }
            )

        with open(self.fine_tune_submit_filename, "w", encoding="utf-8") as f:
            for entry in training_samples:
                f.write(json.dumps(entry) + "\n")

        try:
            with open(self.fine_tune_submit_filename, "rb") as f:
                train_file = openai.files.create(file=f, purpose="fine-tune")
                if self.verbose:
                    print(f"Training file uploaded successfully! File ID: {train_file.id}")
        except openai.APIError as e:
            print(f"An error occurred: {e}")
            return None
        except FileNotFoundError:
            print(f"Error: File not found at {self.fine_tune_submit_filename}. Internal bug as this should be created by system.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        if self.generate_prompt_only:
            print(f"Fine-tuning job not submitted. Prompt only requested. Prompt is in {self.fine_tune_submit_filename}")
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
        Wait for a fine-tuning job to complete.

        Polls the API at regular intervals to check if the fine-tuning job
        has completed, up to a maximum wait time.

        Parameters:
            max_seconds_wait (int): Maximum seconds to wait before timing out.

        Returns:
            Optional[str]: Name of the fine-tuned model if successful, None if failed or timed out.
        """
        seconds_wait = 0
        self._load_details_model()
        if self.details_model is None:
            return None
        while self.details_model.fine_tuned_model is None:
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
        Save the current fine-tuned model details to disk.

        Serializes the fine-tuning job details to the file specified by details_uri.

        Returns:
            None
        """
        if self.details_model is not None:
            with open(self.details_uri, 'wb') as fp:
                pickle.dump(self.details_model, fp)

    def get_write_scene_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Generate narrative text based on a scene prompt.

        Specialized method for generating creative writing content using
        either a fine-tuned model or base model with appropriate system prompts.

        Parameters:
            prompt (str): Scene specification or partial scene content.
            **kwargs: Additional parameters:
                - max_tokens (int, required): Maximum length of generated scene.
                - temperature (float, optional): Sampling temperature (0.0-1.0), default 0.7.

        Returns:
            str: Generated scene text.

        Raises:
            ValueError: If required arguments are missing or invalid.
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
            print(f"Prompt length is {len(prompt)}")
        messages = [{"role": "system", "content": f"You are a fiction writer, writing in the style of '{self.author_name}'"},
                    {"role": "user", "content": prompt}
                    ]

        model_name = self._get_model_name()
        if self.verbose:
            print(f"Using model {model_name}")

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # print completion status
        if self.verbose:
            print(f"Completion details: {completion}")
        response = completion.choices[0].message.content.strip()
        cleaned_response = response.lstrip("```json").rstrip("```").strip()
        resp_wordlen = len(cleaned_response.split())
        if self.verbose:
            print(f"Prompt response length {resp_wordlen} words")
        return cleaned_response

    def _get_model_name(self):
        """
        Determine which model to use for current API requests.

        Decides whether to use the fine-tuned model (if available and successful)
        or fall back to the base model specified during initialization.

        Returns:
            str: Name of the model to use for API calls.
        """
        if (self.details_model is not None and self.details_model.fine_tuned_model is not None
            and self.details_model.status == 'succeeded' and len(self.details_model.fine_tuned_model) > 0):
            return self.details_model.fine_tuned_model
        self._get_details_fine_tuned_model()
        if (self.details_model is not None and self.details_model.fine_tuned_model is not None
            and self.details_model.status == 'succeeded' and len(self.details_model.fine_tuned_model) > 0):
            return self.details_model.fine_tuned_model

        return self.model_name
