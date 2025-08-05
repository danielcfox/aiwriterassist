#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:07:22 2025

Copyright 2025 Daniel C. Fox

@author: dfox
"""

from dotenv import load_dotenv
import openai
import os
import traceback
from typing import List, Optional, Union

from llm_api_handler import LLMAPIHandler
from llm_api_training_state import LLMAPITrainingState

load_dotenv()

class LLMOpenAIAPIHandler(LLMAPIHandler):
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
        details_filename (str): Path to file storing fine-tuned model details.
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
                - details_filename (str, optional): Path to saved fine-tuned model details.
                - author_name (str, optional): Author whose style to emulate.
                - verbose (bool, optional): Enable verbose logging.

        Returns:
            None

        Raises:
            ValueError: If required parameters are missing.
            FileNotFoundError: If specified details_filename doesn't exist.
        """
        super().__init__(**kwargs)
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before using this class.")
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

    def fine_tune_submit(self, corpus_train_prompt_list: List[tuple[str, str]]) -> Optional[object]:
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
        self._write_training_samples(corpus_train_prompt_list)
        if self.generate_prompt_only:
            return None

        train_file_handle = self._upload_training_file()
        if train_file_handle is None:
            print("Error: Failed to upload training file.")
            return None

        self._create_training_job(train_file_handle)

    def _get_models_available(self) -> List[str]:
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
            raise

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

        temperature = kwargs.get("temperature", 0.7)
        if not isinstance(temperature, (float, int)) or temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be a float between 0.0 and 1.0.")

        messages = self._format_inference(prompt)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response = completion.choices[0].message.content.strip()
        return response.lstrip("```json").rstrip("```").strip()

    def _upload_training_file(self):
        """
        Upload the training file to OpenAI for fine-tuning.

        Parameters:
            None

        Returns:
            Optional[object]: Handle to the uploaded training file or None if failed.
        """
        try:
            with open(self.fine_tune_submit_filename, "rb") as f:
                train_file_handle = openai.files.create(file=f, purpose="fine-tune")
                if train_file_handle is None:
                    print("Error: Failed to upload training file.")
                    return None
                if self.verbose:
                    print(f"Training file uploaded successfully! File ID: {train_file_handle.id}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.fine_tune_submit_filename}. Internal bug as this should be created by system.")
            raise

        return train_file_handle

    def _create_training_job(self, train_file_handle: object) -> None:
        """
        Create a fine-tuning job using the uploaded training file.

        Submits a fine-tuning job to OpenAI using the specified training file.

        Parameters:
            train_file_handle (object): Handle to the uploaded training file.

        Returns:
            None
        """
        try:
            if self.verbose:
                print(f"Starting fine-tuning job for model {self.model_name} with training file {train_file_handle.id}")

            # Create the fine-tuning job using OpenAI API
            openai_job_object = self.client.fine_tuning.jobs.create(
                training_file=train_file_handle.id,
                model=self.model_name
            )

            if openai_job_object is None:
                print("Error: Failed to create fine-tuning job.")
                return None

            # Create LLMAPITrainingState instance and set it with the job details
            self.details_fine_tuned_model = LLMAPITrainingState()
            self.details_fine_tuned_model.set(
                base_model_name=self.model_name,
                training_client=self.client,  # Pass the OpenAI client
                training_data_id=train_file_handle.id,
                training_job_object=openai_job_object,
                api_type="openai"
            )

            if self.verbose:
                print(f"Fine-tuning job submitted successfully:")
                print(f"  Job ID: {self.details_fine_tuned_model.job_id}")
                print(f"  Base model: {self.model_name}")
                print(f"  Training file: {train_file_handle.id}")
                print(f"  Status: {self.details_fine_tuned_model.status}")

            self.dump_details_fine_tuned()

        except Exception as e:
            print(f"Error creating fine-tuning job: {e}")
            if self.verbose:
                traceback.print_exc()
            self.details_fine_tuned_model = None
            raise
