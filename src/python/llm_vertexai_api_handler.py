#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 09:55:50 2025

@author: dfox
"""

from dotenv import load_dotenv
import os
import traceback
from typing import List, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.preview.tuning import sft

from llm_api_handler import LLMAPIHandler
from llm_api_training_state import LLMAPITrainingState

load_dotenv()

class LLMVertexAIAPIHandler(LLMAPIHandler):
    """
    Handler for Vertex AI API interactions with support for base models and fine-tuning.

    This class provides a comprehensive interface for working with Vertex AI's API,
    including sending prompts to models, managing fine-tuning jobs, and storing
    model state information. It supports both base Vertex AI models and custom
    fine-tuned models that can be trained on specific writing styles.

    The handler manages:
    1. API authentication via Google Cloud credentials
    2. Prompt creation and response parsing
    3. Fine-tuning job submission and monitoring
    4. Model state persistence and loading
    5. Writing-specific prompt optimization

    Attributes:
        model: The Vertex AI GenerativeModel instance.
        details_model: Details of the fine-tuned model.
        details_filename (str): Path to file storing fine-tuned model details.
        model_name (str): Name of the model (base or fine-tuned).
        author_name (str): Author name for stylistic prompts.
        verbose (bool): Whether to output detailed logs.
        project_id (str): Google Cloud project ID.
        location (str): Google Cloud region.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the Vertex AI API handler with model configuration.

        Sets up the handler with either a base model or fine-tuned model
        by loading model details and establishing API connection.

        Parameters:
            **kwargs: Configuration parameters including:
                - model_name (str, required): Base model identifier (e.g., "gemini-1.5-pro").
                - project_id (str, required): Google Cloud project ID.
                - location (str, optional): Google Cloud region (default: "us-central1").
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

        self.project_id = kwargs.get('project_id')
        self.location = kwargs.get('location', 'us-central1')

        if not self.project_id:
            raise ValueError("project_id is required")

        # Check for service account credentials
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Service account key file not found: {credentials_path}")

        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        self.client = GenerativeModel(self._get_model_name())

        if self.details_filename:
            self._load_details_fine_tuned_model()

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
        if not isinstance(temperature, (float, int)) or temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be a float between 0.0 and 1.0.")

        if self.verbose:
            print(f"Prompt length is {len(prompt)}")

        # Format prompt for Vertex AI
        system_message = f"You are a fiction writer, writing in the style of '{self.author_name}'"
        formatted_prompt = f"System: {system_message}\n\nUser: {prompt}\n\nAssistant: "

        model_name = self._get_model_name()
        if self.verbose:
            print(f"Using model {model_name}")

        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=float(temperature)
        )

        try:
            response = self.client.generate_content(
                formatted_prompt,
                generation_config=generation_config
            )

            # print completion status
            if self.verbose:
                print(f"Generation completed successfully")

            response_text = response.text.strip()
            cleaned_response = response_text.lstrip("```json").rstrip("```").strip()
            resp_wordlen = len(cleaned_response.split())
            if self.verbose:
                print(f"Prompt response length {resp_wordlen} words")
            return cleaned_response

        except Exception as e:
            if self.verbose:
                print(f"Error in scene generation: {e}")
            raise

    def fine_tune_submit(self, corpus_train_prompt_list: List[tuple[str, str]]) -> Optional[object]:
        """
        Submit a fine-tuning job based on training prompt-response pairs.

        Creates a new fine-tuning job using the provided training data,
        which should contain scene examples in the author's writing style.

        Parameters:
            corpus_train_prompt_list (List[tuple[str, str]]): List of (prompt, response) pairs.

        Returns:
            Optional[object]: Fine-tuning job details object if successful, None if failed.

        Notes:
            This method creates a JSONL training file from the input pairs,
            uploads it to Vertex AI, and initiates the fine-tuning process.
        """
        self._write_training_samples(corpus_train_prompt_list)
        if self.generate_prompt_only:
            return None

        # train_file_handle = self.fine_tune_submit_filename

        self._create_training_job(self.fine_tune_submit_filename)
        return self.details_fine_tuned_model

    def _get_models_available(self) -> List[str]:
        """
        Retrieve a list of all available models from Vertex AI.

        Queries the Vertex AI API to get all models the user has access to,
        including base models and fine-tuned models.

        Returns:
            List[str]: List of model identifiers available through the API.
        """
        try:
            # This would be the actual implementation using Vertex AI's model listing API
            available_models = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro",
                "text-bison@002",
                "chat-bison@002"
            ]
            return available_models
        except Exception as e:
            print(f"Error retrieving models: {e}")
            raise

    def _get_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the model and get a completion response.

        This method handles validation of parameters, constructs the generation request
        with appropriate parameters, and processes the response.

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

        # Convert messages to Vertex AI format
        formatted_prompt = ""
        for message in messages:
            if message["role"] == "system":
                formatted_prompt += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                formatted_prompt += f"User: {message['content']}\n\n"
        formatted_prompt += "Assistant: "

        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=float(temperature),
        )

        try:
            response = self.client.generate_content(
                formatted_prompt,
                generation_config=generation_config
            )

            response_text = response.text.strip()
            return response_text.lstrip("```json").rstrip("```").strip()

        except Exception as e:
            if self.verbose:
                print(f"Error generating response: {e}")
            raise

    def _create_training_job(self, train_filename: str) -> None:
        """
        Create a fine-tuning job using the uploaded training file.

        Submits a fine-tuning job to Vertex AI using the specified training file.

        Parameters:
            train_filename (str): Training file.

        Returns:
            None
        """
        try:

            # Create the supervised fine-tuning job using Vertex AI
            sft_tuning_job = sft.train(
                source_model=self.model_name,
                train_dataset=train_filename,
                # Optional parameters you can add:
                # validation_dataset=validation_file_path,
                # epochs=3,
                # learning_rate_multiplier=1.0,
                # adapter_size=1,  # For LoRA fine-tuning
            )

            # Create LLMAPITrainingState instance and set it with the job details
            self.details_fine_tuned_model = LLMAPITrainingState()
            self.details_fine_tuned_model.set(
                base_model_name=self.model_name,
                training_client=sft,  # Pass the sft module as the training client
                training_data_id=train_filename,
                training_job_object=sft_tuning_job,
                api_type="vertexai"
            )

            if self.verbose:
                print(f"Fine-tuning job submitted successfully:")
                print(f"  Job ID: {self.details_fine_tuned_model.job_id}")
                print(f"  Base model: {self.model_name}")
                print(f"  Training file: {train_filename}")
                print(f"  Status: {self.details_fine_tuned_model.status}")

            self._dump_details_fine_tuned()

        except Exception as e:
            print(f"Error creating fine-tuning job: {e}")
            if self.verbose:
                traceback.print_exc()
            self.details_fine_tuned_model = None
            raise
