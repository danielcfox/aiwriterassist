#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:07:22 2025

@author: dfox
"""

import json
import os
import pickle
import time
from typing import List, Optional, Union
from python.llm_api_training_state import LLMAPITrainingState

class LLMAPIHandler:
    """
    Base class for handling LLM interactions.
    This class provides a common interface for different LLM handlers.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialize the LLMAPIHandler with a specific model.

        Parameters:
            **kwargs: Additional keyword arguments for model configuration:
                - details_filename (str, optional): The URI of the fine-tuned model details.
                - author_name (str, optional): The name of the author to include in the prompt.
                - model_name (str, optional): The name of the model to use.
                - verbose (bool, optional): If True, print verbose output.

        Returns:
            None

        Raises:
            ValueError: If the specified model is not available.
        """
        self.details_filename = kwargs.get('details_filename', None)
        self.model_name = kwargs.get('model_name', None)
        self.author_name = kwargs.get('author_name', None)
        self.verbose = kwargs.get('verbose', False)
        self.generate_prompt_only = kwargs.get("generate_prompt_only", False)
        self.fine_tune_submit_filename = kwargs.get("submit_filename", "fine_tune_submit.jsonl")

        self.details_fine_tuned_model: Optional[LLMAPITrainingState] = None
        self.client = None  # Placeholder for the LLM client, to be initialized in subclasses

        # self.client = self._open()

        if self.details_filename is not None and len(self.details_filename) > 0 and os.path.exists(self.details_filename):
            self._load_details_fine_tuned_model()
        if self.model_name is not None and len(self.model_name) > 0:
            model_names = self._get_models_available()
            if self.model_name not in model_names:
                raise ValueError(f"The specified model '{self.model_name}' is not available. Available models: {model_names}")

    def get_timed_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Get a timed response from the LLM for a given prompt.

        This method wraps the _get_prompt_response method with timing functionality
        to measure how long the LLM takes to generate a response.

        Parameters:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional keyword arguments:
                - max_tokens (int): The maximum number of tokens to generate.
                - temperature (float): The sampling temperature to use (0.0 to 1.0).
                - verbose (bool, optional): If True, print verbose output.

        Returns:
            str: The response from the LLM.
        """
        start = time.time()
        response = self._get_prompt_response(prompt, **kwargs)
        end = time.time()
        if self.verbose:
            print(f"elapsed time for prompt response {end - start}")
        return response

    def get_write_scene_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Get a response from the LLM specifically for writing narrative scenes.

        This method is tailored for generating creative content like story scenes,
        with appropriate context and formatting for narrative writing.

        Parameters:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional keyword arguments:
                - author_name (str, optional): The name of the author to include in the prompt.
                - max_tokens (int): The maximum number of tokens to generate.
                - temperature (float): The sampling temperature to use (0.0 to 1.0).
                - verbose (bool, optional): If True, print verbose output.

        Returns:
            str: The generated scene response from the LLM.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_write_scene_prompt_response()' method!")

    def fine_tune_submit(self, corpus_train_prompt_list: List[tuple[str, str]]) -> Optional[object]:
        """
        Submit a fine-tuning job to the LLM provider.

        This method sends training files to the LLM provider and initiates
        a fine-tuning job on a base model.

        Parameters:
            train_filename_list (List[str]): List of paths to training data files.

        Returns:
            Optional[object]: The response from the fine-tuning job submission,
                            typically containing job details.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'fine_tune_submit()' method!")

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
        self._load_details_fine_tuned_model()
        if self.details_fine_tuned_model is None:
            return None

        while self.details_fine_tuned_model.fine_tuned_model_name is None:
            if self.details_fine_tuned_model.is_pending():
                if seconds_wait >= max_seconds_wait:
                    return None
                time.sleep(min(max_seconds_wait - seconds_wait, 60))  # Check every 60 seconds
                seconds_wait += 60
                # Refresh status from server
                try:
                    self.details_fine_tuned_model.refresh_from_server()
                except Exception as e:
                    if self.verbose:
                        print(f"Error refreshing status: {e}")
                continue

            if self.details_fine_tuned_model.is_completed():
                if self.details_fine_tuned_model.fine_tuned_model_name is None:
                    # Generate a model name if not provided by API
                    fine_tuned_model_name = f"{self.model_name}-ft-{self.details_fine_tuned_model.job_id}"
                    print(f"Fine-tuning job completed successfully! Fine-tuned model ID: {self.details_fine_tuned_model.job_id}, "
                        + f"Name: {fine_tuned_model_name}")
                    return fine_tuned_model_name
                else:
                    print(f"Fine-tuning job completed successfully! Fine-tuned model ID: {self.details_fine_tuned_model.job_id}, "
                        + f"Name: {self.details_fine_tuned_model.fine_tuned_model_name}")
                    return self.details_fine_tuned_model.fine_tuned_model_name

            if self.details_fine_tuned_model.is_failed():
                print(f"Fine-tuning job failed. Status: {self.details_fine_tuned_model.status}, error: {self.details_fine_tuned_model.error_message}")
                return None

            # If status is unknown, break
            print(f"Fine-tuning job status unknown: {self.details_fine_tuned_model.status}")
            return None

        # Job already completed
        if self.details_fine_tuned_model.is_completed():
            print(f"Fine-tuning job completed successfully! Fine-tuned model ID: {self.details_fine_tuned_model.job_id}, "
                    + f"Name: {self.details_fine_tuned_model.fine_tuned_model_name}")
            return self.details_fine_tuned_model.fine_tuned_model_name

        print(f"Fine-tuning job did not complete. Status: {self.details_fine_tuned_model.status}, "
            + f"Fine-tuned model ID: {self.details_fine_tuned_model.job_id}, "
            + f"Name: {self.details_fine_tuned_model.fine_tuned_model_name}")
        return self.details_fine_tuned_model.fine_tuned_model_name

    def _load_details_fine_tuned_model(self) -> None:
        """
        Load fine-tuned model details from a file.

        Attempts to load previously saved model details from the specified
        details_filename, which contains serialized information about a fine-tuning job.

        Returns:
            None

        Note:
            If details_filename is invalid or doesn't exist, this method will fall back
            to using the base model specified in model_name.
        """
        if self.details_filename is None or len(self.details_filename) == 0 or not os.path.exists(self.details_filename):
            print(f"Fine-tuned model has not been created yet. Using base model {self.model_name}...")
            return
        print(f"Loading model details from file {self.details_filename}...")
        self.details_fine_tuned_model = LLMAPITrainingState.load(self.details_filename)
        # with open(self.details_filename, 'rb') as fp:
        #     self.details_fine_tuned_model = LLM.load(fp)
        #     if self.verbose:
        #         print(f"self.details_fine_tuned_model: {self.details_fine_tuned_model}")
        if self.details_fine_tuned_model.is_pending():
            self.details_fine_tuned_model.refresh_from_server()

    def _get_models_available(self) -> List[str]:
        """
        Retrieve the list of available models from the LLM provider.

        This method queries the LLM API to get a complete list of models that
        are available for use, including both base and fine-tuned models.

        Returns:
            List[str]: The list of available model names.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the '_get_models_available()' method!")

    def _get_prompt_response(self, prompt: str, **kwargs) -> str:
        """
        Get a response from the LLM for a given prompt.

        This core method handles the actual communication with the LLM API
        to generate text based on the provided prompt.

        Parameters:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional keyword arguments:
                - max_tokens (int): The maximum number of tokens to generate.
                - temperature (float): The sampling temperature to use (0.0 to 1.0).

        Returns:
            str: The response from the LLM.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_prompt_response()' method!")

    def _format_inference(self, prompt: str) -> List[dict]:

        if 'llama' in self.model_name.lower():
            system_msg = "You are an assistant to the author of the work given. Always follow instructions exactly. Be precise."
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        if 'gpt' in self.model_name.lower():
            return [
                {"role": "system",
                 "content":
                    "You are an assistant to the author of the work given. Always follow instructions exactly. Be precise."},
                {"role": "user",
                 "content": prompt}
                 ]

        # For now, all other models are treated as Gemini
        return f"You are an assistant to the author of the work given. Always follow instructions exactly. Be precise. {prompt}"

    def _get_model_name(self) -> str:
        """
        Determine which model to use for current API requests.

        Decides whether to use the fine-tuned model (if available and successful)
        or fall back to the base model specified during initialization.

        Returns:
            str: Name of the model to use for API calls.
        """
        if (self.details_fine_tuned_model is not None and self.details_fine_tuned_model.fine_tuned_model_name is not None
            and self.details_fine_tuned_model.is_completed() and len(self.details_fine_tuned_model.fine_tuned_model_name) > 0):
            return self.details_fine_tuned_model.fine_tuned_model_name

        self._load_details_fine_tuned_model()
        if (self.details_fine_tuned_model is not None and self.details_fine_tuned_model.fine_tuned_model_name is not None
            and self.details_fine_tuned_model.is_completed() and len(self.details_fine_tuned_model.fine_tuned_model_name) > 0):
            return self.details_fine_tuned_model.fine_tuned_model_name

        return self.model_name

    def _write_training_samples(self, corpus_train_prompt_list: List[tuple[str, str]]) -> List[dict]:
        """
        Format training samples for the LLM provider.

        This method takes a list of training samples and formats them into
        a structure suitable for submission to the LLM provider.

        Parameters:
            corpus_train_prompt_list (List[tuple[str, str]]): List of tuples containing
                                                            training prompts and responses.

        Returns:
            List[dict]: A list of formatted training samples.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        training_samples = self._format_training_samples(corpus_train_prompt_list)

        with open(self.fine_tune_submit_filename, "w", encoding="utf-8") as f:
            for entry in training_samples:
                f.write(json.dumps(entry) + "\n")

        if self.generate_prompt_only:
            print(f"Fine-tuning job not submitted. Prompt only requested. Training samples are in {self.fine_tune_submit_filename}")

    def _format_training_samples(self, corpus_train_prompt_list: List[tuple[str, str]]) -> List[dict]:
        """
        Format training samples for fine-tuning.

        Converts a list of tuples containing prompts and responses into
        a structured format suitable for fine-tuning.

        Parameters:
            corpus_train_prompt_list (List[tuple[str, str]]): List of tuples with prompt and response.

        Returns:
            List[dict]: Formatted training samples.
        """
        training_samples = []
        for prompt, response in corpus_train_prompt_list:
            training_samples.append(self._format_training_sample(prompt, response))
        return training_samples

    def _format_training_sample(self, prompt: str, response: Optional[str] = None) -> dict:
        """
        Format training samples for fine-tuning.

        Converts a list of tuples containing prompts and responses into
        a structured format suitable for fine-tuning.

        Parameters:
            corpus_train_prompt_list (List[tuple[str, str]]): List of tuples with prompt and response.

        Returns:
            List[dict]: Formatted training samples.
        """
        # Check if this is a Llama model
        if 'llama' in self.model_name.lower():
            # Llama models use a simpler instruction format
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a fiction writer, writing in the style of '{self.author_name}'<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            formatted_response = f"{response}<|eot_id|>"

            return {
                "input_text": formatted_prompt,
                "output_text": formatted_response
            }

        # For GPT, Gemini, and other models, use the messages format
        return {"messages": [{"role": "system",
                              "content": f"You are a fiction writer, writing in the style of '{self.author_name}'"},
                              {"role": "user", "content": prompt},
                              {"role": "assistant", "content": response}]}

    def _dump_details_fine_tuned(self) -> None:
        """
        Save the current fine-tuned model details to disk.

        Serializes the fine-tuning job details to the file specified by details_filename.

        Returns:
            None
        """
        # if self.details_fine_tuned_model is not None:
        #     with open(self.details_filename, 'wb') as fp:
        #         pickle.dump(self.details_fine_tuned_model, fp)
        if self.details_filename and self.details_fine_tuned_model:
            try:
                self.details_fine_tuned_model.dump(self.details_filename)
                if self.verbose:
                    print(f"Saved model details to {self.details_filename}")
            except Exception as e:
                if self.verbose:
                    print(f"Error saving model details: {e}")
