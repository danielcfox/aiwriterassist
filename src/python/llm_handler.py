#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:07:22 2025

Copyright 2025 Daniel C. Fox

@author: dfox
"""

import os
import time
from typing import List, Optional, Union

class LLMHandler:
    """
    Base class for handling LLM interactions.
    This class provides a common interface for different LLM handlers.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialize the LLMHandler with a specific model.

        Parameters:
            **kwargs: Additional keyword arguments for model configuration:
                - details_uri (str, optional): The URI of the fine-tuned model details.
                - author_name (str, optional): The name of the author to include in the prompt.
                - model_name (str, optional): The name of the model to use.
                - verbose (bool, optional): If True, print verbose output.

        Returns:
            None

        Raises:
            ValueError: If the specified model is not available.
        """
        self.details_uri = kwargs.get('details_uri', None)
        self.model_name = kwargs.get('model_name', None)
        self.author_name = kwargs.get('author_name', None)
        self.verbose = kwargs.get('verbose', False)

        self.client = self._open()

        if self.details_uri is not None and len(self.details_uri) > 0 and os.path.exists(self.details_uri):
            self._load_details_model()
        if self.model_name is not None and len(self.model_name) > 0:
            model_names = self.get_models_available()
            if self.model_name not in model_names:
                raise ValueError(f"The specified model '{self.model_name}' is not available. Available models: {model_names}")

    def _open(self) -> Optional[object]:
        """
        Open a connection to the LLM.

        This method establishes a connection to the LLM service and returns
        a client object that can be used for further interactions.

        Returns:
            object: The client object for the LLM.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'open()' method!")

    def _load_details_model(self) -> None:
        """
        Load the details of a fine-tuned model from a URI.

        This method retrieves model details from the location specified in self.details_uri
        and stores the information in the instance for later use.

        Returns:
            None

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'load_details_model_from_uri()' method!")

    def _get_details_fine_tuned_model(self) -> None:
        """
        Retrieves the details of the fine-tuned model.

        This method fetches the latest information about a fine-tuned model
        and sets it in self.details_model, overwriting any previous value.

        Returns:
            None

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the '_get_details_fine_tuned_model()' method!")

    def get_models_available(self) -> List[str]:
        """
        Retrieve the list of available models from the LLM provider.

        This method queries the LLM API to get a complete list of models that
        are available for use, including both base and fine-tuned models.

        Returns:
            List[str]: The list of available model names.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_models_available()' method!")

    def wait_fine_tuning_model(self, max_seconds_wait: int) -> Optional[str]:
        """
        Wait for the fine-tuning model to be ready.

        This method polls the LLM provider at regular intervals to check if
        the requested fine-tuning job has completed, up to a maximum wait time.

        Parameters:
            max_seconds_wait (int): The maximum number of seconds to wait.

        Returns:
            Optional[str]: The status of the fine-tuning model, or None if the wait timed out.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'wait_fine_tuning_model()' method!")

    def fine_tune_submit(self, train_filename_list: List[str]) -> Optional[object]:
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

    def _get_model_name(self) -> str:
        """
        Get the name of the model currently in use.

        This method returns the identifier of the model being used,
        which might be a base model or a fine-tuned model.

        Returns:
            str: The name of the model.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_model_name()' method!")

    def get_fine_tuned_model_pending(self) -> bool:
        """
        Check if the fine-tuned model is still in a pending state.

        This method queries the LLM provider to determine if a submitted
        fine-tuning job is still processing or has completed.

        Returns:
            bool: True if the fine-tuned model is pending, False if it's completed.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_fine_tuned_model_pending()' method!")
