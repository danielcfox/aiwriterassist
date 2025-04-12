#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:07:22 2025

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
    def __init__(self, model_spec: str, model, **kwargs):
        """
        Initialize the LLMHandler with a specific model.
        :param model_spec: Specifies the type of the model argument.
                        Options are 'openai_details_filename' or 'model_name'.
                        - 'details_filename': Indicates that the model argument is a filename
                            containing the object details of a fine-tuned model.
                        - 'model_name': Indicates that the model argument is the name of the model
                            available through the GPT-4o API.
        :param model: The name of the model to use.
        :param kwargs: Additional keyword arguments for model configuration.
                        - verbose: If True, print verbose output.
        :return: None
        """
        self.verbose = kwargs.get('verbose', False)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.client = self._open()
        self._set_model_name(model_spec, model) # sets self._model_name

    def _set_model_name(self, model_spec: str, model: str):
        """
        Set the model name based on the provided specification.
        :param model_spec: Specifies the type of the model argument.
                        Options are 'openai_details_filename' or 'model_name'.
                        - 'openai_details_filename': Indicates that the model argument is a filename 
                            containing the object details of a fine-tuned model.
                        - 'model_name': Indicates that the model argument is the name of the model 
                            available through the GPT-4o API.
        :param model: The model argument, which can either be a filename (if model_spec is 
                    'openai_details_filename') or the name of the model (if model_spec is 'model_name').
        """
        self.details_model: Optional[object] = None

        if model_spec == 'details_filename':
            self._load_details_model_from_uri(model)
        elif model_spec == 'model_name':
            # model_names = self.get_models_available()
            # if model not in model_names:
                # raise ValueError(f"The specified model '{model}' is not available. Available models: {model_names}")
            self.model_name = model
        else:
            raise ValueError(f"Invalid model_spec '{model_spec}'. Must be 'openai_details_filename' or 'model_name'.")

    def open(self):
        """
        Open a connection to the LLM.
        :return: The client object for the LLM.
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'open()' method!")
    
    def _load_details_model_from_uri(self, model_uri):
        """
        Load the details of a fine-tuned model from a URI.
        :param model_uri: The URI of the fine-tuned model.
        :return: None
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'load_details_model_from_uri()' method!")

    def get_models_available(self):
        """
        Retrieve the list of available models.
        :return: The list of available models.
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_models_available()' method!")

    def fine_tune_submit(self, train_filename_list: List[str], author_name: str) -> Optional[object]:
        """
        Submit a fine-tuning job to the LLM.
        :param train_filename_list: List of training filenames.
        :param author_name: The name of the author for fine-tuning.
        :return: The response from the fine-tuning job submission.
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'fine_tune_submit()' method!")

    def get_timed_prompt_response(self, prompt, **kwargs):
        """
        Get a timed response from the LLM for a given prompt.
        :param prompt: The input prompt for the LLM.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature to use.
        :return: The response from the LLM.
        """
        start = time.time()
        response = self._get_prompt_response(prompt, **kwargs)
        end = time.time()
        if self.verbose:
            print(f"elapsed time for prompt response {end - start}")
        return response
    
    def get_write_scene_prompt_response(self, prompt, author_name, **kwargs):
        """
        Get a response from the LLM for a given prompt with author name.
        :param prompt: The input prompt for the LLM.
        :param author_name: The name of the author to include in the prompt.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature to use.
        :return: The response from the LLM.
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_write_scene_prompt_response()' method!")
   

    def _get_prompt_response(self, prompt, **kwargs):
        """
        Get a response from the LLM for a given prompt.
        :param prompt: The input prompt for the LLM.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature to use.
        :return: The response from the LLM.
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_prompt_response()' method!")

    def _get_details_fine_tuned_model(self) -> None:
        """
        Retrieves the details of the fine-tuned model.
        Sets it in self.details_model, overwriting any previous value.
        :return: None
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the '_get_details_fine_tuned_model()' method!")
    
    def get_fine_tuned_model_pending(self) -> bool:
        """
        Check if the fine-tuned model is pending.
        :return: True if the fine-tuned model is pending, False otherwise.
        """
        # This is a dummy function to be overridden in subclasses.
        # The base class does not implement any functionality.
        # It is expected that subclasses will provide their own implementation.
        """This method must be implemented in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_fine_tuned_model_pending()' method!")
