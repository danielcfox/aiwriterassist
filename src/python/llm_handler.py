#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:07:22 2025

@author: dfox
"""

import time

class LLMHandler:
    """
    Base class for handling LLM interactions.
    This class provides a common interface for different LLM handlers.
    """
    def __init__(self, model='gpt-4o-mini'):
        """
        Initialize the LLMHandler with a specific model.
        :param model: The name of the model to use.
        """
        self.model = model

    def get_timed_prompt_response(self, prompt, max_tokens, temperature):
        """
        Get a timed response from the LLM for a given prompt.
        :param prompt: The input prompt for the LLM.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature to use.
        :return: The response from the LLM.
        """
        # start = time.time()
        response = self.get_prompt_response(prompt, max_tokens, temperature)
        # end = time.time()
        # print(f"elapsed time for prompt response {end - start}")
        return response
    
    def get_write_scene_prompt_response(self, prompt, author_name, max_tokens, temperature):
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
   

    def get_prompt_response(self, prompt, max_tokens, temperature):
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

