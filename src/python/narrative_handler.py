#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""

import os

from narrative_preprocess import NarrativePreprocessResults

class NarrativeScenesHandler():
    """Base class for handling LLM interactions for narrative scenes."""
    def __init__(self, input_filename_list: list[str], verbose: bool = False):
        """Initialize the NarrativeScenesHandler with model and narrative details.
        Args:
            user (str): User name.
            narrative (str): Narrative name.
            author_name (str): Author name.
            api_obj (object): API object for LLM interaction.
            input_train_filename (str): Path to the training input file.
            input_eval_filename (str): Path to the evaluation input file.
            max_input_tokens (int, optional): Maximum input tokens. Defaults to None.
            max_output_tokens (int, optional): Maximum output tokens. Defaults to None.
        """

        self.verbose = verbose
        # self.user = None
        # self.narrative = None
        # self.author_name = None
        # self.api_obj = None
        # self.input_train_filename = None
        # self.input_eval_filename = None
        # self.max_input_tokens = None
        # self.max_output_tokens = None
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        # self.max_input_tokens = max_input_tokens
        # self.max_output_tokens = max_output_tokens
        # self.api_obj = api_obj

        # self.user = user
        # self.narrative = narrative
        # self.author_name = author_name

        # self.input_train_filename = input_train_filename
        # self.input_eval_filename = input_eval_filename

        # if self.user is None or len(self.user) == 0:
        #     raise ValueError("user is not set")
        # if self.narrative is None or len(self.narrative) == 0:
        #     raise ValueError("narrative is not set")
        # # if self.author_name is None or len(self.author_name) == 0: # needs to be enforced in the caller
        # #     raise ValueError("author_name is not set")

        self.preprocess_results = NarrativePreprocessResults()
        if input_filename_list is not None:
            for input_filename in input_filename_list:
                if not os.path.exists(input_filename):
                    raise ValueError(f"Input file {input_filename} does not exist.")
                self.preprocess_results.load(input_filename)

        # if self.input_train_filename is not None and os.path.exists(self.input_train_filename): # needs to be enforced in the caller
        #     if self.verbose:
        #         print("NarrativeScenesHandler: Loading in input train file")
        #     self.preprocess_results.load(self.input_train_filename)
        # if self.input_eval_filename is not None and os.path.exists(self.input_eval_filename):
        #     if self.verbose:
        #         print("NarrativeScenesHandler: Loading in input eval file")
        #     self.preprocess_results.load(self.input_eval_filename)

        if verbose:
            print("Calling NarrativeScenesHandler")
            print(f"input_files: {input_filename_list}")
