#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""

import os

from narrative_preprocess import NarrativePreprocessResults

class LLMNarrativeScenesHandler():
    """Base class for handling LLM interactions for narrative scenes."""
    def __init__(self, user, narrative, author_name, api_obj, train_input_file, eval_input_file, 
                 max_input_tokens=None, max_output_tokens=None):
        """Initialize the LLMNarrativeScenesHandler with model and narrative details."""
        print("Calling LLMNarrativeScenesHandler")
        print(f"train_input_file: {train_input_file}")
        print(f"eval_input_file: {eval_input_file}")
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.api_obj = api_obj

        self.user = user
        self.narrative = narrative
        self.author_name = author_name

        self.preprocess_results = NarrativePreprocessResults()

        self.train_input_file = train_input_file
        self.eval_input_file = eval_input_file

        if train_input_file is not None and os.path.exists(train_input_file):
            print("Loading in input train file")
            self.preprocess_results.load(train_input_file)
        if eval_input_file is not None and os.path.exists(eval_input_file):
            print("Loading in input eval file A")
            self.preprocess_results.load(eval_input_file)

