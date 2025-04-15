#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""
import copy
import os
import random

from llm_narrative_handler import LLMNarrativeScenesHandler
from narrative_preprocess import NarrativePreprocessResults

class LLMNarrativeScenesBuildTestCompose(LLMNarrativeScenesHandler):
    """
    A handler for building test input files for narrative scene composition.

    This class processes evaluation data from preprocessed narrative scenes and
    transforms it into test composition specifications. It selectively removes content
    from scenes while preserving structural elements needed to test scene generation,
    such as character names and setting identifiers.

    The class handles:
    1. Loading preprocessed narrative scenes from evaluation files
    2. Creating modified versions with content elements removed
    3. Preserving scene structure and metadata for testing
    4. Generating composition specification files with controlled content
    5. Supporting scene limit controls for targeted testing

    This enables systematic testing of scene composition capabilities by providing
    consistent and controlled input specifications derived from existing scenes.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the test composition builder with configuration details.

        Sets up the handler with input/output file paths and processing options,
        then loads the narrative scenes from the specified evaluation file.

        Required Parameters:
            - output_compose_filename (str): Path where test composition file will be saved
            - input_eval_filename (str): Path to evaluation data file with preprocessed scenes

        Optional Parameters:
            - scene_limit (int): Maximum number of scenes to include in test file
            - verbose (bool): Whether to output processing details (default: False)

        Raises:
            ValueError: If required parameters are missing or input file doesn't exist
        """
        # Required parameters
        self.output_compose_filename = None
        # self.input_eval_filename = None set by base class

        # Optional parameters
        self.scene_limit = None
        self.verbose = False

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.output_compose_filename is None:
            raise ValueError("output_compose_filename is not set")

        input_filename_list = []
        if self.input_eval_filename is not None and os.path.exists(self.input_eval_filename):
            input_filename_list.append(self.input_eval_filename)
        if len(input_filename_list) == 0:
            raise ValueError("LLMNarrativeScenesCollection: Input eval file does not exist.")

        super().__init__(input_filename_list, self.verbose)

        if self.verbose:
            print("Size of scene list is", len(self.preprocess_results.scene_list))

    def build_test_compose_scene_input_file(self) -> None:
        """
        Create a test input file for scene composition by modifying evaluation scenes.

        This method takes preprocessed evaluation scenes and selectively removes
        content elements (like scene bodies and detailed descriptions) while
        preserving structural elements (like character names and setting identifiers).

        The resulting file provides a structured test specification that can be used
        to evaluate scene composition capabilities with controlled inputs.

        The process:
        1. Creates a deep copy of preprocessed scenes
        2. Identifies evaluation scenes
        3. Clears scene body text
        4. Retains entity names but clears their plot summaries
        5. Respects the scene_limit setting if specified
        6. Saves the modified data to the specified output file

        Returns:
            None
        """
        # take the eval records and dump them to input_filename, but zero out the scene body, character and setting descriptions
        # and plot summaries. Keep the scene index and chron scene index and the top-level plot_summary,
        # which will be used to generate the secne. Also keep the named characters and settings names.
        # this is used to test the compose scene function

        i = 0
        ppr = copy.deepcopy(self.preprocess_results)
        for scene in ppr.scene_list:
            if scene['datamode'] == "eval":
                if self.scene_limit is not None and i >= self.scene_limit:
                    print("breaking out of build_test_compose_scene_input_file")
                    break
                scene['body'] = ""
                for char_dict in scene['named_characters'].values():
                    char_dict['plot_summary'] = ""
                for setting_dict in scene['settings'].values():
                    setting_dict['plot_summary'] = ""
                if 'objects' in scene:
                    for object_dict in scene['objects'].values():
                        object_dict['plot_summary'] = ""
                i += 1
                ppr.dump(self.output_compose_filename)
