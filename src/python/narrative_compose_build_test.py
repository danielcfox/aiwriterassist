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
    def __init__(self, user, narrative, input_eval_filename, output_filename):
        """Initialize the LLMNarrativeScenesCompose with model and narrative details.
        :param type str, user: The name of the user.
        :param type str, narrative: The name of the narrative.
        :param type str, author_name: The name of the author.
        :param type ?, api_obj: The name of the model.
        :param type str, input_train_filename: The name of the input train file.
        :param type str, input_eval_filename: The name of the input eval file.
        :param type str, input_filename: The name of the input file.
        :param type str, output_filename: The name of the output file.
        :param type VectorDBMilvus, vector_db: An instatiation of the vector database singleton class.
        :param type str, use_fine_tuned_model: The name of the fine-tuned model.
        :return: None
        """
        super().__init__(user, narrative, None, None, None, input_eval_filename)

        self.output_filename = output_filename

        print("Size of scene list is", len(self.preprocess_results.scene_list))
    
    def build_test_compose_scene_input_file(self, output_filename, scene_limit):
        """Build a test input file for composing scenes.
        :param type str, output_filename: The name of the output file.
        :param type int, scene_limit: The maximum number of scenes to process.
        :return: None
        """
        # take the eval records and dump them to input_filename, but zero out the scene body, character and setting descriptions
        # and plot summaries. Keep the scene index and chron scene index and the top-level plot_summary,
        # which will be used to generate the secne. Also keep the named characters and settings names.
        # this is used to test the compose scene function
        
        i = 0
        ppr = copy.deepcopy(self.preprocess_results)
        for scene in ppr.scene_list:
            if scene['datamode'] == "eval":
                if scene_limit is not None and i >= scene_limit:
                    print("breaking out of build_test_compose_scene_input_file")
                    break
                scene['body'] = ""
                for char_dict in scene['named_characters'].values():
                    # char_dict['description'] = ""
                    char_dict['plot_summary'] = ""
                for setting_dict in scene['settings'].values():
                    # setting_dict['description'] = ""
                    setting_dict['plot_summary'] = ""
                if 'objects' in scene:
                    for object_dict in scene['objects'].values():
                        # object_dict['description'] = ""
                        object_dict['plot_summary'] = ""
                i += 1
                ppr.dump(output_filename)

