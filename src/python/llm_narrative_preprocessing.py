#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

Copyright 2025 Daniel C. Fox

@author: dfox
"""
import json
import os
from tqdm import tqdm

from narrative_handler import NarrativeScenesHandler
from llm_openai_api_handler import LLMOpenAIAPIHandler

class LLMNarrativeScenesPreprocessing(NarrativeScenesHandler):
    """
    A handler for preprocessing narrative scenes using LLMs to extract structured metadata.

    This class processes raw narrative scenes and uses an LLM to analyze the content,
    extracting key elements such as characters, settings, objects, and plot summaries.
    It transforms unstructured narrative text into structured JSON data that can be used
    for further processing, vector embedding, and scene generation.

    The class handles:
    1. Loading raw narrative scenes from input files
    2. Analyzing scene content with LLM to identify narrative elements
    3. Extracting character descriptions and perspectives
    4. Identifying settings and objects with descriptions
    5. Determining scene focus, tone, and point of view style
    6. Creating structured scene summaries
    7. Saving processed data to output files

    The preprocessing creates rich metadata that powers downstream tasks such as
    vector embedding, semantic search, and contextually-aware scene generation.
    """
    def __init__(self, **kwargs):
        # when doing an unclean run, set input_train_filename to the same as output_train_filename,
        # and also set input_eval_filename to the same as output_eval_filename
        # this means that the output files will be treated as input files, with all of the metadata set from a previous run
        """Initialize the LLMNarrativeScenesPreprocessing with model and narrative details."""

        # Required parameters
        self.author_name = None
        self.api_obj = None
        self.max_output_tokens = None
        self.narrative = None
        self.output_train_filename = None

        # Optional parameters
        self.output_eval_filename = None
        self.verbose = False

        for key, value in kwargs.items():
            setattr(self, key, value)

        if len(self.output_eval_filename) == 0 or not os.path.exists(self.output_eval_filename):
            self.output_eval_filename = None
        if self.author_name is None or len(self.author_name) == 0:
            raise ValueError("author_name is not set")
        if self.narrative is None or len(self.narrative) == 0:
            raise ValueError("narrative is not set")
        if self.output_train_filename is None or not os.path.exists(self.output_train_filename):
            raise ValueError("output_train_filename is not set")
        if self.api_obj is None or not isinstance(self.api_obj, LLMOpenAIAPIHandler):
            raise ValueError("api_obj is not set")
        if self.max_output_tokens is None:
            raise ValueError("max_output_tokens is not set")

        # del kwargs['output_train_filename']
        # del kwargs['output_eval_filename']

        input_filename_list = []
        if self.input_train_filename is not None and os.path.exists(self.input_train_filename):
            input_filename_list.append(self.input_train_filename)
        if self.input_eval_filename is not None and os.path.exists(self.input_eval_filename):
            input_filename_list.append(self.input_eval_filename)
        if len(input_filename_list) == 0:
            raise ValueError("LLMNarrativeScenesCollection: Neither Input train file nor Input eval file exists.")

        super().__init__(input_filename_list, self.verbose)

    def _dump_train(self):
        self.preprocess_results.dump_train(self.output_train_filename)

    def _dump_eval(self):
        if self.output_eval_filename is not None:
            self.preprocess_results.dump_eval(self.output_eval_filename)

    def _get_scene_prompt(self, scene):
        p = "{"
        p += f"narr_scene_index: {scene['narr_scene_index']},\n"
        p += f"chron_scene_index: {scene['chron_scene_index']},\n"
        p += """
                "point_of_view_character": "<name of point-of-view character>",
                "named_characters": {
                    "<named point-of-view character>": {
                        "description": "<(point-of-view character) (named explicitly) traits and backstory only>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end),
                                         from the point of view of the point-of-view character,
                                         approximately one-tenth the length of the scene text;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'tone', 'focus',
                                         'synopsis', or 'summary'>"
                       }
                    "<named character 2, if there is more than one character>": {
                        "description": "<character 2 (named explicitly) traits and backstory only>",
                        "plot_summary": "<events that occur in the scene beginning, middle, and end),
                                         from point of view of character 2;
                                         do not use the words 'character', 'setting', 'object', scene', 'plot', 'synopsis',
                                         or 'summary'>"
                        }
                    "<named character n, if there are more than two characters>": {
                        "description": "<character n (named explicitly) traits and backstory only>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end),
                                         from point of view of character n;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'tone', 'focus',
                                         'synopsis', or 'summary'>"
                        }
                    },
                "objects": {
                    "<object 1, if there are any objects>": {
                        "description": "<object 1 (named explicitly) description>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end), with this specific object;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'tone', 'focus',
                                         'synopsis', or 'summary'>"
                        }
                    "<object 2, if there is more than one object>": {
                        "description": "<object 2 (named explicitly) description>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end), at this specific setting;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'tone', 'focus',
                                         'synopsis', or 'summary'>"
                        }
                    "<setting n, if there are more than two settings>": {
                        "main setting": false
                        "description": "<setting n (named explicitly) description>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end), at this specific setting;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'tone', 'focus',
                                         'synopsis', or 'summary'>"
                        }
                    },
                "settings": {
                    "<main setting>": {
                        "main setting": true
                        "description": "<main setting (named explicitly) description>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end), at this specific setting;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'synopsis',
                                         or 'summary'>"
                        }
                    "<setting 2, if there is more than one setting>": {
                        "main setting": false
                        "description": "<setting 2 (named explicitly) description>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end), at this specific setting;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'synopsis',
                                         or 'summary'>"
                        }
                    "<setting n, if there are more than two settings>": {
                        "main setting": false
                        "description": "<setting n (named explicitly) description>",
                        "plot_summary": "<events that occur in the scene (beginning, middle, and end), at this specific setting;
                                         do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'synopsis',
                                         or 'summary'>"
                        }
                    },
                "focus": "<focus of the scene, if there is one, do not user the word 'focus'>",
                "tone": "<conflict OR suspense OR mysterious OR foreboding OR intriguing OR comedic OR reflective OR descriptive
                         OR emotional OR introspective OR philosophical OR romantic OR action OR adventure OR horror>"
                "point of view style": "<first person OR second person OR third person or fourth person>
                                        <'limited' (knows only thoughts and feelings of point of view character)
                                         OR 'omniscient' (knows thoughts and feelings of more than one character)>"
                "plot_summary": "<events that occur in the scene (beginning, middle, and end);
                                 make sure to reference all named characters;
                                 make the length approximately one-tenth of the scene text;
                                 do not use the words 'character', 'setting', 'object', 'scene', 'plot', 'tone', 'focus',
                                 'synopsis', or 'summary';
                                 stick to the events of the scene with no commentary on character thoughts or feelings>"
                }
            """
        p += "}"
        p = " ".join(p.split())
        print(f"preamble prompt is:\n{p}")
        return p

    def _format_scene_request(self, scene):
        p = f"Given the following scene from a fiction narrative called {self.narrative} "
        p += f"that fiction writer {self.author_name} wrote. The content of the scene is:\n"
        p += f"{scene['body']}\n"
        p += "Analyze and give a response.\n"
        p += "Format the response in the following well-formed JSON format:\n"
        p += self._get_scene_prompt(scene)
        return p

    def _get_scene_response(self, scene):
        """Get a response from the LLM for a given scene.
        :param type dict, scene: The scene to get a response for.
        :param type int, max_tokens: The maximum number of tokens to generate.
        :return: type dict, The response from the LLM.
        """
        if self.api_obj is None:
            raise ValueError("API object is not initialized. Please provide a valid model name.")
        response = self.api_obj.get_timed_prompt_response(self._format_scene_request(scene), max_tokens=self.max_output_tokens,
                                                          temperature=0.0)
        # print(response)
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            return {}
        # print(response_dict)
        return response_dict

    def _get_scene_list_response(self, ppr, scene_limit):
        """Get a response from the LLM for a list of scenes.
        :param type NarrativePreprocessResults, ppr: The preprocessed results of the narrative.
        :param type int, scene_limit: The maximum number of scenes to process.
        :param type int, max_tokens: The maximum number of tokens to generate.
        :return: None
        """
        scene_list = ppr.scene_list
        if scene_limit is None or scene_limit < 0 or scene_limit > len(scene_list):
            scene_limit = len(scene_list)
         # use tqdm on next loop
        for index, scene in tqdm(enumerate(scene_list), total=scene_limit, desc=f"Analyzing scene list"):
            # print(scene_limit, index)
            if scene_limit is not None and index >= scene_limit:
                # print("get_scene_list_response: breaking out of scene lists")
                return
            if len(scene['plot_summary']) == 0:
                # print(f"Using LLM to analyze scene chron_scene_index {scene['chron_scene_index']}")
                num_retries = 0
                while num_retries < 10:
                    try:
                        # sr = self.get_scene_response(scene, prev_scene, max_tokens)
                        sr = self._get_scene_response(scene)
                    except json.JSONDecodeError as e:
                        print(f"JSON decoding failed: {e}")
                        num_retries += 1
                        continue
                    break
                scene.update(sr)
            ppr.dump_train(self.output_train_filename)
            if self.output_eval_filename is not None:
                ppr.dump_eval(self.output_eval_filename)
        return

    def update_scene_list(self, scene_limit):
        """Update the scene list with responses from the LLM.
        :param type int, scene_limit: The maximum number of scenes to process.
        :param type bool, build_summaries_to_date: Whether to build summaries to date.
        :return: None
        """
        ppr = self.preprocess_results
        self._get_scene_list_response(ppr, scene_limit)
        self._dump_train()
        if self.output_eval_filename is not None:
            self._dump_eval()
