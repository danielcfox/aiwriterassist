#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""
import json
import os
from tqdm import tqdm

from llm_narrative_handler import LLMNarrativeScenesHandler

class LLMNarrativeScenesPreprocessing(LLMNarrativeScenesHandler):
    """Class for handling LLM interactions for narrative scenes with preprocessing."""
    def __init__(self, clean, user, narrative, author_name, api_obj, train_input_file, eval_input_file, 
                 train_output_file, eval_output_file, max_input_tokens, max_output_tokens):
        """Initialize the LLMNarrativeScenesPreprocessing with model and narrative details."""
        if (not clean and train_output_file is not None and os.path.exists(train_output_file) and eval_output_file is not None 
            and os.path.exists(eval_output_file)):
            super().__init__(user, narrative, author_name, api_obj, train_output_file, eval_output_file, 
                             max_input_tokens, max_output_tokens)
        else:
            super().__init__(user, narrative, author_name, api_obj, train_input_file, eval_input_file,
                             max_input_tokens, max_output_tokens)

        self.train_output_file = train_output_file
        self.eval_output_file = eval_output_file
        
    def _dump_train(self):        
        self.preprocess_results.dump_train(self.train_output_file)

    def _dump_eval(self):
        self.preprocess_results.dump_eval(self.eval_output_file)

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
        p += f"that fiction writer {self.user} wrote. The content of the scene is:\n"
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
            ppr.dump_train(self.train_output_file)
            ppr.dump_eval(self.eval_output_file)
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
        self._dump_eval()
