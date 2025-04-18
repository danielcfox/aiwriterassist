#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""
import copy
import math
import os
import random
from typing import Optional
import yaml

from narrative_handler import NarrativeScenesHandler
from narrative_preprocess import NarrativePreprocessResults
from vdb_milvus import VectorDBMilvus
from llm_openai_api_handler import LLMOpenAIAPIHandler

class LLMNarrativeScenesCompose(NarrativeScenesHandler):
    """
    A handler for generating narrative scenes using LLMs and contextual information.

    This class uses an LLM API and vector database to create narrative scenes based on
    provided scene specifications. It retrieves relevant context from previous scenes
    and uses that to generate coherent, stylistically appropriate narrative content.

    The class handles:
    1. Loading scene specifications from input files
    2. Retrieving contextual information from vector database
    3. Building prompts with character, setting, and object descriptions
    4. Generating narrative text using LLM completions
    5. Writing completed scenes to output files

    The composition process balances token usage between different narrative elements
    to create rich, contextually-aware narrative content that maintains consistency
    with the existing narrative world and author's style.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialize a scene composition handler for generating narrative scenes using LLM.

        This class uses an LLM API and vector database to create narrative scenes based on
        provided scene specifications. It retrieves relevant context from previous scenes
        and uses that to generate coherent, stylistically appropriate narrative content.

        Parameters:
            - kwargs: Additional keyword arguments for configuration:
            Required:
                - api_obj (LLMOpenAIAPIHandler): Handler for LLM API interactions
                - author_name (str): Name of the author whose style to emulate
                - input_compose_filename (str): Path to file with scene specifications
                - max_input_tokens (int): Maximum tokens for input prompt
                - max_output_tokens (int): Maximum tokens for generated content
                - narrative (str): Narrative identifier
                - output_compose_filename (str): Path for saving composed scenes
                - scene_limit (int): Maximum number of scenes to compose
                - vector_db (VectorDBMilvus): Vector database for retrieving context
                - input_train_filename (str): Path to training data file
                - input_eval_filename (str): Path to evaluation data file
            Optional:
                - links_filename (str): Path to file with links between scenes
                - previous_narrative_fraction (float): Fraction of prompt (after preamble) for recent narrative
                - verbose (bool): Whether to output processing details (default: False)

        Raises:
            ValueError: If any required parameter is missing or invalid
        """
        # super().__init__(user, narrative, author_name, api_obj, input_train_filename, input_eval_filename,
        #                  max_input_tokens, max_output_tokens)

        # Required parameters for parent
        # one of: input_train_filename, input_eval_filename

        # Required parameters
        self.api_obj = None
        self.author_name = None
        self.input_compose_filename = None
        self.max_input_tokens = None
        self.max_output_tokens = None
        self.narrative = None
        self.output_compose_filename = None
        self.scene_limit = None
        self.vector_db = None

        # Optional parameters
        self.recent_event_count = 0
        self.verbose = False
        self.links_filename = None
        self.previous_narrative_fraction = 0.5 # fraction of the prompt after preable that is the most recent text of the narrative
        self.generate_prompt_only = False
        self.request_log_file_template = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.api_obj is None or not isinstance(self.api_obj, LLMOpenAIAPIHandler):
            raise ValueError("api_obj is not set")
        if self.author_name is None or len(self.author_name) == 0:
            raise ValueError("author_name is not set")
        if self.input_compose_filename == None or len(self.input_compose_filename) == 0 or not os.path.exists(self.input_compose_filename):
            raise ValueError("Input compose file is not initialized or does not exist. Please provide a valid input filename.")
        if self.max_input_tokens is None or self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than 0")
        if self.max_output_tokens is None or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0")
        if self.narrative is None or len(self.narrative) == 0:
            raise ValueError("narrative is not set")
        if self.output_compose_filename == None or len(self.output_compose_filename) == 0:
            raise ValueError("Output filename is not initialized. Please provide a valid output filename.")
        if self.scene_limit is None or self.scene_limit <= 0:
            raise ValueError("scene_limit must be greater than 0")
        if self.vector_db == None or not isinstance(self.vector_db, VectorDBMilvus):
            raise ValueError("VectorDBMilvus is not initialized. Please provide a valid URI in the configuration YAML file.")

        input_filename_list = []
        if self.input_train_filename is not None and os.path.exists(self.input_train_filename):
            input_filename_list.append(self.input_train_filename)
        if self.input_eval_filename is not None and os.path.exists(self.input_eval_filename):
            input_filename_list.append(self.input_eval_filename)
        if len(input_filename_list) == 0:
            raise ValueError("LLMNarrativeScenesCollection: Neither Input train file nor Input eval file exists.")

        self.to_compose = NarrativePreprocessResults()
        self.to_compose.load(self.input_compose_filename)

        super().__init__(input_filename_list, self.verbose)
        # super().__init__(**kwargs)

        if self.links_filename is None or len(self.links_filename) == 0 or not os.path.exists(self.links_filename):
            self.previous_narrative_fraction = 0.0
            print(f"No links_filename ({self.links_filename}) provided, setting previous_narrative_fraction to 0.0")
            self.scene_list = None
            self.named_characters = None
            self.links = None
            self.linked_named_characters = None
        else:
            with open(self.links_filename, "r") as fp:
                self.links = yaml.safe_load(fp)  # Use safe_load for security
            self.linked_named_characters = self.links['named_characters']

            self.scene_list = self.preprocess_results.scene_list
            # this class MUST NOT modify the scene list
            # self.verbose = verbose
        self.named_characters = {}

        # self.user = user
        # self.narrative = narrative
        # self.output_filename = output_filename
        # self.vector_db = vector_db

        if self.verbose:
            print("LLMNarrativeScenesCompose: Size of scene list is", len(self.preprocess_results.scene_list))

    def _retrieve_scene_context_records(self, scene: dict) -> list:
        """
        Extract context information directly from the scene specification.

        Gathers character, setting, and object descriptions from the current scene
        specification to use as context for scene generation. This collects only
        the descriptions explicitly provided in the scene request.

        Parameters:
            - scene (dict): Scene specification containing characters, settings, and objects

        Returns:
            list: List of description strings from the scene specification
        """
        character_list = [c for c in scene['named_characters'].keys()]
        setting_list = [s for s in scene['settings'].keys()]
        object_list = [o for o in scene['objects'].keys()]

        fixed_records = []
        for character in character_list:
            if character in scene['named_characters']:
                if ('description' in scene['named_characters'][character]
                    and len(scene['named_characters'][character]['description']) > 0):
                    fixed_records.append(scene['named_characters'][character]['description'])
        for setting in setting_list:
            if ('setting' in scene['settings']
                and len(scene['settings'][setting]['description']) > 0):
                fixed_records.append(scene['settings'][setting]['description'])
        for object in object_list:
            if ('object' in scene['objects']
                and len(scene['objects'][object]['description']) > 0):
                fixed_records.append(scene['objects'][object]['description'])

        return fixed_records

    def _guestimate_num_records(self, scene: dict) -> tuple:
        """
        Calculate the optimal number of description records to retrieve for each entity type.

        This method estimates how many description records to retrieve for each
        character, object, and setting based on available token space in the prompt
        and the number of entities involved in the scene.

        Parameters:
            - scene (dict): Scene specification containing characters, settings, and objects
            - preamble (str): Current prompt preamble

        Returns:
            tuple: Six integers representing record counts for:
                - Main character
                - Other characters
                - First object
                - Other objects
                - First setting
                - Other settings
        """
        character_list = [c for c in scene['named_characters'].keys()]
        setting_list = [s for s in scene['settings'].keys()]
        object_list = [o for o in scene['objects'].keys()]

        # input_tokens_used = len(preamble) * 3
        # input_tokens_remaining = self.max_input_tokens - input_tokens_used
        input_tokens_remaining = self.scene_budget['description']
        num_other_chars = len(character_list) - 1
        if len(object_list) == 0:
            num_other_objects = 0
        else:
            num_other_objects = len(object_list) - 1
        num_other_settings = len(setting_list) - 1
        if len(object_list) > 0:
            total_num_entities = num_other_chars + num_other_settings + num_other_objects + 7
        else:
            total_num_entities = num_other_chars + num_other_settings + 6
        main_char_tokens = (input_tokens_remaining * 5) // total_num_entities
        other_char_tokens = (input_tokens_remaining * 1) // total_num_entities
        if len(object_list) > 0:
            first_object_tokens = (input_tokens_remaining * 1) // total_num_entities
            other_object_tokens = (input_tokens_remaining * 1) // total_num_entities
        else:
            first_object_tokens = 0
            other_object_tokens = 0
        first_setting_tokens = (input_tokens_remaining * 1) // total_num_entities
        other_setting_tokens = (input_tokens_remaining * 1) // total_num_entities
        main_char_num_records = max(main_char_tokens * 3 // 50, 1)
        other_char_num_records = max(other_char_tokens * 3 // 50, 1)
        if len(object_list) > 0:
            first_object_num_records = max(first_object_tokens * 3 // 50, 1)
            other_object_num_records = max(other_object_tokens * 3 // 50, 1)
        else:
            first_object_num_records = 0
            other_object_num_records = 0
        first_setting_num_records = max(first_setting_tokens * 3 // 50, 1)
        other_setting_num_records = max(other_setting_tokens * 3 // 50, 1)

        return (main_char_num_records, other_char_num_records, first_object_num_records,
                other_object_num_records, first_setting_num_records, other_setting_num_records)

    def _get_character_records(self, prev_chron_scene_index: int, character_list: list,
                               main_char_num_records: int, other_char_num_records: int) -> dict:
        """
        Retrieve character descriptions from the vector database.

        Searches the vector database for descriptions of characters from previous scenes,
        retrieving more records for the main character and fewer for supporting characters.

        Parameters:
            - prev_chron_scene_index (int): Index of previous scene (to avoid future information)
            - character_list (list): List of character names to retrieve records for
            - main_char_num_records (int): Number of records to retrieve for the main character
            - other_char_num_records (int): Number of records to retrieve for supporting characters

        Returns:
            dict: Dictionary mapping character names to lists of description records
        """
        if len(character_list) == 0:
            return {}
        res = self.vector_db.search(self.narrative, prev_chron_scene_index, [f"{character_list[0]}"], limit=main_char_num_records)
        for r in res[0]:
            scene_index = r['id'] // 1000
            if scene_index > prev_chron_scene_index:
                print(f"scene_index {scene_index} in _get_character_records() "
                      + f"is greater than prev_chron_scene_index {prev_chron_scene_index}")
                return {}
        main_char_records = []
        word_set_in_characters = set()
        if self.linked_named_characters is not None and len(self.linked_named_characters) > 0:
            entity = self._match_name_to_entity(self.linked_named_characters, character_list[0], scene_index)
        else:
            entity = character_list[0]
        word_set_in_characters.update(entity.lower().split())
        for r in res[0]:
            word_set_in_record = set(r['entity']['text'].lower().split())
            if len(word_set_in_characters.intersection(word_set_in_record)) > 0:
                main_char_records.append(r['entity']['text'])
        # main_char_records = [r['entity']['text'] for r in res[0]]
        char_dict = {character_list[0]: main_char_records}
        for i in range(1, len(character_list)):
            res = self.vector_db.search(self.narrative, prev_chron_scene_index, [f"{character_list[i]}"],
                                        limit=other_char_num_records)
            for r in res[0]:
                scene_index = r['id'] // 1000
                if scene_index > prev_chron_scene_index:
                    print(f"scene_index {scene_index} in _get_character_records() "
                          + f"is greater than prev_chron_scene_index {prev_chron_scene_index}")
                    return {}
            this_char_records = []
            word_set_in_characters = set()
            if self.linked_named_characters is not None and len(self.linked_named_characters) > 0:
                entity = self._match_name_to_entity(self.linked_named_characters, character_list[i], scene_index)
            else:
                entity = character_list[i]
            word_set_in_characters.update(entity.lower().split())
            for r in res[0]:
                word_set_in_record = set(r['entity']['text'].lower().split())
                if len(word_set_in_characters.intersection(word_set_in_record)) > 0:
                    this_char_records.append(r['entity']['text'])
            char_dict[character_list[i]] = this_char_records
        return char_dict

    def _get_setting_records(self, prev_chron_scene_index: int, setting_list: list,
                            first_setting_num_records: int, other_setting_num_records: int) -> dict:
        """
        Retrieve setting descriptions from the vector database.

        Searches the vector database for descriptions of settings from previous scenes,
        retrieving more records for the primary setting and fewer for secondary settings.

        Parameters:
            - prev_chron_scene_index (int): Index of previous scene (to avoid future information)
            - setting_list (list): List of setting names to retrieve records for
            - first_setting_num_records (int): Number of records to retrieve for the primary setting
            - other_setting_num_records (int): Number of records to retrieve for secondary settings

        Returns:
            dict: Dictionary mapping setting names to lists of description records
        """
        if len(setting_list) == 0:
            return {}
        res = self.vector_db.search(self.narrative, prev_chron_scene_index, [f"{setting_list[0]}"], limit=first_setting_num_records)
        for r in res[0]:
            scene_index = r['id'] // 1000
            if scene_index > prev_chron_scene_index:
                print(f"scene_index {scene_index} in _get_setting_records() "
                      + f"is greater than prev_chron_scene_index {prev_chron_scene_index}")
                return {}
        first_setting_records = []
        word_set_in_settings = set()
        word_set_in_settings.update(setting_list[0].lower().split())
        for r in res[0]:
            # print(r)
            word_set_in_record = set(r['entity']['entity_name'].lower().split())
            if len(word_set_in_settings.intersection(word_set_in_record)) > 0:
                first_setting_records.append(r['entity']['text'])
        # main_char_records = [r['entity']['text'] for r in res[0]]
        # first_setting_records = [r['entity']['text'] for r in res[0]]
        setting_dict = {setting_list[0]: first_setting_records}
        for i in range(1, len(setting_list)):
            res = self.vector_db.search(self.narrative, prev_chron_scene_index, [f"{setting_list[i]}"], limit=other_setting_num_records)
            for r in res[0]:
                scene_index = r['id'] // 1000
                if scene_index > prev_chron_scene_index:
                    print(f"scene_index {scene_index} in _get_setting_records() "
                          + f"is greater than prev_chron_scene_index {prev_chron_scene_index}")
                    return {}
            this_setting_records = []
            word_set_in_settings = set()
            word_set_in_settings.update(setting_list[i].lower().split())
            for r in res[0]:
                word_set_in_record = set(r['entity']['entity_name'].lower().split())
                if len(word_set_in_settings.intersection(word_set_in_record)) > 0:
                    this_setting_records.append(r['entity']['text'])
            setting_dict[setting_list[i]] = this_setting_records
        return setting_dict

    def _get_object_records(self, prev_chron_scene_index: int, object_list: list,
                        first_object_num_records: int, other_object_num_records: int) -> dict:
        """
        Retrieve object descriptions from the vector database.

        Searches the vector database for descriptions of objects from previous scenes,
        retrieving more records for the primary object and fewer for secondary objects.

        Parameters:
            - prev_chron_scene_index (int): Index of previous scene (to avoid future information)
            - object_list (list): List of object names to retrieve records for
            - first_object_num_records (int): Number of records to retrieve for the primary object
            - other_object_num_records (int): Number of records to retrieve for secondary objects

        Returns:
            dict: Dictionary mapping object names to lists of description records
        """
        if len(object_list) == 0:
            return {}
        res = self.vector_db.search(self.narrative, prev_chron_scene_index, [f"{object_list[0]}?"], limit=first_object_num_records)
        for r in res[0]:
            scene_index = r['id'] // 1000
            if scene_index > prev_chron_scene_index:
                print(f"scene_index {scene_index} in _get_object_records() "
                      + f"is greater than prev_chron_scene_index {prev_chron_scene_index}")
                return {}
        first_object_records = []
        word_set_in_objects = set()
        word_set_in_objects.update(object_list[0].lower().split())
        for r in res[0]:
            word_set_in_record = set(r['entity']['entity_name'].lower().split())
            if len(word_set_in_objects.intersection(word_set_in_record)) > 0:
                first_object_records.append(r['entity']['text'])
        object_dict = {object_list[0]: first_object_records}
        for i in range(1, len(object_list)):
            res = self.vector_db.search(self.narrative, prev_chron_scene_index, [f"{object_list[i]}?"],
                                        limit=other_object_num_records)
            for r in res[0]:
                scene_index = r['id'] // 1000
                if scene_index > prev_chron_scene_index:
                    print(f"scene_index {scene_index} in _get_object_records() "
                          + f"is greater than prev_chron_scene_index {prev_chron_scene_index}")
                    return {}
            this_object_records = []
            word_set_in_objects = set()
            word_set_in_objects.update(object_list[i].lower().split())
            for r in res[0]:
                word_set_in_record = set(r['entity']['entity_name'].lower().split())
                if len(word_set_in_objects.intersection(word_set_in_record)) > 0:
                    this_object_records.append(r['entity']['text'])
            object_dict[object_list[i]] = this_object_records
        return object_dict

    def _get_scene_creation_introduction_prompt(self):
        return (f"Write a fiction scene in the style of '{self.author_name}':\n\nYou will be presented with contextual information "
                + "that is relevant to the scene you are about to write. First you will be given the relevant portions of the most "
                + "recent narrative that has been written so far.\n"
                + "This narrative will begin with BEGIN NARRATIVE and end with END NARRATIVE.\n"
                + "Then you will be given some contextual information that is relevant to the scene you are about to write.\n"
                + "This contextual information will begin with BEGIN CONTEXT and end with END CONTEXT.\n"
                + "Then you will be given some short descriptions of characters, settings, and objects.\n"
                + "These descriptions will begin with BEGIN DESCRIPTIONS and end with END DESCRIPTIONS.\n"
                + "Then the first paragraph of the scene you will write is provided.\n"
                + "This paragraph will begin with BEGIN OPENING PARAGRAPH OF SCENE TO BE WRITTEN "
                + "and end with END OPENING PARAGRAPH OF SCENE TO BE WRITTEN.\n"
                + "The scene you are about to write will be consistent with the information provided in the prompt.\n"
                + "Do not include any known characters or settings that are not mentioned above, though you may refer to them.\n"
                + "The scene should not use previous content by the author, other than the content provided in the prompt.\n"
                + "The scene should be consistent with the tone, synopsis, and traits and backstory of characters and settings.\n"
                + "Characters not explicitly named may be assigned a name, but "
                + "only if the point of view character becomes aware of the name.\n\n")

    def _get_scene_creation_context_prompt(self, scene_index: int, tone: str, character_list: list,
                                           object_list: list, setting_list: list, synopsis: str,
                                           focus: Optional[str] = None) -> str:
                                          # recent_events: Optional[str] = None) -> str:
        """
        Generate the initial portion of the scene creation prompt.

        Creates a structured prompt that specifies the stylistic, tonal, and narrative
        requirements for the scene to be generated, including character, setting,
        and object lists, synopsis, and recent narrative events.

        Required Parameters:
            - tone (str): The emotional tone of the scene
            - character_list (list): List of characters appearing in the scene
            - object_list (list): List of objects available for interaction
            - setting_list (list): List of settings in chronological order
            - synopsis (str): Brief summary of what should happen in the scene
            - beginning (str): The opening sentence or paragraph for the scene

        Optional Parameters:
            - recent_narrative (str): Summary of recent events in the narrative
            - focus (str): The scene's primary focus or theme (can be None)

        Returns:
            str: Formatted prompt preamble
        """

        new_character_list = [self._match_name_to_entity(self.linked_named_characters, c, scene_index) for c in character_list]

        p = f"Using the following context, write a fiction scene in the style of '{self.author_name}':\n\n"
        p += "\n\nBEGIN CONTEXT\n\n"
        # p = "Compose a scene for a fiction narrative consistent with the following information:\n"
        # p += f"The author is {self.user}, and the scene should be composed in the author's style.\n"
        # p += f"The scene should not use previous content by the author, other than the content provided in the prompt.\n"
        p += f"The tone of the scene is: {tone}\n"
        p += f"The point of view character is: {new_character_list[0]}\n"
        p += f"The named characters are: {', '.join(new_character_list)}\n"
        if len(object_list) > 0:
            p += f"Objects the characters can interact with are: {', '.join(object_list)}\n"
        p += f"The settings in chronological order are: {', '.join(setting_list)}\n"
        p += f"The synopsis of the scene to be written is:\n{synopsis}\n"
        # p += "Do not include any known characters or settings that are not mentioned above, though you may refer to them.\n"
        # p += "The scene should be consistent with the tone, synopsis, and traits and backstory of characters and settings.\n"
        # p += "Characters not explicitly named may be assigned a name, but "
        # p += "only if the point of view character becomes aware of the name.\n"
        if focus is not None:
            p += f"The focus of the scene: {focus}\n"
        # if beginning is not None and len(beginning) > 0:
        #     p += f"The beginning of the scene is:\n{beginning}\n\n"
        # if recent_events is None:
            # p += "There are no recent events that have occurred in the narrative. This is the beginning of the narrative.\n"
        # else:
            # p += f"Here are the most recent events that have occurred in the narrative:\n\n{recent_narrative}\n\n"
        #    p += "This ends the recent events.\n\n"
        # p += "Here is some more information regarding the characters, settings, and objects:\n\n"
        p += "\nEND_CONTEXT\n\n"

        return p

        # preamble = self._get_scene_creation_preamble_prompt(
        #     scene['chron_scene_index'],
        #     tone,
        #     character_list,
        #     object_list,
        #     setting_list,
        #     synopsis,
        #     beginning,
        #     focus)
    def _get_scene_creation_descriptions(self, scene: dict) -> str:
        """
        Generate the character, setting, and object descriptions for the scene.
        This method creates a structured prompt that includes descriptions of characters,
        settings, and objects relevant to the scene being generated.

        Required Parameters:
            - scene (dict): Scene specification containing characters, settings, and objects
        Returns:
            str: Formatted prompt with character, setting, and object descriptions
        """
        if 'plot_summary' not in scene:
            print("No plot summary found")
            return ""
        if 'tone' not in scene:
            print("No tone found")
            return ""
        if 'chron_scene_index' not in scene:
            print("No chron scene index found")
            return ""
        if 'named_characters' not in scene:
            print("No named characters found")
            return ""
        if 'settings' not in scene:
            print("No settings found")
            return ""

        if 'objects' not in scene:
            object_list = []
        else:
            object_list = [o for o in scene['objects'].keys()]
        prev_chron_scene_index = scene['chron_scene_index'] - 1
        print(f"prev_chron_scene_index is {prev_chron_scene_index}")
        synopsis = scene['plot_summary']
        tone = scene['tone']
        character_list = [c for c in scene['named_characters'].keys()]
        setting_list = [s for s in scene['settings'].keys()]

        if len(character_list) == 0:
            print("No characters found")
            return ""
        if len(setting_list) == 0:
            print("No settings found")
            return ""
        if len(synopsis) == 0:
            print("No synopsis found")
            return ""
        if len(tone) == 0:
            print("No tone found")
            return ""

        if 'focus' not in scene:
            focus = None
        else:
            focus = scene['focus']

        # self.scene_budget['preamble'] = len(preamble)
        # if self.scene_budget['preamble'] > self.max_input_tokens:
        #     if self.verbose:
        #         print(f"prompt preamble length is too long: {self.scene_budget['preamble']}")
        #     return preamble[:self.max_input_tokens]
        # # if self.verbose:

        # self.scene_budget['RAG'] = math.floor((self.scene_budget['overall'] - self.scene_budget['preamble'])
        #                                       * (1.0 - self.previous_narrative_fraction))
        # self.scene_budget['recent_narrative'] = (self.scene_budget['overall'] - self.scene_budget['preamble']
        #                                          - self.scene_budget['RAG'])
        if self.verbose:
            print(f"scene budget is {self.scene_budget}")

        # first we include all of the descriptions and plot summaries embedded in the scene request
        fixed_records = self._retrieve_scene_context_records(scene)
        # print(fixed_records, len(fixed_records))
        # print(character_list, object_list, setting_list)

        # guestimate how many records to retrieve from the RAG vector db for each type
        # assume 50 characters for each doc in the vector db, will of course be larger than that, but that's the upper bound
        # this ensures we will retrieve more than we can fit (or all of them), which is fine, we simply cut if off later
        main_char_num_records, other_char_num_records, first_object_num_records, other_object_num_records,\
            first_setting_num_records, other_setting_num_records =\
            self._guestimate_num_records(scene)

        # now we finally retrieve the records from the vector db using RAG
        # the first record for each character, object, and setting is the one we want to use
        # the rest of the records are used to fill in the rest of the scene
        if len(character_list) > 0:
            char_dict = self._get_character_records(prev_chron_scene_index, character_list,
                                                    main_char_num_records, other_char_num_records)
        else:
            char_dict = {}
        if len(object_list) > 0:
            object_dict = self._get_object_records(prev_chron_scene_index, object_list,
                                                   first_object_num_records, other_object_num_records)
        else:
            object_dict = {}
        if len(setting_list) > 0:
            setting_dict = self._get_setting_records(prev_chron_scene_index, setting_list,
                                                     first_setting_num_records, other_setting_num_records)
        else:
            setting_dict = {}

        # print(char_dict, len(char_dict))

        # now we leave the fixed records in order at the top of the stack
        # the first set of records was already added to the fixed_records list and was everything in the scene request
        # now add the first record for each character, object, and setting to ensure there's one historic record for each
        # note all of these records may not be used, but we'll use as many as we can
        if len(char_dict) > 0 and len(char_dict[character_list[0]]) > 0:
                fixed_records.append(char_dict[character_list[0]][0])
        if len(object_dict) > 0 and len(object_dict[object_list[0]]) > 0:
                fixed_records.append(object_dict[object_list[0]][0])
        if len(setting_dict) > 0 and len(setting_dict[setting_list[0]]) > 0:
            fixed_records.append(setting_dict[setting_list[0]][0])
        for i in range(1, len(character_list)):
            if len(char_dict) > 0 and len(char_dict[character_list[i]]) > 0:
                fixed_records.append(char_dict[character_list[i]][0])
        for i in range(1, len(object_list)):
            if len(object_dict) > 0 and len(object_dict[object_list[i]]) > 0:
                fixed_records.append(object_dict[object_list[i]][0])
        for i in range(1, len(setting_list)):
            if len(setting_dict) > 0 and len(setting_dict[setting_list[i]]) > 0:
                fixed_records.append(setting_dict[setting_list[i]][0])

        # now we have all of the rest of the records for each character, object, and setting. we'll shuffle them
        # and add them to the end of the fixed records
        second_records = []
        for character in char_dict:
            if len(char_dict[character]) > 1:
                second_records.extend(char_dict[character][1:])
        for object in object_dict:
            if len(object_dict[object]) > 1:
                second_records.extend(object_dict[object][1:])
        for setting in setting_dict:
            if len(setting_dict[setting]) > 1:
                second_records.extend(setting_dict[setting][1:])
        random.shuffle(second_records)

        records = []
        records.extend(fixed_records)
        records.extend(second_records)

        print(f"Number of records is {len(records)}")
        # print(records)

        prompt_rag_len = self.scene_budget['description']
        prompt_rag_old = ""
        prompt_rag_begin = "\n\nBEGIN DESCRIPTIONS\n\n"
        prompt_rag_end = "\n\nEND DESCRIPTIONS\n\n"
        prompt_rag = prompt_rag_begin
        prompt_rag_old = prompt_rag
        for record in records:
            if len(prompt_rag) + len(prompt_rag_end) > prompt_rag_len:
                return prompt_rag_old + prompt_rag_end
            prompt_rag_old = prompt_rag
            prompt_rag += f"{record}\n"

        # s_new = p
        # s_old = s_new
        # for record in records:
        #     if len(s_new) > (self.max_input_tokens * 3):
        #        return s_old
        #     s_old = s_new
        #     q = f"{record}\n"
        #     s_new = p + q

        # This section for debugging only, uncomment it if you want to debug both the number of records and the prompt
        print(f"Added {len(records)} records to the prompt, but we still had room for more.")
        # this is the final prompt, which is the original prompt plus all of the records

            # scene['chron_scene_index'],
            # tone,
            # character_list,
            # object_list,
            # setting_list,
            # synopsis,
            # beginning,
            # focus)
        return prompt_rag + prompt_rag_end

    def _get_scene_creation_prompt_beginning_paragraph(self, scene: dict) -> str:
        """
        Display the opening paragraph for the scene.
        """
        return (f"BEGIN OPENING PARAGRAPH OF SCENE TO BE WRITTEN\n\n{scene['body'].split("\n")[0]}\n\n"
                + f"END OPENING PARAGRAPH OF SCENE TO BE WRITTEN\n\n")

    def _get_scene_creation_prompt_previous_narrative(self, scene: dict) -> str:
        character_list = [c for c in scene['named_characters'].keys()]
        cur_scene_index = scene['chron_scene_index']
        intro = "\n\nBEGIN PREVIOUS NARRATIVE\n\n"
        end = "\n\nEND PREVIOUS NARRATIVE\n\n"
        if len(intro) > self.scene_budget['recent_narrative']:
            print(f"intro length for recent narrtive is too long: {len(intro)}")
            return
        scene_remaining_len = self.scene_budget['recent_narrative'] - len(intro) - len(end)
        if self.verbose:
            print(f"max_input_tokens {self.max_input_tokens} scene_remaining_len {scene_remaining_len}")
        if scene_remaining_len > 0:
            previous_narrative = self._get_scenes_text_for_characters_prior_to_scene(
                character_list, cur_scene_index, char_count_limit=scene_remaining_len
                )

        return intro + previous_narrative + end

    # def _format_scene_creation_prompt_preamble_with_rag(self, scene: dict, recent_narrative: Optional[str] = None) -> str:
    #     """
    #     Build the complete scene generation prompt with context.

    #     This method combines the scene specification with relevant contextual information
    #     from the vector database. It prioritizes descriptions from the scene itself,
    #     then adds historical context from previous scenes, balancing between different
    #     entity types (characters, settings, objects) to create a rich but token-efficient prompt.

    #     Required Parameters:
    #         - scene (dict): Complete scene specification

    #     Optional Parameters:
    #         - recent_narrative (str): Description of recent events in the narrative (can be None)

    #     Returns:
    #         str: Complete formatted prompt for the LLM
    #     """
    #     # assume max scene length of 3000 tokens
    #     if 'plot_summary' not in scene:
    #         print("No plot summary found")
    #         return ""
    #     if 'tone' not in scene:
    #         print("No tone found")
    #         return ""
    #     if 'chron_scene_index' not in scene:
    #         print("No chron scene index found")
    #         return ""
    #     if 'named_characters' not in scene:
    #         print("No named characters found")
    #         return ""
    #     if 'settings' not in scene:
    #         print("No settings found")
    #         return ""

    #     if 'objects' not in scene:
    #         object_list = []
    #     else:
    #         object_list = [o for o in scene['objects'].keys()]
    #     prev_chron_scene_index = scene['chron_scene_index'] - 1
    #     print(f"prev_chron_scene_index is {prev_chron_scene_index}")
    #     synopsis = scene['plot_summary']
    #     tone = scene['tone']
    #     character_list = [c for c in scene['named_characters'].keys()]
    #     setting_list = [s for s in scene['settings'].keys()]

    #     if len(character_list) == 0:
    #         print("No characters found")
    #         return ""
    #     if len(setting_list) == 0:
    #         print("No settings found")
    #         return ""
    #     if len(synopsis) == 0:
    #         print("No synopsis found")
    #         return ""
    #     if len(tone) == 0:
    #         print("No tone found")
    #         return ""

    #     if 'focus' not in scene:
    #         focus = None
    #     else:
    #         focus = scene['focus']

    #     beginning = scene['body'].split("\n")[0]

    #     preamble = self._get_scene_creation_preamble_prompt(
    #         scene['chron_scene_index'],
    #         tone,
    #         character_list,
    #         object_list,
    #         setting_list,
    #         synopsis,
    #         beginning,
    #         focus)

    #     self.scene_budget['preamble'] = len(preamble)
    #     if self.scene_budget['preamble'] > self.max_input_tokens:
    #         if self.verbose:
    #             print(f"prompt preamble length is too long: {self.scene_budget['preamble']}")
    #         return preamble[:self.max_input_tokens]
    #     # if self.verbose:

    #     self.scene_budget['RAG'] = math.floor((self.scene_budget['overall'] - self.scene_budget['preamble'])
    #                                           * (1.0 - self.previous_narrative_fraction))
    #     self.scene_budget['recent_narrative'] = (self.scene_budget['overall'] - self.scene_budget['preamble']
    #                                              - self.scene_budget['RAG'])
    #     if self.verbose:
    #         print(f"scene budget is {self.scene_budget}")

    #     # first we include all of the descriptions and plot summaries embedded in the scene request
    #     fixed_records = self._retrieve_scene_context_records(scene)
    #     # print(fixed_records, len(fixed_records))
    #     # print(character_list, object_list, setting_list)

    #     # guestimate how many records to retrieve from the RAG vector db for each type
    #     # assume 50 characters for each doc in the vector db, will of course be larger than that, but that's the upper bound
    #     # this ensures we will retrieve more than we can fit (or all of them), which is fine, we simply cut if off later
    #     main_char_num_records, other_char_num_records, first_object_num_records, other_object_num_records,\
    #         first_setting_num_records, other_setting_num_records =\
    #         self._guestimate_num_records(scene, preamble)

    #     # now we finally retrieve the records from the vector db using RAG
    #     # the first record for each character, object, and setting is the one we want to use
    #     # the rest of the records are used to fill in the rest of the scene
    #     if len(character_list) > 0:
    #         char_dict = self._get_character_records(prev_chron_scene_index, character_list,
    #                                                 main_char_num_records, other_char_num_records)
    #     else:
    #         char_dict = {}
    #     if len(object_list) > 0:
    #         object_dict = self._get_object_records(prev_chron_scene_index, object_list,
    #                                                first_object_num_records, other_object_num_records)
    #     else:
    #         object_dict = {}
    #     if len(setting_list) > 0:
    #         setting_dict = self._get_setting_records(prev_chron_scene_index, setting_list,
    #                                                  first_setting_num_records, other_setting_num_records)
    #     else:
    #         setting_dict = {}

    #     # print(char_dict, len(char_dict))

    #     # now we leave the fixed records in order at the top of the stack
    #     # the first set of records was already added to the fixed_records list and was everything in the scene request
    #     # now add the first record for each character, object, and setting to ensure there's one historic record for each
    #     # note all of these records may not be used, but we'll use as many as we can
    #     if len(char_dict) > 0 and len(char_dict[character_list[0]]) > 0:
    #             fixed_records.append(char_dict[character_list[0]][0])
    #     if len(object_dict) > 0 and len(object_dict[object_list[0]]) > 0:
    #             fixed_records.append(object_dict[object_list[0]][0])
    #     if len(setting_dict) > 0 and len(setting_dict[setting_list[0]]) > 0:
    #         fixed_records.append(setting_dict[setting_list[0]][0])
    #     for i in range(1, len(character_list)):
    #         if len(char_dict) > 0 and len(char_dict[character_list[i]]) > 0:
    #             fixed_records.append(char_dict[character_list[i]][0])
    #     for i in range(1, len(object_list)):
    #         if len(object_dict) > 0 and len(object_dict[object_list[i]]) > 0:
    #             fixed_records.append(object_dict[object_list[i]][0])
    #     for i in range(1, len(setting_list)):
    #         if len(setting_dict) > 0 and len(setting_dict[setting_list[i]]) > 0:
    #             fixed_records.append(setting_dict[setting_list[i]][0])

    #     # now we have all of the rest of the records for each character, object, and setting. we'll shuffle them
    #     # and add them to the end of the fixed records
    #     second_records = []
    #     for character in char_dict:
    #         if len(char_dict[character]) > 1:
    #             second_records.extend(char_dict[character][1:])
    #     for object in object_dict:
    #         if len(object_dict[object]) > 1:
    #             second_records.extend(object_dict[object][1:])
    #     for setting in setting_dict:
    #         if len(setting_dict[setting]) > 1:
    #             second_records.extend(setting_dict[setting][1:])
    #     random.shuffle(second_records)

    #     records = []
    #     records.extend(fixed_records)
    #     records.extend(second_records)

    #     print(f"Number of records is {len(records)}")
    #     # print(records)

    #     prompt_rag_len = self.scene_budget['RAG']
    #     prompt_rag_old = ""
    #     prompt_rag = "\n\nHere is some more information regarding the characters, settings, and objects:\n\n"
    #     prompt_rag_old = prompt_rag
    #     for record in records:
    #         if len(prompt_rag) > prompt_rag_len:
    #             return prompt_rag_old
    #         prompt_rag_old = prompt_rag
    #         prompt_rag += f"{record}\n"

    #     # s_new = p
    #     # s_old = s_new
    #     # for record in records:
    #     #     if len(s_new) > (self.max_input_tokens * 3):
    #     #        return s_old
    #     #     s_old = s_new
    #     #     q = f"{record}\n"
    #     #     s_new = p + q

    #     # This section for debugging only, uncomment it if you want to debug both the number of records and the prompt
    #     print(f"Added {len(records)} records to the prompt, but we still had room for more.")
    #     # this is the final prompt, which is the original prompt plus all of the records

    #         # scene['chron_scene_index'],
    #         # tone,
    #         # character_list,
    #         # object_list,
    #         # setting_list,
    #         # synopsis,
    #         # beginning,
    #         # focus)

    #     character_list = [c for c in scene['named_characters'].keys()]
    #     cur_scene_index = scene['chron_scene_index']
    #     intro = "\n\nHere is the most recent relevant narrative, and write the scene assuming this narrative came just before:\n\n"
    #     if len(intro) > self.scene_budget['recent_narrative']:
    #         print(f"intro length for recent narrtive is too long: {len(intro)}")
    #         return
    #     scene_remaining_len = self.scene_budget['recent_narrative'] - len(intro)
    #     if self.verbose:
    #         print(f"max_input_tokens {self.max_input_tokens} scene_remaining_len {scene_remaining_len}, "
    #               + f"scene_creation_request len {len(scene_creation_request)}, intro len {len(intro)}")
    #     if scene_remaining_len > 0:
    #         previous_narrative = self._get_scenes_text_for_characters_prior_to_scene(
    #             character_list, cur_scene_index, char_count_limit=scene_remaining_len
    #             )
    #         narrative_portion = intro + previous_narrative



    #     prompt = preamble + prompt_rag

    #     if len(prompt) > self.scene_budget['overall']:
    #         print(f"prompt length is too long, no room for RAG, budget calculations are off: {len(prompt)}")
    #         prompt = prompt[:self.scene_budget['overall']]
    #     return prompt

    def _match_name_to_entity(self, entity_dict: dict, name: str, scene_index: int) -> str:
        """
        Match a name to an entity in the given dictionary.
        :param entity_dict: Dictionary of entities.
        :param name: Name to match.
        :return: Matched entity or None if not found.
        """
        if name is None or len(name) == 0:
            raise ValueError("name must be populated")
        if entity_dict is None or len(entity_dict) == 0:
            return name

        for entity, name_dict_list in entity_dict.items():
            if self.verbose:
                print(f"Checking entity {entity} for name {name}, name_dict_list is {name_dict_list}")
            for name_dict in name_dict_list:
                if name in name_dict:
                    # check if the scene_index is in the list of scenes for this entity
                    if len(name_dict[name]) == 0 or scene_index in name_dict[name]:
                        return entity
            # for alias, scene_list in name_dict_list.items():
            #     if alias == name and (len(scene_list) == 0 or scene_index in scene_list):
            #         return entity
        # if we didn't find it in the links file, then it's an entity with no aliases that needs to be added
        # new_name_dict = {name: []}
        # entity_dict[name] = new_name_dict
        # entity_dict = {k: v for k, v in sorted(entity_dict.items(), key=lambda k, v: v[0])}
        # entity_dict[name].sort()
        return name

    def _link_metadata_scene_list(self):
        """
        Create a scene list for each metadata entry.
        For now only character entities are supported.
        """
        # Implement the logic to link metadata entities here
        # This is a placeholder for the actual linking logic

        if self.linked_named_characters is None or len(self.linked_named_characters) == 0:
            print("No named characters found in links")
            # self.named_characters = {} already set in constructor, but if called again, want to reset
        else:
            self.named_characters = {k: [] for k in self.linked_named_characters.keys()}
        if len(self.named_characters) == 0:
            if self.verbose:
                print("No named character links found")

        # only characters supported for now
        if self.verbose:
            print(f"Linking metadata entities")

        for scene in self.scene_list:
            for name in scene['named_characters']:
                if self.linked_named_characters is not None and len(self.linked_named_characters) > 0:
                    entity = self._match_name_to_entity(self.linked_named_characters, name, scene['chron_scene_index'])
                else:
                    entity = name
                if entity not in self.named_characters:
                    self.named_characters[entity] = []
                self.named_characters[entity].append(scene['chron_scene_index'])
        for entity in self.named_characters:
            self.named_characters[entity].sort()
        # list for each character MUST be set, it is meaningless otherwise
        self.named_characters = {k: v for k, v in sorted(self.named_characters.items(), key=lambda v: v[0])}

    def _get_scene_list_for_characters(self, character_list: list[str], cur_scene_index: int):
        """
        Link metadata entities across scenes.
        """
        self._link_metadata_scene_list()

        scene_set = set()

        for character in character_list:
            master_character = self._match_name_to_entity(self.linked_named_characters, character, cur_scene_index)
            if master_character not in self.named_characters:
                raise ValueError(f"Character {master_character} not found in named characters.")
            scene_set.update(self.named_characters[master_character])
        scene_list_for_chars = list(scene_set)
        scene_list_for_chars.sort()
        return scene_list_for_chars


    def _get_scene_list_for_charactes_prior_to_scene(self, character_list: list[str], cur_scene_index: int):
        """
        Get the scene list for characters prior to a specific scene index.
        """
        scene_list = self._get_scene_list_for_characters(character_list, cur_scene_index)
        # Filter the scene list to include only those before the specified scene index
        # Implement additional linking logic here if needed
        prior_scene_list = []
        for scene_index in scene_list:
            if scene_index < cur_scene_index:
                prior_scene_list.append(scene_index)
            else:
                return prior_scene_list
        return prior_scene_list

    def _get_scenes_text_for_characters_prior_to_scene(self, character_list: list[str], cur_scene_index: int, **kwargs):
        """
        Get the previous narrative involving any character prior to the specific scene_index.
        """
        if self.scene_list is None or len(self.scene_list) == 0: # we don't have links filename
            return ""

        if cur_scene_index == 0: # no prior scenes
            return ""

        if character_list is None or len(character_list) == 0:
            raise ValueError("character_list must be populated")
        if cur_scene_index is None or cur_scene_index < 0:
            raise ValueError("cur_scene_index must be populated")
        if cur_scene_index >= len(self.scene_list):
            raise ValueError(f"cur_scene_index {cur_scene_index} is out of range for scene_list")

        # both limits are enforced, thus the limit resulting in the smallest string is used
        scene_count_limit = kwargs.get('lookback_scene_count_limit', 0)
        char_count_limit = kwargs.get('char_count_limit', 0)
        if scene_count_limit < 0:
            raise ValueError("lookback_scene_count_limit must be >= 0")
        if char_count_limit < 0:
            raise ValueError("char_count_limit must be >= 0")

        # enforce scene count limit
        prior_scene_list = self._get_scene_list_for_charactes_prior_to_scene(character_list, cur_scene_index)
        prior_scene_list.reverse() # reverse the list to get the most recent scenes first
        if len(prior_scene_list) == 0:
            return ""
        if len(prior_scene_list) > scene_count_limit and scene_count_limit > 0:
            prior_scene_list = prior_scene_list[:scene_count_limit]

        # build each paragraph list from the scene list
        # list of paragraphs will be reversed to get the most recent paragraphs first
        paragraph_list_reversed = []
        for scene_index in prior_scene_list:
            paragraph_list = self.scene_list[scene_index]['body'].split("\n")
            paragraph_list.reverse()
            paragraph_list_reversed.extend(paragraph_list)

        # now enforce the character count limit and build the text from the reversed paragraph list
        text = ""
        for paragraph in paragraph_list_reversed:
            new_text_length = len(text) + len(paragraph) + 1
            if new_text_length > char_count_limit and char_count_limit > 0:
                break
            text = paragraph + "\n" + text

        return text

    def _write_scene(self, scene: dict) -> None:
        """
        Generate the actual scene text using the LLM.

        Creates a prompt from the scene specification and contextual information,
        sends it to the LLM, and stores the response in the scene's 'body' field.

        Required Parameters:
            - scene (dict): Scene specification to generate content for

        Optional Parameters:
            - recent_narrative (str): Description of recent events in the narrative (can be None)

        Returns:
            None

        Raises:
            ValueError: If the API object is not initialized
        """

        ppr = self.preprocess_results
        scene_index = scene['chron_scene_index']
        existing_scene = ppr.scene_list[scene_index]
        introduction = self._get_scene_creation_introduction_prompt()
        first_paragraph = self._get_scene_creation_prompt_beginning_paragraph(existing_scene)
        context = self._get_scene_creation_context_prompt(scene['chron_scene_index'],
                                                          scene['tone'],
                                                          [c for c in scene['named_characters'].keys()],
                                                          [o for o in scene['objects'].keys()],
                                                          [s for s in scene['settings'].keys()],
                                                          scene['plot_summary'],
                                                          scene.get('focus', None)
                                                          )

        self.scene_budget = {'overall': self.max_input_tokens * 3,
                             'introduction': 0,
                             'recent_narrative': 0,
                             'context': 0,
                             'description': 0,
                             'scene_first_paragraph': 0}

        self.scene_budget['introduction'] = len(introduction)
        self.scene_budget['scene_first_paragraph'] = len(first_paragraph)
        self.scene_budget['context'] = len(context)

        remaining_len = (self.scene_budget['overall']
                         - self.scene_budget['introduction']
                         - self.scene_budget['scene_first_paragraph']
                         - self.scene_budget['context'])

        if remaining_len < 0:
            print(f"prompt is probably too big for number of input tokens: {self.max_input_tokens}...aborting")
            return

        self.scene_budget['recent_narrative'] = min(8000, math.floor(remaining_len * self.previous_narrative_fraction))
        self.scene_budget['description'] = min(4000, remaining_len - self.scene_budget['recent_narrative'])

        recent_narrative = self._get_scene_creation_prompt_previous_narrative(scene)
        descriptions = self._get_scene_creation_descriptions(scene)

        scene_creation_request = introduction + recent_narrative + context + descriptions + first_paragraph

        if self.verbose:
            print(f"Writing scene {scene['chron_scene_index']}")
        if self.api_obj is None:
            raise ValueError("API object is not initialized. Please provide a valid model name.")
        # scene_creation_request = self._format_scene_creation_prompt_preamble_with_rag(scene)
        # # print(f"scene_creation_request: {scene_creation_request}, type {type(scene_creation_request)}")
        # # max_output_tokens = min(self.max_output_tokens, 3000)
        # character_list = [c for c in scene['named_characters'].keys()]
        # cur_scene_index = scene['chron_scene_index']
        # intro = "\n\nHere is the most recent relevant narrative, and write the scene assuming this narrative came just before:\n\n"
        # if len(intro) > self.scene_budget['recent_narrative']:
        #     print(f"intro length for recent narrtive is too long: {len(intro)}")
        #     return
        # scene_remaining_len = self.scene_budget['recent_narrative'] - len(intro)
        # if self.verbose:
        #     print(f"max_input_tokens {self.max_input_tokens} scene_remaining_len {scene_remaining_len}, "
        #           + f"scene_creation_request len {len(scene_creation_request)}, intro len {len(intro)}")
        # if scene_remaining_len > 0:
        #     previous_narrative = self._get_scenes_text_for_characters_prior_to_scene(
        #         character_list, cur_scene_index, char_count_limit=scene_remaining_len
        #         )
        #     scene_creation_request = scene_creation_request + intro + previous_narrative
        # else:
        #     print(f"no room for recent narrative in scene request, budget calculations are off")

        log_filename = None
        if self.request_log_file_template is not None and len(self.request_log_file_template) > 0:
            log_filename = self.request_log_file_template.format(chron_scene_index=scene['chron_scene_index'])
        if self.verbose:
            print(f"Log filename for scene creation request {log_filename}")
        if log_filename is not None:
            if os.path.exists(log_filename):
                os.remove(log_filename)
            with open(log_filename, "w+") as f:
                if self.verbose:
                    print(f"Scene creation request for scene chron_scene_index {scene['chron_scene_index']} is "
                          f"{len(scene_creation_request)} characters long"
                          )
                f.write(f"Scene creation request for scene chron_scene_index {scene['chron_scene_index']}:\n\n")
                f.write(f"{scene_creation_request}\n\n")
                f.write("\n\n\n\n")
        if self.verbose:
            print(f"generate_prompt_only set to {self.generate_prompt_only}")
        if not self.generate_prompt_only:
            scene['body'] = self.api_obj.get_write_scene_prompt_response(
                scene_creation_request, max_tokens=self.max_output_tokens, temperature=0.0
                )
            print(scene['body'])

    def write_scenes(self) -> None:
        """
        Generate content for all scenes in the composition list.

        This method iterates through the scenes to be composed, collects relevant
        contextual information for each, and uses the LLM to generate narrative content.
        It respects the scene limit configuration and includes recent events as context.

        No Parameters

        Returns:
            None

        Raises:
            ValueError: If the API object is not initialized
        """
        if self.api_obj is None:
            raise ValueError("API object is not initialized. Please provide a valid model name.")

        tc = self.to_compose
        # ppr = self.preprocess_results
        # index_list = []

        for index, scene in enumerate(tc.scene_list):
            if index >= self.scene_limit:
                if self.verbose:
                    print(f"write_scenes: breaking out of scene lists, index {index} >= scene_limit {self.scene_limit}")
                break
            if len(scene['plot_summary']) == 0:
                if self.verbose:
                    print(f"scene {scene.chron_scene_index} has no plot summary")
                continue
            # recent_events = None
            # recent_events_list = []
            # if self.recent_event_count is not None and self.recent_event_count > 0:
            #     index_recents = scene['chron_scene_index'] - 1
            #     while index_recents >= 0 and len(recent_events_list) < self.recent_event_count:
            #         # print(f"scene {scene['chron_scene_index']} recent events index {i}, scene list len {len(tc.scene_list)}")
            #         plot_summary = ppr.scene_list[index_recents]['plot_summary']
            #         print(index_recents, plot_summary, len(recent_events_list))
            #         if plot_summary is not None and len(plot_summary) > 0:
            #             print("Adding to recent_events_list")
            #             recent_events_list.append(plot_summary)
            #         index_recents -= 1
            #         if self.verbose:
            #             print(f"index_recents {index_recents}")
            #     print(recent_events_list)
            # if len(recent_events_list) > 0:
            #     recent_events = '\n\n'.join(recent_events_list)
            # if self.verbose:
            #     print(f"recent_events {recent_events}")
            # index_list.append(scene['chron_scene_index'])
            self._write_scene(scene)
        tc.dump(self.output_compose_filename)
