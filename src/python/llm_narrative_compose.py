#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""
import copy
import os
import random
from typing import Optional

from llm_narrative_handler import LLMNarrativeScenesHandler
from narrative_preprocess import NarrativePreprocessResults
from vdb_milvus import VectorDBMilvus
from llm_openai_api_handler import LLMOpenAIAPIHandler

class LLMNarrativeScenesCompose(LLMNarrativeScenesHandler):
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
                - recent_event_count (int): Number of recent events to include (default: 0)
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

    def _guestimate_num_records(self, scene: dict, preamble: str) -> tuple:
        """
        Calculate the optimal number of context records to retrieve for each entity type.

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

        input_tokens_used = len(preamble) * 4
        input_tokens_remaining = self.max_input_tokens - input_tokens_used
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
        main_char_num_records = max(main_char_tokens * 4 // 50, 1)
        other_char_num_records = max(other_char_tokens * 4 // 50, 1)
        if len(object_list) > 0:
            first_object_num_records = max(first_object_tokens * 4 // 50, 1)
            other_object_num_records = max(other_object_tokens * 4 // 50, 1)
        else:
            first_object_num_records = 0
            other_object_num_records = 0
        first_setting_num_records = max(first_setting_tokens * 4 // 50, 1)
        other_setting_num_records = max(other_setting_tokens * 4 // 50, 1)

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
        main_char_records = [r['entity']['text'] for r in res[0]]
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
            char_dict[character_list[i]] = [r['entity']['text'] for r in res[0]]
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
        first_setting_records = [r['entity']['text'] for r in res[0]]
        setting_dict = {setting_list[0]: first_setting_records}
        for i in range(1, len(setting_list)):
            res = self.vector_db.search(self.narrative, prev_chron_scene_index, [f"{setting_list[i]}"], limit=other_setting_num_records)
            for r in res[0]:
                scene_index = r['id'] // 1000
                if scene_index > prev_chron_scene_index:
                    print(f"scene_index {scene_index} in _get_setting_records() "
                          + f"is greater than prev_chron_scene_index {prev_chron_scene_index}")
                    return {}
            setting_dict[setting_list[i]] = [r['entity']['text'] for r in res[0]]
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
        first_object_records = [r['entity']['text'] for r in res[0]]
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
            object_dict[object_list[i]] = [r['entity']['text'] for r in res[0]]
        return object_dict

    def _get_scene_creation_preamble_prompt(self, tone: str, character_list: list,
                                        object_list: list, setting_list: list, synopsis: str,
                                        beginning: str, focus: Optional[str] = None,
                                        recent_events: Optional[str] = None) -> str:
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
            - recent_events (str): Summary of recent events in the narrative
            - focus (str): The scene's primary focus or theme (can be None)

        Returns:
            str: Formatted prompt preamble
        """
        p = f"Using the following context, write a fiction scene in the style of '{self.author_name}':\n\n"
        p += "Context:\n"
        # p = "Compose a scene for a fiction narrative consistent with the following information:\n"
        # p += f"The author is {self.user}, and the scene should be composed in the author's style.\n"
        # p += f"The scene should not use previous content by the author, other than the content provided in the prompt.\n"
        p += f"The tone of the scene is: {tone}\n"
        p += f"The point of view character is: {character_list[0]}\n"
        p += f"The named characters are: {', '.join(character_list)}\n"
        if len(object_list) > 0:
            p += f"Objects the characters can interact with are: {', '.join(object_list)}\n"
        p += f"The settings in chronological order are: {', '.join(setting_list)}\n"
        p += f"The synopsis of the scene to be written is:\n{synopsis}\n"
        p += "Do not include any known characters or settings that are not mentioned above, though you may refer to them.\n"
        p += "The scene should be consistent with the tone, synopsis, and traits and backstory of characters and settings.\n"
        p += "Characters not explicitly named may be assigned a name, but "
        p += "only if the point of view character becomes aware of the name.\n"
        if focus is not None:
            p += f"The focus of the scene: {focus}\n"
        if beginning is not None and len(beginning) > 0:
            p += f"The beginning of the scene is:\n{beginning}\n\n"
        if recent_events is None:
            p += "There are no recent events that have occurred in the narrative. This is the beginning of the narrative.\n"
        else:
            p += f"Here are the most recent events that have occurred in the narrative:\n\n{recent_events}\n\n"
            p += "This ends the recent events.\n\n"
        p += "Here is some more information regarding the characters and settings:\n\n"

        return p

    def _format_scene_creation_prompt(self, scene: dict, recent_events: Optional[str] = None) -> str:
        """
        Build the complete scene generation prompt with context.

        This method combines the scene specification with relevant contextual information
        from the vector database. It prioritizes descriptions from the scene itself,
        then adds historical context from previous scenes, balancing between different
        entity types (characters, settings, objects) to create a rich but token-efficient prompt.

        Required Parameters:
            - scene (dict): Complete scene specification

        Optional Parameters:
            - recent_events (str): Description of recent events in the narrative (can be None)

        Returns:
            str: Complete formatted prompt for the LLM
        """
        # assume max scene length of 3000 tokens
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

        beginning = scene['body'].split("\n")[0]

        p = self._get_scene_creation_preamble_prompt(tone, character_list, object_list, setting_list, synopsis,
                                                     beginning, focus, recent_events)

        if len(p) > (self.max_input_tokens * 4):
            print(f"prompt too long: {len(p)}")
            return p[:self.max_input_tokens * 4]

        # first we include all of the descriptions and plot summaries embedded in the scene request
        fixed_records = self._retrieve_scene_context_records(scene)
        print(fixed_records, len(fixed_records))
        print(character_list, object_list, setting_list)

        # guestimate how many records to retrieve from the RAG vector db for each type
        # assume 50 characters for each doc in the vector db, will of course be larger than that, but that's the upper bound
        # this ensures we will retrieve more than we can fit (or all of them), which is fine, we simply cut if off later
        main_char_num_records, other_char_num_records, first_object_num_records, other_object_num_records,\
            first_setting_num_records, other_setting_num_records =\
            self._guestimate_num_records(scene, p)

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

        print(char_dict, len(char_dict))

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
        print(records)

        s_new = p
        s_old = s_new
        for record in records:
            if len(s_new) > (self.max_input_tokens * 4):
               return s_old
            s_old = s_new
            q = f"{record}\n"
            s_new = p + q

        # This section for debugging only, uncomment it if you want to debug both the number of records and the prompt
        print(f"Added {len(records)} records to the prompt, but we still had room for more.")
        with open("format_scene_request.txt", "a") as f:
            f.write("\n\n")
            f.write(f"{s_new}\n")
            f.write("\n\n")

        # this is the final prompt, which is the original prompt plus all of the records
        return s_new

    def _write_scene(self, scene: dict, recent_events: Optional[str] = None) -> None:
        """
        Generate the actual scene text using the LLM.

        Creates a prompt from the scene specification and contextual information,
        sends it to the LLM, and stores the response in the scene's 'body' field.

        Required Parameters:
            - scene (dict): Scene specification to generate content for

        Optional Parameters:
            - recent_events (str): Description of recent events in the narrative (can be None)

        Returns:
            None

        Raises:
            ValueError: If the API object is not initialized
        """
        if self.api_obj is None:
            raise ValueError("API object is not initialized. Please provide a valid model name.")
        scene_creation_request = self._format_scene_creation_prompt(scene, recent_events)
        # max_output_tokens = min(self.max_output_tokens, 3000)
        scene['body'] = self.api_obj.get_write_scene_prompt_response(scene_creation_request,
                                                                     max_tokens=self.max_output_tokens,
                                                                     temperature=0.7)
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
        ppr = self.preprocess_results
        for index, scene in enumerate(tc.scene_list):
            if index >= self.scene_limit:
                if self.verbose:
                    print(f"write_scenes: breaking out of scene lists, index {index} >= scene_limit {self.scene_limit}")
                break
            if len(scene['plot_summary']) == 0:
                if self.verbose:
                    print(f"scene {scene.chron_scene_index} has no plot summary")
                continue
            recent_events = None
            recent_events_list = []
            if self.recent_event_count is not None and self.recent_event_count > 0:
                index_recents = scene['chron_scene_index'] - 1
                while index_recents >= 0 and len(recent_events_list) < self.recent_event_count:
                    # print(f"scene {scene['chron_scene_index']} recent events index {i}, scene list len {len(tc.scene_list)}")
                    plot_summary = ppr.scene_list[index_recents]['plot_summary']
                    print(index_recents, plot_summary, len(recent_events_list))
                    if plot_summary is not None and len(plot_summary) > 0:
                        print("Adding to recent_events_list")
                        recent_events_list.append(plot_summary)
                    index_recents -= 1
                    if self.verbose:
                        print(f"index_recents {index_recents}")
                print(recent_events_list)
            if len(recent_events_list) > 0:
                recent_events = '\n\n'.join(recent_events_list)
            if self.verbose:
                print(f"recent_events {recent_events}")
            self._write_scene(scene, recent_events)
        tc.dump_eval(self.output_compose_filename)
