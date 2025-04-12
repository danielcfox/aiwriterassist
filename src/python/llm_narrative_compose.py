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

class LLMNarrativeScenesCompose(LLMNarrativeScenesHandler):
    def __init__(self, user, narrative, author_name, api_obj, 
                 input_train_filename, input_eval_filename, input_filename, output_filename, 
                 max_input_tokens, max_output_tokens, vector_db):
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
        super().__init__(user, narrative, author_name, api_obj, input_train_filename, input_eval_filename, 
                         max_input_tokens, max_output_tokens)

        if max_input_tokens is None or max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than 0")
        if max_output_tokens is None or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0")
        if vector_db is None:
            raise ValueError("VectorDBMilvus is not initialized. Please provide a valid URI in the configuration YAML file.")

        self.user = user
        self.narrative = narrative
        self.output_filename = output_filename
        self.vector_db = vector_db

        self.to_compose = NarrativePreprocessResults()
        if input_filename is not None and os.path.exists(input_filename):
            self.to_compose.load(input_filename)
        else:
            raise ValueError(f"Input file {input_filename} does not exist.")
        
        if output_filename is None:
            raise ValueError("Output filename cannot be None")
        if len(output_filename) == 0:
            raise ValueError("Output filename cannot be empty")
        
        if input_train_filename is None or not os.path.exists(input_train_filename):
            if input_eval_filename is None or not os.path.exists(input_eval_filename):
                raise ValueError(f"Neither Input train file {input_train_filename} "
                                 + f"or Input eval file {input_eval_filename} exists.")

        print("Size of scene list is", len(self.preprocess_results.scene_list))
    
    def _retrieve_scene_context_records(self, scene):
        """Retrieve context records for a scene.
        :param type dict, scene: The scene to retrieve context records for.
        :return: A list of context records from the descriptions in the scene.
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

    def _guestimate_num_records(self, scene, p):
        """Estimate the number of records to retrieve for a scene.
        :param type dict, scene: The scene to estimate the number of records for.
        :param type str, p: The prompt preamble.
        :return: A tuple of the number of records to retrieve for each entity type.
        """
        character_list = [c for c in scene['named_characters'].keys()]
        setting_list = [s for s in scene['settings'].keys()]
        object_list = [o for o in scene['objects'].keys()]

        input_tokens_used = len(p) * 4
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
    
    def _get_character_records(self, prev_chron_scene_index, character_list, main_char_num_records, other_char_num_records):
        """Get character records for a scene.
        :param type int, prev_chron_scene_index: The chronological scene index of the previous scene.
        :param type list, character_list: The list of characters to retrieve records for.
        :param type int, main_char_num_records: The number of records to retrieve for the main character.
        :param type int, other_char_num_records: The number of records to retrieve for other characters.
        :return: A dictionary of character records.
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
    
    def _get_setting_records(self, prev_chron_scene_index, setting_list, first_setting_num_records, other_setting_num_records):
        """Get setting records for a scene.
        :param type int, prev_chron_scene_index: The chronological scene index of the previous scene.
        :param type list, setting_list: The list of settings to retrieve records for.
        :param type int, first_setting_num_records: The number of records to retrieve for the first setting.
        :param type int, other_setting_num_records: The number of records to retrieve for other settings.
        :return: A dictionary of setting records.
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

    def _get_object_records(self, prev_chron_scene_index, object_list, first_object_num_records, other_object_num_records):
        """Get object records for a scene.
        :param type int, prev_chron_scene_index: The chronological scene index of the previous scene.
        :param type list, object_list: The list of objects to retrieve records for.
        :param type int, first_object_num_records: The number of records to retrieve for the first object.
        :param type int, other_object_num_records: The number of records to retrieve for other objects.
        :return: A dictionary of object records.
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
     
    def _get_scene_creation_preamble_prompt(self, focus, tone, character_list, object_list, setting_list, synopsis,
                                            beginning, recent_events):
        """Get the scene creation preamble prompt.
        :param type str, focus: The focus of the scene.
        :param type str, tone: The tone of the scene.
        :param type list, character_list: The list of characters in the scene.
        :param type list, object_list: The list of objects in the scene.
        :param type list, setting_list: The list of settings in the scene.
        :param type str, synopsis: The synopsis of the scene.
        :param type str, beginning: The beginning of the scene.
        :param type str, recent_events: The recent events in the narrative.
        :return: The scene creation preamble prompt.
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

    def _format_scene_creation_prompt(self, scene, recent_events):
        """Format the scene creation prompt.
        :param type dict, scene: The scene to format the prompt for.
        :param type str, recent_events: The recent events in the narrative.
        :return: The formatted scene creation prompt.
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

        p = self._get_scene_creation_preamble_prompt(tone, focus, character_list, object_list, setting_list, synopsis, 
                                                     beginning, recent_events)

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
    
    def _write_scene(self, scene, recent_events):
        """Write the scene to the output file.
        :param type dict, scene: The scene to write.
        :param type str, recent_events: The recent events in the narrative.
        :return: None
        """
        if self.api_obj is None:
            raise ValueError("API object is not initialized. Please provide a valid model name.")
        scene_creation_request = self._format_scene_creation_prompt(scene, recent_events)
        # max_output_tokens = min(self.max_output_tokens, 3000)
        scene['body'] = self.api_obj.get_write_scene_prompt_response(scene_creation_request, self.author_name,
                                                                     max_tokens=self.max_output_tokens,
                                                                     temperature=0.7) 
    def write_scenes(self, scene_limit, recent_event_count):
        """Write the scenes to the output file.
        :param type int, scene_limit: The maximum number of scenes to process.
        :param type int, recent_event_count: The number of recent events to include in the prompt.
        :return: None
        """
        if self.api_obj is None:
            raise ValueError("API object is not initialized. Please provide a valid model name.")
        
        tc = self.to_compose
        ppr = self.preprocess_results
        for index, scene in enumerate(tc.scene_list):
            if index >= scene_limit:
                print(f"write_scenes: breaking out of scene lists, index {index} >= scene_limit {scene_limit}")
                break
            if len(scene['plot_summary']) == 0:
                print(f"scene {scene.chron_scene_index} has no plot summary")
                continue
            recent_events = None
            recent_events_list = []
            if recent_event_count is not None and recent_event_count > 0:
                index_recents = scene['chron_scene_index'] - 1
                while index_recents >= 0 and len(recent_events_list) < recent_event_count:
                    # print(f"scene {scene['chron_scene_index']} recent events index {i}, scene list len {len(tc.scene_list)}")
                    plot_summary = ppr.scene_list[index_recents]['plot_summary']
                    print(index_recents, plot_summary, len(recent_events_list))
                    if plot_summary is not None and len(plot_summary) > 0:
                        print("Adding to recent_events_list")
                        recent_events_list.append(plot_summary)
                    index_recents -= 1
                    print(f"index_recents {index_recents}")
                print(recent_events_list)
            if len(recent_events_list) > 0:
                recent_events = '\n\n'.join(recent_events_list)
            print(f"recent_events {recent_events}")
            self._write_scene(scene, recent_events)
        tc.dump_eval(self.output_filename)                  
