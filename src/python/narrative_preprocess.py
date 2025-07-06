#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:16:54 2025

Copyright 2025 Daniel C. Fox

@author: dfox
"""
import copy
import json
# import openai
import os
# import time
from word2number import w2n
import yaml

def represent_list(dumper, data):
    """ formatting for yaml """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

class NarrativeScene:
    """
    Represents a single scene from a narrative text.

    This class stores the content and metadata for a narrative scene, including
    its structural position (book and chapter numbers), content body, and extracted
    narrative elements like characters, settings, and objects. It maintains continuity
    with previous scenes through sequential indexing.
    """

    def __init__(self, prev_scene: 'NarrativeScene') -> None:
        """
        Initialize a narrative scene, optionally inheriting properties from a previous scene.

        Creates a new scene object that maintains continuity with previous scenes by
        incrementing indices and inheriting structural information when a previous
        scene is provided.

        Parameters:
            prev_scene (NarrativeScene): Previous scene to inherit properties from.
                                         If None, initializes with default values.
        """
        self.datamode = None
        if prev_scene is not None:
            self.booknum = prev_scene.booknum
            self.chapternum = prev_scene.chapternum
            self.narr_scene_index = prev_scene.narr_scene_index + 1
            self.chron_scene_index = self.narr_scene_index
        else:
            self.booknum = 0
            self.chapternum = None
            self.narr_scene_index = 0
            self.chron_scene_index = 0
        self.body = ''
        self.plot_summary = ''
        self.tone = ''
        self.point_of_view_character = ''
        self.named_characters = {}
        self.objects = {}
        self.settings = {}

    def set_chapter(self, line: str) -> None:
        """
        Extract and set the chapter number from a chapter heading line.

        Parses a chapter heading to identify the chapter number, converting
        from either numeric or word form (e.g., "Chapter 1" or "Chapter One").

        Parameters:
            line (str): Text line containing the chapter heading.
        """
        chapter_words = line.split(' ')
        if chapter_words[1].isnumeric():
            self.chapternum = int(chapter_words[1])
        else:
            self.chapternum = w2n.word_to_num(chapter_words[1])

    def set_booknum(self, booknum: int) -> None:
        """
        Set the book number for the current scene.

        Assigns the scene to a specific book within a multi-book narrative.

        Parameters:
            booknum (int): Book identifier number.
        """
        self.booknum = booknum

    def add_to_narrative(self, text: str) -> None:
        """
        Append text content to the scene's narrative body.

        Adds the provided text to the scene's main content, building up
        the full narrative of the scene incrementally.

        Parameters:
            text (str): Text content to append to the scene body.
        """
        self.body += text

class NarrativePreprocessResults:
    """
    Container for processed narrative data and scene collections.

    This class stores and manages the results of narrative preprocessing,
    including collections of scenes, characters, settings, and metadata.
    It provides methods to filter scenes by data mode (training or evaluation),
    save results to files, and merge data from multiple sources.
    """

    def __init__(self) -> None:
        """
        Initialize an empty narrative preprocessing results container.

        Creates a new container with tracking variables and empty collections
        for scenes and narrative elements.
        """
        self.at_linenum = 0
        self.at_scenenum = 0
        # self.protagonist = ''
        self.named_characters = {}
        self.objects = {}
        self.settings = {}
        self.tones = {}
        self.plot_summary = ''
        self.scene_list = []

    def get_current_scene(self) -> dict:
        """
        Retrieve the most recently added scene.

        Returns the last scene in the scene list, which represents the
        current scene being processed.

        Returns:
            dict: The most recent scene data, or None if no scenes exist.
        """
        if len(self.scene_list) > 0:
            return self.scene_list[-1]
        return None

    def get_train_scene_list(self) -> list:
        """
        Filter scenes marked for training purposes.

        Returns a list containing only scenes that have been designated
        as training data (datamode = 'train').

        Returns:
            list: List of scene dictionaries for training.
        """
        train_scene_list = [scene for scene in self.scene_list if scene['datamode'] == 'train']
        return train_scene_list
    def get_eval_scene_list(self) -> list:
        """
        Filter scenes marked for evaluation purposes.

        Returns a list containing only scenes that have been designated
        as evaluation data (datamode = 'eval').

        Returns:
            list: List of scene dictionaries for evaluation.
        """
        eval_scene_list = [scene for scene in self.scene_list if scene['datamode'] == 'eval']
        return eval_scene_list

    def dump(self, filename: str) -> None:
        """
        Save the complete results to a JSON file.

        Serializes the entire object's data, including all scenes and
        metadata, to the specified file.

        Parameters:
            filename (str): Path to the output JSON file.
        """
        with open(filename, 'w') as fp:
            json.dump(self.__dict__, fp, indent = 4)

    def dump_train(self, filename: str) -> None:
        """
        Save only training scenes to a JSON file.

        Temporarily filters the scene list to include only training scenes,
        saves this filtered data to the specified file, then restores the
        original scene list.

        Parameters:
            filename (str): Path to the output JSON file for training data.
        """
        cur_scene_list = self.scene_list
        train_scene_list = self.get_train_scene_list()
        if len(train_scene_list) == 0:
            return
        self.scene_list = train_scene_list
        with open(filename, 'w') as fp:
            # print("dumping train...")
            json.dump(self.__dict__, fp, indent = 4)
        self.scene_list = cur_scene_list

    def dump_eval(self, filename: str) -> None:
        """
        Save only evaluation scenes to a JSON file.

        Temporarily filters the scene list to include only evaluation scenes,
        saves this filtered data to the specified file, then restores the
        original scene list.

        Parameters:
            filename (str): Path to the output JSON file for evaluation data.
        """
        cur_scene_list = copy.deepcopy(self.scene_list)
        eval_scene_list = self.get_eval_scene_list()
        if len(eval_scene_list) == 0:
            return
        self.scene_list = eval_scene_list
        with open(filename, 'w') as fp:
            json.dump(self.__dict__, fp, indent = 4)
        self.scene_list = copy.deepcopy(cur_scene_list)

    # def dump_idx(self, filename: str, idx_list: list[int]) -> None:
    #     """
    #     Save only evaluation scenes to a JSON file.

    #     Temporarily filters the scene list to include only evaluation scenes,
    #     saves this filtered data to the specified file, then restores the
    #     original scene list.

    #     Parameters:
    #         filename (str): Path to the output JSON file for evaluation data.
    #     """
    #     cur_scene_list = copy.deepcopy(self.scene_list)
    #     for idx in idx_list:
    #         if idx >= len(self.scene_list):
    #             raise ValueError(f"Index {idx} out of range for scene list.")
    #     self.scene_list = [self.scene_list[idx] for idx in idx_list]
    #     with open(filename, 'w') as fp:
    #         json.dump(self.__dict__, fp, indent = 4)
    #     self.scene_list = copy.deepcopy(cur_scene_list)

    def dump_metadata(self, filename: str) -> None:
        """
        Save metadata information to a JSON file.

        Serializes the metadata portion of the object (named characters,
        settings, objects, and tones) to the specified file.

        Parameters:
            filename (str): Path to the output JSON file for metadata.
        """
        metadata = {}
        if len(self.named_characters) == 0:
            self.build_metadata_scene_list()
            # self.build_metadata_no_scene_list()

        metadata['named_characters'] = self.named_characters
        metadata['settings'] = self.settings
        metadata['objects'] = self.objects
        metadata['tones'] = self.tones

        yaml.add_representer(list, represent_list)

        with open(filename, 'w') as fp:
            yaml.dump(metadata, fp, default_flow_style=False, default_style=False, sort_keys=False)
        # with open(filename, 'w') as fp:
        #         yaml.dump(metadata, fp, default_flow_style=False, default_style=False, flow_style={tuple, list}, sort_keys=False)

    def load(self, filename: str) -> None:
        """
        Load narrative data from a JSON file and merge with existing data.

        Reads scene and metadata from the specified file and intelligently
        merges it with existing data, updating existing scenes by chronological
        index and appending new ones.

        Parameters:
            filename (str): Path to the input JSON file.
        """
        print(f"loading {filename}")
        if not os.path.exists(filename):
            print(f"file {filename} does not exist")
            return
        with open(filename, 'r') as fp:
            new_struct = json.load(fp)
        if new_struct is None:
            return
        new_scene_list = copy.deepcopy(new_struct['scene_list'])
        new_len = len(new_scene_list)
        if len(self.scene_list) == 0:
            self.update(new_struct)
            return
        self.update(new_struct)
        # we assume that chron_scene_index is in order in both files
        cur_scene_index = 0
        for index, scene in enumerate(new_scene_list):
            while (cur_scene_index < len(self.scene_list)
                   and scene['chron_scene_index'] != self.scene_list[cur_scene_index]['chron_scene_index']):
                cur_scene_index += 1
            if cur_scene_index >= len(self.scene_list):
                break
            self.scene_list[cur_scene_index].update(scene)
            cur_scene_index += 1
        if index < (len(new_scene_list)-1):
            self.scene_list.extend(new_scene_list[index:])
        self.build_metadata_scene_list()

    def update(self, new_dict: dict) -> None:
        """
        Update the current results with data from another dictionary.

        Merges new narrative data into the current object, handling
        specialized merging for different data categories.

        Parameters:
            new_dict (dict): Dictionary containing new narrative data.
        """
        if 'protagonist' in new_dict:
            self.protagonist = new_dict['protagonist']
        if 'named_characters' in new_dict:
            self.named_characters.update(new_dict['named_characters'])
        if 'settings' in new_dict:
            self.settings.update(new_dict['settings'])
        if 'objects' in new_dict:
            self.objects.update(new_dict['objects'])
        if 'tones' in new_dict:
            self.tones.update(new_dict['tones'])
        if 'plot_summary' in new_dict:
            self.plot_summary = new_dict['plot_summary']
        if 'scene_list' in new_dict:
            self.scene_list.extend(new_dict['scene_list'])

    def build_metadata_scene_list(self) -> None:
        """
        Build a list of metadata from the scene list.

        Extracts named characters, settings, and objects from the scene list
        and organizes them into a structured format for further processing.
        It extracts all characters and places them in self.named_characters.
        Each character is a key in the dictionay, and the value is the list
        of scenes (chron_scene_index) in which the character appears.
        The same is done for settings and objects.

        Also extracts the tones from the scene list and places them in
        self.tones. Each tone is a key in the dictionay, and the value is
        the list of scenes (chron_scene_index) in which the tone appears.
        This method is called after the scene list has been fully populated.
        It is used to build a metadata scene list that can be used for
        training and evaluation purposes.

        Returns:
            None
        """
        self.named_characters = {}
        self.settings = {}
        self.objects = {}
        self.tones = {}
        for scene in self.scene_list:
            for character in scene['named_characters']:
                if character not in self.named_characters:
                    self.named_characters[character] = []
                self.named_characters[character].append(scene['chron_scene_index'])
            for setting in scene['settings']:
                if setting not in self.settings:
                    self.settings[setting] = []
                self.settings[setting].append(scene['chron_scene_index'])
            for obj in scene['objects']:
                if obj not in self.objects:
                    self.objects[obj] = []
                self.objects[obj].append(scene['chron_scene_index'])
            tone = scene['tone']
            if scene['tone'] not in self.tones:
                self.tones[tone] = []
            self.tones[tone].append(scene['chron_scene_index'])

        self.named_characters = {k: v for k, v in sorted(self.named_characters.items(), key=lambda item: item[0])}
        self.settings = {k: v for k, v in sorted(self.settings.items(), key=lambda item: item[0])}
        self.objects = {k: v for k, v in sorted(self.objects.items(), key=lambda item: item[0])}
        self.tones = {k: v for k, v in sorted(self.tones.items(), key=lambda item: item[0])}



    # def build_metadata_no_scene_list(self) -> None:
    #     """
    #     Build a metadata list without using the scene list.

    #     This method is used to build a metadata list that can be used for
    #     training and evaluation purposes. It extracts all characters and
    #     places them in self.named_characters. Each character is a key in
    #     the dictionay, and the value is the list of scenes (chron_scene_index)
    #     in which the character appears. The same is done for settings and
    #     objects.

    #     Returns:
    #         None
    #     """
    #     self.named_characters = {}
    #     self.settings = {}
    #     self.objects = {}
    #     self.tones = {}
    #     for scene in self.scene_list:
    #         for character in scene['named_characters']:
    #             if character not in self.named_characters:
    #                 self.named_characters[character] = []
    #         for setting in scene['settings']:
    #             if setting not in self.settings:
    #                 self.settings[setting] = []
    #             self.settings[setting].append(scene['chron_scene_index'])
    #         for obj in scene['objects']:
    #             if obj not in self.objects:
    #                 self.objects[obj] = []
    #         tone = scene['tone']
    #         if scene['tone'] not in self.tones:
    #             self.tones[tone] = []
    #     self.named_characters = {k: v for k, v in sorted(self.named_characters.items(), key=lambda item: item[0])}
    #     self.settings = {k: v for k, v in sorted(self.settings.items(), key=lambda item: item[0])}
    #     self.objects = {k: v for k, v in sorted(self.objects.items(), key=lambda item: item[0])}
    #     self.tones = {k: v for k, v in sorted(self.tones.items(), key=lambda item: item[0])}

class NarrativePreprocess:
    """
    Base class for narrative text preprocessing.

    This class provides the foundation for converting unstructured narrative text
    into structured scene objects. It implements a state machine for processing
    narrative files line by line, detecting scene boundaries and chapter headings,
    and organizing content into a collection of scene objects.

    The class supports splitting processed scenes into training and evaluation sets,
    and saving the resulting data to JSON files for further processing.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the narrative preprocessor with configuration options.

        Sets up the preprocessor with file paths, processing options, and initial state.

        Parameters:
            **kwargs: Configuration parameters including:
                - narrative_filename_list (list): List of input file paths.
                - narrative_preprocessed_train_filename (str): Output path for training data.
                - narrative_preprocessed_eval_filename (str): Output path for evaluation data.
                - train_split (float): Proportion of scenes to use for training (0.0-1.0).
                - scene_limit (int, optional): Maximum number of scenes to process.
        """
        self.narrative_filename_list = None
        self.narrative_preprocessed_train_filename = None
        self.narrative_preprocessed_eval_filename = None
        self.train_split = None
        self.scene_limit = None
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.chapter_start_str = 'Chapter'
        if os.path.exists(self.narrative_preprocessed_train_filename):
            os.remove(self.narrative_preprocessed_train_filename)
        if os.path.exists(self.narrative_preprocessed_eval_filename):
            os.remove(self.narrative_preprocessed_eval_filename)
        self.scene_delimiter = ''
        self.preprocess_state = 'start'
        self.preprocess_results = NarrativePreprocessResults()
        self.preprocess_train = None
        self.preprocess_eval = None

    def train_eval_split(self) -> None:
        """
        Split processed scenes into training and evaluation sets.

        Divides the complete set of scenes based on the train_split ratio,
        assigning datamode flags to each scene for identification.
        """
        if self.train_split is not None and self.train_split < 1.0:
            num_scenes = len(self.preprocess_results.scene_list)
            num_train_scenes = int(num_scenes * self.train_split)
            # num_eval_scenes = num_scenes - num_train_scenes
            self.preprocess_train = copy.deepcopy(self.preprocess_results)
            self.preprocess_train.scene_list = self.preprocess_train.scene_list[:num_train_scenes]
        for scene in self.preprocess_train.scene_list:
            scene['datamode'] = 'train'
        if self.train_split is not None and self.train_split < 1.0:
            self.preprocess_eval = copy.deepcopy(self.preprocess_results)
            self.preprocess_eval.scene_list = self.preprocess_eval.scene_list[num_train_scenes:]
            for scene in self.preprocess_eval.scene_list:
                scene['datamode'] = 'eval'


    def process(self) -> None:
        """
        Execute the complete preprocessing pipeline.

        Runs the full workflow of preprocessing narrative files, splitting into
        training/evaluation sets, and saving results to output files.
        """
        self.preprocess()
        self.train_eval_split()
        self.dump()

    def preprocess(self) -> None:
        """
        Parse narrative files and extract structured scenes.

        Processes each input file line by line, identifying scene boundaries
        and building a collection of structured scene objects.
        """
        # print(self.narrative_filename_list)
        linenum = 0
        for booknum, narrative_filename in enumerate(self.narrative_filename_list):
            with open(narrative_filename, "r") as fp:
                for line in fp:
                    if self.preprocess_results.at_linenum <= linenum:
                        if self._preprocess_line(booknum + 1, line.strip()) == False:
                            continue
                        self.preprocess_results.at_linenum += 1
                        # print(self.preprocess_results.__dict__)
                    linenum += 1

    def dump(self) -> None:
        """
        Save preprocessing results to output files.

        Writes the training and evaluation data to their respective output files
        in JSON format.
        """
        self.preprocess_train.dump(self.narrative_preprocessed_train_filename)
        self.preprocess_eval.dump(self.narrative_preprocessed_eval_filename)

    def get_current_filename(self) -> str:
        """
        Get the name of the file currently being processed.

        Returns:
            str: Current file path, or None if no file is being processed.
        """
        return self.preprocess_results.get_current_filename()

    def get_next_filename(self) -> str:
        """
        Advance to the next file and return its name.

        Returns:
            str: Next file path, or None if no more files are available.
        """
        return self.preprocess_results.advance_filename()

    def get_current_scene(self) -> NarrativeScene:
        """
        Get the scene currently being processed.

        Retrieves the most recently added scene from the results, converting
        it from a dictionary back to a NarrativeScene object.

        Returns:
            NarrativeScene: Current scene object, or None if no scenes exist.
        """
        pr = self.preprocess_results
        if len(pr.scene_list) == 0:
            return None
        scene_obj = NarrativeScene(None)
        scene_obj.__dict__.update(pr.scene_list[-1])
        return scene_obj

    def update_current_scene(self, scene: NarrativeScene) -> None:
        """
        Update the current scene in the results with new data.

        Applies changes from the provided scene object to the corresponding
        dictionary in the results.

        Parameters:
            scene (NarrativeScene): Updated scene object.
        """
        pr = self.preprocess_results
        if len(pr.scene_list) > 0:
            scene_dict = pr.scene_list[-1]
            scene_dict.update(scene.__dict__)

    def add_scene(self, scene: NarrativeScene) -> NarrativeScene:
        """
        Add a new scene to the results.

        Converts the scene object to a dictionary and appends it to the
        scene list, respecting the optional scene limit.

        Parameters:
            scene (NarrativeScene): Scene object to add.

        Returns:
            NarrativeScene: Added scene as an object, or None if scene limit was reached.
        """
        ppr = self.preprocess_results
        if self.scene_limit is not None and len(ppr.scene_list) >= self.scene_limit:
            return None
        self.preprocess_results.scene_list.append(scene.__dict__)
        return self.get_current_scene()

    def _preprocess_line(self, booknum: int, line: str) -> bool:
        """
        Process a single line of narrative text.

        Implements a state machine that transitions between different processing
        states ('start', 'chapter', 'narrative') based on line content and
        current state.

        Parameters:
            booknum (int): Number of the book containing this line.
            line (str): Text line to process.

        Returns:
            bool: True if processing should continue, False to skip.
        """
        # by default, scenes are simply separated by a line, with chapters
        match self.preprocess_state:
            case 'start':
                if len(line) == 0:
                    return
                if self._preprocess_new_scene(booknum) == False:
                    return False
                if line.startswith(self.chapter_start_str):
                    self._preprocess_chapter(booknum, line)
                    self.preprocess_state = 'chapter'
                elif len(line) > 0:
                    self._preprocess_narrative(booknum, line)
                    self.preprocess_state = 'narrative'

            case 'chapter':
                if len(line) > 0:
                    self._preprocess_narrative(booknum, line)
                    self.preprocess_state = 'narrative'

            case 'narrative':
                if line.strip() == self.scene_delimiter:
                    self.preprocess_state = 'start'
                elif line.startswith(self.chapter_start_str):
                    self._preprocess_chapter(booknum, line)
                    self.preprocess_state = 'chapter'
                elif len(line) > 0:
                    self._preprocess_narrative(booknum, line)

    def _preprocess_chapter(self, booknum: int, line: str) -> None:
        """
        Process a chapter heading.

        Updates the current scene with chapter information extracted from
        the heading line.

        Parameters:
            booknum (int): Number of the book containing this chapter.
            line (str): Chapter heading text.
        """
        scene = self.get_current_scene()
        scene.set_chapter(line)
        scene.set_booknum(booknum)
        self.update_current_scene(scene)

    def _preprocess_new_scene(self, booknum: int) -> bool:
        """
        Create and initialize a new scene.

        Creates a new scene object connected to the previous scene and
        adds it to the results.

        Parameters:
            booknum (int): Number of the book containing this scene.

        Returns:
            bool: True if scene was created, False if scene limit was reached.
        """
        scene = self.get_current_scene()
        new_scene = self.add_scene(NarrativeScene(scene))
        if new_scene is None:
            return False
        new_scene.set_booknum(booknum)
        self.update_current_scene(new_scene)
        return True

    def _preprocess_narrative(self, booknum: int, line: str) -> None:
        """
        Process narrative content.

        Adds a line of text to the current scene's narrative body.

        Parameters:
            booknum (int): Number of the book containing this content.
            line (str): Narrative text to add.
        """
        scene = self.get_current_scene()
        text = line + '\n'
        scene.add_to_narrative(text)
        scene.set_booknum(booknum)
        self.update_current_scene(scene)
