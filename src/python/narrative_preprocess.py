#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:16:54 2025

@author: dfox
"""
import copy
import json
# import openai
import os
# import time
from word2number import w2n

class NarrativeScene:
    """
    Class to represent a scene in a narrative.
    This class contains methods to set the chapter, book number, and add narrative text.
    """
    def __init__(self, prev_scene):
        """
        Initialize the scene based on the previous scene.
        :param type NarrativeScene, prev_scene: The previous scene object.
        """
        # self.llm_process = llm.LLMProcessing()
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

    def set_chapter(self, line):
        """
        Set the chapter number for the scene.
        :param type str, line: The line containing the chapter information.
        """
        chapter_words = line.split(' ')
        if chapter_words[1].isnumeric():
            self.chapternum = int(chapter_words[1])
        else:
            self.chapternum = w2n.word_to_num(chapter_words[1])

    def set_booknum(self, booknum):
        """
        Set the book number for the scene.
        :param type int, booknum: The book number.
        """
        self.booknum = booknum

    def add_to_narrative(self, text):
        """
        Add text to the narrative body of the scene.
        :param type str, text: The text to add to the narrative.
        """
        self.body += text

class NarrativePreprocessResults:
    """
    Class to store the results of the narrative preprocessing.
    This class contains methods to load, dump, and update the narrative data.
    """
    def __init__(self):
        """
        Initialize the narrative preprocess results.
        """
        self.at_linenum = 0
        self.at_scenenum = 0
        # self.protagonist = ''
        self.named_characters = {}
        self.settings = {}
        self.tones = {}
        self.plot_summary = ''
        self.scene_list = []

    def get_current_scene(self):
        """
        Get the current scene from the scene list.
        :return: The current scene object.
        """
        if len(self.scene_list) > 0:
            return self.scene_list[-1]
        return None
    
    def get_train_scene_list(self):
        """
        Get the list of training scenes from the scene list.
        :return: The list of training scenes.
        """
        train_scene_list = [scene for scene in self.scene_list if scene['datamode'] == 'train']
        return train_scene_list
    def get_eval_scene_list(self):
        """
        Get the list of evaluation scenes from the scene list.
        :return: The list of evaluation scenes.
        """
        eval_scene_list = [scene for scene in self.scene_list if scene['datamode'] == 'eval']
        return eval_scene_list

    def dump(self, filename):
        """
        Dump the narrative preprocess results to a JSON file.
        :param type str, filename: The filename to save the results to.
        """
        with open(filename, 'w') as fp:
            json.dump(self.__dict__, fp, indent = 4)

    def dump_train(self, filename):
        """
        Dump the training scenes to a JSON file.
        :param type str, filename: The filename to save the training scenes to.
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

    def dump_eval(self, filename):
        """
        Dump the evaluation scenes to a JSON file.
        :param type str, filename: The filename to save the evaluation scenes to.
        """
        cur_scene_list = self.scene_list
        eval_scene_list = self.get_eval_scene_list()
        if len(eval_scene_list) == 0:
            return
        self.scene_list = eval_scene_list
        with open(filename, 'w') as fp:
            json.dump(self.__dict__, fp, indent = 4)
        self.scene_list = cur_scene_list


    def load(self, filename):
        """
        Load the narrative preprocess results from a JSON file.
        :param type str, filename: The filename to load the results from.
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

    def update(self, new_dict):
        """
        Update the narrative preprocess results with new data.
        :param type dict, new_dict: The new data to update the results with.
        """
        if 'protagonist' in new_dict:
            self.protagonist = new_dict['protagonist']
        if 'named_characters' in new_dict:
            self.named_characters.update(new_dict['named_characters'])
        if 'settings' in new_dict:
            self.settings.update(new_dict['settings'])
        if 'tones' in new_dict:
            self.tones.update(new_dict['tones'])
        if 'plot_summary' in new_dict:
            self.plot_summary = new_dict['plot_summary']
        if 'scene_list' in new_dict:
            self.scene_list.extend(new_dict['scene_list'])

class NarrativePreprocess:
    """
    Class to preprocess narrative files and extract scenes.
    This class contains methods to process the narrative files, extract scenes, and save the results.
    """
    def __init__(self, narrative_filename_list, narrative_preprocessed_train_filename, narrative_preprocessed_eval_filename,
                 train_split):
        """
        Initialize the NarrativePreprocess class.
        :param type list of str, narrative_filename_list: List of narrative filenames.
        :param type str, narrative_preprocessed_train_filename: Filename for the preprocessed training data.
        :param type str, narrative_preprocessed_eval_filename: Filename for the preprocessed evaluation data.
        :param type float, train_split: The split ratio for training and evaluation data.
        """
        clean = True
        self.chapter_start_str = 'Chapter'
        self.narrative_filename_list = narrative_filename_list
        self.narrative_preprocessed_train_filename = narrative_preprocessed_train_filename
        self.narrative_preprocessed_eval_filename = narrative_preprocessed_eval_filename
        self.train_split = train_split
        if clean and os.path.exists(narrative_preprocessed_train_filename):
            os.remove(narrative_preprocessed_train_filename)
        if clean and os.path.exists(narrative_preprocessed_eval_filename):
            os.remove(narrative_preprocessed_eval_filename)
        self.scene_delimiter = ''
        self.preprocess_state = 'start'
        self.preprocess_results = NarrativePreprocessResults()
        self.preprocess_train = None
        self.preprocess_eval = None
        # if os.path.exists(narrative_preprocessed_filename):
        #     self.preprocess_results.load(narrative_preprocessed_filename)

    def train_eval_split(self):
        """
        Split the preprocessed results into training and evaluation sets.
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


    def process(self):
        """
        Process the narrative files and extract scenes.
        This method is the main entry point for preprocessing the narrative files.
        """
        self.preprocess()
        self.train_eval_split()
        self.dump()

    def preprocess(self):
        """
        Preprocess the narrative files and extract scenes.
        This method reads the narrative files line by line and processes them based on the current state.
        """
        # print(self.narrative_filename_list)
        linenum = 0
        for booknum, narrative_filename in enumerate(self.narrative_filename_list):
            with open(narrative_filename, "r") as fp:
                for line in fp:
                    if self.preprocess_results.at_linenum <= linenum:
                        self._preprocess_line(booknum + 1, line.strip())
                        self.preprocess_results.at_linenum += 1
                        # print(self.preprocess_results.__dict__)
                    linenum += 1

    def dump(self):
        """
        Dump the preprocessed results to JSON files.
        This method saves the preprocessed training and evaluation data to separate JSON files.
        """
        self.preprocess_train.dump(self.narrative_preprocessed_train_filename)
        self.preprocess_eval.dump(self.narrative_preprocessed_eval_filename)

    def get_current_filename(self):
        """
        Get the current filename being processed.
        :return: type str, The current filename.
        """
        return self.preprocess_results.get_current_filename()

    def get_next_filename(self):
        """
        Get the next filename to be processed.
        :return: type str, The next filename.
        """
        return self.preprocess_results.advance_filename()

    def get_current_scene(self):
        """
        Get the current scene from the preprocess results.
        :return: type NarrativeScene, The current scene object.
        """
        pr = self.preprocess_results
        if len(pr.scene_list) == 0:
            return None
        scene_obj = NarrativeScene(None)
        scene_obj.__dict__.update(pr.scene_list[-1])
        return scene_obj

    def update_current_scene(self, scene):
        """
        Update the current scene in the preprocess results.
        :param type NarrativeScene, scene: The scene object to update.
        """
        pr = self.preprocess_results
        if len(pr.scene_list) > 0:
            scene_dict = pr.scene_list[-1]
            scene_dict.update(scene.__dict__)

    def add_scene(self, scene):
        """
        Add a new scene to the preprocess results.
        :param type NarrativeScene, scene: The scene object to add.
        :return: type NarrativeScene, The added scene object.
        """
        self.preprocess_results.scene_list.append(scene.__dict__)
        return self.get_current_scene()

    def _preprocess_line(self, booknum, line):
        """
        Preprocess a line of text from the narrative file.
        This method updates the current scene based on the line content and the current state.
        :param type int, booknum: The book number.
        :param type str, line: The line of text to preprocess.
        """
        # by default, scenes are simply separated by a line, with chapters
        match self.preprocess_state:
            case 'start':
                if len(line) == 0:
                    return
                self._preprocess_new_scene(booknum)
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

    def _preprocess_chapter(self, booknum, line):
        """
        Preprocess the chapter information and update the current scene.
        :param type int, booknum: The book number.
        :param type str, line: The line containing the chapter information.
        """
        scene = self.get_current_scene()
        scene.set_chapter(line)
        scene.set_booknum(booknum)
        self.update_current_scene(scene)

    def _preprocess_new_scene(self, booknum):
        """
        Create a new scene object and add it to the preprocess results.
        :param type int, booknum: The book number.
        """
        scene = self.get_current_scene()
        new_scene = self.add_scene(NarrativeScene(scene))
        new_scene.set_booknum(booknum)
        self.update_current_scene(new_scene)

    def _preprocess_narrative(self, booknum, line):
        """
        Preprocess the narrative text and update the current scene.
        :param type int, booknum: The book number.
        :param type str, line: The line containing the narrative text.
        """
        scene = self.get_current_scene()
        text = line + '\n'
        scene.add_to_narrative(text)
        scene.set_booknum(booknum)
        self.update_current_scene(scene)
