#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 08:28:15 2025

@author: dfox
"""

import datetime
from word2number import w2n

from narrative_preprocess import NarrativeScene, NarrativePreprocess

class NarrativeSceneDCPFoxZombieApocalypse(NarrativeScene):
    """
    Class to represent a scene in the DCP Fox Zombie Apocalypse narrative.
    Inherits from the NarrativeScene class.
    """
    def __init__(self, prev_scene):
        """
        Initialize the scene with the previous scene.
        :param type NarrativeScene, prev_scene: The previous scene object.
        """
        super().__init__(prev_scene)
        if prev_scene is not None:
            self.daynum = prev_scene.daynum
            self.datestr = prev_scene.datestr
        else:
            self.daynum = None
            self.datestr = None

    def set_day(self, line):
        """
        Set the day number and date string for the scene.
        :param type str, line: The line containing the day information.
        """
        day_words = line.split(' ')
        self.daynum = w2n.word_to_num(day_words[1])
        # print(line, self.daynum)
        date = datetime.datetime.strptime('2025-08-25', "%Y-%m-%d") + datetime.timedelta(days=self.daynum)
        self.datestr = date.strftime('%Y-%m-%d')

class NarrativePreprocessDCPFoxZombieApocalypse(NarrativePreprocess):
    """
    Class to preprocess the DCP Fox Zombie Apocalypse narrative.
    Inherits from the NarrativePreprocess class.
    """
    def __init__(self, narrative_filename_list, narrative_preprocessed_train_filename, narrative_preprocessed_eval_filename,
                 train_eval_split):
        """
        Initialize the narrative preprocess with the given filenames and split.
        :param type list of str, narrative_filename_list: List of narrative filenames.
        :param type str, narrative_preprocessed_train_filename: Filename for the preprocessed training data.
        :param type str, narrative_preprocessed_eval_filename: Filename for the preprocessed evaluation data.
        :param type float, train_eval_split: The split ratio for training and evaluation data.
        """
        super().__init__(narrative_filename_list, narrative_preprocessed_train_filename, narrative_preprocessed_eval_filename,
                         train_eval_split)

        self.day_start_str = 'Day'
        self.scene_delimiter = '* * *'

    def preprocess(self):
        """
        Preprocess the narrative files and extract scenes.
        """
        print(self.narrative_filename_list)
        linenum = 0
        for booknum, narrative_filename in enumerate(self.narrative_filename_list):
            with open(narrative_filename, "r") as fp:
                for line in fp:
                    if self.preprocess_results.at_linenum <= linenum:
                        self._preprocess_line(booknum + 1, line.strip())
                        self.preprocess_results.at_linenum += 1
                        # print(self.preprocess_results.__dict__)
                    linenum += 1
        self.preprocess_results.scene_list.sort(key=lambda x: (x["daynum"], x["narr_scene_index"]))
        for index, scene in enumerate(self.preprocess_results.scene_list):
            scene['chron_scene_index'] = index

    def _preprocess_line(self, booknum, line):
        """
        Preprocess a single line of the narrative.
        :param type int, booknum: The book number.
        :param type str, line: The line to preprocess.
        """
        match self.preprocess_state:
            case 'start':
                # print(line)
                self._preprocess_new_scene(booknum)
                if line.startswith(self.chapter_start_str):
                    self._preprocess_chapter(booknum, line)
                    self.preprocess_state = 'chapter'
                elif line.startswith(self.day_start_str):
                    self._preprocess_day(booknum, line)
                    self.preprocess_state = 'day'
                elif len(line) > 0:
                    self._preprocess_narrative(booknum, line)
                    self.preprocess_state = 'narrative' # 'title' if processing title

            case 'chapter':
                if line.startswith(self.day_start_str):
                    self._preprocess_day(booknum, line)
                    self.preprocess_state = 'day'
                elif len(line) > 0:
                    self._preprocess_narrative(booknum, line)
                    self.preprocess_state = 'narrative'

            case 'day':
                if len(line) > 0:
                    self._preprocess_narrative(booknum, line)
                    self.preprocess_state = 'narrative'

            case 'narrative':
                if line.strip() == self.scene_delimiter:
                    # print( "scene delimiter found: {self.scene_delimiter}")
                    self.preprocess_state = 'start'
                elif line.startswith(self.chapter_start_str):
                    self._preprocess_new_scene(booknum)
                    self._preprocess_chapter(booknum, line)
                    self.preprocess_state = 'chapter'
                elif line.startswith(self.day_start_str):
                    self._preprocess_new_scene(booknum)
                    self._preprocess_day(booknum, line)
                    self.preprocess_state = 'day'
                elif len(line) > 0:
                    self._preprocess_narrative(booknum, line)

    def get_current_scene(self):
        """
        Get the current scene from the preprocessed results.
        :return: The current scene object.
        """
        pr = self.preprocess_results
        if len(pr.scene_list) == 0:
            return None
        scene_obj = NarrativeSceneDCPFoxZombieApocalypse(None)
        scene_obj.__dict__.update(pr.scene_list[-1])
        # print(scene_obj.daynum)
        return scene_obj

    def _preprocess_new_scene(self, booknum):
        """
        Create a new scene object and add it to the preprocessed results.
        :param type int, booknum: The book number.
        """
        scene = self.get_current_scene()
        new_scene = self.add_scene(NarrativeSceneDCPFoxZombieApocalypse(scene))
        new_scene.set_booknum(booknum)
        self.update_current_scene(new_scene)

    def _preprocess_day(self, booknum, line):
        """
        Preprocess the day information and update the current scene.
        :param type int, booknum: The book number.
        :param type str, line: The line containing the day information.
        """
        scene = self.get_current_scene()
        if scene is None:
            scene = self.add_scene(NarrativeSceneDCPFoxZombieApocalypse(None))
        scene.set_day(line)
        # print(scene.daynum)
        scene.set_booknum(booknum)
        self.update_current_scene(scene)

class NarrativePreprocessDCPFoxFate(NarrativePreprocess):
    """
    Class to preprocess the DCP Fox Fate narrative.
    Inherits from the NarrativePreprocess class.
    """
    def __init__(self, narrative_filename_list, narrative_preprocessed_train_filename, narrative_preprocessed_eval_filename,
                 train_eval_split):
        """
        Initialize the narrative preprocess with the given filenames and split.
        :param type list of str, narrative_filename_list: List of narrative filenames.
        :param type str, narrative_preprocessed_train_filename: Filename for the preprocessed training data.
        :param type str, narrative_preprocessed_eval_filename: Filename for the preprocessed evaluation data.
        :param type float, train_eval_split: The split ratio for training and evaluation data.
        """
        super().__init__(narrative_filename_list, narrative_preprocessed_train_filename, narrative_preprocessed_eval_filename,
                         train_eval_split)
        self.scene_delimiter = '* * *'
