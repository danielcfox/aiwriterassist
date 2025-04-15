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
    A specialized scene class for the DCP Fox Zombie Apocalypse narrative.

    Extends the base NarrativeScene class with additional tracking of temporal
    information specific to the Zombie Apocalypse narrative, including day numbers
    and calendar dates.

    Each scene inherits time information from its predecessor when available,
    maintaining chronological consistency throughout the narrative processing.
    """

    def __init__(self, prev_scene: 'NarrativeScene') -> None:
        """
        Initialize a Zombie Apocalypse scene, inheriting temporal data from previous scene.

        Creates a new scene object that maintains chronological continuity with previous
        scenes by inheriting day number and date information when available.

        Parameters:
            prev_scene (NarrativeScene): Previous scene to inherit temporal data from.
                                         Can be None for the first scene.
        """
        super().__init__(prev_scene)
        if prev_scene is not None:
            self.daynum = prev_scene.daynum
            self.datestr = prev_scene.datestr
        else:
            self.daynum = None
            self.datestr = None

    def set_day(self, line: str) -> None:
        """
        Parse and set the day number and calendar date for the scene.

        Extracts day number from textual representation (e.g., "Day Three")
        and calculates the corresponding calendar date based on a fixed
        starting date of August 25, 2025.

        Parameters:
            line (str): Text line containing day information in format "Day [Number]"
                       where Number can be a word ("Three") or digit ("3").
        """
        day_words = line.split(' ')
        self.daynum = w2n.word_to_num(day_words[1])
        # print(line, self.daynum)
        date = datetime.datetime.strptime('2025-08-25', "%Y-%m-%d") + datetime.timedelta(days=self.daynum)
        self.datestr = date.strftime('%Y-%m-%d')

class NarrativePreprocessDCPFoxZombieApocalypse(NarrativePreprocess):
    """
    Preprocessor for the DCP Fox Zombie Apocalypse narrative format.

    This class implements specialized parsing logic for the Zombie Apocalypse
    narrative structure, which includes day-based organization and scene
    delimiters. It processes raw narrative text files and extracts structured
    scene information with chronological ordering.

    The class handles specific formatting conventions including:
    - Day-based scene organization ("Day One", "Day Two", etc.)
    - Scene separation using asterisk delimiters ("* * *")
    - Chronological ordering using day numbers and scene indices
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the Zombie Apocalypse narrative preprocessor.

        Sets up the preprocessor with format-specific delimiters and parsing logic
        for the Zombie Apocalypse narrative structure.

        Parameters:
            **kwargs: Configuration parameters inherited from NarrativePreprocess:
                - narrative_filename_list (list): Paths to input narrative files
                - narrative_preprocessed_train_filename (str): Output path for training data
                - narrative_preprocessed_eval_filename (str): Output path for evaluation data
                - train_eval_split (float): Ratio of training to evaluation data
                - scene_limit (int, optional): Maximum number of scenes to process
        """
        super().__init__(**kwargs)

        self.day_start_str = 'Day'
        self.scene_delimiter = '* * *'

    def preprocess(self) -> None:
        """
        Process narrative files and extract structured scene information.

        Reads all narrative files line by line, identifies scene boundaries and
        day markers, and builds a chronologically ordered list of structured scenes.
        Each scene includes day number, narrative index, and chronological index.

        After processing all files, scenes are sorted by day number and narrative
        index, then assigned sequential chronological indices.
        """
        print(self.narrative_filename_list)
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
        self.preprocess_results.scene_list.sort(key=lambda x: (x["daynum"], x["narr_scene_index"]))
        for index, scene in enumerate(self.preprocess_results.scene_list):
            scene['chron_scene_index'] = index

    def _preprocess_line(self, booknum: int, line: str) -> bool:
        """
        Process a single line of the narrative text.

        Implements a state machine that transitions between different parsing states
        based on line content. States include 'start', 'chapter', 'day', and 'narrative'.
        The method identifies scene boundaries, day markers, chapter headings, and
        narrative content.

        Parameters:
            booknum (int): Current book number being processed
            line (str): Single line of text from the narrative file

        Returns:
            bool: False if processing should be skipped for this line, True otherwise
        """
        match self.preprocess_state:
            case 'start':
                # print(line)
                if self._preprocess_new_scene(booknum) == False:
                    return False
                if line.startswith(self.chapter_start_str):
                    self._preprocess_chapter(booknum, line)
                    self.preprocess_state = 'chapter'
                elif line.startswith(self.day_start_str):
                    if self._preprocess_day(booknum, line) == False:
                        return False
                    self.preprocess_state = 'day'
                elif len(line) > 0:
                    self._preprocess_narrative(booknum, line)
                    self.preprocess_state = 'narrative' # 'title' if processing title

            case 'chapter':
                if line.startswith(self.day_start_str):
                    if self._preprocess_day(booknum, line) == False:
                        return False
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

    def get_current_scene(self) -> NarrativeSceneDCPFoxZombieApocalypse:
        """
        Retrieve the current scene being processed.

        Returns the most recently added scene as a NarrativeSceneDCPFoxZombieApocalypse
        object, populated with all current scene data. Returns None if no scenes
        have been processed yet.

        Returns:
            NarrativeSceneDCPFoxZombieApocalypse: Current scene object, or None if no scenes exist
        """
        pr = self.preprocess_results
        if len(pr.scene_list) == 0:
            return None
        scene_obj = NarrativeSceneDCPFoxZombieApocalypse(None)
        scene_obj.__dict__.update(pr.scene_list[-1])
        # print(scene_obj.daynum)
        return scene_obj

    def _preprocess_new_scene(self, booknum: int) -> bool:
        """
        Create and initialize a new scene in the narrative.

        Creates a new scene object linked to the previous scene (if any),
        assigns book number information, and updates the current scene pointer.

        Parameters:
            booknum (int): Book number to associate with the new scene

        Returns:
            bool: False if scene couldn't be created, True otherwise
        """
        scene = self.get_current_scene()
        new_scene = self.add_scene(NarrativeSceneDCPFoxZombieApocalypse(scene))
        if new_scene is None:
            return False
        new_scene.set_booknum(booknum)
        self.update_current_scene(new_scene)
        return True

    def _preprocess_day(self, booknum: int, line: str) -> bool:
        """
        Process day information and update the current scene.

        Extracts day number from the text line, assigns it to the current scene,
        and updates the scene's calendar date. Creates a new scene if necessary.

        Parameters:
            booknum (int): Current book number
            line (str): Text line containing day information ("Day One", etc.)

        Returns:
            bool: False if day processing failed, True otherwise
        """
        scene = self.get_current_scene()
        if scene is None:
            scene = self.add_scene(NarrativeSceneDCPFoxZombieApocalypse(None))
        if scene is None:
            return False
        scene.set_day(line)
        # print(scene.daynum)
        scene.set_booknum(booknum)
        self.update_current_scene(scene)
        return True

class NarrativePreprocessDCPFoxFate(NarrativePreprocess):
    """
    Preprocessor for the DCP Fox Fate narrative format.

    This class implements parsing logic for the Fate narrative structure,
    which uses asterisk delimiters to separate scenes. It inherits the general
    preprocessing framework but applies specific formatting rules for this
    narrative style.

    The class maintains a simpler structure compared to the Zombie Apocalypse
    narrative, with only scene delimiter customization.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the Fate narrative preprocessor.

        Sets up the preprocessor with format-specific delimiters for the
        Fate narrative structure.

        Parameters:
            **kwargs: Configuration parameters inherited from NarrativePreprocess:
                - narrative_filename_list (list): Paths to input narrative files
                - narrative_preprocessed_train_filename (str): Output path for training data
                - narrative_preprocessed_eval_filename (str): Output path for evaluation data
                - train_eval_split (float): Ratio of training to evaluation data
                - scene_limit (int, optional): Maximum number of scenes to process
        """
        super().__init__(**kwargs)
        self.scene_delimiter = '* * *'
