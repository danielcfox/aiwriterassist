#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""
from llm_narrative_handler import LLMNarrativeScenesHandler
from vdb_milvus import VectorDBMilvus

import os

class LLMNarrativeScenesCollection(LLMNarrativeScenesHandler):
    """
    A handler for processing narrative scenes and storing them in a vector database.

    This class prepares narrative scenes for semantic search by extracting key elements
    (characters, objects, settings, summaries) and embedding them in a vector database.
    It processes input files containing scene data and organizes the information for
    efficient retrieval during narrative generation.

    The class handles:
    1. Loading scene data from input files
    2. Extracting descriptions of characters, settings, and objects
    3. Creating vector embeddings for narrative elements
    4. Organizing data with consistent ID schemes
    5. Storing information in the vector database

    The collection process preserves the chronological development of narrative elements
    while making them available for semantic search and contextual retrieval.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialize the LLMNarrativeScenesCollection with narrative and database details.

        This class handles the collection and processing of narrative scenes,
        preparing them for vector database storage. It requires at least one input file
        and a vector database connection.

        Parameters:
            **kwargs: Additional keyword arguments for configuration:
                - narrative (str, required): The narrative identifier.
                - vector_db (VectorDBMilvus, required): The vector database object.
                - input_train_filename (str, optional): Path to the training data file.
                - input_eval_filename (str, optional): Path to the evaluation data file.
                - verbose (bool, optional): If True, print verbose output. Default False.

        Returns:
            None

        Raises:
            ValueError: If required parameters are missing or invalid, or if no valid input files exist.
        """

        # Required parameters
        self.vector_db = None
        self.narrative = None

        # Required one of these parameters
        self.input_train_filename = None
        self.input_eval_filename = None

        # Optional parameters
        self.verbose = False

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.narrative is None or len(self.narrative) == 0:
            raise ValueError("LLMNarrativeScenesCollection: narrative is not set")
        if self.vector_db is None:
            raise ValueError("LLMNarrativeScenesCollection: vector_db is not set")
        # for key, value in kwargs.items():
        #     if key == 'vector_db':
        #         if not isinstance(value, VectorDBMilvus):
        #             raise ValueError("vector_db must be an instance of VectorDBMilvus")
        #         self.vector_db = value
        #     else:
        #         setattr(self, key, value)
        if self.vector_db is None or not isinstance(self.vector_db, VectorDBMilvus):
            raise ValueError("LLMNarrativeScenesCollection: VectorDBMilvus is not initialized. "
                             + "Please provide a valid URI in the configuration YAML file.")

        # self.output_filename = None
        # del kwargs['vector_db']
        input_filename_list = []
        if self.input_train_filename is not None and os.path.exists(self.input_train_filename):
            input_filename_list.append(self.input_train_filename)
        if self.input_eval_filename is not None and os.path.exists(self.input_eval_filename):
            input_filename_list.append(self.input_eval_filename)
        if len(input_filename_list) == 0:
            raise ValueError("LLMNarrativeScenesCollection: Neither Input train file nor Input eval file exists.")
        super().__init__(input_filename_list, self.verbose)
        # if self.input_train_filename is None or not os.path.exists(self.input_train_filename):
        #     if self.input_eval_filename is None or not os.path.exists(self.input_eval_filename):
        #         raise ValueError(f"Neither Input train file {self.input_train_filename} "
        #                          + f"or Input eval file {self.input_eval_filename} exists.")
        # super().__init__(user, narrative, author_name, None, input_train_filename, input_eval_filename)

    def build_vector_collection(self, clean: bool = False) -> None:
        """
        Process narrative scenes and build vector embeddings in the database.

        This method extracts narrative elements (characters, objects, settings, scene summaries)
        from all processed scenes and creates vector embeddings for each element. Each scene's
        elements are stored with unique IDs that encode both the scene index and element type.

        Parameters:
            clean (bool): If True, delete existing collection before building. Default is False.

        Returns:
            None

        Raises:
            ValueError: If the vector database is not properly initialized.
        """
        if self.vector_db is None:
            raise ValueError("VectorDBMilvus is not initialized. Please provide a valid URI in the configuration YAML file.")
            # self.vector_db = VectorDB(uri)

        print("Building vector collection")
        if clean:
            print(f"Deleting collection {self.narrative}")
            self.vector_db.delete_collection(self.narrative)
        self.vector_db.create_collection(self.narrative)

        # gather character names and match first and last names
        # character_names = [name for name in self.preprocessing_results.scene_list['named_characters'].keys()]

        character_description_documents = []
        # character_scene_summary_documents = []
        object_description_documents =[]
        # object_scene_summary_documents = []
        setting_description_documents = []
        # setting_scene_summary_documents = []
        scene_summary_documents = []
        for i in range(len(self.preprocess_results.scene_list)):
            scene = self.preprocess_results.scene_list[i]
            scene_index = scene['chron_scene_index']
            character_description_documents.extend(
                [
                {'id': ((scene_index * 1000) + index), 'text': f"{value['description']}"}
                for index, value in enumerate(scene['named_characters'].values())
                ]
            )
            # character_scene_summary_documents.extend(
            #     [
            #     {'id': ((scene_index * 1000) + 100 + index), 'text': f"{value['plot_summary']}"}
            #     for index, value in enumerate(scene['named_characters'].values())
            #     ]
            # )
            if 'objects' in scene:
                object_description_documents.extend(
                    [
                    {'id': ((scene_index * 1000) + 200 + index), 'text': f"{value['description']}"}
                    for index, value in enumerate(scene['objects'].values())
                    ]
                )
                # object_scene_summary_documents.extend(
                #     [
                #     {'id': ((scene_index * 1000) + 300 + index), 'text': f"{value['plot_summary']}"}
                #     for index, value in enumerate(scene['objects'].values())
                #     ]
                # )
            setting_description_documents.extend(
                [
                {'id': ((scene_index * 1000) + 400 + index), 'text': f"{value['description']}"}
                for index, value in enumerate(scene['settings'].values())
                ]
            )
            # setting_scene_summary_documents.extend(
            #     [
            #     {'id': ((scene_index * 1000) + 500 + index), 'text': f"{value['plot_summary']}"}
            #     for index, value in enumerate(scene['settings'].values())
            #     ]
            # )
            if len(scene['plot_summary']) > 0:
                scene_summary_documents.append(
                    {
                    'id': ((scene_index * 1000) + 600), 'text': f"{scene['plot_summary']}"
                    }
                )

        # print(f"length of character descriptions is {len(character_description_documents)}")
        # print(f"length of character scene summaries is {len(character_scene_summary_documents)}")
        # print(f"length of object descriptions is {len(object_description_documents)}")
        # print(f"length of setting descriptions is {len(setting_description_documents)}")
        # # print(f"length of setting scene summaries is {len(setting_scene_summary_documents)}")
        # print(f"length of scene summaries is {len(scene_summary_documents)}")

        doc_types = {"character_descriptions": character_description_documents,
                    #  "character_scene_summaries": character_scene_summary_documents,
                     "object_descriptions": object_description_documents,
                     "setting_descriptions": setting_description_documents,
                    #  "setting_scene_summaries": setting_scene_summary_documents,
                     "scene_summaries": scene_summary_documents}

        print(f"Insert documents into the collection")
        self.vector_db.insert_documents(self.narrative, doc_types)
