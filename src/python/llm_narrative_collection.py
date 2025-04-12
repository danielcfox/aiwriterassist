#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:01:31 2025

@author: dfox
"""
from llm_narrative_handler import LLMNarrativeScenesHandler

class LLMNarrativeScenesCollection(LLMNarrativeScenesHandler):
    def __init__(self, user, narrative, author_name, train_input_file, eval_input_file, 
                 vector_db=None):
        """Initialize the LLMNarrativeScenesHandler with model and narrative details."""
        super().__init__(user, narrative, author_name, None, train_input_file, eval_input_file)

        self.vector_db = vector_db
   
    def build_vector_collection(self, clean):
        """Build the vector collection for the narrative.
        :param type bool, clean: Whether to clean the collection before building.
        :return: None
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

