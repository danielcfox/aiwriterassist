#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 09:17:25 2025

@author: dfox
"""

import argparse
import os
import random
import time

from config import Config
from narrative_preprocess import NarrativePreprocessResults
from narrative_preprocess_dcpfox import NarrativePreprocessDCPFoxZombieApocalypse, NarrativePreprocessDCPFoxFate
from llm_narrative_preprocessing import LLMNarrativeScenesPreprocessing
from llm_narrative_collection import LLMNarrativeScenesCollection
from llm_narrative_compose import LLMNarrativeScenesCompose
from narrative_compose_build_test import LLMNarrativeScenesBuildTestCompose
from llm_openai_api_handler import LLMOpenAIAPIHandler

from vdb_milvus import VectorDBMilvus


LLM_MODELS = {'gpt-4o-mini':
                  {'api_class': LLMOpenAIAPIHandler,
                   'max_tokens': 128000,
                   'max_output_tokens': 16384},
              # 'meta-llama/Meta-Llama-3-8B-Instruct-4bit':
              #     {'api_class': LLMLlama38BInstruct4bit,
              #      'max_tokens': 8192}
              }

def open_api_object(config, model_id, model_spec: str, model_filename: str = None):
    """
    Create an API object for the specified model.
    :param type Config, config: The configuration object.
    :param type str, model_id: The model ID to use.
    :param type str, model_spec: Specifies the type of the model argument.
                         Options are 'openai_details_filename' or 'model_name'.
                         - 'details_filename': Indicates that the model argument is a filename
                             containing the object details of a fine-tuned model.
                         - 'model_name': Indicates that the model argument is the name of the model
                             available through the GPT-4o API.
    :param type str, model: The model to use.
    :raises ValueError: If the model class name specified in the configuration file does not exist.
    :return: An instance of the API class for the specified model.
    """
    api_class_name = config.get_model_class(model_id)
    api_class = globals()[api_class_name]
    if config.verbose:
        print(f"model_spec {model_spec}")
    if model_spec == 'inference':
        model = config.get_model_inference_name(model_id)
        model_spec = 'model_name'
    elif model_spec == 'fine_tune':
        model = config.get_model_fine_tune_name(model_id)
        model_spec = 'model_name'
    elif model_spec == 'details_filename':
        model = model_filename
    else:
        raise ValueError(f"Invalid model_spec '{model_spec}'. Must be 'openai_details_filename', 'model_name', or 'fine_tune.")

    # if use_fine_tuned_model is None:
    #     model_name = config.get_model_inference_name(model_id)
    #     model_spec = 'model_name'
    # else:
    #     model_name = use_fine_tuned_model
    #     model_spec = 'openai_details_filename'
    return api_class(model_spec, model, verbose=config.verbose)

def module_narrative_preprocess(config, user, narrative):
    """Preprocess the narrative data for the given user and narrative.
    :param type Config, config: The configuration object.
    :param type str, user: The user/author identifier.
    :param type str, narrative: The narrative identifier.
    """
    input_filename_list = config.get_user_narrative_input_file_list(user, narrative)
    output_train_filename = config.get_user_narrative_preprocess_output_train_filename(user, narrative)
    output_eval_filename = config.get_user_narrative_preprocess_output_eval_filename(user, narrative)
    train_split = config.get_user_preprocess_fine_tune_train_split(user)
    scene_limit = config.get_user_preprocess_scene_limit(user)

    npp_class_name = config.get_narrative_class(narrative)
    globals_dict = globals()
    if npp_class_name in globals_dict:
        npp_class = globals_dict[npp_class_name]
        nppc = npp_class(input_filename_list, output_train_filename, output_eval_filename, train_split, scene_limit) # clean
        nppc.process()
        # nppc.dump()
    else:
        raise Exception(ValueError, "class name specified in configuration file does not exist")
    print("Narrative preprocess complete")

def module_narrative_scenes_llm_preprocess(config, user, narrative):
    """Preprocess (via LLM) the narrative scenes data for the given user and narrative.
    :param type Config, config: The configuration object.
    :param type str, user: The user/author identifier.
    :param type str, narrative: The narrative identifier.
    """
    train_input_file = config.get_user_narrative_preprocess_output_train_filename(user, narrative)
    eval_input_file = config.get_user_narrative_preprocess_output_eval_filename(user, narrative)
    train_output_file = config.get_user_narrative_scenes_llm_preprocess_output_train_filename(user, narrative)
    eval_output_file = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)

    model_id = config.get_user_narrative_scenes_llm_preprocess_model_id(user)
    model_name = config.get_model_inference_name(model_id)
    clean = config.get_user_narrative_scenes_llm_preprocess_clean(user)
    author_name = config.get_user_author_name(user)
    max_input_tokens = config.get_model_max_input_tokens(model_id)
    max_output_tokens = config.get_model_max_output_tokens(model_id)

    api_obj = open_api_object(config, model_id, 'inference')
    llmp = LLMNarrativeScenesPreprocessing(clean, user, narrative, author_name, api_obj, train_input_file, eval_input_file, 
                                           train_output_file, eval_output_file, max_input_tokens, max_output_tokens)
    scene_limit = config.get_user_narrative_scenes_llm_preprocess_scene_limit_per_narrative(user)
    llmp.update_scene_list(scene_limit)
    print("Narrative scenes LLM preprocess complete")

def module_narrative_into_vector_db(config, vector_db, user, narrative):
    """Place the narrative data into the milvus vector database.
    :param type Config, config: The configuration object.
    :param type VectorDBMilvus, vector_db: The milvus vector database object.
    :param type str, user: The user/author identifier.
    :param type str, narrative: The narrative identifier.
    """
    model_id = config.get_user_narrative_scenes_llm_preprocess_model_id(user)
    # model_name = config.get_model_inference_name(model_id)
    input_train_file = config.get_user_narrative_scenes_llm_preprocess_output_train_filename(user, narrative)
    input_eval_file = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)
    author_name = config.get_user_author_name(user)

    # api_obj = open_api_object(config, model_id, 'inference')

    llmnsp = LLMNarrativeScenesCollection(user, narrative, author_name, input_train_file, input_eval_file, vector_db)
    llmnsp.build_vector_collection(config.get_user_narrative_into_vector_db_clean(user))

def module_corpus_llm_fine_tuning(config, user, corpus):
    """Fine-tune the LLM model for the given user and corpus.
    :param type Config, config: The configuration object.
    :param type str, user: The user/author identifier.
    :param type list, corpus: The list of narratives for the user.
    :param type int, maxwait: The maximum wait time (in seconds) for the fine-tuning process.
    :return: None
    """
    train_filename_list = [config.get_user_narrative_preprocess_output_train_filename(user, narrative) for narrative in corpus]
    fine_tune_filename = config.get_user_fine_tuned_filename(user)
    model_id = config.get_user_fine_tune_model_id(user)
    # model_name = config.get_model_fine_tune_name(model_id)
    api_obj = open_api_object(config, model_id, 'fine_tune')
    if api_obj is None:
        print("API object is None")
        return
    author_name = config.get_user_author_name(user)
    maxwait = config.get_user_fine_tune_maxwait(user)
    api_obj.fine_tune_submit(train_filename_list, author_name)
    print(api_obj.wait_fine_tuning_model(fine_tune_filename, maxwait))

def module_corpus_llm_fine_tuning_check(config, user):
    """Check the status of the fine-tuning process for the given user and corpus.
    :param type Config, config: The configuration object.
    :param type str, user: The user/author identifier.
    :param type list, corpus: The list of narratives for the user.
    """
    model_id = config.get_user_fine_tune_model_id(user)
    fine_tune_filename = config.get_user_fine_tuned_filename(user)
    api_obj = open_api_object(config, model_id, 'details_filename', fine_tune_filename)
    if api_obj is None:
        print("API object is None")
        return
    print(api_obj.wait_fine_tuning_model(fine_tune_filename, 0))

def module_narrative_scenes_build_test_set(config, user, narrative):
    """Build the test set for the narrative scenes for the given user and narrative.
    :param type Config, config: The configuration object.
    :param type str, user: The user/author identifier.
    :param type str, narrative: The narrative identifier.
    """
    input_eval_file = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)

    output_filename = config.get_user_narrative_scenes_build_test_set_output_filename(user, narrative)

    llmnsp = LLMNarrativeScenesBuildTestCompose(user, narrative, input_eval_file, output_filename)
    scene_limit = config.get_user_narrative_scenes_build_test_set_scene_limit(user)
    llmnsp.build_test_compose_scene_input_file(output_filename, scene_limit)
    print("Narrative scenes build test set complete")

def module_compose_scene_llm_narrative_handler(config, vector_db, user, narrative):
    details_filename = config.get_user_fine_tuned_filename(user)
    input_filename = config.get_user_narrative_compose_scene_llm_handler_input_filename(user, narrative)
    output_filename = config.get_user_narrative_compose_scene_llm_handler_output_filename(user, narrative)
    model_id = config.get_user_narrative_compose_scene_llm_handler_model_id(user)
    # fine_tune_model_name = config.get_user_fine_tune_model_name(user)
    author_name = config.get_user_author_name(user)
    api_obj = open_api_object(config, model_id, "details_filename", details_filename)
    max_input_tokens = config.get_model_max_input_tokens(model_id)
    max_output_tokens = config.get_model_max_output_tokens(model_id)
    input_train_filename = config.get_user_narrative_scenes_llm_preprocess_output_train_filename(user, narrative)
    input_eval_filename = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)
    llmnsp = LLMNarrativeScenesCompose(user, narrative, author_name, api_obj, input_train_filename, input_eval_filename, 
                                       input_filename, output_filename, max_input_tokens, max_output_tokens, vector_db) 
    scene_limit = config.get_user_narrative_compose_scene_llm_handler_scene_limit(user)
    recent_event_count = config.get_user_narrative_compose_scene_llm_handler_recent_event_count(user)
    llmnsp.write_scenes(scene_limit, recent_event_count)
    print("Narrative scenes compose complete")

def main():
    """
    Main function to run the pipeline for determining optimal satellite imagery resolution.
    This function handles command-line arguments, configuration loading, and the execution of various modules in the pipeline.
    It orchestrates the entire process by calling the appropriate functions based on the configuration settings.
    The pipeline includes preprocessing narratives, fine-tuning LLMs, processing narrative scenes, and handling vector databases.
    The function also measures the duration of the operation for each user and narrative.
    :return: None
    """
    print("Running...")

    start = time.time()

    random.seed(17)

    parser = argparse.ArgumentParser(description="Pipeline for determining optimal satellite imagery resolution.")

    parser.add_argument("config_filename", type=str, help="The path to the configuration file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Specify verbosity", default=False)

    args = parser.parse_args()

    # config.args = args
    # config_filename = args.config_filename
    config = Config(args)

    vector_db_filename = config.get_vector_db_filename()
    vector_db = None
    
    users = config.get_user_list()

    for user in users:
        os.makedirs(config.get_user_cwd(user), exist_ok=True)

        corpus = config.get_user_preprocess_corpus(user)

        # This preprocessing step must come first before any other modules can be run. It prepares the data from the narrative
        # for the LLM to process. It also splits each narrative into the training and evaluation sets.
        # The training set is used to fine-tune the LLM and the evaluation set is used to test the LLM.
        # The training set and evaluation set are split chronologically.
        # That is, the training set scenes occur before the evaluation set scenes.
        for narrative in corpus:
            if config.run_narrative_preprocess():
                module_narrative_preprocess(config, user, narrative)

        # The completion of the fine tuning step is a prerequisite for building the vector database and the scene compose step.
        # Note that this module "kicks off" the fine tuning process only waits for the configured amount of time
        # The fine tuning process is run in the background and will take some time to complete
        # Therefore, the fine tuning process may not be complete after this module is run
        # The scene compose step will fail if the fine tuning process is not complete
        if config.run_corpus_llm_fine_tuning():
            module_corpus_llm_fine_tuning(config, user, corpus)

        # The fine tuning check step is a way for the user to check the status of the fine tuning process, if it did not complete
        # in the time specified in the fine tuning step
        # This module will also dump the details of the fine tuning process to a file
        if config.check_corpus_llm_fine_tuning():
            module_corpus_llm_fine_tuning_check(config, user)

        # The LLM needs to complete the preprocessing for each scene in the narrative
        # This step analyzes each scene in the narrative and generates metadata
        # The metadata is later used to build the vector database
        # The metadata is also used to build the test set for the narrative
        # The metadata consists of:
        #     the tone of the scene
        #     the summary of the scene
        #     the characters in the scene (and their descriptions plus plot summary from their point of view)
        #     the objects in the scene (and their descriptions plus plot summary from the object's point of view) 
        #     the setting of the scene (and its description plus plot summary from the setting's point of view)
        for narrative in corpus:
            if config.run_narrative_scenes_llm_preprocess():
                module_narrative_scenes_llm_preprocess(config, user, narrative)

        # The vector database is used to store the metadata for each scene in the narrative
        # This step must be run after the LLM preprocessing step,
        # and is a prerequisite for the scene compose step
        for narrative in corpus:
            if config.run_narrative_into_vector_db():
                if vector_db_filename is None:
                    raise Exception(ValueError, "vector_db_name not specified in configuration file")
                if vector_db is None:
                    vector_db = VectorDBMilvus(vector_db_filename)
                module_narrative_into_vector_db(config, vector_db, user, narrative)

        # This step prepares the test set for scene composition
        # The test set is used to evaluate the performance of the LLM
        # The test set consists of a subset of the scenes in the narrative
        # This step must be run after the LLM preprocessing step
        # and is a prerequisite for the scene compose step
        # This step strips some of the metadata from the scenes (characters, objects, and settings descriptions and summaries)
        # This list of characters, objects, and settings remain in the test set
        # In this way, we test the following scenario:
        # The author provides a scene with characters, objects, and settings, along with a synopsis of the scene
        # The LLM is asked to compose a scene based on this information
        # Characters, objects, and settings descriptions are retrieved from the vector database
        for narrative in corpus:
            if config.run_narrative_scenes_build_test_set():
                module_narrative_scenes_build_test_set(config, user, narrative)

        # This step is the final step in the pipeline and composes a new scene based on the test set
        # OR a provided set of scenes if not testing but actually composing a scene for the author
        # The author provides a scene with characters, objects, and settings, along with a synopsis of the scene
        # The LLM is asked to compose a scene based on this information
        # Characters, objects, and settings descriptions are retrieved from the vector database
        for narrative in corpus:
            if config.run_compose_scene_llm_narrative_handler():
                if vector_db_filename is None:
                    raise Exception(ValueError, "vector_db_name not specified in configuration file")
                if vector_db is None:
                    vector_db = VectorDBMilvus(vector_db_filename)
                module_compose_scene_llm_narrative_handler(config, vector_db, user, narrative)

    duration = time.time() - start
    print(f"Duration of operation for user {user} is {duration}")

if __name__ == "__main__":
    main()
