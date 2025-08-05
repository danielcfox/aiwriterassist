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
from llm_narrative_collection import LLMNarrativeScenesCollection
from llm_narrative_compose import LLMNarrativeScenesCompose
from llm_narrative_preprocessing import LLMNarrativeScenesPreprocessing
from llm_openai_api_handler import LLMOpenAIAPIHandler
from narrative_compose_build_test import LLMNarrativeScenesBuildTestCompose
from narrative_preprocess import NarrativePreprocessResults
from narrative_preprocess_dcpfox import NarrativePreprocessDCPFoxZombieApocalypse, NarrativePreprocessDCPFoxFate
from vdb_milvus import VectorDBMilvus

def open_api_object(config: Config, model_id: str, **kwargs) -> object:
    """
    Create an API object for the specified model.

    Parameters:
        config (Config): The configuration object containing model settings.
        model_id (str): The identifier of the model to create.
        **kwargs: Additional keyword arguments for model configuration:
            - details_filename (str, optional): The URI/path to the model details file.
            - model_name (str, optional): The name of the model to use.
            - author_name (str, optional): The name of the author for personalized models.

    Returns:
        object: An instance of the model class specified by the configuration.

    Raises:
        KeyError: If the model class specified in config doesn't exist in globals().
    """
    # details_filename = getattr(kwargs, 'details_filename', None)
    # model_name = getattr(kwargs, 'model_name', None)
    # verbose = getattr(kwargs, 'verbose', False)
    api_class_name = config.get_model_class(model_id)
    api_class = globals()[api_class_name]

    # if verbose:
    #     print(f"model_id {model_id}")
    #     print(f"details_filename {details_filename}")
    #     print(f"model_name {model_name}")

    return api_class(**kwargs)

def module_narrative_preprocess(config: Config, user: str, narrative: str) -> None:
    """
    Preprocess the narrative data for the given user and narrative.

    This function loads the appropriate narrative preprocessing class based on configuration,
    and processes the input files into training and evaluation datasets.

    Parameters:
        config (Config): The configuration object with processing settings.
        user (str): The user/author identifier.
        narrative (str): The narrative identifier.

    Returns:
        None

    Raises:
        ValueError: If the class name specified in configuration file does not exist.
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
        nppc = npp_class(
            narrative_filename_list=input_filename_list,
            narrative_preprocessed_train_filename=output_train_filename,
            narrative_preprocessed_eval_filename=output_eval_filename,
            train_split=train_split,
            scene_limit=scene_limit
        )
        nppc.process()
        # nppc.dump()
    else:
        raise Exception(ValueError, "class name specified in configuration file does not exist")
    print("Narrative preprocess complete")

def module_narrative_scenes_llm_preprocess(config: Config, user: str, narrative: str) -> None:
    """
    Preprocess (via LLM) the narrative scenes data for the given user and narrative.

    This function uses an LLM to analyze and enhance narrative scenes with metadata
    such as tone, summaries, character descriptions, etc.

    Parameters:
        config (Config): The configuration object with processing settings.
        user (str): The user/author identifier.
        narrative (str): The narrative identifier.

    Returns:
        None
    """
    train_input_file = config.get_user_narrative_preprocess_output_train_filename(user, narrative)
    eval_input_file = config.get_user_narrative_preprocess_output_eval_filename(user, narrative)
    train_output_file = config.get_user_narrative_scenes_llm_preprocess_output_train_filename(user, narrative)
    eval_output_file = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)

    model_id = config.get_user_narrative_scenes_llm_preprocess_model_id(user)
    clean = config.get_user_narrative_scenes_llm_preprocess_clean(user)
    author_name = config.get_user_author_name(user)
    max_input_tokens = config.get_model_max_input_tokens(model_id)
    max_output_tokens = config.get_model_max_output_tokens(model_id)
    inference_name = config.get_model_inference_name(model_id)

    api_obj = open_api_object(config, model_id, model_name=inference_name, verbose=config.verbose)
    if clean:
        llmp = LLMNarrativeScenesPreprocessing(
            api_obj=api_obj,
            narrative=narrative,
            author_name=author_name,
            input_train_filename=train_input_file,
            input_eval_filename=eval_input_file,
            output_train_filename=train_output_file,
            output_eval_filename=eval_output_file,
            max_output_tokens=max_output_tokens
        )
    else:
        llmp = LLMNarrativeScenesPreprocessing(
            narrative=narrative,
            author_name=author_name,
            api_obj=api_obj,
            input_train_filename=train_output_file, # not clean uses output files as input
            input_eval_filename=eval_output_file, # not clean uses output files as input
            output_train_filename=train_output_file,
            output_eval_filename=eval_output_file,
            max_output_tokens=max_output_tokens
        )
    scene_limit = config.get_user_narrative_scenes_llm_preprocess_scene_limit_per_narrative(user)
    llmp.update_scene_list(scene_limit)
    print("Narrative scenes LLM preprocess complete")

def module_narrative_into_vector_db(config: Config, vector_db: VectorDBMilvus, user: str, narrative: str) -> None:
    """
    Place the narrative data into the Milvus vector database.

    This function loads processed narrative data and stores embeddings and metadata
    in the vector database for later retrieval.

    Parameters:
        config (Config): The configuration object with database settings.
        vector_db (VectorDBMilvus): The Milvus vector database object.
        user (str): The user/author identifier.
        narrative (str): The narrative identifier.

    Returns:
        None
    """
    input_train_filename = config.get_user_narrative_scenes_llm_preprocess_output_train_filename(user, narrative)
    input_eval_filename = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)
    author_name = config.get_user_author_name(user)

    llmnsp = LLMNarrativeScenesCollection(
        narrative=narrative,
        input_train_filename=input_train_filename,
        input_eval_filename=input_eval_filename,
        vector_db=vector_db,
        verbose=config.verbose
    )

    llmnsp.build_vector_collection(config.get_user_narrative_into_vector_db_clean(user))

def module_corpus_llm_fine_tuning(config: Config, user: str, corpus: list[str], vector_db: VectorDBMilvus) -> None:
    """
    Fine-tune the LLM model for the given user and corpus.

    This function initiates the fine-tuning process on the specified model using
    the preprocessed training data from the user's narrative corpus.

    Parameters:
        config (Config): The configuration object with fine-tuning settings.
        user (str): The user/author identifier.
        corpus (list): The list of narrative identifiers for the user.

    Returns:
        None
    """
    if vector_db is None:
        raise Exception(ValueError, "vector_db is None")
    # train_filename_list = [config.get_user_narrative_preprocess_output_train_filename(user, narrative) for narrative in corpus]
    fine_tune_filename = config.get_user_fine_tuned_filename(user)
    model_id = config.get_user_fine_tune_model_id(user)
    model_name = config.get_model_fine_tune_name(model_id)
    author_name = config.get_user_author_name(user)
    generate_prompt_only = config.get_user_fine_tune_generate_prompt_only(user)
    fine_tune_submit_filename = config.get_user_fine_tune_submit_filename(user)
    api_obj = open_api_object(
        config,
        model_id,
        author_name=author_name,
        model_name=model_name,
        details_filename=fine_tune_filename,
        generate_prompt_only=generate_prompt_only,
        fine_tune_submit_filename=fine_tune_submit_filename,
        verbose=config.verbose
    )
    if api_obj is None:
        print("API object is None")
        return
    maxwait = config.get_user_fine_tune_maxwait(user)
    lookback_scene_limit_count = config.get_user_narrative_compose_scene_llm_handler_lookback_scene_limit_count(user)
    # build the fine-tuning dataset
    # details_filename = config.get_user_fine_tuned_filename(user)
    # model_id = config.get_user_narrative_compose_scene_llm_handler_model_id(user)
    # fine_tune_model_name = config.get_user_fine_tune_model_name(user)
    # author_name = config.get_user_author_name(user)
    # api_obj = open_api_object(config, model_id, details_filename=details_filename, verbose=config.verbose)
    max_input_tokens = config.get_model_max_input_tokens(model_id)
    max_output_tokens = config.get_model_max_output_tokens(model_id)

    corpus_train_prompt_list = []
    for narrative in corpus:
        # input_compose_filename = config.get_user_narrative_compose_scene_llm_handler_input_filename(user, narrative)
        # output_compose_filename = config.get_user_narrative_compose_scene_llm_handler_output_filename(user, narrative)
        input_train_filename = config.get_user_narrative_scenes_llm_preprocess_output_train_filename(user, narrative)
        input_eval_filename = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)
        # scene_compose_limit = config.get_user_narrative_compose_scene_llm_handler_scene_limit(user)
        # recent_event_count = config.get_user_narrative_compose_scene_llm_handler_recent_event_count(user)
        links_filename = config.get_user_narrative_compose_scene_llm_handler_links_filename(user, narrative)
        # request_log_file_template = config.get_user_narrative_compose_scene_request_log_file_template(user, narrative)
        verbose = config.verbose
        if verbose:
            print(f"links_filename {links_filename}")

        lnsc = LLMNarrativeScenesCompose(
            api_obj=api_obj,
            narrative=narrative,
            author_name=author_name,
            input_train_filename=input_train_filename,
            input_eval_filename=input_eval_filename,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            vector_db=vector_db,
            links_filename=links_filename,
            lookback_scene_limit_count=lookback_scene_limit_count,
            previous_narrative_fraction=0.5,
            generate_prompt_only=generate_prompt_only,
            verbose=verbose
        )
        corpus_train_prompt_list.extend(lnsc.get_fine_tune_narrative_prompt_response_list())

    api_obj.fine_tune_submit(corpus_train_prompt_list)
    print(api_obj.wait_fine_tuning_model(maxwait))

def module_corpus_llm_fine_tuning_check(config: Config, user: str) -> None:
    """
    Check the status of the fine-tuning process for the given user.

    This function allows checking the progress of an ongoing fine-tuning job
    and saves model details when complete.

    Parameters:
        config (Config): The configuration object with fine-tuning settings.
        user (str): The user/author identifier.

    Returns:
        None
    """
    model_id = config.get_user_fine_tune_model_id(user)
    fine_tune_filename = config.get_user_fine_tuned_filename(user)
    maxwait = config.get_user_fine_tune_maxwait(user)
    api_obj = open_api_object(config, model_id, details_filename=fine_tune_filename, verbose=config.verbose)
    if api_obj is None:
        print("API object is None")
        return
    print(api_obj.wait_fine_tuning_model(maxwait))

def module_narrative_scenes_build_test_set(config: Config, user: str, narrative: str) -> None:
    """
    Build the test set for the narrative scenes for the given user and narrative.

    This function creates a test dataset by selecting a subset of scenes and
    removing certain metadata to test the LLM's ability to reconstruct scenes.

    Parameters:
        config (Config): The configuration object with test set settings.
        user (str): The user/author identifier.
        narrative (str): The narrative identifier.

    Returns:
        None
    """
    input_compose_filename = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)

    output_compose_filename = config.get_user_narrative_scenes_build_test_set_output_filename(user, narrative)
    scene_limit = config.get_user_narrative_scenes_build_test_set_scene_limit(user)
    verbose = config.verbose

    llmnsp = LLMNarrativeScenesBuildTestCompose(
        input_eval_filename=input_compose_filename, # yes, input conpose treated like input eval
        output_compose_filename=output_compose_filename,
        scene_limit=scene_limit,
        verbose=config.verbose
    )
    llmnsp.build_test_compose_scene_input_file()
    print("Narrative scenes build test set complete")

def module_compose_scene_llm_narrative_handler(config: Config, vector_db: VectorDBMilvus, user: str, narrative: str) -> None:
    """
    Handle the composition of narrative scenes using the LLM and vector database.

    This function uses a fine-tuned LLM to compose new narrative scenes based on
    input specifications, retrieving relevant context from the vector database.

    Parameters:
        config (Config): The configuration object with composition settings.
        vector_db (VectorDBMilvus): The Milvus vector database object.
        user (str): The user/author identifier.
        narrative (str): The narrative identifier.

    Returns:
        None
    """
    details_filename = config.get_user_fine_tuned_filename(user)
    input_compose_filename = config.get_user_narrative_compose_scene_llm_handler_input_filename(user, narrative)
    output_compose_filename = config.get_user_narrative_compose_scene_llm_handler_output_filename(user, narrative)
    model_id = config.get_user_narrative_compose_scene_llm_handler_model_id(user)
    # fine_tune_model_name = config.get_user_fine_tune_model_name(user)
    author_name = config.get_user_author_name(user)
    lookback_scene_limit_count = config.get_user_narrative_compose_scene_llm_handler_lookback_scene_limit_count(user)
    api_obj = open_api_object(config, model_id, details_filename=details_filename, verbose=config.verbose)
    max_input_tokens = config.get_model_max_input_tokens(model_id)
    max_output_tokens = config.get_model_max_output_tokens(model_id)
    input_train_filename = config.get_user_narrative_scenes_llm_preprocess_output_train_filename(user, narrative)
    input_eval_filename = config.get_user_narrative_scenes_llm_preprocess_output_eval_filename(user, narrative)
    scene_compose_limit = config.get_user_narrative_compose_scene_llm_handler_scene_limit(user)
    # recent_event_count = config.get_user_narrative_compose_scene_llm_handler_recent_event_count(user)
    links_filename = config.get_user_narrative_compose_scene_llm_handler_links_filename(user, narrative)
    generate_prompt_only = config.get_user_narrative_compose_scene_llm_handler_generate_prompt_only(user)
    request_log_file_template = config.get_user_narrative_compose_scene_request_log_file_template(user, narrative)
    verbose = config.verbose
    if verbose:
        print(f"links_filename {links_filename}")

    llmnsp = LLMNarrativeScenesCompose(
        api_obj=api_obj,
        narrative=narrative,
        author_name=author_name,
        input_train_filename=input_train_filename,
        input_eval_filename=input_eval_filename,
        input_compose_filename=input_compose_filename,
        output_compose_filename=output_compose_filename,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        vector_db=vector_db,
        scene_compose_limit=scene_compose_limit,
        links_filename=links_filename,
        lookback_scene_limit_count=lookback_scene_limit_count,
        previous_narrative_fraction=0.5,
        generate_prompt_only=generate_prompt_only,
        request_log_file_template=request_log_file_template,
        verbose=verbose
    )
    llmnsp.write_scenes()
    print("Narrative scenes compose complete")

def main() -> None:
    """
    Main function to run the narrative processing pipeline.

    This function handles command-line arguments, configuration loading, and the
    execution of various modules in the pipeline. It orchestrates the entire process
    by calling the appropriate functions based on the configuration settings.

    The pipeline includes:
    - Preprocessing narratives
    - Fine-tuning LLMs
    - Processing narrative scenes
    - Building vector databases
    - Composing new narrative scenes

    The function measures the duration of operations for each user and narrative.

    Parameters:
        None

    Returns:
        None
    """
    start = time.time()

    print("Done with imports. Running...")

    random.seed(17)

    parser = argparse.ArgumentParser(description="Pipeline for determining optimal satellite imagery resolution.")

    parser.add_argument("config_filename", type=str, help="The path to the configuration file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Specify verbosity", default=False)

    args = parser.parse_args()

    # config.args = args
    # config_filename = args.config_filename
    config = Config(args)

    vector_db_filename = config.get_vector_db_filename()
    if vector_db_filename is None or not os.path.exists(vector_db_filename):
        vector_db = None
    else:
        vector_db = VectorDBMilvus(vector_db_filename)
        if vector_db.open() is None:
            print("vector_db file failed to open")
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

        # This step processes the narrative metadata and saves it to a file
        # for narrative in corpus:
        #     if config.run_narrative_metadata_process():
        #         module_narrative_metadata_process(config, user, narrative)

        # The vector database is used to store the metadata for each scene in the narrative
        # This step must be run after the LLM preprocessing step,
        # and is a prerequisite for the scene compose step
        for narrative in corpus:
            if config.run_narrative_into_vector_db():
                if vector_db_filename is None:
                    raise Exception(ValueError, "vector_db_name not specified in configuration file")
                if vector_db is None:
                    vector_db = VectorDBMilvus(vector_db_filename)
                    if vector_db.open() is None:
                        raise Exception(ValueError, "vector_db file failed to open")
                module_narrative_into_vector_db(config, vector_db, user, narrative)

        # The completion of the fine tuning step is a prerequisite for building the vector database and the scene compose step.
        # Note that this module "kicks off" the fine tuning process only waits for the configured amount of time
        # The fine tuning process is run in the background and will take some time to complete
        # Therefore, the fine tuning process may not be complete after this module is run
        # The scene compose step will fail if the fine tuning process is not complete
        if config.run_corpus_llm_fine_tuning():
            module_corpus_llm_fine_tuning(config, user, corpus, vector_db)

        # The fine tuning check step is a way for the user to check the status of the fine tuning process, if it did not complete
        # in the time specified in the fine tuning step
        # This module will also dump the details of the fine tuning process to a file
        if config.check_corpus_llm_fine_tuning():
            module_corpus_llm_fine_tuning_check(config, user)

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
