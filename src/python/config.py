#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:52:44 2025

@author: dfox
"""

import os
import yaml

class Config:
    """
    Configuration class to handle user-specific settings and model configurations.
    This class loads the configuration from a YAML file and provides methods to access
    various settings related to users, models, and narratives.
    """
    def __init__(self, args):
        """
        Initialize the Config class with the given configuration filename.
        :param config_filename: Path to the YAML configuration file.
        """
        with open(args.config_filename, "r") as fp:
            self.config = yaml.safe_load(fp)  # Use safe_load for security

        self.verbose = args.verbose # not used yet

    """Internal methods to access top-level configurations."""
    """They all return dictionaries of sub-parameters."""
    def _get_model_config(self, model_id):
        """Get the model configuration for the given model ID."""
        if 'models' not in self.config:
            raise ValueError("Model configuration not found in the global config.")
        if model_id not in self.config['models']:
            raise ValueError(f"Model ID {model_id} not found in the global config.")
        print(self.config['models'])
        return self.config['models'][model_id]

    def _get_user_config(self, user):
        return self.config['users'][user]

    def _get_user_fine_tune_config(self, user):
        """Get the fine-tune configuration for the user."""
        user_config = self._get_user_config(user)
        if 'fine_tune' not in user_config:
            raise ValueError(f"Fine-tune configuration not found for user {user}")
        return user_config['fine_tune']

    def _get_user_model_fine_tune_config(self, user):
        """Get the model fine-tune configuration for the user."""
        user_fine_tune_config = self._get_user_fine_tune_config(user)
        if 'model' not in user_fine_tune_config:
            raise ValueError(f"Model configuration not found for user {user}")
        model_name = user_fine_tune_config['model']
        return self._get_model_config(model_name)['fine_tune']

    def _get_user_model_fine_tune_ext(self, user):
        fine_tune_config = self._get_user_model_fine_tune_config(user)
        if 'ext' not in fine_tune_config:
            raise ValueError(f"Fine-tune extension not found for user {user}")
        return fine_tune_config['ext']

    def _get_narrative_config(self, narrative):
        return self.config[narrative]

    def _get_user_preprocess_config(self, user):
        """Get the preprocess configuration for the user."""
        if 'preprocess' not in self._get_user_config(user):
            raise ValueError(f"Preprocess configuration not found for user {user}")
        return self._get_user_config(user)['preprocess']

    def _get_user_narrative_scenes_llm_preprocess_config(self, user):
        return self._get_user_config(user)['narrative_scenes_llm_preprocess']

    def _get_user_narrative_into_vector_db_config(self, user):
        if 'narrative_into_vector_db' not in self._get_user_config(user):
            return None
        return self._get_user_config(user)['narrative_into_vector_db']

    def _get_user_narrative_scenes_build_test_set_config(self, user):
        if 'narrative_scenes_build_test_set' not in self._get_user_config(user):
            return None
        return self._get_user_config(user)['narrative_scenes_build_test_set']

    def _get_model_inference_config(self, model_id):
        """Get the inference configuration for the model."""
        model_config = self._get_model_config(model_id)
        if 'inference' not in model_config:
            raise ValueError(f"Inference configuration not found for model ID {model_id}")
        return model_config['inference']

    def _get_user_narrative_metadata_process_config(self, user):
        """Get the metadata process configuration for the narrative."""
        if 'narrative_metadata_process' not in self._get_user_config(user):
            raise ValueError(f"Metadata process configuration not found for user {user}")
        return self._get_user_config(user)['narrative_metadata_process']

    """End of internal methods"""

    def get_model_class(self, model_id):
        """Get the class name for the model with the given model ID."""
        print(self._get_model_config(model_id))
        if 'api_class' not in self._get_model_config(model_id):
            raise ValueError(f"API class not found for model ID {model_id}")
        return self._get_model_config(model_id)['api_class']

    def get_model_fine_tune_name(self, model_id):
        """Get the fine-tune name for the model with the given model ID."""
        if 'fine_tune' not in self._get_model_config(model_id):
            raise ValueError(f"Fine-tune name not found for model ID {model_id}")
        if 'name' not in self._get_model_config(model_id)['fine_tune']:
            raise ValueError(f"Fine-tune name not found for model ID {model_id}")
        return self._get_model_config(model_id)['fine_tune']['name']

    def get_user_preprocess_scene_limit(self, user):
        """Get the scene limit for the user's preprocess configuration."""
        if 'scene_limit_per_narrative' not in self._get_user_preprocess_config(user):
            raise ValueError(f"Scene limit not found in preprocess configuration for user {user}")
        return self._get_user_preprocess_config(user)['scene_limit_per_narrative']

    def get_vector_db_filename(self):
        """Get the vector database name from the global configuration."""
        if 'vector_db_filename' not in self.config:
            return None
        return self.config['vector_db_filename']

    def get_user_cwd(self, user):
        """Get the current working directory for the user."""
        user_config = self._get_user_config(user)
        if 'working_directory' in user_config:
            return user_config['working_directory']
        return os.getcwd()

    def get_user_list(self):
        """Get the list of users from the global configuration."""
        if 'users' not in self.config:
            return []
        return [user for user in self.config['users'] if user != 'default']

    def get_user_preprocess_fine_tune_train_split(self, user):
        """Get the training split for fine-tuning from the user's preprocess configuration."""
        user_config = self._get_user_preprocess_config(user)
        if 'train_split' not in user_config:
            return None
        return user_config['train_split']

    def get_model_inference_name(self, model_id):
        """Get the model inference name from the model configuration."""
        inference_config = self._get_model_inference_config(model_id)
        if 'name' not in inference_config:
            raise ValueError(f"Inference name not found for model ID {model_id}")
        return inference_config['name']

    def get_user_fine_tune_model_id(self, user):
        """Get the fine-tuned model ID from the user's fine-tune configuration."""
        return self._get_user_fine_tune_config(user).get('model', None)

    def get_user_fine_tune_maxwait(self, user):
        """Get the maximum wait time while the user's model is being fine-tuned.
        Note that after the time has expired, the model is not guaranteed to be ready.
        And then the user must check the status of the model periodically.
        """
        return self._get_user_fine_tune_config(user).get('maxwait', 0)

    def get_user_fine_tune_model_name(self, user):
        """Get the fine-tuned model name from the user's fine-tune configuration."""
        model_id = self.get_user_fine_tune_model_id(user)
        if model_id is None:
            raise ValueError(f"Model ID for fine tuning not found in config for user {user}")
        model_config = self._get_model_config(model_id)
        if model_config is None:
            raise ValueError(f"Model configuration not found for model ID {model_id}")
        if 'inference' not in model_config:
            raise ValueError(f"Inference configuration not found for model ID {model_id}")
        if 'name' not in model_config['inference']:
            raise ValueError(f"Model name not found in inference configuration for model ID {model_id}")
        return model_config['inference']['name']

    def get_user_summary_recent_scene_count(self, user):
        """Get the recent scene count from the user's fine-tune configuration."""
        config = self._get_user_fine_tune_config(user)
        if config is None:
            raise ValueError(f"Fine-tune configuration not found for user {user}")
        if 'recent_scene_count' not in config:
            return None
        return config['recent_scene_count']

    def get_user_author_name(self, user):
        """Get the author's name from the user's configuration."""
        if 'author_name' not in self._get_user_config(user):
            return user
        return self._get_user_config(user)['author_name']

    def _get_user_fine_tune_output_file_template(self, user):
        return self._get_user_fine_tune_config(user).get('output_file_template', None)

    def get_user_fine_tuned_filename(self, user):
        """Get the filename for the fine-tuned model."""
        fine_tune_ext = self._get_user_model_fine_tune_ext(user)
        if fine_tune_ext is None:
            raise ValueError(f"Fine-tune extension not found for user {user}")
        model_id = self.get_user_fine_tune_model_id(user)
        if model_id is None:
            raise ValueError(f"Model ID for fine tuning not found in config for user {user}")
        output_file_template = self._get_user_fine_tune_config(user).get('output_file_template', None)
        if output_file_template is None:
            raise ValueError(f"Output file template not found in fine-tune configuration for user {user}")
        basename = output_file_template.format(user=user, model=model_id, fine_tune_ext=fine_tune_ext)
        return os.path.join(self.get_user_cwd(user), basename)

    def get_model_max_input_tokens(self, model_id):
        """Get the maximum input tokens for the fine-tuned model."""
        if 'max_tokens' not in self.config['models'][model_id]:
            raise ValueError(f"Max tokens not found in model configuration for model ID {model_id}")
        if 'input' not in self.config['models'][model_id]['max_tokens']:
            raise ValueError(f"Input max tokens not found in model configuration for model ID {model_id}")
        return self.config['models'][model_id]['max_tokens']['input']

    def get_model_max_output_tokens(self, model_id):
        """Get the maximum output tokens for the fine-tuned model."""
        if 'max_tokens' not in self.config['models'][model_id]:
            raise ValueError(f"Max tokens not found in model configuration for model ID {model_id}")
        if 'output' not in self.config['models'][model_id]['max_tokens']:
            raise ValueError(f"Output max tokens not found in model configuration for model ID {model_id}")
        return self.config['models'][model_id]['max_tokens']['output']

    def get_user_preprocess_corpus(self, user):
        """Get the corpus narrative list from the user's preprocess configuration."""
        return self._get_user_preprocess_config(user)['corpus_narrative_list']

    def get_user_narratives_directory(self, user):
        """Get the directory for narratives from the user's configuration."""
        if 'narratives_directory' not in self._get_user_config(user):
            return self.get_user_cwd(user)
        return self._get_user_config(user)['narratives_directory']

    def get_narrative_class(self, narrative):
        """Get the narrative class from the narrative configuration."""
        if 'class' not in self._get_narrative_config(narrative):
            return None
        return self._get_narrative_config(narrative)['class']

    def get_user_narrative_input_file_list(self, user, narrative):
        """Get the list of input files for the narrative from the user's configuration."""
        input_files = self._get_narrative_config(narrative)['input_files']
        return [os.path.join(self.get_user_narratives_directory(user), filename) for filename in input_files]

    def get_user_narrative_preprocess_output_train_filename(self, user, narrative):
        """Get the output filename for the training set from the user's preprocess configuration."""
        output_basename_template = self._get_user_preprocess_config(user)['output_file_train_template']
        return os.path.join(self.get_user_cwd(user), output_basename_template.format(narrative=narrative))

    def get_user_narrative_preprocess_output_eval_filename(self, user, narrative):
        """Get the output filename for the evaluation set from the user's preprocess configuration."""
        output_basename_template = self._get_user_preprocess_config(user)['output_file_eval_template']
        return os.path.join(self.get_user_cwd(user), output_basename_template.format(narrative=narrative))

    def get_user_narrative_scenes_llm_preprocess_clean(self, user):
        """Get the clean flag from the user's narrative scenes LLM preprocess configuration."""
        config = self._get_user_narrative_scenes_llm_preprocess_config(user)
        # print(f"Config: {config}")
        if 'clean' not in config:
            return False
        return config['clean']

    def get_user_narrative_scenes_llm_preprocess_model_id(self, user):
        """Get the model ID from the user's narrative scenes LLM preprocess configuration."""
        return self._get_user_narrative_scenes_llm_preprocess_config(user)['model']

    def get_user_narrative_scenes_llm_preprocess_model_name(self, user):
        """Get the model name from the user's narrative scenes LLM preprocess configuration."""
        return self.config['models'][self.get_user_narrative_scenes_llm_preprocess_model_id(user)]['name']

    def get_user_narrative_scenes_llm_preprocess_output_train_filename(self, user, narrative):
        """Get the output filename for the training set from the user's narrative scenes LLM preprocess configuration."""
        output_basename_template = self._get_user_narrative_scenes_llm_preprocess_config(user)['output_file_train_template']
        return os.path.join(self.get_user_cwd(user), output_basename_template.format(narrative=narrative))

    def get_user_narrative_scenes_llm_preprocess_output_eval_filename(self, user, narrative):
        """Get the output filename for the evaluation set from the user's narrative scenes LLM preprocess configuration."""
        output_basename_template = self._get_user_narrative_scenes_llm_preprocess_config(user)['output_file_eval_template']
        return os.path.join(self.get_user_cwd(user), output_basename_template.format(narrative=narrative))

    def get_user_narrative_scenes_llm_preprocess_scene_limit_per_narrative(self, user):
        """Get the scene limit per narrative from the user's narrative scenes LLM preprocess configuration."""
        config = self._get_user_narrative_scenes_llm_preprocess_config(user)
        if 'scene_limit_per_narrative' not in config:
            return None
        return config['scene_limit_per_narrative']

    def get_user_narrative_scenes_llm_preprocess_scene_limit_per_narrative(self, user):
        """Get the scene limit per narrative from the user's narrative scenes LLM preprocess configuration."""
        config = self._get_user_narrative_scenes_llm_preprocess_config(user)
        if 'scene_limit_per_narrative' not in config:
            return None
        return config['scene_limit_per_narrative']

    def get_user_narrative_into_vector_db_clean(self, user):
        """Get the clean flag from the user's narrative into vector DB configuration."""
        config = self._get_user_narrative_into_vector_db_config(user)
        if 'clean' not in config:
            return False
        return config['clean']

    def get_user_narrative_scenes_build_test_set_scene_limit(self, user):
        """Get the scene limit from the user's narrative scenes build test set configuration."""
        config = self._get_user_narrative_scenes_build_test_set_config(user)
        if 'scene_limit_per_narrative' not in config:
            return None
        return config['scene_limit_per_narrative']

    def get_user_narrative_compose_scene_llm_handler_config(self, user):
        """Get the configuration for the narrative compose scene LLM handler."""
        if 'compose_scene_llm_handler' not in self._get_user_config(user):
            return None
        return self._get_user_config(user)['compose_scene_llm_handler']

    def get_user_narrative_compose_scene_llm_handler_input_directory(self, user):
        """Get the input directory for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'input_directory' not in config:
            return self.get_user_cwd(user)
        return config['input_directory']

    def get_user_narrative_compose_scene_llm_handler_output_directory(self, user):
        """Get the output directory for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'output_directory' not in config:
            return self.get_user_cwd(user)
        return config['output_directory']

    def get_user_narrative_compose_scene_llm_handler_input_filename(self, user, narrative):
        """Get the input filename for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'input_file_compose_template' not in config:
            raise ValueError(f"Input file compose template not found in config for user {user}")
        input_basename_template = config['input_file_compose_template']
        input_directory = self.get_user_narrative_compose_scene_llm_handler_input_directory(user)
        if not os.path.exists(input_directory):
            raise ValueError(f"Input directory {input_directory} does not exist for user {user}")
        if not os.path.isdir(input_directory):
            raise ValueError(f"Input directory {input_directory} is not a directory for user {user}")
        return os.path.join(input_directory, input_basename_template.format(user=user, narrative=narrative))

    def get_user_narrative_compose_scene_llm_handler_output_filename(self, user, narrative):
        """Get the output filename for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'output_file_compose_template' not in config:
            raise ValueError(f"Output file compose template not found in config for user {user}")
        output_directory = self.get_user_narrative_compose_scene_llm_handler_output_directory(user)
        if not os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory} does not exist for user {user}")
        if not os.path.isdir(output_directory):
            raise ValueError(f"Output directory {output_directory} is not a directory for user {user}")
        output_basename_template = config['output_file_compose_template']
        return os.path.join(output_directory, output_basename_template.format(user=user, narrative=narrative))

    def get_user_narrative_compose_scene_llm_handler_model_id(self, user):
        """Get the model name for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'model' not in config:
            raise ValueError(f"Model not found in config for user {user}")
        return config['model']
        # return self.config['models'][model_id]['name']

    def get_user_narrative_compose_scene_llm_handler_test(self, user):
        """Get the test flag for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'test' not in config:
            return False
        return config['test']

    def get_user_narrative_compose_scene_llm_handler_scene_limit(self, user):
        """Get the scene limit for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'scene_limit_per_narrative' not in config:
            raise ValueError(f"Scene limit not found in config for user {user}")
        return config['scene_limit_per_narrative']

    def get_user_narrative_compose_scene_llm_handler_recent_event_count(self, user):
        """Get the recent event count (number of previous scene summaries to include in the compose scene prompt)
        for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'recent_event_count' not in config:
            return None
        return config['recent_event_count']

    def get_user_narrative_compose_scene_llm_handler_links_filename(self, user, narrative):
        """Get the links filename for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if self.verbose:
            print(f"Compose config: {config}")
        if 'links_file_template' not in config:
            return None
        links_basename_template = config['links_file_template']
        links_directory = self.get_user_narratives_directory(user)
        if not os.path.exists(links_directory):
            raise ValueError(f"Links directory {links_directory} does not exist for user {user}")
        if not os.path.isdir(links_directory):
            raise ValueError(f"Links directory {links_directory} is not a directory for user {user}")
        return os.path.join(links_directory, links_basename_template.format(narrative=narrative))

    def get_user_narrative_compose_scene_llm_handler_generate_prompt_only(self, user):
        """Get the generate prompt only flag for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if self.verbose:
            print(f"Compose config: {config}")
        if 'generate_prompt_only' not in config:
            return False
        return config['generate_prompt_only']

    def get_user_narrative_compose_scene_request_log_file_template(self, user, narrative):
        """Get the request log filename for the narrative compose scene LLM handler."""
        config = self.get_user_narrative_compose_scene_llm_handler_config(user)
        if 'request_log_file_template' not in config:
            return None
        request_log_basename_template = config['request_log_file_template']
        request_log_directory = self.get_user_narratives_directory(user)
        if not os.path.exists(request_log_directory):
            raise ValueError(f"Request log directory {request_log_directory} does not exist for user {user}")
        if not os.path.isdir(request_log_directory):
            raise ValueError(f"Request log directory {request_log_directory} is not a directory for user {user}")
        return os.path.join(
            request_log_directory, request_log_basename_template.format(narrative=narrative, chron_scene_index='chron_scene_index')
        )

    def get_user_narrative_scenes_build_test_set_input_directory(self, user):
        """Get the input directory for the narrative scenes build test set."""
        config = self.get_user_narrative_scenes_build_test_set_config(user)
        if 'input_directory' not in config:
            return self.get_user_cwd(user)
        return config['input_directory']

    def get_user_narrative_scenes_build_test_set_output_directory(self, user):
        """Get the output directory for the narrative scenes build test set."""
        config = self._get_user_narrative_scenes_build_test_set_config(user)
        if 'output_directory' not in config:
            return self.get_user_cwd(user)
        return config['output_directory']

    def get_user_narrative_scenes_build_test_set_output_filename(self, user, narrative):
        """Get the output filename for the narrative scenes build test set."""
        config = self._get_user_narrative_scenes_build_test_set_config(user)
        if 'output_file_template' not in config:
            raise ValueError(f"Output file template not found in config for user {user}")
        output_directory = self.get_user_narrative_scenes_build_test_set_output_directory(user)
        if not os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory} does not exist for user {user}")
        if not os.path.isdir(output_directory):
            raise ValueError(f"Output directory {output_directory} is not a directory for user {user}")
        output_basename_template = config['output_file_template']
        return os.path.join(output_directory, output_basename_template.format(user=user, narrative=narrative))

    def get_user_narrative_metadata_process_output_filename(self, user, narrative):
        """Get the output filename for the narrative metadata process."""
        config = self._get_user_narrative_metadata_process_config(user)
        if 'output_file_template' not in config:
            raise ValueError(f"Output file template not found in config for user {user}")
        output_directory = self.get_user_cwd(user)
        if not os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory} does not exist for user {user}")
        if not os.path.isdir(output_directory):
            raise ValueError(f"Output directory {output_directory} is not a directory for user {user}")
        output_basename_template = config['output_file_template']
        return os.path.join(output_directory, output_basename_template.format(user=user, narrative=narrative))

    def run_narrative_preprocess(self):
        """Check if narrative preprocessing should be run."""
        if 'run_narrative_preprocess' not in self.config:
            return False
        return self.config['run_narrative_preprocess']

    def run_corpus_llm_fine_tuning(self):
        """Check if corpus LLM fine-tuning should be run."""
        if 'run_corpus_llm_fine_tuning' not in self.config:
            return False
        return self.config['run_corpus_llm_fine_tuning']

    def check_corpus_llm_fine_tuning(self):
        """Check if corpus LLM fine-tuning should be checked."""
        if 'check_corpus_llm_fine_tuning' not in self.config:
            return False
        return self.config['check_corpus_llm_fine_tuning']

    def run_narrative_scenes_llm_preprocess(self):
        """Check if narrative scenes LLM preprocessing should be run."""
        if 'run_narrative_scenes_llm_preprocess' not in self.config:
            return False
        return self.config['run_narrative_scenes_llm_preprocess']

    def run_narrative_metadata_process(self):
        """Check if narrative metadata processing should be run."""
        if 'run_narrative_metadata_process' not in self.config:
            return False
        return self.config['run_narrative_metadata_process']

    def run_narrative_into_vector_db(self):
        """Check if narrative should be placed into the vector DB."""
        if 'run_narrative_into_vector_db' not in self.config:
            return False
        return self.config['run_narrative_into_vector_db']

    def run_compose_scene_llm_narrative_handler(self):
        """Check if the narrative compose scene LLM handler should be run."""
        if 'run_compose_scenes' not in self.config:
            return False
        return self.config['run_compose_scenes']

    def run_narrative_scenes_build_test_set(self):
        """Check if the narrative scenes build test set should be run."""
        if 'run_narrative_scenes_build_test_set' not in self.config:
            return False
        return self.config['run_narrative_scenes_build_test_set']
