#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Training State Management

@author: dfox
"""

import pickle
import os
from typing import Optional, Any, Union
from datetime import datetime

class LLMAPITrainingState:
    """
    Generic training state management for LLM fine-tuning jobs.

    This class provides a unified interface for tracking fine-tuning job state
    across different APIs (OpenAI, Vertex AI, etc.). It stores generic state
    information and can be persisted to/from pickle files.

    The class is designed to be API-agnostic, with specific handlers accessing
    the state variables directly as needed.

    Attributes:
        job_id (str): Unique identifier for the training job
        base_model_name (str): Name of the base model being fine-tuned
        training_data_id (str): Identifier for training data (file ID or filename)
        status (str): Current status of the training job
        fine_tuned_model_name (Optional[str]): Name of resulting fine-tuned model
        created_at (datetime): When the job was created
        updated_at (datetime): When the state was last updated
        completed_at (Optional[datetime]): When the job completed
        error_message (Optional[str]): Error message if job failed
        training_job_object (Any): Raw training job object from API
        api_type (str): Type of API used ('openai', 'vertexai', etc.)
    """

    def __init__(self):
        """Initialize empty training state."""
        self.job_id: Optional[str] = None
        self.base_model_name: Optional[str] = None
        self.training_data_id: Optional[str] = None
        self.status: str = "unknown"
        self.fine_tuned_model_name: Optional[str] = None
        self.created_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.training_job_object: Optional[Any] = None
        self.api_type: Optional[str] = None

    def set(self, base_model_name: str, training_data_id: str, training_job_object: Any, api_type: str = "unknown") -> None:
        """
        Set the training state from API response objects.

        Parameters:
            base_model_name (str): Name of the base model being fine-tuned
            training_data_id (str): Identifier for training data (file ID or filename)
            training_job_object (Any): Training job object returned from API
            api_type (str): Type of API ('openai', 'vertexai', etc.)
        """
        self.base_model_name = base_model_name
        self.training_data_id = training_data_id
        self.training_job_object = training_job_object
        self.api_type = api_type.lower()
        self.updated_at = datetime.now()

        # Extract common fields based on API type
        if self.api_type == "openai":
            self._extract_openai_fields()
        elif self.api_type == "vertexai":
            self._extract_vertexai_fields()
        else:
            # Generic extraction - try common field names
            self._extract_generic_fields()

    def _extract_openai_fields(self) -> None:
        """Extract fields from OpenAI FineTuningJob object."""
        if self.training_job_object is None:
            return

        self.job_id = getattr(self.training_job_object, 'id', None)
        self.status = getattr(self.training_job_object, 'status', 'unknown')
        self.fine_tuned_model_name = getattr(self.training_job_object, 'fine_tuned_model', None)

        # Convert timestamps
        created_timestamp = getattr(self.training_job_object, 'created_at', None)
        if created_timestamp:
            self.created_at = datetime.fromtimestamp(created_timestamp)

        finished_timestamp = getattr(self.training_job_object, 'finished_at', None)
        if finished_timestamp:
            self.completed_at = datetime.fromtimestamp(finished_timestamp)

        # Extract error information
        error_obj = getattr(self.training_job_object, 'error', None)
        if error_obj:
            self.error_message = getattr(error_obj, 'message', str(error_obj))

    def _extract_vertexai_fields(self) -> None:
        """Extract fields from Vertex AI tuning job object."""
        if self.training_job_object is None:
            return

        self.job_id = getattr(self.training_job_object, 'resource_name', None)
        self.status = getattr(self.training_job_object, 'state', 'unknown')

        # Extract tuned model reference
        tuned_model = getattr(self.training_job_object, 'tuned_model', None)
        if tuned_model:
            self.fine_tuned_model_name = str(tuned_model)

        # Convert timestamps (Vertex AI uses different field names)
        create_time = getattr(self.training_job_object, 'create_time', None)
        if create_time:
            self.created_at = create_time

        end_time = getattr(self.training_job_object, 'end_time', None)
        if end_time:
            self.completed_at = end_time

        # Extract error information
        error_obj = getattr(self.training_job_object, 'error', None)
        if error_obj:
            self.error_message = getattr(error_obj, 'message', str(error_obj))

    def _extract_generic_fields(self) -> None:
        """Extract fields using common field names."""
        if self.training_job_object is None:
            return

        # Try common field names
        possible_id_fields = ['id', 'job_id', 'resource_name', 'name']
        for field in possible_id_fields:
            if hasattr(self.training_job_object, field):
                self.job_id = getattr(self.training_job_object, field)
                break

        possible_status_fields = ['status', 'state', 'job_status']
        for field in possible_status_fields:
            if hasattr(self.training_job_object, field):
                self.status = getattr(self.training_job_object, field)
                break

    def is_pending(self) -> bool:
        """
        Check if the training job is still in progress.

        Returns:
            bool: True if job is still running, False otherwise
        """
        pending_statuses = {
            # OpenAI statuses
            'validating_files', 'pending', 'running', 'queued',
            # Vertex AI statuses
            'JOB_STATE_PENDING', 'JOB_STATE_RUNNING', 'JOB_STATE_PREPARING',
            # Generic statuses
            'in_progress', 'training', 'processing'
        }
        return self.status.lower() in {s.lower() for s in pending_statuses}

    def is_completed(self) -> bool:
        """
        Check if the training job completed successfully.

        Returns:
            bool: True if job completed successfully, False otherwise
        """
        success_statuses = {'succeeded', 'JOB_STATE_SUCCEEDED', 'completed', 'success'}
        return self.status.lower() in {s.lower() for s in success_statuses}

    def is_failed(self) -> bool:
        """
        Check if the training job failed.

        Returns:
            bool: True if job failed, False otherwise
        """
        failed_statuses = {'failed', 'JOB_STATE_FAILED', 'cancelled', 'error'}
        return self.status.lower() in {s.lower() for s in failed_statuses}

    def save(self, filepath: str) -> None:
        """
        Save the training state to a pickle file.

        Parameters:
            filepath (str): Path to save the pickle file

        Raises:
            IOError: If file cannot be written
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            raise IOError(f"Failed to save training state to {filepath}: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'LLMAPITrainingState':
        """
        Load training state from a pickle file.

        Parameters:
            filepath (str): Path to the pickle file

        Returns:
            LLMAPITrainingState: Loaded training state object

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read or is corrupted
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training state file not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # Ensure it's the right type
            if not isinstance(state, cls):
                raise IOError(f"File {filepath} does not contain a valid LLMAPITrainingState object")

            # Update the state from the stored training job object if available
            if state.training_job_object is not None:
                state.set(
                    state.base_model_name,
                    state.training_data_id,
                    state.training_job_object,
                    state.api_type
                )

            return state

        except Exception as e:
            raise IOError(f"Failed to load training state from {filepath}: {e}")

    def update_from_api(self, updated_job_object: Any) -> None:
        """
        Update state from a refreshed API job object.

        Parameters:
            updated_job_object (Any): Updated job object from API
        """
        if updated_job_object is not None:
            self.training_job_object = updated_job_object
            self.set(self.base_model_name, self.training_data_id, updated_job_object, self.api_type)

    def __str__(self) -> str:
        """String representation of the training state."""
        return (f"LLMAPITrainingState(job_id={self.job_id}, "
                f"status={self.status}, "
                f"base_model={self.base_model_name}, "
                f"fine_tuned_model={self.fine_tuned_model_name})")

    def __repr__(self) -> str:
        """Detailed representation of the training state."""
        return (f"LLMAPITrainingState(job_id='{self.job_id}', "
                f"base_model_name='{self.base_model_name}', "
                f"training_data_id='{self.training_data_id}', "
                f"status='{self.status}', "
                f"api_type='{self.api_type}', "
                f"created_at={self.created_at}, "
                f"fine_tuned_model_name='{self.fine_tuned_model_name}')")