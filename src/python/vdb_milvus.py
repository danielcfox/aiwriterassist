#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 05:27:26 2025

@author: dfox
"""
import numpy as np
import os
import pymilvus
from pymilvus import MilvusClient
from pymilvus import model
from tqdm import tqdm

class VectorDBMilvus:
    """
    A vector database interface for Milvus to store and retrieve document embeddings.

    This class provides methods to create collections, insert document embeddings,
    and perform semantic searches within the Milvus vector database. It handles the
    vector embedding process and maintains collection organization for narrative elements.

    The class supports multiple document types within collections and ensures that
    document IDs remain unique across insertions.
    """
    def __init__(self, uri: str, embedding_function=pymilvus.model.DefaultEmbeddingFunction()) -> None:
        """
        Initialize a connection to the Milvus vector database.

        Sets up a client connection to the specified Milvus server and configures
        the embedding function to use for converting text to vectors.

        Parameters:
            uri (str): Connection string for the Milvus server
            embedding_function: Function to convert text to vector embeddings
                               (defaults to DefaultEmbeddingFunction)
        """
        self.uri = uri
        # Initialize Milvus client
        self.client = MilvusClient(uri)
        # self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.collection_names = []

        # Drop collection if it exists
        # if self.client.has_collection(collection_name=self.collection_name):
        #     self.client.drop_collection(collection_name=self.collection_name)

        # Create collection
        # self.client.create_collection(
        #     collection_name=self.collection_name, dimension=self.embedding_function.dim
        # )

    def create_collection(self, collection_name: str) -> None:
        """
        Create a new collection in the Milvus database if it doesn't exist.

        Creates a vector collection with dimensions matching the embedding function,
        skipping creation if the collection already exists.

        Parameters:
            collection_name (str): Name of the collection to create
        """
        if self.client.has_collection(collection_name=collection_name):
            print(f"Collection {collection_name} already exists.")
            return

        # Create collection
        self.client.create_collection(
            collection_name=collection_name, dimension=self.embedding_function.dim
        )
        print(f"Collection {collection_name} created.")
        # self.collection_names.append(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """
        Remove a collection from the Milvus database if it exists.

        Deletes the specified collection and all its contents from the database,
        with no effect if the collection doesn't exist.

        Parameters:
            collection_name (str): Name of the collection to delete
        """
        if not self.client.has_collection(collection_name=collection_name):
            print(f"Collection {collection_name} does not exist.")
            return

        # Drop collection
        self.client.drop_collection(collection_name=collection_name)
        print(f"Collection {collection_name} deleted.")
        # self.collection_names.remove(collection_name)

    def insert_documents(self, collection_name: str, doc_types: dict) -> dict:
        """
        Convert documents to vector embeddings and insert them into a collection.

        Processes multiple document types, converts their text to vector embeddings,
        and inserts them into the specified collection with metadata including
        document type and unique IDs.

        Parameters:
            collection_name (str): Name of the collection to insert documents into
            doc_types (dict): Dictionary mapping document types to lists of document dicts,
                             where each document contains at minimum 'id' and 'text' fields

        Returns:
            dict: Result information from the Milvus insert operation

        Raises:
            ValueError: If vector dimensions don't match or if duplicate IDs are found
        """
        data = []
        test_id_dict = {}

        for doc_type, doc_list in doc_types.items():
            docs = []
            for doc_dict in doc_list:
                docs.append(doc_dict['text'])
            print(f"encoding {len(docs)} documents of type {doc_type}")
            for i in tqdm(range(len(docs)), desc=f"Encoding {doc_type} documents"):
                # print(f"Encoding document {i+1}/{len(docs)}")
                vectors = self.embedding_function.encode_documents([docs[i]])
                # Ensure the vector is a list or numpy array
                if not isinstance(vectors[0], (list, np.ndarray)):
                    raise ValueError("The encoded vector must be a list or numpy array.")
                doc_dict = doc_list[i]
                vector = vectors[0]
                if len(vector) != self.embedding_function.dim:
                    raise ValueError(
                        f"The encoded vector (dimension {len(vectors[0])}) must have a dimension of {self.embedding_function.dim}."
                    )
                id = doc_dict['id']
                # Ensure the document ID is unique
                if id in test_id_dict:
                    raise ValueError(f"Duplicate document ID found: {id}")
                test_id_dict[id] = True
                name = doc_dict.get("name", "")
                data.append({"id": id, "vector": vector, "text": docs[i], "entity_name": name, "subject": doc_type})
                # what are the other fields?
                # "title": doc_dict.get("title", ""),
                # "author": doc_dict.get("author", ""),
                # "date": doc_dict.get("date", ""),
                # "source": doc_dict.get("source", ""),
                # "summary": doc_dict.get("summary", ""),
                # "keywords": doc_dict.get("keywords", ""),
                # "tags": doc_dict.get("tags", ""),
                # "url": doc_dict.get("url", ""),
                # "image": doc_dict.get("image", ""),
                # "video": doc_dict.get("video", ""),
                # "audio": doc_dict.get("audio", ""),
                # "location": doc_dict.get("location", ""),
                # "coordinates": doc_dict.get("coordinates", ""),
                # "language": doc_dict.get("language", ""),
                # "category": doc_dict.get("category", ""),
                # "subcategory": doc_dict.get("subcategory", ""),
                # "type": doc_dict.get("type", ""),
                # "format": doc_dict.get("format", ""),
                # "size": doc_dict.get("size", ""),
                # "length": doc_dict.get("length", ""),
                # "width": doc_dict.get("width", ""),
                # "height": doc_dict.get("height", ""),
                # "duration": doc_dict.get("duration", ""),
                # "bitrate": doc_dict.get("bitrate", ""),
                # "resolution": doc_dict.get("resolution", ""),
                # "aspect_ratio": doc_dict.get("aspect_ratio", ""),
                # "frame_rate": doc_dict.get("frame_rate", ""),
                # "codec": doc_dict.get("codec", ""),
                # "compression": doc_dict.get("compression", ""),
                # "encryption": doc_dict.get("encryption", ""),
                # "watermark": doc_dict.get("watermark", ""),
                # "license": doc_dict.get("license", ""),

        # Insert data into the collection
        insert_res = self.client.insert(collection_name=collection_name, data=data)
        print(f"Inserted {len(data)} documents into the collection.")
        return insert_res

    def search(self, collection_name: str, prev_chron_scene_index: int, query_texts: list, limit: int = 3) -> list:
        """
        Search for documents similar to the query texts, respecting chronological order.

        Performs a semantic similarity search for the provided query texts, only
        returning documents with IDs lower than a threshold based on the previous
        chronological scene index (to avoid future information in searches).

        Parameters:
            collection_name (str): Name of the collection to search
            prev_chron_scene_index (int): Maximum scene index to include in results
            query_texts (list): List of text strings to search for
            limit (int): Maximum number of results to return per query (default: 3)

        Returns:
            list: Search results with document matches and similarity scores

        Raises:
            ValueError: If search results are not in the expected format
        """
        query_vectors = self.embedding_function.encode_queries(query_texts)
        prev_id = (prev_chron_scene_index+1) * 1000
        results = self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            filter=(f"id < {prev_id}"),
            limit=limit,
            output_fields=["text", "subject", "entity_name"]
        )
        if not isinstance(results, list):
            raise ValueError("The search results must be a list.")
        return results
