# aiwriterassist
AI Assistant for Authors

The current implementation is a **pipeline-based system**.  
In progress is a transition to an **Agentic AI architecture** that will be more flexible, resilient, and scalable.

## Architecture Diagram

The diagram below shows the architecture currently being developed.

Some may argue this is not a “fully agentic” system, since the **Orchestrator Agent** is implemented as a **deterministic state machine**. However, the system as a whole is **non-deterministic** because it delegates key reasoning and generation steps to LLMs.  

This design choice is deliberate:  
- It keeps orchestration logic **transparent, testable, and debuggable**.  
- It avoids unnecessary complexity at this stage.  
- It ensures that as long as requirements are clear, the system executes predictably while still benefiting from the creative variability of LLMs.  

In the future, if we add a **chatbot-style user interface** where the user can request arbitrary tasks, the Orchestrator Agent itself may evolve into a **non-deterministic, LLM-driven planner**. For now, a deterministic orchestrator offers the right balance between **simplicity** and **scalability**.

```mermaid
flowchart TD

    %% ===== Frontend =====
    subgraph Frontend
        F[frontend_client]
    end

    %% ===== Orchestrator =====
    subgraph Orchestrator
        O[orchestrator_agent]
    end

    %% ===== AAA (future) =====
    subgraph AAA
        AAA_TOOL[aaa_policy_tool]
    end

    %% ===== Summary lane =====
    subgraph Summary_Cluster
        SA[summary_agent] --> SSA[scene_summary_agent]
        SSA --> LPM1[llm_policy_tool]
        LPM1 --> LT1[llm_tool]
        SA --> MST1[metadata_storage_tool]
    end

    %% ===== Entity lane =====
    subgraph Entity_Cluster
        EA[entity_recognition_agent] --> SERA[scene_entity_recognition_agent]
        SERA --> LPM2[llm_policy_tool]
        LPM2 --> LT2[llm_tool]
        EA --> MST2[metadata_storage_tool]
    end

    %% ===== Compose lane =====
    subgraph Compose_Cluster
        CA[compose_agent]
        CA --> LPM3[llm_policy_tool]
        LPM3 --> LT3[llm_tool]
    end

    %% ===== Fine-tuning lane =====
    subgraph FineTuning_Cluster
        FTA[fine_tuning_agent]
        FTA --> Split[train_test_dataset_split_tool]
        FTA --> Kickoff[kickoff_fine_tuning_task_tool]
        FTA --> Status[status_fine_tuning_task_tool]
        FTA --> Wait[wait_fine_tuning_task_tool]
        Kickoff --> LPM4[llm_policy_tool] --> LT4[llm_tool]
        Status  --> LPM5[llm_policy_tool] --> LT5[llm_tool]
        Wait    --> LPM6[llm_policy_tool] --> LT6[llm_tool]
    end

    %% ===== RAG lane =====
    subgraph RAG_Cluster
        RAG[rag_agent]
        RAG --> AddDoc[add_doc_tool]
        RAG --> Retrieve[retrieve_docs_tool]
        RAG --> VecDB[vector_db_tool]
        RAG --> GraphDB[graph_db_tool]
    end

    %% ===== Format lane =====
    subgraph Format_Cluster
        BFT[book_format_tool]
        BFT --> PDF[book_pdf_format_tool]
        BFT --> DOCX[book_docx_format_tool]
        BFT --> EPUB[book_epub_format_tool]
        BFT --> TXT[book_txt_format_tool]
    end

    %% ===== Wiring =====
    F --> O
    O --> AAA_TOOL
    O --> SA
    O --> EA
    O --> CA
    O --> FTA
    O --> RAG
    O --> BFT
