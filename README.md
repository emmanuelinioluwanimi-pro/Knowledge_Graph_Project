# Danfoss Assembly Knowledge Graph

A project to build a **Knowledge Graph** from Danfoss assembly manuals (text + images) and enable intelligent question-answering using hybrid Retrieval-Augmented Generation (RAG).

## 🎯 Project Goal

Transform Danfoss assembly manuals into a structured **Knowledge Graph** (using Neo4j) that captures parts, steps, tools, warnings, and visual information. Then build a hybrid RAG system (Vector Search + Graph Traversal) that allows users to ask natural questions like:

- "What tools do I need for Step 5?"
- "Show me the warning for tightening the screws."

---

## 🧩 Work Split

| Role              | Responsibilities |
|-------------------|------------------|
| **Inioluwa** | Data collection, PDF processing, VLM extraction, Ontology definition, Knowledge Graph population (Neo4j) |
| **Gargi**      | Vector database, Embeddings, Retrieval logic, Hybrid RAG implementation |
| **Both**          | Final integration, Streamlit UI, Testing & Iteration |

---

## 📋 Current Status

- We are starting with **one shared manual** (EZ Clip to 5400 Series).
- After completing one manual successfully, we will scale to the rest.
- **3 new assembly manuals** are available in the `manuals/` folder.

---

## 🗂 Project Structure (Current)

```bash
Knowledge_Graph_Project/
├── README.md
├── requirements.txt
├── main.py
├── manuals/                  # Raw PDFs
│   ├── DanfossReact_RA_click.pdf
│   ├── EZ_Clip_to_5400_Series.pdf
│   ├── Filter_drier_shell.pdf
│   └── ...
├── src
│   ├── kg               
│   │   ├── pdf_extractor.py
│   │   ├──vlm_prompts.py
│   │   ├──neo4j_loader.py
│   └── raw...
└── config.py                     
