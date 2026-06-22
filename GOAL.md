# GOAL.md

# VisionWorkbench

## One Sentence

Build the best lightweight AI Vision R&D Workbench for students, makers, competition teams and small research groups.

VisionWorkbench should manage the entire computer vision workflow:

Dataset → Training → Experiment → Model → Evaluation → Demo

instead of only running a YOLO training command.

---

# Background

Current computer vision workflows are fragmented.

A typical student project often uses:

* YOLO for training
* LabelImg / CVAT for annotation
* Excel for experiment records
* Local folders for model storage
* PPT screenshots for reporting

This causes several problems:

* datasets become disorganized
* experiments cannot be reproduced
* model versions are lost
* training history disappears
* failure analysis is difficult
* collaboration becomes messy

VisionWorkbench aims to solve these problems with a lightweight local-first approach.

---

# Target Users

Primary Users:

* university students
* competition teams
* makers
* OpenMV users
* OpenART users
* YOLO users
* undergraduate researchers

Secondary Users:

* small labs
* startup prototype teams
* hobbyist AI developers

Non-target Users:

* enterprise MLOps teams
* Kubernetes users
* large-scale cloud platforms

---

# Product Principles

## Principle 1

Local First

The system must work completely on a personal computer.

Internet access should be optional.

---

## Principle 2

Student Friendly

A freshman should be able to understand the workflow.

Avoid enterprise complexity.

Avoid DevOps-heavy architecture.

---

## Principle 3

AI Native

Claude

Codex

Cursor

future AI agents

should become first-class citizens.

The system should be designed assuming AI assistance exists.

---

## Principle 4

Engineering Over Papers

The goal is not publishing papers.

The goal is improving productivity.

Focus on:

* reproducibility
* organization
* experimentation
* evaluation
* reporting

---

## Principle 5

Incremental Development

Every version must be usable.

Never build a giant architecture before proving value.

---

# Core Vision

VisionWorkbench is NOT:

* another YOLO fork
* another annotation platform
* another enterprise MLOps system
* another cloud training service

VisionWorkbench IS:

A personal AI vision operating system.

---

# Core Modules

## Dataset Center

Responsibilities:

* dataset registration
* dataset metadata
* class statistics
* image statistics
* dataset validation
* dataset version tracking

Questions it should answer:

* What datasets do I have?
* What classes exist?
* Are labels broken?
* Which dataset trained this model?

---

## Training Center

Responsibilities:

* launch training
* record training parameters
* manage training tasks
* collect logs
* track status

Questions it should answer:

* What is currently training?
* What failed?
* Why did it fail?
* What parameters were used?

---

## Experiment Center

Responsibilities:

* experiment records
* experiment comparison
* experiment notes
* experiment conclusions

Questions it should answer:

* What changed?
* Did it improve?
* Which experiment was best?

---

## Model Registry

Responsibilities:

* model versioning
* metadata tracking
* model comparison
* model lineage

Questions it should answer:

* Where did this model come from?
* Which dataset produced it?
* Which experiment produced it?
* Which model should I deploy?

---

## Evaluation Center

Responsibilities:

* image evaluation
* video evaluation
* batch evaluation
* metric comparison
* error analysis

Questions it should answer:

* What mistakes does the model make?
* Which version performs best?
* What should be improved?

---

## Demo Center

Responsibilities:

* image demo
* video demo
* webcam demo
* model selection

Questions it should answer:

* Can I quickly demonstrate my work?

---

# Future AI Agent Layer

Future versions should support:

Dataset Agent

Training Agent

Experiment Agent

Evaluation Agent

Documentation Agent

Presentation Agent

Potential capabilities:

* detect annotation problems
* analyze failed training
* compare experiments
* generate reports
* generate PPT outlines
* generate technical documentation
* suggest next experiments

---

# Technical Constraints

Preferred Stack:

* Python
* Ultralytics YOLO
* PyTorch
* OpenCV
* Gradio
* Streamlit
* SQLite

Avoid:

* Kubernetes
* microservices
* distributed systems
* cloud-only architecture
* enterprise DevOps complexity

---

# MVP Scope

Version 0.1

Goal:

Single-model image inference.

Requirements:

* load YOLO model
* image inference
* save results

---

Version 0.2

Goal:

Dataset validation.

Requirements:

* validate dataset structure
* validate labels
* generate reports

---

Version 0.3

Goal:

Training task tracking.

Requirements:

* launch training
* task status
* log storage

---

Version 0.4

Goal:

Model registry.

Requirements:

* model versions
* metadata
* model selection

---

Version 0.5

Goal:

Experiment comparison.

Requirements:

* metrics parsing
* visualization
* comparison

---

Version 1.0

Goal:

Complete personal vision workbench.

Requirements:

* all core modules connected
* local deployment
* usable by students
* documented
* reproducible

---

# Success Criteria

A user should be able to answer:

1. Which dataset trained this model?
2. Which parameters were used?
3. Which experiment improved performance?
4. Which model should be deployed?
5. What mistakes does the model still make?
6. How do I demonstrate results quickly?

If VisionWorkbench can answer these questions clearly, the project succeeds.

---

# Development Rule

Before writing code:

Research existing repositories.

Before adding features:

Ask whether the feature helps students and makers.

Before increasing complexity:

Ask whether a freshman could still understand the project.

When uncertain:

Choose simplicity.
