# AI Declaration

I used AI-assisted tools during the development and documentation of this project, mainly as a support aid for code review, debugging, explanation, and drafting. The core project decisions, implementation choices, testing, validation, and final submission remain my own responsibility.

AI assistance was used in the following limited ways:

- To help review and improve the clarity of Python code, configuration files, and documentation.
- To suggest possible fixes for errors encountered while building the DVC pipeline, Docker Compose stack, FastAPI services, and MLflow/Airflow workflow.
- To help structure written material such as the project report, runbook, test documentation, and explanations of the MLOps architecture.
- To generate alternative wording for technical descriptions, which I then checked and edited to match the actual project implementation.
- To support troubleshooting by explaining error messages, command outputs, and possible causes during development.
- To provide occasional code completion suggestions from comments or partially written code, especially for repetitive helper logic, docstrings, and small validation checks.

AI was not used as a substitute for understanding the project or for blindly generating the final solution. I reviewed and adapted all AI-assisted suggestions before including them. The model architecture, data processing workflow, training and evaluation pipeline, API behavior, deployment setup, and monitoring design were checked against the working repository and executed outputs.

The project uses a reproducible DVC pipeline for Galaxy Zoo image preprocessing, PyTorch-based galaxy morphology classification, MLflow experiment tracking and registry management, Airflow orchestration, FastAPI/Streamlit application services, Docker Compose deployment, and monitoring/logging components. Any AI-assisted text or code changes were validated against these project components to ensure they were consistent with the actual system.

## Example Prompts and AI-Assisted Tasks

The following are representative examples of prompts or comment-based instructions used during development. They are included to show the type of AI assistance used, not as a complete transcript.

| Area | Example prompt or comment | How it was used |
| --- | --- | --- |
| Documentation | "Improve this project report paragraph so it clearly explains the DVC, MLflow, and Airflow workflow." | Used to refine wording after the technical workflow had already been implemented. |
| Debugging | "This Docker Compose service is not connecting to MLflow. What configuration values should I check?" | Used to identify likely environment variables, ports, service names, and network settings to inspect manually. |
| Code review | "Review this FastAPI endpoint for possible validation or error-handling issues." | Used as a checklist for improving API behavior before testing the endpoint. |
| Testing | "Suggest unit test cases for preprocessing image splits and schema validation." | Used to think through missing test cases and edge cases. |
| Error explanation | "Explain this DVC error and what it means for pipeline dependencies." | Used to understand command output before deciding the correction. |

I also used Codex-style autocomplete in some places where comments or partially written code described the intended logic. For example:

```python
# Validate that the uploaded file extension is supported
```

This kind of comment could produce a small completion for checking allowed image extensions such as `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`. I reviewed and adjusted any such suggestion before using it.

```python
# Build a summary dictionary for preprocessing output counts
```

This could assist with writing repetitive summary fields for train, validation, and test image counts. The final values and structure were checked against the actual preprocessing outputs.

```python
# Log training metrics and model artifacts to MLflow
```

This could help complete routine logging calls, while the metric names, artifact paths, and model behavior were verified against the implemented training pipeline.

In all cases, autocomplete suggestions were treated as draft code. I accepted only suggestions that matched the project design, edited them when needed, and tested or checked them in context.

I understand that I am responsible for the correctness, originality, and academic integrity of the submitted work. AI assistance was used as a productivity and learning aid, while the final judgement, verification, and submission decisions were made by me.
