
# High-Level Design

## Objective

Build a local full-stack AI application for galaxy morphology classification with a real MLOps control loop.

## Why DVC + Airflow together?

Using only Airflow would make the operational scheduler also responsible for ML artifact lineage.
Using only DVC would make reproducibility strong, but runtime control weak.

So the system is intentionally split:

- **DVC** for build reproducibility
- **Airflow** for control-plane logic

## Main modules

### Data acquisition
Downloads Galaxy Zoo metadata and image archive, then materializes only the configured subset.

### Preprocess v1
Creates a first normalized processed dataset.

### Preprocess final
Creates the final training split and baseline drift statistics.

### Training
Fine-tunes a classifier and stores metrics, model artifacts, and confusion outputs.

### Evaluation
Computes offline test metrics and live feedback metrics.

### Reporting
Creates Markdown and HTML reports used by the demo and email step.

### Continuous improvement
Airflow monitors current metrics and new feedback volume and retriggers the DVC pipeline only when needed.

## Justification

This design demonstrates:

- reproducibility
- modularity
- loose coupling
- observability
- a realistic retraining loop
- a clear distinction between build and control responsibilities
