# Assessing Large Language Models' Susceptibility to Induced Doubt

This repository contains the code used to analyze and visualize the experimental results presented in our paper, **"Assessing Large Language Models' Susceptibility to Induced Doubt"**. Authored by **Gal Ashkenazi**, **Dan Feldman**, and **Jonathan Nethanel**, the paper investigates the effects of adversarial questioning on the confidence and accuracy of Large Language Models (LLMs).

## Project Overview

Large Language Models (LLMs) exhibit impressive capabilities but remain susceptible to external influences such as repeated questioning and assertive contradictions. This project aims to shed light on their vulnerabilities by systematically exploring how these adversarial strategies affect model outputs across various knowledge domains.

Key features of the repository include:
- **Experiment Visualization**: Code to generate plots of results presented in the paper.
- **Automation Attempt**: A Jupyter notebook (`automation_attempt.ipynb`) exploring automated methodologies for running experiments.

## Automation and Manual Experiments

While an initial attempt was made to automate the experimental process via API calls, challenges arose due to inconsistent responses from the LLMs. As a result:
- The automation effort, documented in `automation_attempt.ipynb`, provides insights into the obstacles encountered and outlines the approach taken.
- For improved accuracy and consistency, experiments were conducted manually. This manual approach ensured coherent results while minimizing variance.

## Usage

- To explore the automated experimental approach, run the `automation_attempt.ipynb` notebook. While the automation was ultimately unsuccessful, it remains a valuable resource for understanding the limitations and challenges in this context.
- Refer to the scripts in this repository for data visualization and analysis used in the paper.

## Citation

If you use the code or findings from this repository in your work, please cite our paper:

```
@article{ashkenazi2024induceddoubt,
  title={Assessing Large Language Models' Susceptibility to Induced Doubt},
  author={Gal Ashkenazi, Dan Feldman, Jonathan Nethanel},
  journal={NLP Research},
  year={2024}
}
```
