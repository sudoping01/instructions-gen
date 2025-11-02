# Bambara Instruction Dataset Creation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/sudoping01/bambara-instructions)
[![Model](https://img.shields.io/badge/🤗-Model-green)](https://huggingface.co/sudoping01/bambara-llm-exp3)


A framework for creating high-quality instruction datasets for low-resource languages by combining large language model reasoning capabilities with structured linguistic knowledge. This repository contains the implementation used to create over 2 million Bambara conversations for language model training.

## Overview

This project addresses the critical shortage of instruction datasets for Bambara through a novel methodology that combines linguistic knowledge injection with the reasoning capabilities of large language models. Rather than relying on direct translation, our approach enables deliberative linguistic transformation that respects Bambara's complex morphology and syntax.

## Key Features

- **Knowledge-Enhanced Translation**: Integrates glossaries, grammar rules, and annotated examples
- **Reasoning-Based Processing**: Leverages LLM reasoning for complex linguistic transformations
- **High-Performance Architecture**: Concurrent processing with caching, fault tolerance, and checkpointing
- **Scale**: Successfully generated 2M+ Bambara conversations from diverse English/French datasets



## Results

Our system generated over 2 million Bambara conversations across multiples source datasets, achieving:
- 98.5% processing success rate
- Validation through successful language model training (93.4% loss reduction)
- Grammatically coherent outputs respecting Bambara's SOV word order and morphology

## Adaptation for Other Languages

The framework can be adapted to other low-resource languages by providing:
1. Lexical resources (glossary mapping source to target language)
2. Grammatical rule specifications (structured rules covering morphology and syntax)
3. Annotated examples (Universal Dependencies or similar annotations)

## Citation

```bibtex
@article{diallo2025bambara,
  title={Linguistically-Informed Large Language Models for Low-Resource Instruction Dataset Creation},
  author={Diallo, Seydou},
  journal={[Unpublished manuscript]},
  year={2025},
  month={July},
  url={Unpublished}
}
```



