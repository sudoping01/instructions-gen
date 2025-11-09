# MALIBA-Instructions: Source Datasets Catalog

This document provides an overview of the diverse source datasets used to create the MALIBA-Instructions dataset for Bambara. Each source dataset was carefully selected to ensure broad coverage across different topics, task types, and linguistic features.

## Source Datasets Overview

| Dataset | Primary Focus | Task Types | Languages Used | Approx. Size |
|---------|--------------|------------|----------------|--------------|
| Anthropic/hh-rlhf | Helpfulness & harmlessness | Conversational alignment | English | 160K+ conversations |
| Anthropic/model-written-evals | Model behavior assessment | Persona, ethics, bias evaluation | English | 154+ datasets |
| OpenAssistant/oasst1 | General conversation | Open-domain dialogue | Multilingual (35 languages) | 161K messages |
| OpenAssistant/oasst2 | Enhanced conversation | Open-domain dialogue | Multilingual | 208K messages |
| google/smol | Translation | Linguistic transfer | 221 low-resource languages | Multiple configurations |
| CohereLabs/aya_collection_language_split | Instruction-following | Multilingual instruction tuning | 101 languages (French) | 513M instances |
| openai/gsm8k | Mathematical reasoning | Multi-step problem solving | English | 8.5K problems |
| HuggingFaceH4/CodeAlpaca_20K | Code generation | Programming assistance | English | 20K instructions |
| angeluriot/french_instruct | French instruction-tuning | General instruction-following | French | Diverse tasks |
| MemGPT/MSC-Self-Instruct | Conversation generation | Self-instruct dialogue | English | Diverse tasks |
| HuggingFaceH4/instruction-dataset | General instruction-tuning | Diverse tasks | English | Multiple subtasks |
| HuggingFaceTB/instruct-data-basics-smollm-H4 | Basic instruction tuning | Foundational tasks | English | Core instructions |
| lmsys/lmsys-chat-1m | Conversational data | Dialogue generation | Primarily English | 1M+ conversations |
| everyday-conversations-llama3.1-2k | Everyday dialogue | Casual conversation | English | 2K conversations |
| IrfanHamid/chat-dataset | Chat interactions | Dialogue generation | English | Various chat formats |

## Detailed Dataset Descriptions

### Anthropic/hh-rlhf
**Topics Covered**: General conversational assistance, ethical reasoning, preference data for reinforcement learning.
**Description**: A dataset designed to train models to be helpful while also refusing harmful requests. Contains human preference data about helpfulness and harmlessness from diverse conversations.
**Key Strengths**: Strong focus on safety and alignment, diverse conversation topics, high-quality human evaluation.

### Anthropic/model-written-evals
**Topics Covered**: AI ethics, model behavior, persona evaluation, sycophancy testing, safety alignment.
**Description**: Collection of datasets written by language models to discover and evaluate model behaviors. Includes tests for political/religious views, personality traits, moral beliefs, and responses to potentially dangerous goals.
**Key Strengths**: Targeted evaluation of specific model behaviors, comprehensive coverage of bias and alignment issues.

### OpenAssistant/oasst1
**Topics Covered**: Open-domain dialogue, general knowledge, creative content, instruction-following.
**Description**: Human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 languages, annotated with 461,292 quality ratings.
**Key Strengths**: Multilingual support, conversational structure, human annotation, diverse topics.

### OpenAssistant/oasst2
**Topics Covered**: Enhanced open-domain dialogue, improved instruction-following, diverse conversation topics.
**Description**: An expanded version of oasst1 with 208,584 total messages, featuring improved conversation trees and quality control.
**Key Strengths**: Built upon oasst1 with refinements, expanded language coverage, more structured conversations.

### google/smol
**Topics Covered**: Language translation, low-resource language support, cross-lingual knowledge transfer.
**Description**: SMOL (Set for Maximal Overall Leverage) is a collection of professional translations into 221 low-resource languages. Contains multiple subsets:
- SmolDoc: document-level translations (104 language pairs)
- SmolSent: sentence-level translations (90 language pairs)
- GATITOS: token-level translations (179 language pairs)
**Key Strengths**: Extensive coverage of low-resource languages, multiple granularity levels (document, sentence, token).

### CohereLabs/aya_collection_language_split (French)
**Topics Covered**: Multilingual instruction tuning, diverse tasks, French language instruction-following.
**Description**: Part of a massive multilingual collection consisting of 513 million instances of prompts and completions covering various tasks in 101 languages. For our work, we focused on the French subset to ensure high-quality French-to-Bambara adaptation.
**Key Strengths**: Large scale, diverse task coverage, high-quality templates from fluent speakers, excellent resource for French language.

### openai/gsm8k
**Topics Covered**: Mathematical reasoning, step-by-step problem solving, logical thinking.
**Description**: A dataset of 8.5K high-quality grade school math word problems requiring multi-step reasoning (2-8 steps). Available in two configurations:
- main: Standard format with step-by-step solutions
- socratic: Enhanced version with "Socratic subquestions" before each solution step
**Key Strengths**: Tests complex reasoning, step-by-step solution paths, calculator annotations for verification.

### HuggingFaceH4/CodeAlpaca_20K
**Topics Covered**: Programming, code generation, technical problem-solving.
**Description**: A dataset of 20K coding instructions and solutions covering various programming languages and tasks.
**Key Strengths**: Specialized for code generation, diverse programming challenges, technical explanation quality.

### angeluriot/french_instruct
**Topics Covered**: French language instruction-tuning, diverse general knowledge tasks.
**Description**: Dedicated instruction dataset in French, providing valuable language-specific instruction examples for French-to-Bambara adaptation.
**Key Strengths**: High-quality French instructions, native language patterns, cultural context.

### MemGPT/MSC-Self-Instruct
**Topics Covered**: Conversational AI, memory augmentation, self-guided instruction generation.
**Description**: Dataset focused on enhancing conversational abilities with improved contextual awareness and memory handling.
**Key Strengths**: Long-context reasoning, memory-augmented dialogue patterns.

### HuggingFaceH4/instruction-dataset
**Topics Covered**: General instruction-following, diverse tasks across multiple domains.
**Description**: Comprehensive instruction dataset covering a wide range of tasks from simple queries to complex reasoning.
**Key Strengths**: Broad task coverage, varied instruction styles, comprehensive instruction-following examples.

### HuggingFaceTB/instruct-data-basics-smollm-H4
**Topics Covered**: Fundamental instruction-following for smaller language models, core tasks.
**Description**: Basic instruction dataset designed for efficient training of smaller language models on essential tasks.
**Key Strengths**: Concise, focused instructions, core capability coverage, optimized for efficient learning.

### lmsys/lmsys-chat-1m
**Topics Covered**: Large-scale conversational data, diverse dialogue patterns.
**Description**: Collection of 1M+ conversations covering a wide range of topics and interaction styles.
**Key Strengths**: High volume, diverse conversation formats, real-world interaction patterns.

### everyday-conversations-llama3.1-2k
**Topics Covered**: Casual dialogue, everyday conversations, natural language interaction.
**Description**: Dataset of 2K everyday conversations designed to improve natural conversational abilities.
**Key Strengths**: Natural dialogue flow, casual conversation patterns, everyday topics.

### IrfanHamid/chat-dataset
**Topics Covered**: General chat interactions, dialogue generation across various formats.
**Description**: Diverse collection of chat-style interactions for improving conversational capabilities.
**Key Strengths**: Various chat formats, interactive dialogue patterns.

## Topic Coverage Analysis

The combined datasets provide comprehensive coverage across the following major domains:

1. **General Knowledge & Conversation** (oasst1/2, lmsys-chat-1m, everyday-conversations)
   - Open-domain dialogue
   - Factual information
   - Natural conversation flow
   - Casual interactions

2. **Reasoning & Problem-solving** (gsm8k, CodeAlpaca, hh-rlhf)
   - Mathematical reasoning
   - Step-by-step problem solving
   - Logical deduction
   - Programming and technical reasoning

3. **Instruction-following** (Anthropic datasets, HuggingFace datasets)
   - Task execution
   - Prompt adherence
   - Complex instruction handling
   - Multi-step directive following

4. **Safety & Ethics** (hh-rlhf, model-written-evals)
   - Harmful request refusal
   - Ethical reasoning
   - Bias mitigation
   - Cultural sensitivity

5. **Multilingual & Translation** (smol, aya_collection, french_instruct)
   - Cross-lingual knowledge transfer
   - Language-specific patterns
   - Cultural context preservation
   - Translation quality

6. **Specialized Domains**
   - Programming (CodeAlpaca)
   - Mathematics (gsm8k)
   - Memory augmentation (MSC-Self-Instruct)
   - Cross-cultural adaptation

This diverse set of source datasets ensures that MALIBA-Instructions provides comprehensive coverage across a wide range of skills and knowledge domains, enabling robust instruction-following capabilities in Bambara.
