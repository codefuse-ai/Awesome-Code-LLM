# Awesome-Code-LLM

This is the repo for our [TMLR](https://jmlr.org/tmlr/) survey [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989) - a comprehensive review of LLM researches for code. Works in each category are ordered chronologically. If you have a basic understanding of machine learning but are new to NLP, we also provide a list of recommended readings in [section 9](#9-recommended-readings).

## News

🔥🔥🔥 [2025/05/29] ICLR 2025 papers have been added. Search for the keyword "ICLR 2025"!

🔥🔥🔥 [2025/05/29] Featured papers:

- 🔥🔥 [Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks](https://arxiv.org/abs/2505.16901) from Ant Group.

- 🔥🔥 [rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset](https://arxiv.org/abs/2505.21297) from Microsoft Research Asia.

- 🔥🔥 [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](https://arxiv.org/abs/2505.20411) from Nebius.

- 🔥🔥 [Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL](https://arxiv.org/abs/2505.20315) from Snowflake AI Research.

- 🔥 [SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development](https://arxiv.org/abs/2505.16975) from Shanghai Jiao Tong University.

- 🔥 [AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning](https://arxiv.org/abs/2505.16400) from NVIDIA.

- 🔥 [MLZero: A Multi-Agent System for End-to-end Machine Learning Automation](https://arxiv.org/abs/2505.13941) from Amazon Web Services.

🔥🔥🔥 Recent works from Codefuse:

- Graph - Aligned LLM for Improved Source Code Understanding: [codefuse-ai/GALLa](https://github.com/codefuse-ai/GALLa)
- Code Graph Model: [codefuse-ai/CodeFuse-CGM](https://github.com/codefuse-ai/CodeFuse-CGM)
- Rodimus: [codefuse-ai/rodimus](https://github.com/codefuse-ai/rodimus)

🔥🔥&nbsp;&nbsp;&nbsp;&nbsp; [2025/01/11] 20 papers from NeurIPS 2024 have been collected. You may search for the keyword "NeurIPS 2024" in this page.

🔥&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [2024/09/06] **Our survey has been accepted for publication by [Transactions on Machine Learning Research (TMLR)](https://jmlr.org/tmlr/).**

#### How to Contribute

If you find a paper to be missing from this repository, misplaced in a category, or lacking a reference to its journal/conference information, please do not hesitate to create an issue. If you find this repo helpful, please cite our survey:

```
@article{zhang2024unifying,
   title={Unifying the Perspectives of {NLP} and Software Engineering: A Survey on Language Models for Code},
   author={Ziyin Zhang and Chaoyu Chen and Bingchang Liu and Cong Liao and Zi Gong and Hang Yu and Jianguo Li and Rui Wang},
   journal={Transactions on Machine Learning Research},
   issn={2835-8856},
   year={2024},
   url={https://openreview.net/forum?id=hkNnGqZnpa},
   note={}
}
```

## Table of Contents

1. [Surveys](#1-surveys)

2. [Models](#2-models)

   2.1 [Base LLMs and Pretraining Strategies](#21-base-llms-and-pretraining-strategies)

   2.2 [Existing LLM Adapted to Code](#22-existing-llm-adapted-to-code)

   2.3 [General Pretraining on Code](#23-general-pretraining-on-code)

   - [Encoder](#encoder)
   - [Decoder](#decoder)
   - [Encoder-Decoder](#encoder-decoder)
   - [UniLM](#unilm)

   <!-- prettier ignore -->

   2.4 [(Instruction) Fine-Tuning on Code](#24-instruction-fine-tuning-on-code)

   2.5 [Reinforcement Learning on Code](#25-reinforcement-learning-on-code)

3. [When Coding Meets Reasoning](#3-when-coding-meets-reasoning)

   3.1 [Coding for Reasoning](#31-coding-for-reasoning)

   3.2 [Code Simulation](#32-code-simulation)

   3.3 [Code Agents](#33-code-agents)

   3.4 [Interactive Coding](#34-interactive-coding)

   3.5 [Frontend Navigation](#35-frontend-navigation)

4. [Code LLM for Low-Resource, Low-Level, and Domain-Specific Languages](#4-code-llm-for-low-resource-low-level-and-domain-specific-languages)

5. [Methods/Models for Downstream Tasks](#5-methodsmodels-for-downstream-tasks)

   - Programming

     - [Code Generation](#code-generation)
     - [Code RAG](#code-rag)
     - [Code Ranking](#code-ranking)
     - [Code Translation](#code-translation)
     - [Code Commenting and Summarization](#code-commenting-and-summarization)
     - [Program Repair](#program-repair)
     - [Code Similarity and Embedding (Clone Detection, Code Search)](#code-similarity-and-embedding-clone-detection-code-search)
     - [Code Refactoring and Migration](#code-refactoring-and-migration)
     - [Type Prediction](#type-prediction)
     - [Repository-Level Coding](#repository-level-coding)
     - [Frontend Development](#frontend-development)
     - [Automated Machine Learning](#automated-machine-learning)
     - [Text-To-SQL](#text-to-sql)
     - [Program Proof](#program-proof)

   - Testing and Deployment

     - [Test Generation](#test-generation)
     - [Oracle Generation](#oracle-generation)
     - [Mutation Testing](#mutation-testing)
     - [Fuzz Testing](#fuzz-testing)
     - [Vulnerability Detection](#vulnerability-detection)
     - [Malicious Code Detection](#malicious-code-detection)
     - [Compiler Optimization](#compiler-optimization)
     - [Binary Analysis and Decompilation](#binary-analysis-and-decompilation)

   - DevOps

     - [Commit Message Generation](#commit-message-generation)
     - [Code Review](#code-review)
     - [Log Analysis](#log-analysis)
     - [Software Configuration](#software-configuration)
     - [Code QA & Reasoning](#code-qa--reasoning)

   - Requirement

     - [Software Modeling](#software-modeling)
     - [Requirement Engineering](#requirement-engineering)

6. [Analysis of AI-Generated Code](#6-analysis-of-ai-generated-code)

   - [Security and Vulnerabilities](#security-and-vulnerabilities)
   - [Correctness](#correctness)
   - [Hallucination](#hallucination)
   - [Efficiency](#efficiency)
   - [Robustness](#robustness)
   - [Interpretability](#interpretability)
   - [API Usage](#api-usage)
   - [Privacy](#privacy)
   - [Bias](#bias)
   - [AI-Generated Code Detection](#ai-generated-code-detection)
   - [Others](#others)

7. [Human-LLM Interaction](#7-human-llm-interaction)

8. [Datasets](#8-datasets)

   8.1 [Pretraining](#81-pretraining)

   8.2 [Benchmarks](#82-benchmarks)

   - [Integrated Benchmarks](#integrated-benchmarks)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Program Synthesis](#program-synthesis)
   - [Visually Grounded Program Synthesis](#visually-grounded-program-synthesis)
   - [Code Reasoning and QA](#code-reasoning-and-qa)
   - [Text-to-SQL](#text-to-sql-1)
   - [Code Translation](#code-translation-1)
   - [Program Repair](#program-repair-1)
   - [Code Summarization](#code-summarization)
   - [Defect/Vulnerability Detection](#defectvulnerability-detection)
   - [Code Retrieval](#code-retrieval)
   - [Type Inference](#type-inference)
   - [Commit Message Generation](#commit-message-generation-1)
   - [Repo-Level Coding](#repo-level-coding)

9. [Recommended Readings](#9-recommended-readings)

10. [Citation](#citation)

11. [Star History](#star-history)

12. [Join Us](#join-us)

## 1. Surveys

We list several recent surveys on similar topics. While they are all about language models for code, 1-2 focus on NLP side; 3-6 focus on SE side; 7-11 are released after ours.

1. "Large Language Models Meet NL2Code: A Survey" [2022-12] [ACL 2023] [[paper](https://arxiv.org/abs/2212.09420)]

2. "A Survey on Pretrained Language Models for Neural Code Intelligence" [2022-12] [[paper](https://arxiv.org/abs/2212.10079)]

3. "An Empirical Comparison of Pre-Trained Models of Source Code" [2023-02] [ICSE 2023] [[paper](https://arxiv.org/abs/2302.04026)]

4. "Large Language Models for Software Engineering: A Systematic Literature Review" [2023-08] [[paper](https://arxiv.org/abs/2308.10620)]

5. "Towards an Understanding of Large Language Models in Software Engineering Tasks" [2023-08] [[paper](https://arxiv.org/abs/2308.11396)]

6. "Pitfalls in Language Models for Code Intelligence: A Taxonomy and Survey" [2023-10] [[paper](https://arxiv.org/abs/2310.17903)]

7. "A Survey on Large Language Models for Software Engineering" [2023-12] [[paper](https://arxiv.org/abs/2312.15223)]

8. "Deep Learning for Code Intelligence: Survey, Benchmark and Toolkit" [2023-12] [[paper](https://arxiv.org/abs/2401.00288)]

9. "A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond" [2024-03] [[paper](https://arxiv.org/abs/2403.14734)]

10. "Tasks People Prompt: A Taxonomy of LLM Downstream Tasks in Software Verification and Falsification Approaches" [2024-04] [[paper](https://arxiv.org/abs/2404.09384)]

11. "Automatic Programming: Large Language Models and Beyond" [2024-05] [[paper](https://arxiv.org/abs/2405.02213)]

12. "Software Engineering and Foundation Models: Insights from Industry Blogs Using a Jury of Foundation Models" [2024-10] [[paper](https://arxiv.org/abs/2410.09012)]

13. "Deep Learning-based Software Engineering: Progress, Challenges, and Opportunities" [2024-10] [[paper](https://arxiv.org/abs/2410.13110)]

14. "Large Language Models (LLMs) for Source Code Analysis: applications, models and datasets" [2025-03] [[paper](https://arxiv.org/abs/2503.17502)]

15. "Challenges and Paths Towards AI for Software Engineering" [2025-03] [[paper](https://arxiv.org/abs/2503.22625)]

16. "Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents" [2025-05] [[paper](https://arxiv.org/abs/2505.05283)]

## 2. Models

<p align='center'>
<img src='imgs/overview.png' style='width: 80%; '>
</p>

### 2.1 Base LLMs and Pretraining Strategies

These LLMs are not specifically trained for code, but have demonstrated varying coding capability.

1. **LaMDA**: "LaMDA: Language Models for Dialog Applications" [2022-01] [[paper](https://arxiv.org/abs/2201.08239)]

2. **PaLM**: "PaLM: Scaling Language Modeling with Pathways" [2022-04] [JMLR] [[paper](https://arxiv.org/abs/2204.02311)]

3. **GPT-NeoX**: "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" [2022-04] [ACL 2022 Workshop on Challenges & Perspectives in Creating LLMs] [[paper](https://arxiv.org/abs/2204.06745)] [[repo](https://github.com/EleutherAI/gpt-neox)]

4. **BLOOM**: "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" [2022-11] [[paper](https://arxiv.org/abs/2211.05100)] [[model](https://huggingface.co/models?search=bigscience/bloom)]

5. **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" [2023-02] [[paper](https://arxiv.org/abs/2302.13971)]

6. **GPT-4**: "GPT-4 Technical Report" [2023-03] [[paper](https://arxiv.org/abs/2303.08774)]

7. **LLaMA 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" [2023-07] [[paper](https://arxiv.org/abs/2307.09288)] [[repo](https://github.com/facebookresearch/llama)]

8. **Phi-1.5**: "Textbooks Are All You Need II: phi-1.5 technical report" [2023-09] [[paper](https://arxiv.org/abs/2309.05463)] [[model](https://huggingface.co/microsoft/phi-1_5)]

9. **Baichuan 2**: "Baichuan 2: Open Large-scale Language Models" [2023-09] [[paper](https://arxiv.org/abs/2309.10305)] [[repo](https://github.com/baichuan-inc/Baichuan2)]

10. **Qwen**: "Qwen Technical Report" [2023-09] [[paper](https://arxiv.org/abs/2309.16609)] [[repo](https://github.com/QwenLM/Qwen)]

11. **Mistral**: "Mistral 7B" [2023-10] [[paper](https://arxiv.org/abs/2310.06825)] [[repo](https://github.com/mistralai/mistral-src)]

12. **Gemini**: "Gemini: A Family of Highly Capable Multimodal Models" [2023-12] [[paper](https://arxiv.org/abs/2312.11805)]

13. **Phi-2**: "Phi-2: The surprising power of small language models" [2023-12] [[blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)]

14. **YAYI2**: "YAYI 2: Multilingual Open-Source Large Language Models" [2023-12] [[paper](https://arxiv.org/abs/2312.14862)] [[repo](https://github.com/wenge-research/YAYI2)]

15. **DeepSeek**: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism" [2024-01] [[paper](https://arxiv.org/abs/2401.02954)] [[repo](https://github.com/deepseek-ai/DeepSeek-LLM)]

16. **Mixtral**: "Mixtral of Experts" [2024-01] [[paper](https://arxiv.org/abs/2401.04088)] [[blog](https://mistral.ai/news/mixtral-of-experts/)]

17. **DeepSeekMoE**: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.12246)] [[repo](https://github.com/deepseek-ai/DeepSeek-MoE)]

18. **Orion**: "Orion-14B: Open-source Multilingual Large Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.06066)] [[repo](https://github.com/OrionStarAI/Orion)]

19. **OLMo**: "OLMo: Accelerating the Science of Language Models" [2024-02] [[paper](https://arxiv.org/abs/2402.00838)] [[repo](https://github.com/allenai/OLMo)]

20. **Gemma**: "Gemma: Open Models Based on Gemini Research and Technology" [2024-02] [[paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)] [[blog](https://blog.google/technology/developers/gemma-open-models/)]

21. **Claude 3**: "The Claude 3 Model Family: Opus, Sonnet, Haiku" [2024-03] [[paper](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)] [[blog](https://www.anthropic.com/news/claude-3-family)]

22. **Yi**: "Yi: Open Foundation Models by 01.AI" [2024-03] [[paper](https://arxiv.org/abs/2403.04652)] [[repo](https://github.com/01-ai/Yi)]

23. **Poro**: "Poro 34B and the Blessing of Multilinguality" [2024-04] [[paper](https://arxiv.org/abs/2404.01856)] [[model](https://huggingface.co/LumiOpen/Poro-34B)]

24. **JetMoE**: "JetMoE: Reaching Llama2 Performance with 0.1M Dollars" [2024-04] [[paper](https://arxiv.org/abs/2404.07413)] [[repo](https://github.com/myshell-ai/JetMoE)]

25. **LLaMA 3**: "The Llama 3 Herd of Models" [2024-04] [[blog](https://ai.meta.com/blog/meta-llama-3/)] [[repo](https://github.com/meta-llama/llama3)] [[paper](https://arxiv.org/abs/2407.21783)]

26. **Reka Core**: "Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models" [2024-04] [[paper](https://arxiv.org/abs/2404.12387)]

27. **Phi-3**: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone" [2024-04] [[paper](https://arxiv.org/abs/2404.14219)]

28. **OpenELM**: "OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework" [2024-04] [[paper](https://arxiv.org/abs/2404.14619)] [[repo](https://github.com/apple/corenet/tree/main/projects/openelm)]

29. **Tele-FLM**: "Tele-FLM Technical Report" [2024-04] [[paper](https://arxiv.org/abs/2404.16645)] [[model](https://huggingface.co/CofeAI/Tele-FLM)]

30. **DeepSeek-V2**: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" [2024-05] [[paper](https://arxiv.org/abs/2405.04434)] [[repo](https://github.com/deepseek-ai/DeepSeek-V2)]

31. **GECKO**: "GECKO: Generative Language Model for English, Code and Korean" [2024-05] [[paper](https://arxiv.org/abs/2405.15640)] [[model](https://huggingface.co/kifai/GECKO-7B)]

32. **MAP-Neo**: "MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series" [2024-05] [[paper](https://arxiv.org/abs/2405.19327)] [[repo](https://github.com/multimodal-art-projection/MAP-NEO)]

33. **Zyda**: "Zyda: A 1.3T Dataset for Open Language Modeling" [2024-06] [[paper](https://arxiv.org/abs/2406.01981)]

34. **Skywork-MoE**: "Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models" [2024-06] [[paper](https://arxiv.org/abs/2406.06563)]

35. **Xmodel-LM**: "Xmodel-LM Technical Report" [2024-06] [[paper](https://arxiv.org/abs/2406.02856)]

36. **GEB**: "GEB-1.3B: Open Lightweight Large Language Model" [2024-06] [[paper](https://arxiv.org/abs/2406.09900)]

37. **HARE**: "HARE: HumAn pRiors, a key to small language model Efficiency" [2024-06] [[paper](https://arxiv.org/abs/2406.11410)]

38. **DCLM**: "DataComp-LM: In search of the next generation of training sets for language models" [2024-06] [[paper](https://arxiv.org/abs/2406.11794)]

39. **Nemotron-4**: "Nemotron-4 340B Technical Report" [2024-06] [[paper](https://arxiv.org/abs/2406.11704)]

40. **ChatGLM**: "ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools" [2024-06] [[paper](https://arxiv.org/abs/2406.12793)]

41. **FineWeb**: "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" [2024-06] [[paper](https://arxiv.org/abs/2406.17557)]

42. **YuLan**: "YuLan: An Open-source Large Language Model" [2024-06] [[paper](https://arxiv.org/abs/2406.19853)]

43. **Gemma 2**: "Gemma 2: Improving Open Language Models at a Practical Size" [2024-06] [[paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)]

44. **H2O-Danube3**: "H2O-Danube3 Technical Report" [2024-07] [[paper](https://arxiv.org/abs/2407.09276)]

45. **Qwen2**: "Qwen2 Technical Report" [2024-07] [[paper](https://arxiv.org/abs/2407.10671)]

46. **ALLaM**: "ALLaM: Large Language Models for Arabic and English" [2024-07] [[paper](https://arxiv.org/abs/2407.15390)]

47. **SeaLLMs 3**: "SeaLLMs 3: Open Foundation and Chat Multilingual Large Language Models for Southeast Asian Languages" [2024-07] [[paper](https://arxiv.org/abs/2407.19672)]

48. **AFM**: "Apple Intelligence Foundation Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.21075)]

49. "To Code, or Not To Code? Exploring Impact of Code in Pre-training" [2024-08] [ICLR 2025] [[paper](https://arxiv.org/abs/2408.10914)]

50. **OLMoE**: "OLMoE: Open Mixture-of-Experts Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.02060)]

51. "How Does Code Pretraining Affect Language Model Task Performance?" [2024-09] [[paper](https://arxiv.org/abs/2409.04556)]

52. **EuroLLM**: "EuroLLM: Multilingual Language Models for Europe" [2024-09] [[paper](https://arxiv.org/abs/2409.16235)]

53. "Which Programming Language and What Features at Pre-training Stage Affect Downstream Logical Inference Performance?" [2024-10] [EMNLP 2024] [[paper](https://arxiv.org/abs/2410.06735)]

54. **GPT-4o**: "GPT-4o System Card" [2024-10] [[paper](https://arxiv.org/abs/2410.21276)]

55. **Hunyuan-Large**: "Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent" [2024-11] [[paper](https://arxiv.org/abs/2411.02265)]

56. **Crystal**: "Crystal: Illuminating LLM Abilities on Language and Code" [2024-11] [[paper](https://arxiv.org/abs/2411.04156)]

57. **Zyda-2**: "Zyda-2: a 5 Trillion Token High-Quality Dataset" [2024-11] [[paper](https://arxiv.org/abs/2411.06068)]

58. **Xmodel-1.5**: "Xmodel-1.5: An 1B-scale Multilingual LLM" [2024-11] [[paper](https://arxiv.org/abs/2411.10083)]

59. **Yi-Lightning**: "Yi-Lightning Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.01253)]

60. "RedStone: Curating General, Code, Math, and QA Data for Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.03398)]

61. **EXAONE 3.5**: "EXAONE 3.5: Series of Large Language Models for Real-world Use Cases" [2024-12] [[paper](https://arxiv.org/abs/2412.04862)]

62. "The Rise and Down of Babel Tower: Investigating the Evolution Process of Multilingual Code Large Language Model" [2024-12] [ICLR 2025] [[paper](https://arxiv.org/abs/2412.07298)]

63. **Phi-4**: "Phi-4 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.08905)]

64. **Typhoon 2**: "Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.13702)]

65. **Qwen2.5**: "Qwen2.5 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.15115)]

66. **YuLan-Mini**: "YuLan-Mini: An Open Data-efficient Language Model" [2024-12] [[paper](https://arxiv.org/abs/2412.17743)]

67. **DeepSeek-V3**: "DeepSeek-V3 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.19437)]

68. **OLMo 2**: "2 OLMo 2 Furious" [2024-12] [[paper](https://arxiv.org/abs/2501.00656)]

69. **FinerWeb**: "FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering" [2025-01] [[paper](https://arxiv.org/abs/2501.07314)]

70. **MiniMax-01**: "MiniMax-01: Scaling Foundation Models with Lightning Attention" [2025-01] [[paper](https://arxiv.org/abs/2501.08313)]

71. **SmolLM2**: "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model" [2025-02] [[paper](https://arxiv.org/abs/2502.02737)]

72. **Salamandra**: "Salamandra Technical Report" [2025-02] [[paper](https://arxiv.org/abs/2502.08489)]

73. **Kanana**: "Kanana: Compute-efficient Bilingual Language Models" [2025-02] [[paper](https://arxiv.org/abs/2502.18934)]

74. **Phi-4-Mini**: "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs" [2025-03] [[paper](https://arxiv.org/abs/2503.01743)]

75. **Ling**: "Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs" [2025-03] [[paper](https://arxiv.org/abs/2503.05139)]

76. **Gemma 3**: "Gemma 3 Technical Report" [2025-03] [[paper](https://arxiv.org/abs/2503.19786)]

77. **Command A**: "Command A: An Enterprise-Ready Large Language Model" [2025-04] [[paper](https://arxiv.org/abs/2504.00698)]

78. **Llama-Nemotron**: "Llama-Nemotron: Efficient Reasoning Models" [2025-05] [[paper](https://arxiv.org/abs/2505.00949)]

79. **MiMo**: "MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining" [2025-05] [[paper](https://arxiv.org/abs/2505.07608)]

80. **xGen-small**: "xGen-small Technical Report" [2025-05] [[paper](https://arxiv.org/abs/2505.06496)]

81. **Qwen3**: "Qwen3 Technical Report" [2025-05] [[paper](https://arxiv.org/abs/2505.09388)]

82. **Hunyuan-TurboS**: "Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought" [2025-05] [[paper](https://arxiv.org/abs/2505.15431)]

### 2.2 Existing LLM Adapted to Code

These models are general-purpose LLMs further pretrained on code-related data.

- **Codex** (GPT-3): "Evaluating Large Language Models Trained on Code" [2021-07] [[paper](https://arxiv.org/abs/2107.03374)]

- **PaLM Coder** (PaLM): "PaLM: Scaling Language Modeling with Pathways" [2022-04] [JMLR] [[paper](https://arxiv.org/abs/2204.02311)]

- **Minerva** (PaLM): "Solving Quantitative Reasoning Problems with Language Models" [2022-06] [[paper](https://arxiv.org/abs/2206.14858)]

- **PaLM 2 \*** (PaLM 2): "PaLM 2 Technical Report" [2023-05] [[paper](https://arxiv.org/abs/2305.10403)]

- **Code LLaMA** (LLaMA 2): "Code Llama: Open Foundation Models for Code" [2023-08] [[paper](https://arxiv.org/abs/2308.12950)] [[repo](https://github.com/facebookresearch/codellama)]

- **Lemur** (LLaMA 2): "Lemur: Harmonizing Natural Language and Code for Language Agents" [2023-10] [ICLR 2024 Spotlight] [[paper](https://arxiv.org/abs/2310.06830)]

- **BTX** (LLaMA 2): "Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM" [2024-03] [[paper](https://arxiv.org/abs/2403.07816)]

- **HiRoPE**: "HiRoPE: Length Extrapolation for Code Models Using Hierarchical Position" [2024-03] [ACL 2024] [[paper](https://arxiv.org/abs/2403.19115)]

- "Mastering Text, Code and Math Simultaneously via Fusing Highly Specialized Language Models" [2024-03] [[paper](https://arxiv.org/abs/2403.08281)]

- **CodeGemma**: "CodeGemma: Open Code Models Based on Gemma" [2024-04] [[paper](https://storage.googleapis.com/deepmind-media/gemma/codegemma_report.pdf)] [[model](https://huggingface.co/models?search=google/codegemma)]

- **DeepSeek-Coder-V2**: "DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence" [2024-06] [[paper](https://arxiv.org/abs/2406.11931)]

- "Promise and Peril of Collaborative Code Generation Models: Balancing Effectiveness and Memorization" [2024-09] [[paper](https://arxiv.org/abs/2409.12020)]

- **Qwen2.5-Coder**: "Qwen2.5-Coder Technical Report" [2024-09] [[paper](https://arxiv.org/abs/2409.12186)]

- **Lingma SWE-GPT**: "Lingma SWE-GPT: An Open Development-Process-Centric Language Model for Automated Software Improvement" [2024-11] [[paper](https://arxiv.org/abs/2411.00622)]

- **Ling-Coder-Lite**: "Every Sample Matters: Leveraging Mixture-of-Experts and High-Quality Data for Efficient and Accurate Code LLM" [2025-03] [[paper](https://arxiv.org/abs/2503.17793)]

### 2.3 General Pretraining on Code

These models are Transformer encoders, decoders, and encoder-decoders pretrained from scratch using existing objectives for general language modeling.

<p align='center'>
<img src='imgs/model_detail.png' style='width: 90%; '>
</p>

#### Encoder

1. **CuBERT** (MLM + NSP): "Learning and Evaluating Contextual Embedding of Source Code" [2019-12] [ICML 2020] [[paper](https://arxiv.org/abs/2001.00059)] [[repo](https://github.com/google-research/google-research/tree/master/cubert)]

2. **CodeBERT** (MLM + RTD): "CodeBERT: A Pre-Trained Model for Programming and Natural Languages" [2020-02] [EMNLP 2020 findings] [[paper](https://arxiv.org/abs/2002.08155)] [[repo](https://github.com/microsoft/CodeBERT)]

3. **GraphCodeBERT** (MLM + DFG Edge Prediction + DFG Node Alignment): "GraphCodeBERT: Pre-training Code Representations with Data Flow" [2020-09] [ICLR 2021] [[paper](https://arxiv.org/abs/2009.08366)] [[repo](https://github.com/microsoft/CodeBERT)]

4. **SynCoBERT** (MLM + Identifier Prediction + AST Edge Prediction + Contrastive Learning): "SynCoBERT: Syntax-Guided Multi-Modal Contrastive Pre-Training for Code Representation" [2021-08] [[paper](https://arxiv.org/abs/2108.04556)]

5. **DISCO** (MLM + Node Type MLM + Contrastive Learning): "Towards Learning (Dis)-Similarity of Source Code from Program Contrasts" [2021-10] [ACL 2022] [[paper](https://arxiv.org/abs/2110.03868)]

6. **Code-MVP** (MLM + Type Inference + Contrastive Learning): "CODE-MVP: Learning to Represent Source Code from Multiple Views with Contrastive Pre-Training" [2022-05] [NAACL 2022 Technical Track] [[paper](https://arxiv.org/abs/2205.02029)]

7. **CodeSage** (MLM + Deobfuscation + Contrastive Learning): "Code Representation Learning At Scale" [2024-02] [ICLR 2024] [[paper](https://arxiv.org/abs/2402.01935)]

8. **CoLSBERT** (MLM): "Scaling Laws Behind Code Understanding Model" [2024-02] [[paper](https://arxiv.org/abs/2402.12813)]

9. **BiGSCoder**: "BiGSCoder: State Space Model for Code Understanding" [2025-05] [[paper](https://arxiv.org/abs/2505.01475)]

#### Decoder

1. **GPT-C** (CLM): "IntelliCode Compose: Code Generation Using Transformer" [2020-05] [ESEC/FSE 2020] [[paper](https://arxiv.org/abs/2005.08025)]

2. **CodeGPT** (CLM): "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation" [2021-02] [NeurIPS Datasets and Benchmarks 2021] [[paper](https://arxiv.org/abs/2102.04664)] [[repo](https://github.com/microsoft/CodeXGLUE)]

3. **CodeParrot** (CLM) [2021-12] [[blog](https://huggingface.co/blog/codeparrot)]

4. **PolyCoder** (CLM): "A Systematic Evaluation of Large Language Models of Code" [2022-02] [DL4C@ICLR 2022] [[paper](https://arxiv.org/abs/2202.13169)] [[repo](https://github.com/VHellendoorn/Code-LMs)]

5. **CodeGen** (CLM): "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis" [2022-03] [ICLR 2023] [[paper](https://arxiv.org/abs/2203.13474)] [[repo](https://github.com/salesforce/CodeGen)]

6. **InCoder** (Causal Masking): "InCoder: A Generative Model for Code Infilling and Synthesis" [2022-04] [ICLR 2023] [[paper](https://arxiv.org/abs/2204.05999)] [[repo](https://github.com/dpfried/incoder)]

7. **PyCodeGPT** (CLM): "CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation" [2022-06] [IJCAI-ECAI 2022] [[paper](https://arxiv.org/abs/2206.06888)] [[repo](https://github.com/microsoft/PyCodeGPT)]

8. **PanGu-Coder** (CLM): "PanGu-Coder: Program Synthesis with Function-Level Language Modeling" [2022-07] [[paper](https://arxiv.org/abs/2207.11280)]

9. **SantaCoder** (FIM): "SantaCoder: don't reach for the stars!" [2023-01] [[paper](https://arxiv.org/abs/2301.03988)] [[model](https://huggingface.co/bigcode/santacoder)]

10. **CodeGeeX** (CLM): "CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X" [2023-03] [[paper](https://arxiv.org/abs/2303.17568)] [[repo](https://github.com/THUDM/CodeGeeX)]

11. **StarCoder** (FIM): "StarCoder: may the source be with you!" [2023-05] [[paper](https://arxiv.org/abs/2305.06161)] [[model](https://huggingface.co/bigcode/starcoder)]

12. **Phi-1** (CLM): "Textbooks Are All You Need" [2023-06] [[paper](https://arxiv.org/abs/2306.11644)] [[model](https://huggingface.co/microsoft/phi-1)]

13. **CodeFuse** (CLM): "CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model" [2023-10] [[paper](https://arxiv.org/abs/2310.06266)] [[model](https://huggingface.co/codefuse-ai/CodeFuse-13B)]

14. **DeepSeek Coder** (CLM+FIM): "DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence" [2024-01] [[paper](https://arxiv.org/abs/2401.14196)] [[repo](https://github.com/deepseek-ai/DeepSeek-Coder)]

15. **StarCoder2** (CLM+FIM): "StarCoder 2 and The Stack v2: The Next Generation" [2024-02] [[paper](https://arxiv.org/abs/2402.19173)] [[repo](https://github.com/bigcode-project/starcoder2)]

16. **CodeShell** (CLM+FIM): "CodeShell Technical Report" [2024-03] [[paper](https://arxiv.org/abs/2403.15747)] [[repo](https://github.com/WisdomShell/codeshell)]

17. **CodeQwen1.5** [2024-04] [[blog](https://qwenlm.github.io/blog/codeqwen1.5/)]

18. **Granite**: "Granite Code Models: A Family of Open Foundation Models for Code Intelligence" [2024-05] [[paper](https://arxiv.org/abs/2405.04324)] "Scaling Granite Code Models to 128K Context" [2024-07] [[paper](https://arxiv.org/abs/2407.13739)]

19. **NT-Java**: "Narrow Transformer: Starcoder-Based Java-LM For Desktop" [2024-07] [[paper](https://arxiv.org/abs/2407.03941)]

20. **Arctic-SnowCoder**: "Arctic-SnowCoder: Demystifying High-Quality Data in Code Pretraining" [2024-09] [[paper](https://arxiv.org/abs/2409.02326)]

21. **aiXcoder**: "aiXcoder-7B: A Lightweight and Effective Large Language Model for Code Completion" [2024-10] [[paper](https://arxiv.org/abs/2410.13187)]

22. **OpenCoder**: "OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.04905)]

23. **ObscuraCoder**: "ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding" [2025-03] [ICLR 2025] [[paper](https://arxiv.org/abs/2504.00019)]

#### Encoder-Decoder

1. **PyMT5** (Span Corruption): "PyMT5: multi-mode translation of natural language and Python code with transformers" [2020-10] [EMNLP 2020] [[paper](https://arxiv.org/abs/2010.03150)]

2. **Mastropaolo et al.** (MLM + Deobfuscation): "DOBF: A Deobfuscation Pre-Training Objective for Programming Languages" [2021-02] [ICSE 2021] [[paper](https://arxiv.org/abs/2102.02017)] [[repo](https://github.com/antonio-mastropaolo/TransferLearning4Code)]

3. **DOBF** (Span Corruption): "Studying the Usage of Text-To-Text Transfer Transformer to Support Code-Related Tasks" [2021-02] [NeurIPS 2021] [[paper](https://arxiv.org/abs/2102.07492)] [[repo](https://github.com/facebookresearch/CodeGen/blob/main/docs/dobf.md)]

4. **PLBART** (DAE): "Unified Pre-training for Program Understanding and Generation" [2021-03] [NAACL 2021] [[paper](https://arxiv.org/abs/2103.06333)] [[repo](https://github.com/wasiahmad/PLBART)]

5. **CodeT5** (Span Corruption + Identifier Tagging + Masked Identifier Prediction + Text2Code + Code2Text): "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation" [2021-09] [EMNLP 2021] [[paper](https://arxiv.org/abs/2109.00859)] [[repo](https://github.com/salesforce/CodeT5)]

6. **SPT-Code** (Span Corruption + NSP + Method Name Prediction): "SPT-Code: Sequence-to-Sequence Pre-Training for Learning Source Code Representations" [2022-01] [ICSE 2022 Technical Track] [[paper](https://arxiv.org/abs/2201.01549)]

7. **AlphaCode** (MLM + CLM): "Competition-Level Code Generation with AlphaCode" [2022-02] [Science] [[paper](https://arxiv.org/abs/2203.07814)] [[blog](https://deepmind.google/discover/blog/competitive-programming-with-alphacode/)]

8. **NatGen** (Code Naturalization): "NatGen: Generative pre-training by "Naturalizing" source code" [2022-06] [ESEC/FSE 2022] [[paper](https://arxiv.org/abs/2206.07585)] [[repo](https://github.com/saikat107/NatGen)]

9. **ERNIE-Code** (Span Corruption + Pivot-based Translation LM): "ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages" [2022-12] [ACL23 (Findings)] [[paper](https://aclanthology.org/2023.findings-acl.676.pdf)][[repo](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-code)]

10. **CodeT5+** (Span Corruption + CLM + Text-Code Contrastive Learning + Text-Code Translation): "CodeT5+: Open Code Large Language Models for Code Understanding and Generation" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.07922)] [[repo](https://github.com/salesforce/CodeT5)]

11. **AST-T5** (Span Corruption): "AST-T5: Structure-Aware Pretraining for Code Generation and Understanding" [2024-01] [ICML 2024] [[paper](https://arxiv.org/abs/2401.03003)]

12. **DivoT5**: "Directional Diffusion-Style Code Editing Pre-traini" [2025-01] [[paper](https://arxiv.org/abs/2501.12079)]

#### UniLM

1. **CugLM** (MLM + NSP + CLM): "Multi-task Learning based Pre-trained Language Model for Code Completion" [2020-12] [ASE 2020] [[paper](https://arxiv.org/abs/2012.14631)]

2. **UniXcoder** (MLM + NSP + CLM + Span Corruption + Contrastive Learning + Code2Text): "UniXcoder: Unified Cross-Modal Pre-training for Code Representation" [2022-03] [ACL 2022] [[paper](https://arxiv.org/abs/2203.03850)] [[repo](https://github.com/microsoft/CodeBERT)]

### 2.4 (Instruction) Fine-Tuning on Code

These models apply Instruction Fine-Tuning techniques to enhance the capacities of Code LLMs.

1. **WizardCoder** (StarCoder + Evol-Instruct): "WizardCoder: Empowering Code Large Language Models with Evol-Instruct" [2023-06] [ICLR 2024] [[paper](https://arxiv.org/abs/2306.08568)] [[repo](https://github.com/nlpxucan/WizardLM)]

2. **PanGu-Coder 2** (StarCoder + Evol-Instruct + RRTF): "PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback" [2023-07] [[paper](https://arxiv.org/abs/2307.14936)]

3. **OctoCoder** (StarCoder) / **OctoGeeX** (CodeGeeX2): "OctoPack: Instruction Tuning Code Large Language Models" [2023-08] [ICLR 2024 Spotlight] [[paper](https://arxiv.org/abs/2308.07124)] [[repo](https://github.com/bigcode-project/octopack)]

4. "At Which Training Stage Does Code Data Help LLMs Reasoning" [2023-09] [ICLR 2024 Spotlight] [[paper](https://arxiv.org/abs/2309.16298)]

5. **InstructCoder**: "InstructCoder: Instruction Tuning Large Language Models for Code Editing" [[paper](https://arxiv.org/abs/2310.20329)] [[repo](https://github.com/qishenghu/CodeInstruct)]

6. **MFTCoder**: "MFTCoder: Boosting Code LLMs with Multitask Fine-Tuning" [2023-11] [KDD 2024] [[paper](https://arxiv.org/abs/2311.02303)] [[repo](https://github.com/codefuse-ai/MFTCoder)]

7. "LLM-Assisted Code Cleaning For Training Accurate Code Generators" [2023-11] [ICLR 2024] [[paper](https://arxiv.org/abs/2311.14904)]

8. **Magicoder**: "Magicoder: Empowering Code Generation with OSS-Instruct" [2023-12] [ICML 2024] [[paper](https://arxiv.org/abs/2312.02120)]

9. **WaveCoder**: "WaveCoder: Widespread And Versatile Enhancement For Code Large Language Models By Instruction Tuning" [2023-12] [ACL 2024] [[paper](https://arxiv.org/abs/2312.14187)]

10. **Astraios**: "Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.00788)]

11. **DolphCoder**: "DolphCoder: Echo-Locating Code Large Language Models with Diverse and Multi-Objective Instruction Tuning" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.09136)]

12. **SafeCoder**: "Instruction Tuning for Secure Code Generation" [2024-02] [ICML 2024] [[paper](https://arxiv.org/abs/2402.09497)]

13. "Code Needs Comments: Enhancing Code LLMs with Comment Augmentation" [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.13013)]

14. **CCT**: "Code Comparison Tuning for Code Large Language Models" [2024-03] [[paper](https://arxiv.org/abs/2403.19121)]

15. **SAT**: "Structure-aware Fine-tuning for Code Pre-trained Models" [2024-04] [[paper](https://arxiv.org/abs/2404.07471)]

16. **CodeFort**: "CodeFort: Robust Training for Code Generation Models" [2024-04] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2405.01567)]

17. **XFT**: "XFT: Unlocking the Power of Code Instruction Tuning by Simply Merging Upcycled Mixture-of-Experts" [2024-04] [ACL 2024] [[paper](https://arxiv.org/abs/2404.15247)] [[repo](https://github.com/ise-uiuc/xft)]

18. **AIEV-Instruct**: "AutoCoder: Enhancing Code Large Language Model with AIEV-Instruct" [2024-05] [[paper](https://arxiv.org/abs/2405.14906)]

19. **AlchemistCoder**: "AlchemistCoder: Harmonizing and Eliciting Code Capability by Hindsight Tuning on Multi-source Data" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.19265)]

20. "From Symbolic Tasks to Code Generation: Diversification Yields Better Task Performers" [2024-05] [[paper](https://arxiv.org/abs/2405.19787)]

21. "Unveiling the Impact of Coding Data Instruction Fine-Tuning on Large Language Models Reasoning" [2024-05] [[paper](https://arxiv.org/abs/2405.20535)]

22. **SemCoder**: "SemCoder: Training Code Language Models with Comprehensive Semantics Reasoning" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.01006)]

23. **PLUM**: "PLUM: Preference Learning Plus Test Cases Yields Better Code Language Models" [2024-06] [[paper](https://arxiv.org/abs/2406.06887)]

24. **mCoder**: "McEval: Massively Multilingual Code Evaluation" [2024-06] [ICLR 2025] [[paper](https://arxiv.org/abs/2406.07436)]

25. "Unlock the Correlation between Supervised Fine-Tuning and Reinforcement Learning in Training Code Large Language Models" [2024-06] [[paper](https://arxiv.org/abs/2406.10305)]

26. **Code-Optimise**: "Code-Optimise: Self-Generated Preference Data for Correctness and Efficiency" [2024-06] [[paper](https://arxiv.org/abs/2406.12502)]

27. **UniCoder**: "UniCoder: Scaling Code Large Language Model via Universal Code" [2024-06] [ACL 2024] [[paper](https://arxiv.org/abs/2406.16441)]

28. "Brevity is the soul of wit: Pruning long files for code generation" [2024-06] [[paper](https://arxiv.org/abs/2407.00434)]

29. "Code Less, Align More: Efficient LLM Fine-tuning for Code Generation with Data Pruning" [2024-07] [[paper](https://arxiv.org/abs/2407.05040)]

30. **InverseCoder**: "InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruct" [2024-07] [[paper](https://arxiv.org/abs/2407.05700)]

31. "Curriculum Learning for Small Code Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.10194)]

32. **Genetic-Instruct**: "Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.21077)]

33. **DataScope**: "API-guided Dataset Synthesis to Finetune Large Code Models" [2024-08] [[paper](https://arxiv.org/abs/2408.08343)]

34. **XCoder**: "How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data" [2024-09] [EMNLP 2024] [[paper](https://arxiv.org/abs/2409.03810)]

35. **GALLa**: "GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding" [2024-09] [[paper](https://arxiv.org/abs/2409.04183)]

36. **HexaCoder**: "HexaCoder: Secure Code Generation via Oracle-Guided Synthetic Training Data" [2024-09] [[paper](https://arxiv.org/abs/2409.06446)]

37. **AMR-Evol**: "AMR-Evol: Adaptive Modular Response Evolution Elicits Better Knowledge Distillation for Large Language Models in Code Generation" [2024-10] [EMNLP 2024] [[paper](https://arxiv.org/abs/2410.00558)]

38. **LintSeq**: "Training Language Models on Synthetic Edit Sequences Improves Code Synthesis" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.02749)]

39. **CoBa**: "CoBa: Convergence Balancer for Multitask Finetuning of Large Language Models" [2024-10] [EMNLP 2024] [[paper](https://arxiv.org/abs/2410.06741)]

40. **CursorCore**: "CursorCore: Assist Programming through Aligning Anything" [2024-10] [[paper](https://arxiv.org/abs/2410.07002)]

41. **SelfCodeAlign**: "SelfCodeAlign: Self-Alignment for Code Generation" [2024-10] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2410.24198)]

42. "Mastering the Craft of Data Synthesis for CodeLLMs" [2024-10] [[paper](https://arxiv.org/abs/2411.00005)]

43. **CodeLutra**: "CodeLutra: Boosting LLM Code Generation via Preference-Guided Refinement" [2024-11] [[paper](https://arxiv.org/abs/2411.05199)]

44. **DSTC**: "DSTC: Direct Preference Learning with Only Self-Generated Tests and Code to Improve Code LMs" [2024-11] [[paper](https://arxiv.org/abs/2411.13611)]

45. **WarriorCoder**: "WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.17395)]

46. **EpiCoder**: "EpiCoder: Encompassing Diversity and Complexity in Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.04694)]

47. **Qwen2.5-xCoder**: "Multi-Agent Collaboration for Multilingual Code Instruction Tuning" [2025-02] [[paper](https://arxiv.org/abs/2502.07487)]

48. **UnitCoder**: "UnitCoder: Scalable Iterative Code Synthesis with Unit Test Guidance" [2025-02] [[paper](https://arxiv.org/abs/2502.11460)]

49. **GiFT**: "GiFT: Gibbs Fine-Tuning for Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.11466)]

50. **KODCODE**: "KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding" [2025-03] [[paper](https://arxiv.org/abs/2503.02951)]

51. **NextCoder**: "Robust Learning of Diverse Code Edits" [2025-03] [[paper](https://arxiv.org/abs/2503.03656)]

52. **FAIT**: "FAIT: Fault-Aware Fine-Tuning for Better Code Generation" [2025-03] [[paper](https://arxiv.org/abs/2503.16913)]

53. **Z1**: "Z1: Efficient Test-time Scaling with Code" [2025-04] [[paper](https://arxiv.org/abs/2504.00810)]

54. **OpenCodeReasoning**: "OpenCodeReasoning: Advancing Data Distillation for Competitive Coding" [2025-04] [[paper](https://arxiv.org/abs/2504.01943)]

55. **OpenCodeInstruct**: "OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs" [2025-04] [[paper](https://arxiv.org/abs/2504.04030)]

56. "Data-efficient LLM Fine-tuning for Code Generation" [2025-04] [[paper](https://arxiv.org/abs/2504.12687)]

57. "AKD : Adversarial Knowledge Distillation For Large Language Models Alignment on Coding tasks" [2025-05] [[paper](https://arxiv.org/abs/2505.06267)]

58. "CRPE: Expanding The Reasoning Capability of Large Language Model for Code Generation" [2025-05] [[paper](https://arxiv.org/abs/2505.10594)]

### 2.5 Reinforcement Learning on Code

1. **CompCoder**: "Compilable Neural Code Generation with Compiler Feedback" [2022-03] [ACL 2022] [[paper](https://arxiv.org/abs/2203.05132)]

2. **CodeRL**: "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning" [2022-07] [NeurIPS 2022] [[paper](https://arxiv.org/abs/2207.01780)] [[repo](https://github.com/salesforce/CodeRL)]

3. **PPOCoder**: "Execution-based Code Generation using Deep Reinforcement Learning" [2023-01] [TMLR 2023] [[paper](https://arxiv.org/abs/2301.13816)] [[repo](https://github.com/reddy-lab-code-research/PPOCoder)]

4. **RLTF**: "RLTF: Reinforcement Learning from Unit Test Feedback" [2023-07] [[paper](https://arxiv.org/abs/2307.04349)] [[repo](https://github.com/Zyq-scut/RLTF)]

5. **B-Coder**: "B-Coder: Value-Based Deep Reinforcement Learning for Program Synthesis" [2023-10] [ICLR 2024] [[paper](https://arxiv.org/abs/2310.03173)]

6. **IRCoCo**: "IRCoCo: Immediate Rewards-Guided Deep Reinforcement Learning for Code Completion" [2024-01] [FSE 2024] [[paper](https://arxiv.org/abs/2401.16637)]

7. **StepCoder**: "StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.01391)]

8. **RLPF & DPA**: "Performance-Aligned LLMs for Generating Fast Code" [2024-04] [[paper](https://arxiv.org/abs/2404.18864)]

9. "Measuring memorization in RLHF for code completion" [2024-06] [ICLR 2025] [[paper](https://arxiv.org/abs/2406.11715)]

10. "Applying RLAIF for Code Generation with API-usage in Lightweight LLMs" [2024-06] [[paper](https://arxiv.org/abs/2406.20060)]

11. **RLCoder**: "RLCoder: Reinforcement Learning for Repository-Level Code Completion" [2024-07] [[paper](https://arxiv.org/abs/2407.19487)]

12. **PF-PPO**: "Policy Filtration in RLHF to Fine-Tune LLM for Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.06957)]

13. **Coffee-Gym**: "Coffee-Gym: An Environment for Evaluating and Improving Natural Language Feedback on Erroneous Code" [2024-09] [EMNLP 2024] [[paper](https://arxiv.org/abs/2409.19715)]

14. **RLEF**: "RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning" [2024-10] [[paper](https://arxiv.org/abs/2410.02089)]

15. **CodePMP**: "CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning" [2024-10] [[paper](https://arxiv.org/abs/2410.02229)]

16. **CodeDPO**: "CodeDPO: Aligning Code Models with Self Generated and Verified Source Code" [2024-10] [[paper](https://arxiv.org/abs/2410.05605)]

17. "Process Supervision-Guided Policy Optimization for Code Generation" [2024-10] [[paper](https://arxiv.org/abs/2410.17621)]

18. "Aligning CodeLLMs with Direct Preference Optimization" [2024-10] [[paper](https://arxiv.org/abs/2410.18585)]

19. **FALCON**: "FALCON: Feedback-driven Adaptive Long/short-term memory reinforced Coding Optimization system" [2024-10] [[paper](https://arxiv.org/abs/2410.21349)]

20. **PFPO**: "Preference Optimization for Reasoning with Pseudo Feedback" [2024-11] [[paper](https://arxiv.org/abs/2411.16345)]

21. **o1-Coder**: "o1-Coder: an o1 Replication for Coding" [2024-11] [[paper](https://arxiv.org/abs/2412.00154)]

22. **PRLCoder**: "Process-Supervised Reinforcement Learning for Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.01715)]

23. **AceCoder**: "ACECODER: Acing Coder RL via Automated Test-Case Synthesis" [2025-02] [[paper](https://arxiv.org/abs/2502.01718)]

24. **Focused-DPO**: "Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points" [2025-02] [[paper](https://arxiv.org/abs/2502.11475)]

25. **SWE-RL**: "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution" [2025-02] [[paper](https://arxiv.org/abs/2502.18449)]

26. **AceReason-Nemotron**: "AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning" [2025-05] [[paper](https://arxiv.org/abs/2505.16400)]

27. **rStar-Coder**: "rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset" [2025-05] [[paper](https://arxiv.org/abs/2505.21297)]

## 3. When Coding Meets Reasoning

### 3.1 Coding for Reasoning

1. **PAL**: "PAL: Program-aided Language Models" [2022-11] [ICML 2023] [[paper](https://arxiv.org/abs/2211.10435)] [[repo](https://github.com/reasoning-machines/pal)]

2. **PoT**: "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks" [2022-11] [TMLR 2023] [[paper](https://arxiv.org/abs/2211.12588)] [[repo](https://github.com/wenhuchen/Program-of-Thoughts)]

3. **PaD**: "PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning" [2023-05] [NAACL 2024] [[paper](https://arxiv.org/abs/2305.13888)]

4. **CSV**: "Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification" [2023-08] [ICLR 2024] [[paper](https://arxiv.org/abs/2308.07921)]

5. **MathCoder**: "MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning" [2023-10] [ICLR 2024] [[paper](https://arxiv.org/abs/2310.03731)]

6. **CoC**: "Chain of Code: Reasoning with a Language Model-Augmented Code Emulator" [2023-12] [ICML 2024] [[paper](https://arxiv.org/abs/2312.04474)]

7. **EHRAgent**: "EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records" [2024-01] [EMNLP 2024] [[paper](https://arxiv.org/abs/2401.07128)]

8. **MARIO**: "MARIO: MAth Reasoning with code Interpreter Output -- A Reproducible Pipeline" [2024-01] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2401.08190)]

9. "Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs" [2024-01] [EMNLP 2024] [[paper](https://arxiv.org/abs/2401.10065)]

10. **ReGAL**: "ReGAL: Refactoring Programs to Discover Generalizable Abstractions" [2024-01] [ICML 2024] [[paper](https://arxiv.org/abs/2401.16467)]

11. **CodeAct**: "Executable Code Actions Elicit Better LLM Agents" [2024-02] [ICML 2024] [[paper](https://arxiv.org/abs/2402.01030)]

12. **MultiPoT**: "Python is Not Always the Best Choice: Embracing Multilingual Program of Thoughts" [2024-02] [EMNLP 2024] [[paper](https://arxiv.org/abs/2402.10691)]

13. **HProPro**: "Exploring Hybrid Question Answering via Program-based Prompting" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.10812)]

14. **HTL**: "How Do Humans Write Code? Large Models Do It the Same Way Too" [2024-02] [EMNLP 2024] [[paper](https://arxiv.org/abs/2402.15729)]

15. **xSTREET**: "Eliciting Better Multilingual Structured Reasoning from LLMs through Code" [2024-03] [ACL 2024] [[paper](https://arxiv.org/abs/2403.02567)]

16. **FlowMind**: "FlowMind: Automatic Workflow Generation with LLMs" [2024-03] [[paper](https://arxiv.org/abs/2404.13050)]

17. **Think-and-Execute**: "Language Models as Compilers: Simulating Pseudocode Execution Improves Algorithmic Reasoning in Language Models" [2024-04] [EMNLP 2024] [[paper](https://arxiv.org/abs/2404.02575)]

18. **CoRE**: "CoRE: LLM as Interpreter for Natural Language Programming, Pseudo-Code Programming, and Flow Programming of AI Agents" [2024-05] [[paper](https://arxiv.org/abs/2405.06907)]

19. **MuMath-Code**: "MuMath-Code: Combining Tool-Use Large Language Models with Multi-perspective Data Augmentation for Mathematical Reasoning" [2024-05] [EMNLP 2024] [[paper](https://arxiv.org/abs/2405.07551)]

20. **COGEX**: "Learning to Reason via Program Generation, Emulation, and Search" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.16337)]

21. "Arithmetic Reasoning with LLM: Prolog Generation & Permutation" [2024-05] [[paper](https://arxiv.org/abs/2405.17893)]

22. "Can LLMs Reason in the Wild with Programs?" [2024-06] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2406.13764)]

23. **DotaMath**: "DotaMath: Decomposition of Thought with Code Assistance and Self-correction for Mathematical Reasoning" [2024-07] [[paper](https://arxiv.org/abs/2407.04078)]

24. **CIBench**: "CIBench: Evaluating Your LLMs with a Code Interpreter Plugin" [2024-07] [[paper](https://arxiv.org/abs/2407.10499)]

25. **PyBench**: "PyBench: Evaluating LLM Agent on various real-world coding tasks" [2024-07] [[paper](https://arxiv.org/abs/2407.16732)]

26. **AdaCoder**: "AdaCoder: Adaptive Prompt Compression for Programmatic Visual Question Answering" [2024-07] [[paper](https://arxiv.org/abs/2407.19410)]

27. **PyramidCoder**: "Pyramid Coder: Hierarchical Code Generator for Compositional Visual Question Answering" [2024-07] [[paper](https://arxiv.org/abs/2407.20563)]

28. **CodeGraph**: "CodeGraph: Enhancing Graph Reasoning of LLMs with Code" [2024-08] [[paper](https://arxiv.org/abs/2408.13863)]

29. **SIaM**: "SIaM: Self-Improving Code-Assisted Mathematical Reasoning of Large Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.15565)]

30. **CodePlan**: "CodePlan: Unlocking Reasoning Potential in Large Langauge Models by Scaling Code-form Planning" [2024-09] [ICLR 2025] [[paper](https://arxiv.org/abs/2409.12452)]

31. **PoT**: "Proof of Thought : Neurosymbolic Program Synthesis allows Robust and Interpretable Reasoning" [2024-09] [[paper](https://arxiv.org/abs/2409.17270)]

32. **MetaMath**: "MetaMath: Integrating Natural Language and Code for Enhanced Mathematical Reasoning in Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.19381)]

33. "BabelBench: An Omni Benchmark for Code-Driven Analysis of Multimodal and Multistructured Data" [2024-10] [[paper](https://arxiv.org/abs/2410.00773)]

34. **CodeSteer**: "Steering Large Language Models between Code Execution and Textual Reasoning" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.03524)]

35. **MathCoder2**: "MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.08196)]

36. **LLMFP**: "Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming" [2024-10] [[paper](https://arxiv.org/abs/2410.12112)]

37. **Prove**: "Not All Votes Count! Programs as Verifiers Improve Self-Consistency of Language Models for Math Reasoning" [2024-10] [[paper](https://arxiv.org/abs/2410.12608)]

38. **PROVE**: "Trust but Verify: Programmatic VLM Evaluation in the Wild" [2024-10] [[paper](https://arxiv.org/abs/2410.13121)]

39. **GeoCoder**: "GeoCoder: Solving Geometry Problems by Generating Modular Code through Vision-Language Models" [2024-10] [[paper](https://arxiv.org/abs/2410.13510)]

40. **ReasonAgain**: "ReasonAgain: Using Extractable Symbolic Programs to Evaluate Mathematical Reasoning" [2024-10] [[paper](https://arxiv.org/abs/2410.19056)]

41. **GFP**: "Gap-Filling Prompting Enhances Code-Assisted Mathematical Reasoning" [2024-11] [[paper](https://arxiv.org/abs/2411.05407)]

42. **UTMath**: "UTMath: Math Evaluation with Unit Test via Reasoning-to-Coding Thoughts" [2024-11] [[paper](https://arxiv.org/abs/2411.07240)]

43. **CoCoP**: "CoCoP: Enhancing Text Classification with LLM through Code Completion Prompt" [2024-11] [[paper](https://arxiv.org/abs/2411.08979)]

44. **REPL-Plan**: "Interactive and Expressive Code-Augmented Planning with Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.13826)]

45. **CrossPAL**: "Empowering Multi-step Reasoning across Languages via Program-Aided Language Models" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.678/)]

46. "From Code to Play: Benchmarking Program Search for Games Using Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.04057)]

47. **CoinMath**: "CoinMath: Harnessing the Power of Coding Instruction for Math LLMs" [2024-12] [[paper](https://arxiv.org/abs/2412.11699)]

48. **MultiLingPoT**: "MultiLingPoT: Enhancing Mathematical Reasoning with Multilingual Program Fine-tuning" [2024-12] [[paper](https://arxiv.org/abs/2412.12609)]

49. **ProgCo**: "ProgCo: Program Helps Self-Correction of Large Language Models" [2025-01] [[paper](https://arxiv.org/abs/2501.01264)]

50. **PIE**: "Pseudocode-Injection Magic: Enabling LLMs to Tackle Graph Computational Tasks" [2025-01] [[paper](https://arxiv.org/abs/2501.13731)]

51. **AutoCode4Math**: "Learning Autonomous Code Integration for Math Language Models" [2025-02] [[paper](https://arxiv.org/abs/2502.00691)]

52. **MIHTCCT**: "MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training" [2025-02] [[paper](https://arxiv.org/abs/2502.08904)]

53. **ToolCoder**: "ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models" [2025-02] [[paper](https://arxiv.org/abs/2502.11404)]

54. **RM-PoT**: "RM-PoT: Reformulating Mathematical Problems and Solving via Program of Thoughts" [2025-02] [[paper](https://arxiv.org/abs/2502.12589)]

55. **SBSC**: "SBSC: Step-By-Step Coding for Improving Mathematical Olympiad Performance" [2025-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2502.16666)]

56. "Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments" [2025-02] [[paper](https://arxiv.org/abs/2502.17956)]

57. "Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs" [2025-02] [[paper](https://arxiv.org/abs/2502.19411)]

58. "The KoLMogorov Test: Compression by Code Generation" [2025-03] [ICLR 2025] [[paper](https://arxiv.org/abs/2503.13992)]

59. **MathCoder-VL**: "MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning" [2025-05] [[paper](https://arxiv.org/abs/2505.10557)]

60. **R1-Code-Interpreter**: "R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning" [2025-05] [[paper](https://arxiv.org/abs/2505.21668)]

### 3.2 Code Simulation

- "Code Simulation Challenges for Large Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.09074)]

- "CodeMind: A Framework to Challenge Large Language Models for Code Reasoning" [2024-02] [[paper](https://arxiv.org/abs/2402.09664)]

- "Executing Natural Language-Described Algorithms with Large Language Models: An Investigation" [2024-02] [[paper](https://arxiv.org/abs/2403.00795)]

- "Can Language Models Pretend Solvers? Logic Code Simulation with LLMs" [2024-03] [[paper](https://arxiv.org/abs/2403.16097)]

- "Evaluating Large Language Models with Runtime Behavior of Program Execution" [2024-03] [[paper](https://arxiv.org/abs/2403.16437)]

- "NExT: Teaching Large Language Models to Reason about Code Execution" [2024-04] [ICML 2024] [[paper](https://arxiv.org/abs/2404.14662)]

- "SelfPiCo: Self-Guided Partial Code Execution with LLMs" [2024-07] [[paper](https://arxiv.org/abs/2407.16974)]

- "Large Language Models as Code Executors: An Exploratory Study" [2024-10] [[paper](https://arxiv.org/abs/2410.06667)]

- "VISUALCODER: Guiding Large Language Models in Code Execution with Fine-grained Multimodal Chain-of-Thought Reasoning" [2024-10] [[paper](https://arxiv.org/abs/2410.23402)]

- "CoCoNUT: Structural Code Understanding does not fall out of a tree" [2025-01] [[paper](https://arxiv.org/abs/2501.16456)]

- "CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction" [2025-02] [[paper](https://arxiv.org/abs/2502.07316)]

- "SURGE: On the Potential of Large Language Models as General-Purpose Surrogate Code Executors" [2025-02] [[paper](https://arxiv.org/abs/2502.11167)]

- "What I cannot execute, I do not understand: Training and Evaluating LLMs on Program Execution Traces" [2025-02] [[paper](https://arxiv.org/abs/2503.05703)]

- "L0-Reasoning Bench: Evaluating Procedural Correctness in Language Models via Simple Program Execution" [2025-03] [[paper](https://arxiv.org/abs/2503.22832)]

### 3.3 Code Agents

1. **Self-collaboration**: "Self-collaboration Code Generation via ChatGPT" [2023-04] [[paper](https://arxiv.org/abs/2304.07590)]

2. **ChatDev**: "Communicative Agents for Software Development" [2023-07] [[paper](https://arxiv.org/abs/2307.07924)] [[repo](https://github.com/OpenBMB/ChatDev)]

3. **MetaGPT**: "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework" [2023-08] [[paper](https://arxiv.org/abs/2308.00352)] [[repo](https://github.com/geekan/MetaGPT)]

4. **CodeChain**: "CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules" [2023-10] [ICLR 2024] [[paper](https://arxiv.org/abs/2310.08992)]

5. **CodeAgent**: "CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges" [2024-01] [ACL 2024] [[paper](https://arxiv.org/abs/2401.07339)]

6. **CONLINE**: "CoCoST: Automatic Complex Code Generation with Online Searching and Correctness Testing" [2024-03] [EMNLP 2024] [[paper](https://arxiv.org/abs/2403.13583)]

7. **LCG**: "When LLM-based Code Generation Meets the Software Development Process" [2024-03] [[paper](https://arxiv.org/abs/2403.15852)]

8. **RepairAgent**: "RepairAgent: An Autonomous, LLM-Based Agent for Program Repair" [2024-03] [[paper](https://arxiv.org/abs/2403.17134)]

9. **MAGIS:**: "MAGIS: LLM-Based Multi-Agent Framework for GitHub Issue Resolution" [2024-03] [[paper](https://arxiv.org/abs/2403.17927)]

10. **SoA**: "Self-Organized Agents: A LLM Multi-Agent Framework toward Ultra Large-Scale Code Generation and Optimization" [2024-04] [[paper](https://arxiv.org/abs/2404.02183)]

11. **AutoCodeRover**: "AutoCodeRover: Autonomous Program Improvement" [2024-04] [[paper](https://arxiv.org/abs/2404.05427)]

12. **SWE-agent**: "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" [2024-05] [[paper](https://arxiv.org/abs/2405.15793)]

13. **MapCoder**: "MapCoder: Multi-Agent Code Generation for Competitive Problem Solving" [2024-05] [ACL 2024] [[paper](https://arxiv.org/abs/2405.11403)]

14. "Fight Fire with Fire: How Much Can We Trust ChatGPT on Source Code-Related Tasks?" [2024-05] [[paper](https://arxiv.org/abs/2405.12641)]

15. **FunCoder**: "Divide-and-Conquer Meets Consensus: Unleashing the Power of Functions in Code Generation" [2024-05] [[paper](https://arxiv.org/abs/2405.20092)]

16. **CTC**: "Multi-Agent Software Development through Cross-Team Collaboration" [2024-06] [[paper](https://arxiv.org/abs/2406.08979)]

17. **MASAI**: "MASAI: Modular Architecture for Software-engineering AI Agents" [2024-06] [[paper](https://arxiv.org/abs/2406.11638)]

18. **AgileCoder**: "AgileCoder: Dynamic Collaborative Agents for Software Development based on Agile Methodology" [2024-06] [[paper](https://arxiv.org/abs/2406.11912)]

19. **CodeNav**: "CodeNav: Beyond tool-use to using real-world codebases with LLM agents" [2024-06] [[paper](https://arxiv.org/abs/2406.12276)]

20. **INDICT**: "INDICT: Code Generation with Internal Dialogues of Critiques for Both Security and Helpfulness" [2024-06] [[paper](https://arxiv.org/abs/2407.02518)]

21. **AppWorld**: "AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents" [2024-07] [[paper](https://arxiv.org/abs/2407.18901)]

22. **CortexCompile**: "CortexCompile: Harnessing Cortical-Inspired Architectures for Enhanced Multi-Agent NLP Code Synthesis" [2024-08] [[paper](https://arxiv.org/abs/2409.02938)]

23. **DEI**: "Diversity Empowers Intelligence: Integrating Expertise of Software Engineering Agents" [2024-08] [ICLR 2025] [[paper](https://arxiv.org/abs/2408.07060)]

24. **Survey**: "Large Language Model-Based Agents for Software Engineering: A Survey" [2024-09] [[paper](https://arxiv.org/abs/2409.02977)]

25. **PairCoder**: "A Pair Programming Framework for Code Generation via Multi-Plan Exploration and Feedback-Driven Refinement" [2024-09] [ASE 2024] [[paper](https://arxiv.org/abs/2409.05001)] [[repo](https://github.com/nju-websoft/PairCoder)]

26. **AutoSafeCoder**: "AutoSafeCoder: A Multi-Agent Framework for Securing LLM Code Generation through Static Analysis and Fuzz Testing" [2024-09] [[paper](https://arxiv.org/abs/2409.10737)]

27. **SuperCoder2.0**: "SuperCoder2.0: Technical Report on Exploring the feasibility of LLMs as Autonomous Programmer" [2024-09] [[paper](https://arxiv.org/abs/2409.11190)]

28. **Survey**: "Agents in Software Engineering: Survey, Landscape, and Vision" [2024-09] [[paper](https://arxiv.org/abs/2409.09030)]

29. **MOSS**: "MOSS: Enabling Code-Driven Evolution and Context Management for AI Agents" [2024-09] [[paper](https://arxiv.org/abs/2409.16120)]

30. **HyperAgent**: "HyperAgent: Generalist Software Engineering Agents to Solve Coding Tasks at Scale" [2024-09] [[paper](https://arxiv.org/abs/2409.16299)]

31. "Compositional Hardness of Code in Large Language Models -- A Probabilistic Perspective" [2024-09] [[paper](https://arxiv.org/abs/2409.18028)]

32. **RGD**: "RGD: Multi-LLM Based Agent Debugger via Refinement and Generation Guidance" [2024-10] [[paper](https://arxiv.org/abs/2410.01242)]

33. **Seeker**: "Seeker: Enhancing Exception Handling in Code with LLM-based Multi-Agent Approach" [2024-10] [[paper](https://arxiv.org/abs/2410.06949)]

34. **REDO**: "REDO: Execution-Free Runtime Error Detection for COding Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.09117)]

35. "Evaluating Software Development Agents: Patch Patterns, Code Quality, and Issue Complexity in Real-World GitHub Scenarios" [2024-10] [[paper](https://arxiv.org/abs/2410.12468)]

36. **EvoMAC**: "Self-Evolving Multi-Agent Collaboration Networks for Software Development" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.16946)]

37. **VisionCoder**: "VisionCoder: Empowering Multi-Agent Auto-Programming for Image Processing with Hybrid LLMs" [2024-10] [[paper](https://arxiv.org/abs/2410.19245)]

38. **AutoKaggle**: "AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions" [2024-10] [[paper](https://arxiv.org/abs/2410.20424)]

39. **Watson**: "Watson: A Cognitive Observability Framework for the Reasoning of Foundation Model-Powered Agents" [2024-11] [[paper](https://arxiv.org/abs/2411.03455)]

40. **CodeTree**: "CodeTree: Agent-guided Tree Search for Code Generation with Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.04329)]

41. **EvoCoder**: "LLMs as Continuous Learners: Improving the Reproduction of Defective Code in Software Issues" [2024-11] [[paper](https://arxiv.org/abs/2411.13941)]

42. **AEGIS**: "AEGIS: An Agent-based Framework for General Bug Reproduction from Issue Descriptions" [2024-11] [[paper](https://arxiv.org/abs/2411.18015)]

43. **ExecutionAgent**: "You Name It, I Run It: An LLM Agent to Execute Tests of Arbitrary Projects" [2024-12] [[paper](https://arxiv.org/abs/2412.10133)]

44. **GHIssueMarket**: "GHIssuemarket: A Sandbox Environment for SWE-Agents Economic Experimentation" [2024-12] [[paper](https://arxiv.org/abs/2412.11722)]

45. **SWE-Gym**: "Training Software Engineering Agents and Verifiers with SWE-Gym" [2024-12] [[paper](https://arxiv.org/abs/2412.21139)]

46. **SWE-Fixer**: "SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution" [2025-01] [[paper](https://arxiv.org/abs/2501.05040)]

47. **CodeCoR**: "CodeCoR: An LLM-Based Self-Reflective Multi-Agent Framework for Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.07811)]

48. **QualityFlow**: "QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks" [2025-01] [[paper](https://arxiv.org/abs/2501.17167)]

49. **Cogito**: "Cogito, ergo sum: A Neurobiologically-Inspired Cognition-Memory-Growth System for Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.18653)]

50. **OrcaLoca**: "OrcaLoca: An LLM Agent Framework for Software Issue Localization" [2025-02] [[paper](https://arxiv.org/abs/2502.00350)]

51. **BRT Agent**: "Agentic Bug Reproduction for Effective Automated Program Repair at Google" [2025-02] [[paper](https://arxiv.org/abs/2502.01821)]

52. **CodeSim**: "CODESIM: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging" [2025-02] [[paper](https://arxiv.org/abs/2502.05664)]

53. **SyncMind**: "SyncMind: Measuring Agent Out-of-Sync Recovery in Collaborative Software Engineering" [2025-02] [[paper](https://arxiv.org/abs/2502.06994)]

54. **SoRFT**: "SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning" [2025-02] [[paper](https://arxiv.org/abs/2502.20127)]

55. "Is Multi-Agent Debate (MAD) the Silver Bullet? An Empirical Analysis of MAD in Code Summarization and Translation" [2025-03] [[paper](https://arxiv.org/abs/2503.12029)]

56. **DARS**: "DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal" [2025-03] [[paper](https://arxiv.org/abs/2503.14269)]

57. **SEAlign**: "SEAlign: Alignment Training for Software Engineering Agent" [2025-03] [[paper](https://arxiv.org/abs/2503.18455)]

58. **SWE-SynInfer**: "Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compute" [2025-03] [[paper](https://arxiv.org/abs/2503.23803)]

59. **AdaCoder**: "AdaCoder: An Adaptive Planning and Multi-Agent Framework for Function-Level Code Generation" [2025-04] [[paper](https://arxiv.org/abs/2504.04220)]

60. **SICA**: "A Self-Improving Coding Agent" [2025-04] [[paper](https://arxiv.org/abs/2504.15228)]

61. **SWE-smith**: "SWE-smith: Scaling Data for Software Engineering Agents" [2025-04] [[paper](https://arxiv.org/abs/2504.21798)]

62. "Enhancing LLM Code Generation: A Systematic Evaluation of Multi-Agent Collaboration and Runtime Debugging for Improved Accuracy, Reliability, and Latency" [2025-05] [[paper](https://arxiv.org/abs/2505.02133)]

63. **SEW**: "SEW: Self-Evolving Agentic Workflows for Automated Code Generation" [2025-05] [[paper](https://arxiv.org/abs/2505.18646)]

64. **RepoMaster**: "RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories for Complex Task Solving" [2025-05] [[paper](https://arxiv.org/abs/2505.21577)]

### 3.4 Interactive Coding

- "Interactive Program Synthesis" [2017-03] [[paper](https://arxiv.org/abs/1703.03539)]

- "Question selection for interactive program synthesis" [2020-06] [PLDI 2020] [[paper](https://dl.acm.org/doi/10.1145/3385412.3386025)]

- "Interactive Code Generation via Test-Driven User-Intent Formalization" [2022-08] [[paper](https://arxiv.org/abs/2208.05950)]

- "Improving Code Generation by Training with Natural Language Feedback" [2023-03] [TMLR] [[paper](https://arxiv.org/abs/2303.16749)]

- "Self-Refine: Iterative Refinement with Self-Feedback" [2023-03] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2303.17651)]

- "Teaching Large Language Models to Self-Debug" [2023-04] [[paper](https://arxiv.org/abs/2304.05128)]

- "Self-Edit: Fault-Aware Code Editor for Code Generation" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.04087)]

- "LeTI: Learning to Generate from Textual Interactions" [2023-05] [[paper](https://arxiv.org/abs/2305.10314)]

- "Is Self-Repair a Silver Bullet for Code Generation?" [2023-06] [ICLR 2024] [[paper](https://arxiv.org/abs/2306.09896)]

- "InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback" [2023-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2306.14898)]

- "INTERVENOR: Prompting the Coding Ability of Large Language Models with the Interactive Chain of Repair" [2023-11] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2311.09868)]

- "OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement" [2024-02] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.14658)]

- "Iterative Refinement of Project-Level Code Context for Precise Code Generation with Compiler Feedback" [2024-03] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2403.16792)]

- "CYCLE: Learning to Self-Refine the Code Generation" [2024-03] [[paper](https://arxiv.org/abs/2403.18746)]

- "LLM-based Test-driven Interactive Code Generation: User Study and Empirical Evaluation" [2024-04] [[paper](https://arxiv.org/abs/2404.10100)]

- "SOAP: Enhancing Efficiency of Generated Code via Self-Optimization" [2024-05] [[paper](https://arxiv.org/abs/2405.15189)]

- "Code Repair with LLMs gives an Exploration-Exploitation Tradeoff" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.17503)]

- "ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation" [2024-05] [[paper](https://arxiv.org/abs/2405.17057)]

- "Training LLMs to Better Self-Debug and Explain Code" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.18649)]

- "Requirements are All You Need: From Requirements to Code with LLMs" [2024-06] [[paper](https://arxiv.org/abs/2406.10101)]

- "I Need Help! Evaluating LLM's Ability to Ask for Users' Support: A Case Study on Text-to-SQL Generation" [2024-07] [EMNLP 2024] [[paper](https://arxiv.org/abs/2407.14767)]

- "An Empirical Study on Self-correcting Large Language Models for Data Science Code Generation" [2024-08] [[paper](https://arxiv.org/abs/2408.15658)]

- "RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.09584)]

- "From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging" [2024-10] [[paper](https://arxiv.org/abs/2410.01215)] [[repo](https://github.com/YerbaPage/MGDebugger)]

- "What Makes Large Language Models Reason in (Multi-Turn) Code Generation?" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.08105)]

- "The First Prompt Counts the Most! An Evaluation of Large Language Models on Iterative Example-based Code Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.06774)]

- "Planning-Driven Programming: A Large Language Model Programming Workflow" [2024-11] [[paper](https://arxiv.org/abs/2411.14503)]

- "ConAIR:Consistency-Augmented Iterative Interaction Framework to Enhance the Reliability of Code Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.15587)]

- "Socratic Human Feedback (SoHF): Expert Steering Strategies for LLM Code Generation" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.908/)]

- "PerfCodeGen: Improving Performance of LLM Generated Code with Execution Feedback" [2024-11] [[paper](https://arxiv.org/abs/2412.03578)]

- "GenX: Mastering Code and Test Generation with Execution Feedback" [2024-12] [[paper](https://arxiv.org/abs/2412.13464)]

- "Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis" [2024-12] [[paper](https://arxiv.org/abs/2412.14841)]

- "Outcome-Refining Process Supervision for Code Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.15118)]

- "Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation and Execution in Complex Task Handling" [2024-12] [[paper](https://arxiv.org/abs/2412.15305)]

- "Dynamic Scaling of Unit Tests for Code Reward Modeling" [2025-01] [[paper](https://arxiv.org/abs/2501.01054)]

- "Revisit Self-Debugging with Self-Generated Tests for Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.12793)]

- "Learning to Generate Unit Tests for Automated Debugging" [2025-02] [[paper](https://arxiv.org/abs/2502.01619)]

- "Large Language Model Guided Self-Debugging Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.02928)]

- "On Iterative Evaluation and Enhancement of Code Quality Using GPT-4o" [2025-02] [[paper](https://arxiv.org/abs/2502.07399)]

- "Intention is All You Need: Refining Your Code from Your Intention" [2025-02] [[paper](https://arxiv.org/abs/2502.08172)]

- "RefineCoder: Iterative Improving of Large Language Models via Adaptive Critique Refinement for Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.09183)]

- "VisPath: Automated Visualization Code Synthesis via Multi-Path Reasoning and Feedback-Driven Optimization" [2025-02] [[paper](https://arxiv.org/abs/2502.11140)]

- "S\*: Test Time Scaling for Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.14382)]

- "LLM4EFFI: Leveraging Large Language Models to Enhance Code Efficiency and Correctness" [2025-02] [[paper](https://arxiv.org/abs/2502.18489)]

- "Multi-Turn Code Generation Through Single-Step Rewards" [2025-02] [[paper](https://arxiv.org/abs/2502.20380)]

- "ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments" [2025-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2502.19852)]

- "IterPref: Focal Preference Learning for Code Generation via Iterative Debugging" [2025-03] [[paper](https://arxiv.org/abs/2503.02783)]

- "debug-gym: A Text-Based Environment for Interactive Debugging" [2025-03] [[paper](https://arxiv.org/abs/2503.21557)]

- "CodeIF-Bench: Evaluating Instruction-Following Capabilities of Large Language Models in Interactive Code Generation" [2025-03] [[paper](https://arxiv.org/abs/2503.22688)]

- "CodeFlowBench: A Multi-turn, Iterative Benchmark for Complex Code Generation" [2025-04] [[paper](https://arxiv.org/abs/2504.21751)]

### 3.5 Frontend Navigation

- "MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding" [2021-10] [ACL 2022] [[paper](https://arxiv.org/abs/2110.08518)]

- "WebKE: Knowledge Extraction from Semi-structured Web with Pre-trained Markup Language Model" [2021-10] [CIKM 2021] [[paper](https://dl.acm.org/doi/abs/10.1145/3459637.3482491)]

- "WebGPT: Browser-assisted question-answering with human feedback" [2021-12] [[paper](https://arxiv.org/abs/2112.09332)]

- "CM3: A Causal Masked Multimodal Model of the Internet" [2022-01] [[paper](https://arxiv.org/abs/2201.07520)]

- "DOM-LM: Learning Generalizable Representations for HTML Documents" [2022-01] [[paper](https://arxiv.org/abs/2201.10608)]

- "WebFormer: The Web-page Transformer for Structure Information Extraction" [2022-02] [WWW 2022] [[paper](https://arxiv.org/abs/2202.00217)]

- "A Dataset for Interactive Vision-Language Navigation with Unknown Command Feasibility" [2022-02] [ECCV 2022] [[paper](https://arxiv.org/abs/2202.02312)]

- "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents" [2022-07] [NeurIPS 2022] [[paper](https://arxiv.org/abs/2207.01206)]

- "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding" [2022-10] [ICML 2023] [[paper](https://arxiv.org/abs/2210.03347)]

- "Understanding HTML with Large Language Models" [2022-10] [EMNLP 2023 findings] [[paper](https://arxiv.org/abs/2210.03945)]

- "WebUI: A Dataset for Enhancing Visual UI Understanding with Web Semantics" [2023-01] [CHI 2023] [[paper](https://arxiv.org/abs/2301.13280)]

- "Mind2Web: Towards a Generalist Agent for the Web" [2023-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2306.06070)]

- "A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis", [2023-07] [ICLR 2024] [[paper](https://arxiv.org/abs/2307.12856)]

- "WebArena: A Realistic Web Environment for Building Autonomous Agents" [2023-07] [[paper](https://arxiv.org/abs/2307.13854)]

- "CogAgent: A Visual Language Model for GUI Agents" [2023-12] [[paper](https://arxiv.org/abs/2312.08914)]

- "GPT-4V(ision) is a Generalist Web Agent, if Grounded" [2024-01] [[paper](https://arxiv.org/abs/2401.01614)]

- "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models" [2024-01] [[paper](https://arxiv.org/abs/2401.13919)]

- "WebLINX: Real-World Website Navigation with Multi-Turn Dialogue" [2024-02] [[paper](https://arxiv.org/abs/2402.05930)]

- "OmniACT: A Dataset and Benchmark for Enabling Multimodal Generalist Autonomous Agents for Desktop and Web" [2024-02] [[paper](https://arxiv.org/abs/2402.17553)]

- "AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Web Navigating Agent" [2024-04] [[paper](https://arxiv.org/abs/2404.03648)]

- "WILBUR: Adaptive In-Context Learning for Robust and Accurate Web Agents" [2024-04] [[paper](https://arxiv.org/abs/2404.05902)]

- "AutoCrawler: A Progressive Understanding Web Agent for Web Crawler Generation" [2024-04] [[paper](https://arxiv.org/abs/2404.12753)]

- "GUICourse: From General Vision Language Models to Versatile GUI Agents" [2024-06] [[paper](https://arxiv.org/abs/2406.11317)]

- "NaviQAte: Functionality-Guided Web Application Navigation" [2024-09] [[paper](https://arxiv.org/abs/2409.10741)]

- "MobileVLM: A Vision-Language Model for Better Intra- and Inter-UI Understanding" [2024-09] [[paper](https://arxiv.org/abs/2409.14818)]

- "Multimodal Auto Validation For Self-Refinement in Web Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.00689)]

- "Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.05243)]

- "Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation" [2024-10] [[paper](https://arxiv.org/abs/2410.13232)]

- "Harnessing Webpage UIs for Text-Rich Visual Understanding" [2024-10] [[paper](https://arxiv.org/abs/2410.13824)]

- "AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.13825)]

- "Beyond Browsing: API-Based Web Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.16464)]

- "Large Language Models Empowered Personalized Web Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.17236)]

- "AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.17401)]

- "Auto-Intent: Automated Intent Discovery and Self-Exploration for Large Language Model Web Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.22552)]

- "OS-ATLAS: A Foundation Action Model for Generalist GUI Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.23218)]

- "From Context to Action: Analysis of the Impact of State Representation and Context on the Generalization of Multi-Turn Web Navigation Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.23555)]

- "AutoGLM: Autonomous Foundation Agents for GUIs" [2024-10] [[paper](https://arxiv.org/abs/2411.00820)]

- "WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning" [2024-11] [[paper](https://arxiv.org/abs/2411.02337)]

- "The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use" [2024-11] [[paper](https://arxiv.org/abs/2411.10323)]

- "ScribeAgent: Towards Specialized Web Agents Using Production-Scale Workflow Data" [2024-11] [[paper](https://arxiv.org/abs/2411.15004)]

- "ShowUI: One Vision-Language-Action Model for GUI Visual Agent" [2024-11] [[paper](https://arxiv.org/abs/2411.17465)]

- "Large Language Model-Brained GUI Agents: A Survey" [2024-11] [[paper](https://arxiv.org/abs/2411.18279)]

- "Free your mouse! Command Large Language Models to Generate Code to Format Word Documents" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.902/)]

- "Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction" [2024-12] [[paper](https://arxiv.org/abs/2412.04454)]

- "Falcon-UI: Understanding GUI Before Following User Instructions" [2024-12] [[paper](https://arxiv.org/abs/2412.09362)]

- "WEPO: Web Element Preference Optimization for LLM-based Web Navigation" [2024-12] [[paper](https://arxiv.org/abs/2412.10742)]

- "AutoDroid-V2: Boosting SLM-based GUI Agents via Code Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.18116)]

- "Beyond Pass or Fail: A Multi-dimensional Benchmark for Mobile UI Navigation" [2025-01] [[paper](https://arxiv.org/abs/2501.02863)]

- "WebWalker: Benchmarking LLMs in Web Traversal" [2025-01] [[paper](https://arxiv.org/abs/2501.07572)]

- "GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration" [2025-01] [[paper](https://arxiv.org/abs/2501.13896)]

- "UI-TARS: Pioneering Automated GUI Interaction with Native Agents" [2025-01] [[paper](https://arxiv.org/abs/2501.12326)]

- "Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks" [2025-03] [[paper](https://arxiv.org/abs/2503.00401)]

## 4. Code LLM for Low-Resource, Low-Level, and Domain-Specific Languages

- [**Ruby**] "On the Transferability of Pre-trained Language Models for Low-Resource Programming Languages" [2022-04] [ICPC 2022] [[paper](https://arxiv.org/abs/2204.09653)]

- [**Verilog**] "Benchmarking Large Language Models for Automated Verilog RTL Code Generation" [2022-12] [DATE 2023] [[paper](https://arxiv.org/abs/2212.11140)]

- [**OCL**] "On Codex Prompt Engineering for OCL Generation: An Empirical Study" [2023-03] [MSR 2023] [[paper](https://arxiv.org/abs/2303.16244)]

- [**Ansible-YAML**] "Automated Code generation for Information Technology Tasks in YAML through Large Language Models" [2023-05] [DAC 2023] [[paper](https://arxiv.org/abs/2305.02783)]

- [**Hansl**] "The potential of LLMs for coding with low-resource and domain-specific programming languages" [2023-07] [[paper](https://arxiv.org/abs/2307.13018)]

- [**Verilog**] "VeriGen: A Large Language Model for Verilog Code Generation" [2023-07] [[paper](https://arxiv.org/abs/2308.00708)]

- [**Verilog**] "RTLLM: An Open-Source Benchmark for Design RTL Generation with Large Language Model" [2023-08] [[paper](https://arxiv.org/abs/2308.05345)]

- [**Racket, OCaml, Lua, R, Julia**] "Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs" [2023-08] [[paper](https://arxiv.org/abs/2308.09895)]

- [**Verilog**] "VerilogEval: Evaluating Large Language Models for Verilog Code Generation" [2023-09] [ICCAD 2023] [[paper](https://arxiv.org/abs/2309.07544)]

- [**Verilog**] "RTLFixer: Automatically Fixing RTL Syntax Errors with Large Language Models" [2023-11] [[paper](https://arxiv.org/abs/2311.16543)]

- [**Verilog**] "Advanced Large Language Model (LLM)-Driven Verilog Development: Enhancing Power, Performance, and Area Optimization in Code Synthesis" [2023-12] [[paper](https://arxiv.org/abs/2312.01022)]

- [**Verilog**] "RTLCoder: Outperforming GPT-3.5 in Design RTL Generation with Our Open-Source Dataset and Lightweight Solution" [2023-12] [[paper](https://arxiv.org/abs/2312.08617)]

- [**Verilog**] "BetterV: Controlled Verilog Generation with Discriminative Guidance" [2024-02] [ICML 2024] [[paper](https://arxiv.org/abs/2402.03375)]

- [**R**] "Empirical Studies of Parameter Efficient Methods for Large Language Models of Code and Knowledge Transfer to R" [2024-03] [[paper](https://arxiv.org/abs/2405.01553)]

- [**Haskell**] "Investigating the Performance of Language Models for Completing Code in Functional Programming Languages: a Haskell Case Study" [2024-03] [[paper](https://arxiv.org/abs/2403.15185)]

- [**Verilog**] "A Multi-Expert Large Language Model Architecture for Verilog Code Generation" [2024-04] [[paper](https://arxiv.org/abs/2404.08029)]

- [**Verilog**] "CreativEval: Evaluating Creativity of LLM-Based Hardware Code Generation" [2024-04] [[paper](https://arxiv.org/abs/2404.08806)]

- [**Alloy**] "An Empirical Evaluation of Pre-trained Large Language Models for Repairing Declarative Formal Specifications" [2024-04] [[paper](https://arxiv.org/abs/2404.11050)]

- [**Verilog**] "Evaluating LLMs for Hardware Design and Test" [2024-04] [[paper](https://arxiv.org/abs/2405.02326)]

- [**Kotlin, Swift, and Rust**] "Software Vulnerability Prediction in Low-Resource Languages: An Empirical Study of CodeBERT and ChatGPT" [2024-04] [[paper](https://arxiv.org/abs/2404.17110)]

- [**Verilog**] "MEIC: Re-thinking RTL Debug Automation using LLMs" [2024-05] [[paper](https://arxiv.org/abs/2405.06840)]

- [**Bash**] "Tackling Execution-Based Evaluation for NL2Bash" [2024-05] [[paper](https://arxiv.org/abs/2405.06807)]

- [**Fortran, Julia, Matlab, R, Rust**] "Evaluating AI-generated code for C++, Fortran, Go, Java, Julia, Matlab, Python, R, and Rust" [2024-05] [[paper](https://arxiv.org/abs/2405.13101)]

- [**OpenAPI**] "Optimizing Large Language Models for OpenAPI Code Completion" [2024-05] [[paper](https://arxiv.org/abs/2405.15729)]

- [**Kotlin**] "Kotlin ML Pack: Technical Report" [2024-05] [[paper](https://arxiv.org/abs/2405.19250)]

- [**UCLID5**] "Synthetic Programming Elicitation for Text-to-Code in Very Low-Resource Programming and Formal Languages" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.03636)]

- [**Verilog**] "VerilogReader: LLM-Aided Hardware Test Generation" [2024-06] [[paper](https://arxiv.org/abs/2406.04373)]

- "Benchmarking Generative Models on Computational Thinking Tests in Elementary Visual Programming" [2024-06] [[paper](https://arxiv.org/abs/2406.09891)]

- [**Logo**] "Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment" [2024-06] [[paper](https://arxiv.org/abs/2406.11334)]

- [**Ansible YAML, Bash**] "DocCGen: Document-based Controlled Code Generation" [2024-06] [EMNLP 2024] [[paper](https://arxiv.org/abs/2406.11925)]

- [**Qiskit**] "Qiskit HumanEval: An Evaluation Benchmark For Quantum Code Generative Models" [2024-06] [[paper](https://arxiv.org/abs/2406.14712)]

- [**Perl, Golang, Swift**] "DistiLRR: Transferring Code Repair for Low-Resource Programming Languages" [2024-06] [[paper](https://arxiv.org/abs/2406.14867)]

- [**Verilog**] "AssertionBench: A Benchmark to Evaluate Large-Language Models for Assertion Generation" [2024-06] [[paper](https://arxiv.org/abs/2406.18627)]

- "A Comparative Study of DSL Code Generation: Fine-Tuning vs. Optimized Retrieval Augmentation" [2024-07] [[paper](https://arxiv.org/abs/2407.02742)]

- [**Json, XLM, YAML**] "ConCodeEval: Evaluating Large Language Models for Code Constraints in Domain-Specific Languages" [2024-07] [[paper](https://arxiv.org/abs/2407.03387)]

- [**Verilog**] "AutoBench: Automatic Testbench Generation and Evaluation Using LLMs for HDL Design" [2024-07] [[paper](https://arxiv.org/abs/2407.03891)]

- [**Verilog**] "CodeV: Empowering LLMs for Verilog Generation through Multi-Level Summarization" [2024-07] [[paper](https://arxiv.org/abs/2407.10424)]

- [**Verilog**] "ITERTL: An Iterative Framework for Fine-tuning LLMs for RTL Code Generation" [2024-07] [[paper](https://arxiv.org/abs/2407.12022)]

- [**Verilog**] "OriGen:Enhancing RTL Code Generation with Code-to-Code Augmentation and Self-Reflection" [2024-07] [[paper](https://arxiv.org/abs/2407.16237)]

- [**Verilog**] "Large Language Model for Verilog Generation with Golden Code Feedback" [2024-07] [[paper](https://arxiv.org/abs/2407.18271)]

- [**Verilog**] "AutoVCoder: A Systematic Framework for Automated Verilog Code Generation using LLMs" [2024-07] [[paper](https://arxiv.org/abs/2407.18333)]

- [**RPA**] "Plan with Code: Comparing approaches for robust NL to DSL generation" [2024-08] [[paper](https://arxiv.org/abs/2408.08335)]

- [**Verilog**] "VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Tool" [2024-08] [[paper](https://arxiv.org/abs/2408.08927)]

- [**Verilog**] "Revisiting VerilogEval: Newer LLMs, In-Context Learning, and Specification-to-RTL Tasks" [2024-08] [[paper](https://arxiv.org/abs/2408.11053)]

- [**MaxMSP, Web Audio**] "Benchmarking LLM Code Generation for Audio Programming with Visual Dataflow Languages" [2024-09] [[paper](https://arxiv.org/abs/2409.00856)]

- [**Verilog**] "RTLRewriter: Methodologies for Large Models aided RTL Code Optimization" [2024-09] [[paper](https://arxiv.org/abs/2409.11414)]

- [**Verilog**] "CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair" [2024-09] [ICLR 2025] [[paper](https://arxiv.org/abs/2409.12993)]

- [**Bash**] "ScriptSmith: A Unified LLM Framework for Enhancing IT Operations via Automated Bash Script Generation, Assessment, and Refinement" [2024-09] [[paper](https://arxiv.org/abs/2409.17166)]

- [**Survey**] "Survey on Code Generation for Low resource and Domain Specific Programming Languages" [2024-10] [[paper](https://arxiv.org/abs/2410.03981)]

- [**R**] "Do Current Language Models Support Code Intelligence for R Programming Language?" [2024-10] [[paper](https://arxiv.org/abs/2410.07793)]

- [**PLC**] "Agents4PLC: Automating Closed-loop PLC Code Generation and Verification in Industrial Control Systems using LLM-based Agents" [2024-10] [[paper](https://arxiv.org/abs/2410.14209)]

- [**Lua**] "Evaluating Quantized Large Language Models for Code Generation on Low-Resource Language Benchmarks" [2024-10] [[paper](https://arxiv.org/abs/2410.14766)]

- "Improving Parallel Program Performance Through DSL-Driven Code Generation with LLM Optimizers" [2024-10] [[paper](https://arxiv.org/abs/2410.15625)]

- [**R, D, Racket, Bash**]: "Bridge-Coder: Unlocking LLMs' Potential to Overcome Language Gaps in Low-Resource Code" [2024-10] [[paper](https://arxiv.org/abs/2410.18957)]

- [**SPICE**]: "SPICEPilot: Navigating SPICE Code Generation and Simulation with AI Guidance" [2024-10] [[paper](https://arxiv.org/abs/2410.20553)]

- [**IEC 61131-3 ST**]: "Training LLMs for Generating IEC 61131-3 Structured Text with Online Feedback" [2024-10] [[paper](https://arxiv.org/abs/2410.22159)]

- [**Verilog**] "MetRex: A Benchmark for Verilog Code Metric Reasoning Using LLMs" [2024-11] [[paper](https://arxiv.org/abs/2411.03471)]

- [**Verilog**] "CorrectBench: Automatic Testbench Generation with Functional Self-Correction using LLMs for HDL Design" [2024-11] [[paper](https://arxiv.org/abs/2411.08510)]

- [**MUMPS, ALC**] "Leveraging LLMs for Legacy Code Modernization: Challenges and Opportunities for LLM-Generated Documentation" [2024-11] [[paper](https://arxiv.org/abs/2411.14971)]

- [**Power Query M, OfficeScript, Excel formulas**] "RAR: Retrieval-augmented retrieval for code generation in low resource languages" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.1199/)]

- [**ST**] "A Multi-Agent Framework for Extensible Structured Text Generation in PLCs" [2024-12] [[paper](https://arxiv.org/abs/2412.02410)]

- [**Verilog**] "PromptV: Leveraging LLM-powered Multi-Agent Prompting for High-quality Verilog Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.11014)]

- [**HPC**] "HPC-Coder-V2: Studying Code LLMs Across Low-Resource Parallel Languages" [2024-12] [[paper](https://arxiv.org/abs/2412.15178)]

- [**Verilog**] "RTLSquad: Multi-Agent Based Interpretable RTL Design" [2025-01] [[paper](https://arxiv.org/abs/2501.05470)]

- [**G**] "GLLM: Self-Corrective G-Code Generation using Large Language Models with User Feedback" [2025-01] [[paper](https://arxiv.org/abs/2501.17584)]

- [**Julia, Lua, R, Racket**] "Enhancing Code Generation for Low-Resource Languages: No Silver Bullet" [2025-01] [[paper](https://arxiv.org/abs/2501.19085)]

- [**F***] "Building A Proof-Oriented Programmer That Is 64% Better Than GPT-4o Under Data Scarsity" [2025-02] [[paper](https://arxiv.org/abs/2502.11901)]

- "Exploring Code Language Models for Automated HLS-based Hardware Generation: Benchmark, Infrastructure and Analysis" [2025-02] [ASP-DAC 2025] [[paper](https://arxiv.org/abs/2502.13921)]

- [**Alloy***] "On the Effectiveness of Large Language Models in Writing Alloy Formulas" [2025-02] [[paper](https://arxiv.org/abs/2502.15441)]

- [**Solidity**] "SolEval: Benchmarking Large Language Models for Repository-level Solidity Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.18793)]

- [**PennyLane**] "PennyLang: Pioneering LLM-Based Quantum Code Generation with a Novel PennyLane-Centric Dataset" [2025-03] [[paper](https://arxiv.org/abs/2503.02497)]

- [**Verilog**] "VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric" [2025-03] [[paper](https://arxiv.org/abs/2503.16514)]

- [**Modelica**] "ModiGen: A Large Language Model-Based Workflow for Multi-Task Modelica Code Generation" [2025-03] [[paper](https://arxiv.org/abs/2503.18460)]

- [**Excel**] "Synthetic Function Demonstrations Improve Generation in Low-Resource Programming Languages" [2025-03] [[paper](https://arxiv.org/abs/2503.18760)]

- "RTLRepoCoder: Repository-Level RTL Code Completion through the Combination of Fine-Tuning and Retrieval Augmentation" [2025-04] [[paper](https://arxiv.org/abs/2504.08862)]

- [**Verilog**] "SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning" [2025-04] [[paper](https://arxiv.org/abs/2504.10369)]

- [**Verilog**] "ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model" [2025-04] [[paper](https://arxiv.org/abs/2504.14560)]

- [**Verilog**] "VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation" [2025-04] [[paper](https://arxiv.org/abs/2504.15659)]

- [**Chisel**] "ChiseLLM: Unleashing the Power of Reasoning LLMs for Chisel Agile Hardware Development" [2025-04] [[paper](https://arxiv.org/abs/2504.19144)]

- [**Verilog**] "ComplexVCoder: An LLM-Driven Framework for Systematic Generation of Complex Verilog Code" [2025-04] [[paper](https://arxiv.org/abs/2504.20653)]

- [**Lean**] "CLEVER: A Curated Benchmark for Formally Verified Code Generation" [2025-05] [[paper](https://arxiv.org/abs/2505.13938)]

- [**Verilog**] "VeriThoughts: Enabling Automated Verilog Code Generation using Reasoning and Formal Verification" [2025-05] [[paper](https://arxiv.org/abs/2505.20302)]

## 5. Methods/Models for Downstream Tasks

For each task, the first column contains non-neural methods (e.g. n-gram, TF-IDF, and (occasionally) static program analysis); the second column contains non-Transformer neural methods (e.g. LSTM, CNN, GNN); the third column contains Transformer based methods (e.g. BERT, GPT, T5).

<p align='center'>
<img src='imgs/downstream-1.png' style='width: 100%; '>
<img src='imgs/downstream-2.png' style='width: 100%; '>
<img src='imgs/downstream-3.png' style='width: 100%; '>
<img src='imgs/downstream-4.png' style='width: 100%; '>
<img src='imgs/downstream-5.png' style='width: 100%; '>
</p>

### Code Generation

- "Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency" [2023-09] [ACL 2024] [[paper](https://arxiv.org/abs/2309.17272)]

- "Self-Infilling Code Generation" [2023-11] [ICML 2024] [[paper](https://arxiv.org/abs/2311.17972)]

- "JumpCoder: Go Beyond Autoregressive Coder via Online Modification" [2024-01] [ACL 2024] [[paper](https://arxiv.org/abs/2401.07870)]

- "The Larger the Better? Improved LLM Code-Generation via Budget Reallocation" [2024-03] [[paper](https://arxiv.org/abs/2404.00725)]

- "Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models" [2024-03] [ACL 2024] [[paper](https://arxiv.org/abs/2403.04811)]

- "Comments as Natural Logic Pivots: Improve Code Generation via Comment Perspective" [2024-04] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2404.07549)]

- "Distilling Algorithmic Reasoning from LLMs via Explaining Solution Programs" [2024-04] [[paper](https://arxiv.org/abs/2404.08148)]

- "Quality Assessment of Prompts Used in Code Generation" [2024-04] [[paper](https://arxiv.org/abs/2404.10155)]

- "Model Cascading for Code: Reducing Inference Costs with Model Cascading for LLM Based Code Generation" [2024-05] [[paper](https://arxiv.org/abs/2405.15842)]

- "A Survey on Large Language Models for Code Generation" [2024-06] [[paper](https://arxiv.org/abs/2406.00515)]

- "Is Programming by Example solved by LLMs?" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.08316)]

- "Benchmarks and Metrics for Evaluations of Code Generation: A Critical Review" [2024-06] [[paper](https://arxiv.org/abs/2406.12655)]

- "MPCODER: Multi-user Personalized Code Generator with Explicit and Implicit Style Representation Learning" [2024-06] [ACL 2024] [[paper](https://arxiv.org/abs/2406.17255)]

- "Revisiting the Impact of Pursuing Modularity for Code Generation" [2024-07] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2407.11406)]

- "Evaluating Long Range Dependency Handling in Code Generation Models using Multi-Step Key Retrieval" [2024-07] [[paper](https://arxiv.org/abs/2407.21049)]

- "When to Stop? Towards Efficient Code Generation in LLMs with Excess Token Prevention" [2024-07] [[paper](https://arxiv.org/abs/2407.20042)]

- "Assessing Programming Task Difficulty for Efficient Evaluation of Large Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.21227)]

- "ArchCode: Incorporating Software Requirements in Code Generation with Large Language Models" [2024-08] [ACL 2024] [[paper](https://www.arxiv.org/abs/2408.00994)]

- "Fine-tuning Language Models for Joint Rewriting and Completion of Code with Potential Bugs" [2024-08] [ACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-acl.938/)]

- "Selective Prompt Anchoring for Code Generation" [2024-08] [[paper](https://arxiv.org/abs/2408.09121)]

- "Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer" [2024-08] [[paper](https://arxiv.org/abs/2408.09701)]

- "No Man is an Island: Towards Fully Automatic Programming by Code Search, Code Generation and Program Repair" [2024-09] [[paper](https://arxiv.org/abs/2409.03267)]

- "Planning In Natural Language Improves LLM Search For Code Generation" [2024-09] [ICLR 2025] [[paper](https://arxiv.org/abs/2409.03733)]

- "Multi-Programming Language Ensemble for Code Generation in Large Language Model" [2024-09] [[paper](https://arxiv.org/abs/2409.04114)]

- "USCD: Improving Code Generation of LLMs by Uncertainty-Aware Selective Contrastive Decoding" [2024-09] [[paper](https://arxiv.org/abs/2409.05923)]

- "Eliciting Instruction-tuned Code Language Models' Capabilities to Utilize Auxiliary Function for Code Generation" [2024-09] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2409.13928)]

- "Selection of Prompt Engineering Techniques for Code Generation through Predicting Code Complexity" [2024-09] [[paper](https://arxiv.org/abs/2409.16416)]

- "Horizon-Length Prediction: Advancing Fill-in-the-Middle Capabilities for Code Generation with Lookahead Planning" [2024-10] [[paper](https://arxiv.org/abs/2410.03103)]

- "Showing LLM-Generated Code Selectively Based on Confidence of LLMs" [2024-10] [[paper](https://arxiv.org/abs/2410.03234)]

- "AutoFeedback: An LLM-based Framework for Efficient and Accurate API Request Generation" [2024-10] [[paper](https://arxiv.org/abs/2410.06943)]

- "Enhancing LLM Agents for Code Generation with Possibility and Pass-rate Prioritized Experience Replay" [2024-10] [[paper](https://arxiv.org/abs/2410.12236)]

- "From Solitary Directives to Interactive Encouragement! LLM Secure Code Generation by Natural Language Prompting" [2024-10] [[paper](https://arxiv.org/abs/2410.14321)]

- "Self-Explained Keywords Empower Large Language Models for Code Generation" [2024-10] [[paper](https://arxiv.org/abs/2410.15966)]

- "Context-Augmented Code Generation Using Programming Knowledge Graphs" [2024-10] [[paper](https://arxiv.org/abs/2410.18251)]

- "In-Context Code-Text Learning for Bimodal Software Engineering" [2024-10] [[paper](https://arxiv.org/abs/2410.18107)]

- "Combining LLM Code Generation with Formal Specifications and Reactive Program Synthesis" [2024-10] [[paper](https://arxiv.org/abs/2410.19736)]

- "Less is More: DocString Compression in Code Generation" [2024-10] [[paper](https://arxiv.org/abs/2410.22793)]

- "Multi-Programming Language Sandbox for LLMs" [2024-10] [[paper](https://arxiv.org/abs/2410.23074)]

- "Personality-Guided Code Generation Using Large Language Models" [2024-10] [[paper](https://arxiv.org/abs/2411.00006)]

- "Scattered Forest Search: Smarter Code Space Exploration with LLMs" [2024-11] [[paper](https://arxiv.org/abs/2411.05010)]

- "Anchor Attention, Small Cache: Code Generation with Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.06680)]

- "ROCODE: Integrating Backtracking Mechanism and Program Analysis in Large Language Models for Code Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.07112)]

- "SRA-MCTS: Self-driven Reasoning Aurmentation with Monte Carlo Tree Search for Enhanced Code Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.11053)]

- "Language-to-Code Translation with a Single Labeled Example" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.462/)]

- "VeCoGen: Automating Generation of Formally Verified C Code with Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.19275)]

- "Seed-CTS: Unleashing the Power of Tree Search for Superior Performance in Competitive Coding Tasks" [2024-12] [[paper](https://arxiv.org/abs/2412.12544)]

- "LoGFiLM: Fine-Tuning A Large Language Model for Automated Generation of Log Statements" [2024-12] [[paper](https://arxiv.org/abs/2412.18835)]

- "The Impact of Prompt Programming on Function-Level Code Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.20545)]

- "Leveraging Metamemory Mechanisms for Enhanced Data-Free Code Generation in LLMs" [2025-01] [[paper](https://arxiv.org/abs/2501.07892)]

- "GREEN-CODE: Optimizing Energy Efficiency in Large Language Models for Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.11006)]

- "Chain of Grounded Objectives: Bridging Process and Goal-oriented Prompting for Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.13978)]

- "CoCoEvo: Co-Evolution of Programs and Test Cases to Enhance Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.10802)]

- "Boost, Disentangle, and Customize: A Robust System2-to-System1 Pipeline for Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.12492)]

- "Learning to Solve and Verify: A Self-Play Framework for Code and Test Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.14948)]

- "Thinking Before Running! Efficient Code Generation with Thorough Exploration and Optimal Refinement" [2025-02] [[paper](https://arxiv.org/abs/2502.17442)]

- "Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs?" [2025-03] [[paper](https://arxiv.org/abs/2503.05507)]

- "Enhancing High-Quality Code Generation in Large Language Models with Comparative Prefix-Tuning" [2025-03] [[paper](https://arxiv.org/abs/2503.09020)]

- "Modularization is Better: Effective Code Generation with Modular Prompting" [2025-03] [[paper](https://arxiv.org/abs/2503.12483)]

- "A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation" [2025-03] [[paper](https://arxiv.org/abs/2503.12899)]

- "Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs" [2025-03] [[paper](https://arxiv.org/abs/2503.15341)]

- "CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis" [2025-03] [[ppaer](https://arxiv.org/abs/2503.23145)]

- "Prism: Dynamic and Flexible Benchmarking of LLMs Code Generation with Monte Carlo Tree Search" [2025-04] [[paper](https://arxiv.org/abs/2504.05500)]

- "Enhancing Chart-to-Code Generation in Multimodal Large Language Models via Iterative Dual Preference Learning" [2025-04] [[paper](https://arxiv.org/abs/2504.02906)]

- "From Token to Line: Enhancing Code Generation with a Long-Term Perspective" [2025-04] [[paper](https://arxiv.org/abs/2504.07433)]

### Code RAG

- "CodeGRAG: Extracting Composed Syntax Graphs for Retrieval Augmented Cross-Lingual Code Generation" [2024-05] [[paper](https://arxiv.org/abs/2405.02355)]

- "Prompt-based Code Completion via Multi-Retrieval Augmented Generation" [2024-05] [[paper](https://arxiv.org/abs/2405.07530)]

- "A Lightweight Framework for Adaptive Retrieval In Code Completion With Critique Model" [2024-06] [[papaer](https://arxiv.org/abs/2406.10263)]

- "Preference-Guided Refactored Tuning for Retrieval Augmented Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.15895)]

- "Building A Coding Assistant via the Retrieval-Augmented Language Model" [2024-10] [[paper](https://arxiv.org/abs/2410.16229)]

- "DroidCoder: Enhanced Android Code Completion with Context-Enriched Retrieval-Augmented Generation" [2024-10] [ASE 2024] [[paper](https://dl.acm.org/doi/10.1145/3691620.3695063)]

- "Assessing the Answerability of Queries in Retrieval-Augmented Code Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.05547)]

- "EvoR: Evolving Retrieval for Code Generation" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.143/)]

- "PERC: Plan-As-Query Example Retrieval for Underrepresented Code Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.12447)]

- "An Empirical Study of Retrieval-Augmented Code Generation: Challenges and Opportunities" [2025-01] [[paper](https://arxiv.org/abs/2501.13742)]

- "CODEPROMPTZIP: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs" [2025-02] [[paper](https://arxiv.org/abs/2502.14925)]

- "CodeSwift: Accelerating LLM Inference for Efficient Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.17139)]

- "SOSecure: Safer Code Generation with RAG and StackOverflow Discussions" [2025-03] [[paper](https://arxiv.org/abs/2503.13654)]

- "When LLMs Meet API Documentation: Can Retrieval Augmentation Aid Code Generation Just as It Helps Developers?" [2025-03] [[paper](https://arxiv.org/abs/2503.15231)]

- "What to Retrieve for Effective Retrieval-Augmented Code Generation? An Empirical Study and Beyond" [2025-03] [[paper](https://arxiv.org/abs/2503.20589)]

### Code Ranking

- "Fault-Aware Neural Code Rankers" [2022-06] [NeurIPS 2022] [[paper](https://arxiv.org/abs/2206.03865)]

- "Coder Reviewer Reranking for Code Generation" [2022-11] [ICML 2023] [[paper](https://arxiv.org/abs/2211.16490)]

- "LEVER: Learning to Verify Language-to-Code Generation with Execution" [2023-02] [ICML 2023] [[paper](https://arxiv.org/abs/2302.08468)]

- "Functional Overlap Reranking for Neural Code Generation" [2023-10] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2311.03366)]

- "Top Pass: Improve Code Generation by Pass@k-Maximized Code Ranking" [2024-08] [[paper](https://arxiv.org/abs/2408.05715)]

- "DOCE: Finding the Sweet Spot for Execution-Based Code Generation" [2024-08] [[paper](https://arxiv.org/abs/2408.13745)]

- "Sifting through the Chaff: On Utilizing Execution Feedback for Ranking the Generated Code Candidates" [2024-08] [[paper](https://arxiv.org/abs/2408.13976)]

- "B4: Towards Optimal Assessment of Plausible Code Solutions with Plausible Tests" [2024-09] [[paper](https://arxiv.org/abs/2409.08692)]

- "Learning Code Preference via Synthetic Evolution" [2024-10] [[paper](https://arxiv.org/abs/2410.03837)]

- "Condor: A Code Discriminator Integrating General Semantics with Code Details" [2024-12] [[paper](https://arxiv.org/abs/2412.17429)]

- "Scoring Verifiers: Evaluating Synthetic Verification in Code and Reasoning" [2025-02] [[paper](https://arxiv.org/abs/2502.13820)]

- "Pragmatic Reasoning improves LLM Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.15835)]

- "SweRank: Software Issue Localization with Code Ranking" [2025-05] [[paper](https://arxiv.org/abs/2505.07849)]

### Code Translation

- "Tree-to-tree Neural Networks for Program Translation" [2018-02] [NeurIPS 2018] [[paper](https://arxiv.org/abs/1802.03691)]

- "Program Language Translation Using a Grammar-Driven Tree-to-Tree Model" [2018-07] [[paper](https://arxiv.org/abs/1807.01784)]

- "Unsupervised Translation of Programming Languages" [2020-06] [NeurIPS 2020] [[paper](https://arxiv.org/abs/2006.03511)]

- "Leveraging Automated Unit Tests for Unsupervised Code Translation" [2021-10] [ICLR 2022] [paper](https://arxiv.org/abs/2110.06773)]

- "Code Translation with Compiler Representations" [2022-06] [ICLR 2023] [[paper](https://arxiv.org/abs/2207.03578)]

- "Multilingual Code Snippets Training for Program Translation" [2022-06] [AAAI 2022] [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21434)]

- "BabelTower: Learning to Auto-parallelized Program Translation" [2022-07] [ICML 2022] [[paper](https://proceedings.mlr.press/v162/wen22b.html)]

- "Syntax and Domain Aware Model for Unsupervised Program Translation" [2023-02] [ICSE 2023] [[paper](https://arxiv.org/abs/2302.03908)]

- "CoTran: An LLM-based Code Translator using Reinforcement Learning with Feedback from Compiler and Symbolic Execution" [2023-06] [[paper](https://arxiv.org/abs/2306.06755)]

- "Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code" [2023-08] [ICSE 2024] [[paper](https://arxiv.org/abs/2308.03109)]

- "On the Evaluation of Neural Code Translation: Taxonomy and Benchmark", 2023-08, ASE 2023, [[paper](https://arxiv.org/abs/2308.08961)]

- "Program Translation via Code Distillation" [2023-10] [EMNLP 2023] [[paper](https://arxiv.org/abs/2310.11476)]

- "Explain-then-Translate: An Analysis on Improving Program Translation with Self-generated Explanations" [2023-11] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2311.07070)]

- "Exploring the Impact of the Output Format on the Evaluation of Large Language Models for Code Translation" [2024-03] [[paper](https://arxiv.org/abs/2403.17214)]

- "Exploring and Unleashing the Power of Large Language Models in Automated Code Translation" [2024-04] [[paper](https://arxiv.org/abs/2404.14646)]

- "VERT: Verified Equivalent Rust Transpilation with Few-Shot Learning" [2024-04] [[paper](https://arxiv.org/abs/2404.18852)]

- "Towards Translating Real-World Code with LLMs: A Study of Translating to Rust" [2024-05] [[paper](https://arxiv.org/abs/2405.11514)]

- "An interpretable error correction method for enhancing code-to-code translation" [2024-05] [ICLR 2024] [[paper](https://openreview.net/forum?id=fVxIEHGnVT&noteId=CyxZE2UbHF)]

- "LASSI: An LLM-based Automated Self-Correcting Pipeline for Translating Parallel Scientific Codes" [2024-06] [[paper](https://arxiv.org/abs/2407.01638)]

- "Rectifier: Code Translation with Corrector via LLMs" [2024-07] [[paper](https://arxiv.org/abs/2407.07472)]

- "Enhancing Code Translation in Language Models with Few-Shot Learning via Retrieval-Augmented Generation" [2024-07] [[paper](https://arxiv.org/abs/2407.19619)]

- "A Joint Learning Model with Variational Interaction for Multilingual Program Translation" [2024-08] [[paper](https://arxiv.org/abs/2408.14515)]

- "Automatic Library Migration Using Large Language Models: First Results" [2024-08] [[paper](https://arxiv.org/abs/2408.16151)]

- "Context-aware Code Segmentation for C-to-Rust Translation using Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.10506)]

- "TRANSAGENT: An LLM-Based Multi-Agent System for Code Translation" [2024-10] [[paper](https://arxiv.org/abs/2409.19894)]

- "Unraveling the Potential of Large Language Models in Code Translation: How Far Are We?" [2024-10] [[paper](https://arxiv.org/abs/2410.09812)]

- "CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming" [2024-10] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2410.20527)]

- "A test-free semantic mistakes localization framework in Neural Code Translation" [2024-10] [[paper](https://arxiv.org/abs/2410.22818)]

- "Repository-Level Compositional Code Translation and Validation" [2024-10] [[paper](https://arxiv.org/abs/2410.24117)]

- "Leveraging Large Language Models for Code Translation and Software Development in Scientific Computing" [2024-10] [[paper](https://arxiv.org/abs/2410.24119)]

- "InterTrans: Leveraging Transitive Intermediate Translations to Enhance LLM-based Code Translation" [2024-11] [[paper](https://arxiv.org/abs/2411.01063)]

- "Translating C To Rust: Lessons from a User Study" [2024-11] [[paper](https://arxiv.org/abs/2411.14174)]

- "Specification-Driven Code Translation Powered by Large Language Models: How Far Are We?" [2024-12] [[paper](https://arxiv.org/abs/2412.04590)]

- "Enhancing Cross-Language Code Translation via Task-Specific Embedding Alignment in Retrieval-Augmented Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.05159)]

- "Scalable, Validated Code Translation of Entire Projects using Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.08035)]

- "Syzygy: Dual Code-Test C to (safe) Rust Translation using LLMs and Dynamic Analysis" [2024-12] [[paper](https://arxiv.org/abs/2412.14234)]

- "I Can't Share Code, but I need Translation -- An Empirical Study on Code Translation through Federated LLM" [2025-01] [[paper](https://arxiv.org/abs/2501.05724)]

- "Guided Debugging of Auto-Translated Code Using Differential Testing" [2025-01] [[paper](https://arxiv.org/abs/2501.09475)]

- "C2SaferRust: Transforming C Projects into Safer Rust with NeuroSymbolic Techniques" [2025-01] [[paper](https://arxiv.org/abs/2501.14257)]

- "ExeCoder: Empowering Large Language Models with Executability Representation for Code Translation" [2025-01] [[paper](https://arxiv.org/abs/2501.18460)]

- "LLM-Driven Multi-step Translation from C to Rust using Static Analysis" [2025-03] [[paper](https://arxiv.org/abs/2503.12511)]

- "Enhancing LLM-based Code Translation in Repository Context via Triple Knowledge-Augmented" [2025-03] [[paper](https://arxiv.org/abs/2503.18305)]

- "Enhancing LLMs in Long Code Translation through Instrumentation and Program State Alignment" [2025-04] [[paper](https://arxiv.org/abs/2504.02017)]

- "A Systematic Literature Review on Neural Code Translation" [2025-05] [[paper](https://arxiv.org/abs/2505.07425)]

- "Large Language Model-Powered Agent for C to Rust Code Translation" [2025-05] [[paper](https://arxiv.org/abs/2505.15858)]

- "On-Policy Optimization with Group Equivalent Preference for Multi-Programming Language Understanding" [2025-05] [[paper](https://arxiv.org/abs/2505.12723)]

### Code Commenting and Summarization

- "A Transformer-based Approach for Source Code Summarization" [2020-05] [ACL 2020] [[paper](https://arxiv.org/abs/2005.00653)]

- "Code Summarization with Structure-induced Transformer" [2020-12] [ACL 2021 Findings] [[paper](https://arxiv.org/abs/2012.14710)]

- "Code Structure Guided Transformer for Source Code Summarization" [2021-04] [ACM TSEM] [[paper](https://arxiv.org/abs/2104.09340)]

- "M2TS: Multi-Scale Multi-Modal Approach Based on Transformer for Source Code Summarization" [2022-03] [ICPC 2022] [[paper](https://arxiv.org/abs/2203.09707)]

- "AST-trans: code summarization with efficient tree-structured attention" [2022-05] [ICSE 2022] [[paper](https://dl.acm.org/doi/abs/10.1145/3510003.3510224)]

- "CoSS: Leveraging Statement Semantics for Code Summarization" [2023-03] [IEEE TSE] [[paper](https://ieeexplore.ieee.org/document/10068264)]

- "Automatic Code Summarization via ChatGPT: How Far Are We?" [2023-05] [[paper](https://arxiv.org/abs/2305.12865)]

- "Semantic Similarity Loss for Neural Source Code Summarization" [2023-08] [[paper](https://arxiv.org/abs/2308.07429)]

- "Distilled GPT for Source Code Summarization" [2023-08] [ASE] [[paper](https://arxiv.org/abs/2308.14731)]

- "CSA-Trans: Code Structure Aware Transformer for AST" [2024-04] [[paper](https://arxiv.org/abs/2404.05767)]

- "Analyzing the Performance of Large Language Models on Code Summarization" [2024-04] [[paper](https://arxiv.org/abs/2404.08018)]

- "Enhancing Trust in LLM-Generated Code Summaries with Calibrated Confidence Scores" [2024-04] [[paper](https://arxiv.org/abs/2404.19318)]

- "DocuMint: Docstring Generation for Python using Small Language Models" [2024-05] [[paper](https://arxiv.org/abs/2405.10243)] [[repo](https://github.com/Docu-Mint/DocuMint)]

- "Natural Is The Best: Model-Agnostic Code Simplification for Pre-trained Large Language Models" [2024-05] [[paper](https://arxiv.org/abs/2405.11196)]

- "Exploring the Efficacy of Large Language Models (GPT-4) in Binary Reverse Engineering" [2024-06] [[paper](https://arxiv.org/abs/2406.06637)]

- "Identifying Inaccurate Descriptions in LLM-generated Code Comments via Test Execution" [2024-06] [[paper](https://arxiv.org/abs/2406.14836)]

- "MALSIGHT: Exploring Malicious Source Code and Benign Pseudocode for Iterative Binary Malware Summarization" [2024-06] [[paper](https://arxiv.org/abs/2406.18379)]

- "Source Code Summarization in the Era of Large Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.07959)]

- "Natural Language Outlines for Code: Literate Programming in the LLM Era" [2024-08] [[paper](https://arxiv.org/abs/2408.04820)]

- "Context-aware Code Summary Generation" [2024-08] [[paper](https://arxiv.org/abs/2408.09006)]

- "AUTOGENICS: Automated Generation of Context-Aware Inline Comments for Code Snippets on Programming Q&A Sites Using LLM" [2024-08] [[paper](https://arxiv.org/abs/2408.15411)]

- "LLMs as Evaluators: A Novel Approach to Evaluate Bug Report Summarization" [2024-09] [[paper](https://arxiv.org/abs/2409.00630)]

- "Generating Equivalent Representations of Code By A Self-Reflection Approach" [2024-10] [[paper](https://arxiv.org/abs/2410.03351)]

- "A review of automatic source code summarization" [2024-10] [Empirical Software Engineering] [[paper](https://link.springer.com/article/10.1007/s10664-024-10553-6)]

- "Can Large Language Models Serve as Evaluators for Code Summarization?" [2024-12] [[paper](https://arxiv.org/abs/2412.01333)]

- "Selective Shot Learning for Code Explanation" [2024-12] [[paper](https://arxiv.org/abs/2412.12852)]

- "Semantic Captioning: Benchmark Dataset and Graph-Aware Few-Shot In-Context Learning for SQL2Text" [2025-01] [[paper](https://arxiv.org/abs/2501.03166)]

- "Hierarchical Repository-Level Code Summarization for Business Applications Using Local LLMs" [2025-01] [[paper](https://arxiv.org/abs/2501.07857)]

- "From Critique to Clarity: A Pathway to Faithful and Personalized Code Explanations with Large Language Models" [2025-01] [[paper](https://arxiv.org/abs/2501.14731)]

- "METAMON: Finding Inconsistencies between Program Documentation and Behavior using Metamorphic LLM Queries" [2025-02] [[paper](https://arxiv.org/abs/2502.02794)]

- "Should Code Models Learn Pedagogically? A Preliminary Evaluation of Curriculum Learning for Real-World Software Engineering Tasks" [2025-02] [[paper](https://arxiv.org/abs/2502.03806)]

- "Optimizing Datasets for Code Summarization: Is Code-Comment Coherence Enough?" [2025-02] [[paper](https://arxiv.org/abs/2502.07611)]

- "Code Summarization Beyond Function Level" [2025-02] [[paper](https://arxiv.org/abs/2502.16704)]

- "ReadMe.LLM: A Framework to Help LLMs Understand Your Library" [2025-04] [[paper](https://arxiv.org/abs/2504.09798)]

- "DocAgent: A Multi-Agent System for Automated Code Documentation Generation" [2025-04] [[paper](https://arxiv.org/abs/2504.08725)]

### Program Repair

- "CURE: Code-Aware Neural Machine Translation for Automatic Program Repair" [2021-02] [ICSE 2021] [[paper](https://arxiv.org/abs/2103.00073)]

- "DeepDebug: Fixing Python Bugs Using Stack Traces, Backtranslation, and Code Skeletons" [2021-05] [[paper](https://arxiv.org/abs/2105.09352)]

- "Break-It-Fix-It: Unsupervised Learning for Program Repair" [2021-06] [ICML 2021] [[paper](https://arxiv.org/abs/2106.06600)]

- "TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer" [2021-07] [ICML 2021] [[paper](https://proceedings.mlr.press/v139/berabi21a.html)]

- "Automated Repair of Programs from Large Language Models" [2022-05] [ICSE 2023] [[paper](https://arxiv.org/abs/2205.10583)]

- "Less Training, More Repairing Please: Revisiting Automated Program Repair via Zero-shot Learning" [2022-07] [ESEC/FSE 2022] [[paper](https://arxiv.org/abs/2207.08281)]

- "Repair Is Nearly Generation: Multilingual Program Repair with LLMs" [2022-08] [AAAI 2023] [[paper](https://arxiv.org/abs/2208.11640)]

- "Practical Program Repair in the Era of Large Pre-trained Language Models" [2022-10] [[paper](https://arxiv.org/abs/2210.14179)]

- "VulRepair: a T5-based automated software vulnerability repair" [2022-11] [ESEC/FSE 2022] [[paper](https://dl.acm.org/doi/abs/10.1145/3540250.3549098)]

- "Conversational Automated Program Repair" [2023-01] [[paper](https://arxiv.org/abs/2301.13246)]

- "Impact of Code Language Models on Automated Program Repair" [2023-02] [ICSE 2023] [[paper](https://arxiv.org/abs/2302.05020)]

- "InferFix: End-to-End Program Repair with LLMs" [2023-03] [ESEC/FSE 2023] [[paper](https://arxiv.org/abs/2303.07263)]

- "Enhancing Automated Program Repair through Fine-tuning and Prompt Engineering" [2023-04] [[paper](https://arxiv.org/abs/2304.07840)]

- "A study on Prompt Design, Advantages and Limitations of ChatGPT for Deep Learning Program Repair" [2023-04] [[paper](https://arxiv.org/abs/2304.08191)]

- "Domain Knowledge Matters: Improving Prompts with Fix Templates for Repairing Python Type Errors" [2023-06] [ICSE 2024] [[paper](https://arxiv.org/abs/2306.01394)]

- "RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair" [2023-12] [[paper](https://arxiv.org/abs/2312.15698)]

- "The Fact Selection Problem in LLM-Based Program Repair" [2024-04] [[paper](https://arxiv.org/abs/2404.05520)]

- "Aligning LLMs for FL-free Program Repair" [2024-04] [[paper](https://arxiv.org/abs/2404.08877)]

- "A Deep Dive into Large Language Models for Automated Bug Localization and Repair" [2024-04] [[paper](https://arxiv.org/abs/2404.11595)]

- "Multi-Objective Fine-Tuning for Enhanced Program Repair with LLMs" [2024-04] [[paper](https://arxiv.org/abs/2404.12636)]

- "How Far Can We Go with Practical Function-Level Program Repair?" [2024-04] [[paper](https://arxiv.org/abs/2404.12833)]

- "Revisiting Unnaturalness for Automated Program Repair in the Era of Large Language Models" [2024-04] [[paper](https://arxiv.org/abs/2404.15236)]

- "A Unified Debugging Approach via LLM-Based Multi-Agent Synergy" [2024-04] [[paper](https://arxiv.org/abs/2404.17153)]

- "A Systematic Literature Review on Large Language Models for Automated Program Repair" [2024-05] [[paper](https://arxiv.org/abs/2405.01466)]

- "NAVRepair: Node-type Aware C/C++ Code Vulnerability Repair" [2024-05] [[paper](https://arxiv.org/abs/2405.04994)]

- "Automated Program Repair: Emerging trends pose and expose problems for benchmarks" [2024-05] [[paper](https://arxiv.org/abs/2405.05455)]

- "Automated Repair of AI Code with Large Language Models and Formal Verification" [2024-05] [[paper](https://arxiv.org/abs/2405.08848)]

- "A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback" [2024-05] [[paper](https://arxiv.org/abs/2405.15690)]

- "CREF: An LLM-based Conversational Software Repair Framework for Programming Tutors" [2024-06] [[paper](https://arxiv.org/abs/2406.13972)]

- "Towards Practical and Useful Automated Program Repair for Debugging" [2024-07] [[paper](https://arxiv.org/abs/2407.08958)]

- "ThinkRepair: Self-Directed Automated Program Repair" [2024-07] [[paper](https://arxiv.org/abs/2407.20898)]

- "MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair" [2024-08] [[paper](https://arxiv.org/abs/2408.09568)]

- "RePair: Automated Program Repair with Process-based Feedback" [2024-08] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2408.11296)]

- "Enhancing LLM-Based Automated Program Repair with Design Rationales" [2024-08] [[paper](https://arxiv.org/abs/2408.12056)]

- "Automated Software Vulnerability Patching using Large Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.13597)]

- "Enhancing Source Code Security with LLMs: Demystifying The Challenges and Generating Reliable Repairs" [2024-09] [[paper](https://arxiv.org/abs/2409.00571)]

- "MarsCode Agent: AI-native Automated Bug Fixing" [2024-09] [[paper](https://arxiv.org/abs/2409.00899)]

- "Co-Learning: Code Learning for Multi-Agent Reinforcement Collaborative Framework with Conversational Natural Language Interfaces" [2024-09] [[paper](https://arxiv.org/abs/2409.00985)]

- "Debugging with Open-Source Large Language Models: An Evaluation" [2024-09] [[paper](https://arxiv.org/abs/2409.03031)]

- "VulnLLMEval: A Framework for Evaluating Large Language Models in Software Vulnerability Detection and Patching" [2024-09] [[paper](https://arxiv.org/abs/2409.10756)]

- "Can GPT-O1 Kill All Bugs? An Evaluation of GPT-Family LLMs on QuixBugs" [2024-09] [[paper](https://arxiv.org/abs/2409.10033)]

- "Exploring and Lifting the Robustness of LLM-powered Automated Program Repair with Metamorphic Testing" [2024-10] [[paper](https://arxiv.org/abs/2410.07516)]

- "LecPrompt: A Prompt-based Approach for Logical Error Correction with CodeBERT" [2024-10] [[paper](https://arxiv.org/abs/2410.08241)]

- "Semantic-guided Search for Efficient Program Repair with Large Language Models" [2024-10] [[paper](https://arxiv.org/abs/2410.16655)]

- "A Comprehensive Survey of AI-Driven Advancements and Techniques in Automated Program Repair and Code Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.07586)]

- "From Defects to Demands: A Unified, Iterative, and Heuristically Guided LLM-Based Framework for Automated Software Repair and Requirement Realization" [2024-12] [[paper](https://arxiv.org/abs/2412.05098)]

- "Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair" [2024-12] [[paper](https://arxiv.org/abs/2412.03905)]

- "LLM4CVE: Enabling Iterative Automated Vulnerability Repair with Large Language Models" [2025-01] [[paper](https://arxiv.org/abs/2501.03446)]

- "Evaluating Agent-based Program Repair at Google" [2025-01] [[paper](https://arxiv.org/abs/2501.07531)]

- "HAFix: History-Augmented Large Language Models for Bug Fixing" [2025-01] [[paper](https://arxiv.org/abs/2501.09135)]

- "MultiMend: Multilingual Program Repair with Context Augmentation and Multi-Hunk Patch Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.16044)]

- "PATCH: Empowering Large Language Model with Programmer-Intent Guidance and Collaborative-Behavior Simulation for Automatic Bug Fixing" [2025-01] [[paper](https://arxiv.org/abs/2501.16149)]

- "Counterexample Guided Program Repair Using Zero-Shot Learning and MaxSAT-based Fault Localization" [2025-02] [[paper](https://arxiv.org/abs/2502.07786)]

- "Knowledge-Enhanced Program Repair for Data Science Code" [2025-02] [[paper](https://arxiv.org/abs/2502.09771)]

- "AuPair: Golden Example Pairs for Code Repair" [2025-02] [[paper](https://arxiv.org/abs/2502.18487)]

- "Less is More: Adaptive Program Repair with Bug Localization and Preference Learning" [2025-03] [[paper](https://arxiv.org/abs/2503.06510)]

- "UTFix: Change Aware Unit Test Repairing using LLM" [2025-03] [[paper](https://arxiv.org/abs/2503.14924)]

- "Unlocking LLM Repair Capabilities in Low-Resource Programming Languages Through Cross-Language Translation and Multi-Agent Refinement" [2025-03] [[paper](https://arxiv.org/abs/2503.22512)]

- "FeedbackEval: A Benchmark for Evaluating Large Language Models in Feedback-Driven Code Repair Tasks" [2025-03] [[paper](https://arxiv.org/abs/2504.06939)]

- "NL-Debugging: Exploiting Natural Language as an Intermediate Representation for Code Debugging" [2025-05] [[paper](https://arxiv.org/abs/2505.15356)]

### Code Similarity and Embedding (Clone Detection, Code Search)

- "Self-Supervised Contrastive Learning for Code Retrieval and Summarization via Semantic-Preserving Transformations" [2020-09] [SIGIR 2021] [[paper](https://arxiv.org/abs/2009.02731)]

- "REINFOREST: Reinforcing Semantic Code Similarity for Cross-Lingual Code Search Models" [2023-05] [[paper](https://arxiv.org/abs/2305.03843)]

- "Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search" [2024-01] [ACL 2024] [[paper](https://arxiv.org/abs/2401.04514)]

- "Revisiting Code Similarity Evaluation with Abstract Syntax Tree Edit Distance" [2024-04] [ACL 2024 short] [[paper](https://arxiv.org/abs/2404.08817)]

- "Is Next Token Prediction Sufficient for GPT? Exploration on Code Logic Comprehension" [2024-04] [[paper](https://arxiv.org/abs/2404.08885)]

- "Refining Joint Text and Source Code Embeddings for Retrieval Task with Parameter-Efficient Fine-Tuning" [2024-05] [[paper](https://arxiv.org/abs/2405.04126)]

- "Typhon: Automatic Recommendation of Relevant Code Cells in Jupyter Notebooks" [2024-05] [[paper](https://arxiv.org/abs/2405.09075)]

- "Toward Exploring the Code Understanding Capabilities of Pre-trained Code Generation Models" [2024-06] [[paper](https://arxiv.org/abs/2406.12326)]

- "Aligning Programming Language and Natural Language: Exploring Design Choices in Multi-Modal Transformer-Based Embedding for Bug Localization" [2024-06] [[paper](https://arxiv.org/abs/2406.17615)]

- "Assessing the Code Clone Detection Capability of Large Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.02402)]

- "CodeCSE: A Simple Multilingual Model for Code and Comment Sentence Embeddings" [2024-07] [[paper](https://arxiv.org/abs/2407.06360)]

- "Large Language Models for cross-language code clone detection" [2024-08] [[paper](https://arxiv.org/abs/2408.04430)]

- "Coding-PTMs: How to Find Optimal Code Pre-trained Models for Code Embedding in Vulnerability Detection?" [2024-08] [[paper](https://arxiv.org/abs/2408.04863)]

- "You Augment Me: Exploring ChatGPT-based Data Augmentation for Semantic Code Search" [2024-08] [[paper](https://arxiv.org/abs/2408.05542)]

- "Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features" [2024-08] [[paper](https://arxiv.org/abs/2408.08903)]

- "LLM Agents Improve Semantic Code Search" [2024-08] [[paper](https://arxiv.org/abs/2408.11058)]

- "zsLLMCode: An Effective Approach for Functional Code Embedding via LLM with Zero-Shot Learning" [2024-09] [[paper](https://arxiv.org/abs/2409.14644)]

- "Exploring Demonstration Retrievers in RAG for Coding Tasks: Yeas and Nays!" [2024-10] [[paper](https://arxiv.org/abs/2410.09662)]

- "Instructive Code Retriever: Learn from Large Language Model's Feedback for Code Intelligence Tasks" [2024-10] [[paper](https://arxiv.org/abs/2410.11300)]

- "Binary Code Similarity Detection via Graph Contrastive Learning on Intermediate Representations" [2024-10] [[paper](https://arxiv.org/abs/2410.18561)]

- "Are Decoder-Only Large Language Models the Silver Bullet for Code Search?" [2024-10] [[paper](https://arxiv.org/abs/2410.22240)]

- "CodeXEmbed: A Generalist Embedding Model Family for Multiligual and Multi-task Code Retrieval" [2024-11] [[paper](https://arxiv.org/abs/2411.12644)]

- "CodeSAM: Source Code Representation Learning by Infusing Self-Attention with Multi-Code-View Graphs" [2024-11] [[paper](https://arxiv.org/abs/2411.14611)]

- "EnStack: An Ensemble Stacking Framework of Large Language Models for Enhanced Vulnerability Detection in Source Code" [2024-11] [[paper](https://arxiv.org/abs/2411.16561)]

- "Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search" [2024-11] [[paper](https://arxiv.org/abs/2411.17538)]

- "Optimizing Code Retrieval: High-Quality and Scalable Dataset Annotation through Large Language Models" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.123/)]

- "Enhancing Learning-Based Binary Code Similarity Detection Model through Adversarial Training with Multiple Function Variants" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.673/)]

- "CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking" [2024-12] [ICLR 2025] [[paper](https://arxiv.org/abs/2412.01007)]

- "OASIS: Order-Augmented Strategy for Improved Code Search" [2025-03] [[paper](https://arxiv.org/abs/2503.08161)]

- "Zero-Shot Cross-Domain Code Search without Fine-Tuning" [2025-04] [[paper](https://arxiv.org/abs/2504.07740)]

### Code Refactoring and Migration

- "An Empirical Study on the Code Refactoring Capability of Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.02320)]

- "Automated Update of Android Deprecated API Usages with Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.04387)]

- "An Empirical Study on the Potential of LLMs in Automated Software Refactoring" [2024-11] [[paper](https://arxiv.org/abs/2411.04444)]

- "CODECLEANER: Elevating Standards with A Robust Data Contamination Mitigation Toolkit" [2024-11] [[paper](https://arxiv.org/abs/2411.10842)]

- "Instruct or Interact? Exploring and Eliciting LLMs' Capability in Code Snippet Adaptation Through Prompt Engineering" [2024-11] [[paper](https://arxiv.org/abs/2411.15501)]

- "Generating refactored code accurately using reinforcement learning" [2024-12] [[paper](https://arxiv.org/abs/2412.18035)]

- "How is Google using AI for internal code migrations?" [2025-01] [[paper](https://arxiv.org/abs/2501.06972)]

- "Automated Refactoring of Non-Idiomatic Python Code: A Differentiated Replication with LLMs" [2025-01] [[paper](https://arxiv.org/abs/2501.17024)]

- "Autonomous Legacy Web Application Upgrades Using a Multi-Agent System" [2025-01] [[paper](https://arxiv.org/abs/2501.19204)]

- "Evaluating the Effectiveness of LLMs in Fixing Maintainability Issues in Real-World Projects" [2025-02] [[paper](https://arxiv.org/abs/2502.02368)]

- "Distributed Approach to Haskell Based Applications Refactoring with LLMs Based Multi-Agent Systems" [2025-02] [[paper](https://arxiv.org/abs/2502.07928)]

- "RefactorBench: Evaluating Stateful Reasoning in Language Agents Through Code" [2025-03] [ICLR 2025] [[paper](https://arxiv.org/abs/2503.07832)]

- "MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration" [2025-03] [[paper](https://arxiv.org/abs/2503.14340)]

- "ECO: An LLM-Driven Efficient Code Optimizer for Warehouse Scale Computers" [2025-03] [[paper](https://arxiv.org/abs/2503.15669)]

- "Migrating Code At Scale With LLMs At Google" [2025-04] [[paper](https://arxiv.org/abs/2504.09691)]

- "Using LLMs for Library Migration" [2025-04] [[paper](https://arxiv.org/abs/2504.13272)]

- "Leveraging LLMs to Automate Energy-Aware Refactoring of Parallel Scientific Codes" [2025-05] [[paper](https://arxiv.org/abs/2505.02184)]

- "MIGRATION-BENCH: Repository-Level Code Migration Benchmark from Java 8" [2025-05] [[paper](https://arxiv.org/abs/2505.09569)]

### Type Prediction

- "Learning type annotation: is big data enough?" [2021-08] [ESEC/FSE 2021] [[paper](https://dl.acm.org/doi/10.1145/3468264.3473135)]

- "Do Machine Learning Models Produce TypeScript Types That Type Check?" [2023-02] [ECOOP 2023] [[paper](https://arxiv.org/abs/2302.12163)]

- "TypeT5: Seq2seq Type Inference using Static Analysis" [2023-03] [ICLR 2023] [[paper](https://arxiv.org/abs/2303.09564)]

- "Type Prediction With Program Decomposition and Fill-in-the-Type Training" [2023-05] [[paper](https://arxiv.org/abs/2305.17145)]

- "Generative Type Inference for Python" [2023-07] [ASE 2023] [[paper](https://arxiv.org/abs/2307.09163)]

- "Activation Steering for Robust Type Prediction in CodeLLMs" [2024-04] [[paper](https://arxiv.org/abs/2404.01903)]

- "An Empirical Study of Large Language Models for Type and Call Graph Analysis" [2024-10] [[paper](https://arxiv.org/abs/2410.00603)]

### Repository-Level Coding

- "Repository-Level Prompt Generation for Large Language Models of Code" [2022-06] [ICML 2023] [[paper](https://arxiv.org/abs/2206.12839)]

- "CoCoMIC: Code Completion By Jointly Modeling In-file and Cross-file Context" [2022-12] [[paper](https://arxiv.org/abs/2212.10007)]

- "RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation" [2023-03] [EMNLP 2023] [[paper](https://arxiv.org/abs/2303.12570)]

- "Coeditor: Leveraging Repo-level Diffs for Code Auto-editing" [2023-05] [ICLR 2024 Spotlight] [[paper](https://arxiv.org/abs/2305.18584)]

- "RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems" [2023-06] [ICLR 2024] [[paper](https://arxiv.org/abs/2306.03091)]

- "Guiding Language Models of Code with Global Context using Monitors" [2023-06] [[paper](https://arxiv.org/abs/2306.10763)]

- "RepoFusion: Training Code Models to Understand Your Repository" [2023-06] [[paper](https://arxiv.org/abs/2306.10998)]

- "CodePlan: Repository-level Coding using LLMs and Planning" [2023-09] [[paper](https://arxiv.org/abs/2306.10998)]

- "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" [2023-10] [ICLR 2024] [[paper](https://arxiv.org/abs/2310.06770)]

- "CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.11248)]

- "A^3-CodGen: A Repository-Level Code Generation Framework for Code Reuse with Local-Aware, Global-Aware, and Third-Party-Library-Aware" [2023-12] [[paper](https://arxiv.org/abs/2312.05772)]

- "Teaching Code LLMs to Use Autocompletion Tools in Repository-Level Code Generation" [2024-01] [[paper](https://arxiv.org/abs/2401.06391)]

- "RepoHyper: Better Context Retrieval Is All You Need for Repository-Level Code Completion" [2024-03] [[paper](https://arxiv.org/abs/2403.06095)]

- "Repoformer: Selective Retrieval for Repository-Level Code Completion" [2024-03] [ICML 2024] [[paper](https://arxiv.org/abs/2403.10059)]

- "CodeS: Natural Language to Code Repository via Multi-Layer Sketch" [2024-03] [[paper](https://arxiv.org/abs/2403.16443)]

- "Class-Level Code Generation from Natural Language Using Iterative, Tool-Enhanced Reasoning over Repository" [2024-04] [[paper](https://arxiv.org/abs/2405.01573)]

- "Contextual API Completion for Unseen Repositories Using LLMs" [2024-05] [[paper](https://arxiv.org/abs/2405.04600)]

- "Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion" [2024-05][ACL 2024] [[paper](https://arxiv.org/abs/2405.19782)]

- "How to Understand Whole Software Repository?" [2024-06] [[paper](https://arxiv.org/abs/2406.01422)]

- "R2C2-Coder: Enhancing and Benchmarking Real-world Repository-level Code Completion Abilities of Code Large Language Models" [2024-06] [[paper](https://arxiv.org/abs/2406.01359)]

- "CodeR: Issue Resolving with Multi-Agent and Task Graphs" [2024-06] [[paper](https://arxiv.org/abs/2406.01304)]

- "Enhancing Repository-Level Code Generation with Integrated Contextual Information" [2024-06] [[paper](https://arxiv.org/abs/2406.03283)]

- "On The Importance of Reasoning for Context Retrieval in Repository-Level Code Editing" [2024-06] [[paper](https://arxiv.org/abs/2406.04464)]

- "GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model" [2024-06] [ASE 2024] [[paper](https://arxiv.org/abs/2406.07003)]

- "STALL+: Boosting LLM-based Repository-level Code Completion with Static Analysis" [2024-06] [[paper](https://arxiv.org/abs/2406.10018)]

- "Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs" [2024-06] [[paper](https://arxiv.org/abs/2406.18294)]

- "Agentless: Demystifying LLM-based Software Engineering Agents" [2024-07] [[paper](https://arxiv.org/abs/2407.01489)]

- "RLCoder: Reinforcement Learning for Repository-Level Code Completion" [2024-07] [[paper](https://arxiv.org/abs/2407.19487)]

- "CoEdPilot: Recommending Code Edits with Learned Prior Edit Relevance, Project-wise Awareness, and Interactive Nature" [2024-08] [[paper](https://arxiv.org/abs/2408.01733)] [[repo](https://github.com/code-philia/CoEdPilot)]

- "RAMBO: Enhancing RAG-based Repository-Level Method Body Completion" [2024-09] [[paper](https://arxiv.org/abs/2409.15204)]

- "Exploring the Potential of Conversational Test Suite Based Program Repair on SWE-bench" [2024-10] [[paper](https://arxiv.org/abs/2410.04485)]

- "RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.14684)]

- "See-Saw Generative Mechanism for Scalable Recursive Code Generation with Generative AI" [2024-11] [[paper](https://arxiv.org/abs/2411.10861)]

- "Beyond pip install: Evaluating LLM Agents for the Automated Installation of Python Projects" [2024-12] [[paper](https://arxiv.org/abs/2412.06294)]

- "ContextModule: Improving Code Completion via Repository-level Contextual Information" [2024-12] [[paper](https://arxiv.org/abs/2412.08063)]

- "Repository Structure-Aware Training Makes SLMs Better Issue Resolver" [2024-12] [[paper](https://arxiv.org/abs/2412.19031)]

- "MutaGReP: Execution-Free Repository-Grounded Plan Search for Code-Use" [2025-02] [[paper](https://arxiv.org/abs/2502.15872)]

- "aiXcoder-7B-v2: Training LLMs to Fully Utilize the Long Context in Repository-level Code Completion" [2025-03] [[paper](https://arxiv.org/abs/2503.15301)]

- "Enhancing Repository-Level Software Repair via Repository-Aware Knowledge Graphs" [2025-03] [[paper](https://arxiv.org/abs/2503.21710)]

- "SRLCG: Self-Rectified Large-Scale Code Generation with Multidimensional Chain-of-Thought and Dynamic Backtracking" [2025-04] [[paper](https://arxiv.org/abs/2504.00532)]

- "SWE-Synth: Synthesizing Verifiable Bug-Fix Data to Enable Large Language Models in Resolving Real-World Bugs" [2025-04] [[paper](https://arxiv.org/abs/2504.14757)]

- "Knowledge Graph Based Repository-Level Code Generation" [2025-05] [[paper](https://arxiv.org/abs/2505.14394)]

- "Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks" [2025-05] [[paper](https://arxiv.org/abs/2505.16901)]

- "An Empirical Study on Strong-Weak Model Collaboration for Repo-level Code Generation" [2025-05] [[paper](https://arxiv.org/abs/2505.20182)]

### Frontend Development

- "Seeking the user interface", 2014-09, ASE 2014, [[paper](https://dl.acm.org/doi/10.1145/2642937.2642976)]

- "pix2code: Generating Code from a Graphical User Interface Screenshot", 2017-05, EICS 2018, [[paper](https://arxiv.org/abs/1705.07962)]

- "Machine Learning-Based Prototyping of Graphical User Interfaces for Mobile Apps", 2018-02, TSE 2020, [[paper](https://arxiv.org/abs/1802.02312)]

- "Automatic HTML Code Generation from Mock-Up Images Using Machine Learning Techniques", 2019-04, EBBT 2019, [[paper](https://ieeexplore.ieee.org/abstract/document/8741736)]

- "Sketch2code: Generating a website from a paper mockup", 2019-05, [[paper](https://arxiv.org/abs/1905.13750)]

- "HTLM: Hyper-Text Pre-Training and Prompting of Language Models", 2021-07, ICLR 2022, [[paper](https://arxiv.org/abs/2107.06955)]

- "Learning UI-to-Code Reverse Generator Using Visual Critic Without Rendering", 2023-05, [[paper](https://arxiv.org/abs/2305.14637)]

- "Design2Code: How Far Are We From Automating Front-End Engineering?" [2024-03] [[paper](https://arxiv.org/abs/2403.03163)]

- "Unlocking the conversion of Web Screenshots into HTML Code with the WebSight Dataset" [2024-03] [[paper](https://arxiv.org/abs/2403.09029)]

- "VISION2UI: A Real-World Dataset with Layout for Code Generation from UI Designs" [2024-04] [[paper](https://arxiv.org/abs/2404.06369)]

- "LogoMotion: Visually Grounded Code Generation for Content-Aware Animation" [2024-05] [[paper](https://arxiv.org/abs/2405.07065)]

- "PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM" [2024-06] [[paper](https://arxiv.org/abs/2406.02884)]

- "UICoder: Finetuning Large Language Models to Generate User Interface Code through Automated Feedback" [2024-06] [[paper](https://arxiv.org/abs/2406.07739)]

- "On AI-Inspired UI-Design" [2024-06] [[paper](https://arxiv.org/abs/2406.13631)]

- "Identifying User Goals from UI Trajectories" [2024-06] [[paper](https://arxiv.org/abs/2406.14314)]

- "Automatically Generating UI Code from Screenshot: A Divide-and-Conquer-Based Approach" [2024-06] [[paper](https://arxiv.org/abs/2406.16386)]

- "Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.20098)]

- "Vision-driven Automated Mobile GUI Testing via Multimodal Large Language Model" [2024-07] [[paper](https://arxiv.org/abs/2407.03037)]

- "AUITestAgent: Automatic Requirements Oriented GUI Function Testing" [2024-07] [[paper](https://arxiv.org/abs/2407.09018)]

- "LLM-based Abstraction and Concretization for GUI Test Migration" [2024-09] [[paper](https://arxiv.org/abs/2409.05028)]

- "Enabling Cost-Effective UI Automation Testing with Retrieval-Based LLMs: A Case Study in WeChat" [2024-09] [[paper](https://arxiv.org/abs/2409.07829)]

- "Self-Elicitation of Requirements with Automated GUI Prototyping" [2024-09] [[paper](https://arxiv.org/abs/2409.16388)]

- "Infering Alt-text For UI Icons With Large Language Models During App Development" [2024-09] [[paper](https://arxiv.org/abs/2409.18060)]

- "Leveraging Large Vision Language Model For Better Automatic Web GUI Testing" [2024-10] [[paper](https://arxiv.org/abs/2410.12157)]

- "Sketch2Code: Evaluating Vision-Language Models for Interactive Web Design Prototyping" [2024-10] [[paper](https://arxiv.org/abs/2410.16232)]

- "WAFFLE: Multi-Modal Model for Automated Front-End Development" [2024-10] [[paper](https://arxiv.org/abs/2410.18362)]

- "DesignRepair: Dual-Stream Design Guideline-Aware Frontend Repair with Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.01606)]

- "Interaction2Code: How Far Are We From Automatic Interactive Webpage Generation?" [2024-11] [[paper](https://arxiv.org/abs/2411.03292)]

- "A Multi-Agent Approach for REST API Testing with Semantic Graphs and LLM-Driven Inputs" [2024-11] [[paper](https://arxiv.org/abs/2411.07098)]

- "Automated Test Transfer Across Android Apps Using Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.17933)]

- "Zero-Shot Prompting Approaches for LLM-based Graphical User Interface Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.11328)]

- "MRWeb: An Exploration of Generating Multi-Page Resource-Aware Web Code from UI Designs" [2024-12] [[paper](https://arxiv.org/abs/2412.15310)]

- "Combining Language and App UI Analysis for the Automated Assessment of Bug Reproduction Steps" [2025-02] [[paper](https://arxiv.org/abs/2502.04251)]

- "ReaderLM-v2: Small Language Model for HTML to Markdown and JSON" [2025-03] [[paper](https://arxiv.org/abs/2503.01151)]

- "WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch" [2025-05] [[paper](https://arxiv.org/abs/2505.03733)]

- "Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks" [2025-05] [[paper](https://arxiv.org/abs/2505.07473)]

### Automated Machine Learning

- "Large Language Models Synergize with Automated Machine Learning" [2024-05] [[paper](https://arxiv.org/abs/2405.03727)]

- "AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML" [2024-10] [[paper](https://arxiv.org/abs/2410.02958)]

- "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering" [2024-10] [[paper](https://arxiv.org/abs/2410.07095)]

- "UniAutoML: A Human-Centered Framework for Unified Discriminative and Generative AutoML with Large Language Models" [2024-10] [[paper](https://arxiv.org/abs/2410.12841)]

- "BudgetMLAgent: A Cost-Effective LLM Multi-Agent system for Automating Machine Learning Tasks" [2024-11] [[paper](https://arxiv.org/abs/2411.07464)]

- "The Performance of the LSTM-based Code Generated by Large Language Models (LLMs) in Forecasting Time Series Data" [2024-11] [[paper](https://arxiv.org/abs/2411.18731)]

- "ML-Dev-Bench: Comparative Analysis of AI Agents on ML development workflows" [2025-02] [[paper](https://arxiv.org/abs/2502.00964)]

- "CSR-Bench: Benchmarking LLM Agents in Deployment of Computer Science Research Repositories" [2025-02] [[paper](https://arxiv.org/abs/2502.06111)]

- "AIDE: AI-Driven Exploration in the Space of Code" [2025-02] [[paper](https://arxiv.org/abs/2502.13138)]

- "MLGym: A New Framework and Benchmark for Advancing AI Research Agents" [2025-02] [[paper](https://arxiv.org/abs/2502.14499)]

- "MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems" [2025-03] [[paper](https://arxiv.org/abs/2503.03686)]

- "SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers" [2025-03] [[paper](https://arxiv.org/abs/2504.00255)]

- "PaperBench: Evaluating AI's Ability to Replicate AI Research" [2025-04] [[paper](https://arxiv.org/abs/2504.01848)]

- "Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning" [2025-04] [[paper](https://arxiv.org/abs/2504.17192)]

- "AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers" [2025-04] [[paper](https://arxiv.org/abs/2504.20115)]

- "ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies" [2025-04] [[paper](https://arxiv.org/abs/2504.20117)]

- "MLZero: A Multi-Agent System for End-to-end Machine Learning Automation" [2025-05] [[paper](https://arxiv.org/abs/2505.13941)]

- "MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research" [2025-05] [[paper](https://arxiv.org/abs/2505.19955)]

### Text-To-SQL

- "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models" [2021-09] [EMNLP 2021] [[paper](https://arxiv.org/abs/2109.05093)]

- "CodexDB: Generating Code for Processing SQL Queries using GPT-3 Codex" [2022-04] [[paper](https://arxiv.org/abs/2204.08941)]

- "T5QL: Taming language models for SQL generation" [2022-09] [[paper](https://arxiv.org/abs/2209.10254)]

- "Towards Generalizable and Robust Text-to-SQL Parsing" [2022-10] [EMNLP 2022 Findings] [[paper](https://arxiv.org/abs/2210.12674)]

- "XRICL: Cross-lingual Retrieval-Augmented In-Context Learning for Cross-lingual Text-to-SQL Semantic Parsing" [2022-10] [EMNLP 2022 Findings] [[paper](https://arxiv.org/abs/2210.13693)]

- "A comprehensive evaluation of ChatGPT's zero-shot Text-to-SQL capability" [2023-03] [[paper](https://arxiv.org/abs/2303.13547)]

- "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction" [2023-04] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2304.11015)]

- "How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings" [2023-05] [[paper](https://arxiv.org/abs/2305.11853)]

- "Enhancing Few-shot Text-to-SQL Capabilities of Large Language Models: A Study on Prompt Design Strategies" [2023-05] [[paper](https://arxiv.org/abs/2305.12586)]

- "SQL-PaLM: Improved Large Language Model Adaptation for Text-to-SQL" [2023-05] [[paper](https://arxiv.org/abs/2306.00739)]

- "Retrieval-augmented GPT-3.5-based Text-to-SQL Framework with Sample-aware Prompting and Dynamic Revision Chain" [2023-07] [ICONIP 2023] [[paper](https://arxiv.org/abs/2307.05074)]

- "Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation" [2023-08] [[paper](https://arxiv.org/abs/2308.15363)]

- "MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL" [2023-12] [[paper](https://arxiv.org/abs/2312.11242)]

- "DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models" [2024-02] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2402.01117)]

- "Investigating the Impact of Data Contamination of Large Language Models in Text-to-SQL Translation" [2024-02] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.08100)]

- "Improving Demonstration Diversity by Human-Free Fusing for Text-to-SQL" [2024-02] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2402.10663)]

- "Decomposition for Enhancing Attention: Improving LLM-based Text-to-SQL through Workflow Paradigm" [2024-02] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.10671)]

- "Knowledge-to-SQL: Enhancing SQL Generation with Data Expert LLM" [2024-02] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.11517)]

- "Understanding the Effects of Noise in Text-to-SQL: An Examination of the BIRD-Bench Benchmark" [2024-02] [ACL 2024 short] [[paper](https://arxiv.org/abs/2402.12243)]

- "SQL-Encoder: Improving NL2SQL In-Context Learning Through a Context-Aware Encoder" [2024-03] [[paper](https://arxiv.org/abs/2403.16204)]

- "LLM-R2: A Large Language Model Enhanced Rule-based Rewrite System for Boosting Query Efficiency" [2024-04] [[paper](https://arxiv.org/abs/2404.12872)]

- "Dubo-SQL: Diverse Retrieval-Augmented Generation and Fine Tuning for Text-to-SQL" [2024-04] [[paper](https://arxiv.org/abs/2404.12560)]

- "EPI-SQL: Enhancing Text-to-SQL Translation with Error-Prevention Instructions" [2024-04] [[paper](https://arxiv.org/abs/2404.14453)]

- "ProbGate at EHRSQL 2024: Enhancing SQL Query Generation Accuracy through Probabilistic Threshold Filtering and Error Handling" [2024-04] [[paper](https://arxiv.org/abs/2404.16659)]

- "CoE-SQL: In-Context Learning for Multi-Turn Text-to-SQL with Chain-of-Editions" [2024-05] [[paper](https://arxiv.org/abs/2405.02712)]

- "Open-SQL Framework: Enhancing Text-to-SQL on Open-source Large Language Models" [2024-05] [[paper](https://arxiv.org/abs/2405.06674)]

- "MCS-SQL: Leveraging Multiple Prompts and Multiple-Choice Selection For Text-to-SQL Generation" [2024-05] [[paper](https://arxiv.org/abs/2405.07467)]

- "PromptMind Team at EHRSQL-2024: Improving Reliability of SQL Generation using Ensemble LLMs" [2024-05] [[paper](https://arxiv.org/abs/2405.08839)]

- "LG AI Research & KAIST at EHRSQL 2024: Self-Training Large Language Models with Pseudo-Labeled Unanswerable Questions for a Reliable Text-to-SQL System on EHRs" [2024-05] [[paper](https://arxiv.org/abs/2405.11162)]

- "Before Generation, Align it! A Novel and Effective Strategy for Mitigating Hallucinations in Text-to-SQL Generation" [2024-05] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2405.15307)]

- "CHESS: Contextual Harnessing for Efficient SQL Synthesis" [2024-05] [[paper](https://arxiv.org/abs/2405.16755)]

- "DeTriever: Decoder-representation-based Retriever for Improving NL2SQL In-Context Learning" [2024-06] [[paper](https://arxiv.org/abs/2406.07913)]

- "Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL" [2024-06] [[paper](https://arxiv.org/abs/2406.08426)]

- "RH-SQL: Refined Schema and Hardness Prompt for Text-to-SQL" [2024-06] [[paper](https://arxiv.org/abs/2406.09133)]

- "QDA-SQL: Questions Enhanced Dialogue Augmentation for Multi-Turn Text-to-SQL" [2024-06] [[paper](https://arxiv.org/abs/2406.10593)]

- "End-to-end Text-to-SQL Generation within an Analytics Insight Engine" [2024-06] [[paper](https://arxiv.org/abs/2406.12104)]

- "MAGIC: Generating Self-Correction Guideline for In-Context Text-to-SQL" [2024-06] [[paper](https://arxiv.org/abs/2406.12692)]

- "SQLFixAgent: Towards Semantic-Accurate SQL Generation via Multi-Agent Collaboration" [2024-06] [[paper](https://arxiv.org/abs/2406.13408)]

- "Unmasking Database Vulnerabilities: Zero-Knowledge Schema Inference Attacks in Text-to-SQL Systems" [2024-06] [[paper](https://arxiv.org/abs/2406.14545)]

- "Improving Retrieval-augmented Text-to-SQL with AST-based Ranking and Schema Pruning" [2024-07] [EMNLP 2024] [[paper](https://arxiv.org/abs/2407.03227)]

- "Lucy: Think and Reason to Solve Text-to-SQL" [2024-07] [[paper](https://arxiv.org/abs/2407.05153)]

- "ESM+: Modern Insights into Perspective on Text-to-SQL Evaluation in the Age of Large Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.07313)]

- "RB-SQL: A Retrieval-based LLM Framework for Text-to-SQL" [2024-07] [[paper](https://arxiv.org/abs/2407.08273)]

- "AI-Assisted SQL Authoring at Industry Scale" [2024-07] [[paper](https://arxiv.org/abs/2407.13280)]

- "SQLfuse: Enhancing Text-to-SQL Performance through Comprehensive LLM Synergy" [2024-07] [[paper](https://arxiv.org/abs/2407.14568)]

- "A Survey on Employing Large Language Models for Text-to-SQL Tasks" [2024-07] [[paper](https://arxiv.org/abs/2407.15186)]

- "Towards Automated Data Sciences with Natural Language and SageCopilot: Practices and Lessons Learned" [2024-07] [[paper](https://arxiv.org/abs/2407.21040)]

- "Evaluating LLMs for Text-to-SQL Generation With Complex SQL Workload" [2024-07] [[paper](https://arxiv.org/abs/2407.19517)]

- "Synthesizing Text-to-SQL Data from Weak and Strong LLMs" [2024-08] [ACL 2024] [[paper](https://arxiv.org/abs/2408.03256)]

- "Improving Relational Database Interactions with Large Language Models: Column Descriptions and Their Impact on Text-to-SQL Performance" [2024-08] [[paper](https://arxiv.org/abs/2408.04691)]

- "The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.07702)]

- "MAG-SQL: Multi-Agent Generative Approach with Soft Schema Linking and Iterative Sub-SQL Refinement for Text-to-SQL" [2024-08] [[paper](https://arxiv.org/abs/2408.07930)]

- "Enhancing Text-to-SQL Parsing through Question Rewriting and Execution-Guided Refinement" [2024-08] [ACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-acl.120/)]

- "DAC: Decomposed Automation Correction for Text-to-SQL" [2024-08] [[paper](https://arxiv.org/abs/2408.08779)]

- "Interactive-T2S: Multi-Turn Interactions for Text-to-SQL with Large Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.11062)]

- "SQL-GEN: Bridging the Dialect Gap for Text-to-SQL Via Synthetic Data And Model Merging" [2024-08] [[paper](https://arxiv.org/abs/2408.12733)]

- "Enhancing SQL Query Generation with Neurosymbolic Reasoning" [2024-08] [[paper](https://arxiv.org/abs/2408.13888)]

- "Text2SQL is Not Enough: Unifying AI and Databases with TAG" [2024-08] [[paper](https://arxiv.org/abs/2408.14717)]

- "Tool-Assisted Agent on SQL Inspection and Refinement in Real-World Scenarios" [2024-08] [[paper](https://arxiv.org/abs/2408.16991)]

- "SelECT-SQL: Self-correcting ensemble Chain-of-Thought for Text-to-SQL" [2024-09] [[paper](https://arxiv.org/abs/2409.10007)]

- "You Only Read Once (YORO): Learning to Internalize Database Knowledge for Text-to-SQL" [2024-09] [[paper](https://arxiv.org/abs/2409.12172)]

- "PTD-SQL: Partitioning and Targeted Drilling with LLMs in Text-to-SQL" [2024-09] [EMNLP 2024] [[paper](https://arxiv.org/abs/2409.14082)]

- "Enhancing Text-to-SQL Capabilities of Large Language Models via Domain Database Knowledge Injection" [2024-09] [[paper](https://arxiv.org/abs/2409.15907)]

- "DataGpt-SQL-7B: An Open-Source Language Model for Text-to-SQL" [2024-09] [[paper](https://arxiv.org/abs/2409.15985)]

- "E-SQL: Direct Schema Linking via Question Enrichment in Text-to-SQL" [2024-09] [[paper](https://arxiv.org/abs/2409.16751)]

- "FLEX: Expert-level False-Less EXecution Metric for Reliable Text-to-SQL Benchmark" [2024-09] [[paper](https://arxiv.org/abs/2409.19014)]

- "Enhancing LLM Fine-tuning for Text-to-SQLs by SQL Quality Measurement" [2024-10] [[paper](https://arxiv.org/abs/2410.01869)]

- "From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems" [2024-10] [[paper](https://arxiv.org/abs/2410.01066)]

- "CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.01943)]

- "Context-Aware SQL Error Correction Using Few-Shot Learning -- A Novel Approach Based on NLQ, Error, and SQL Similarity" [2024-10] [[paper](https://arxiv.org/abs/2410.09174)]

- "Learning from Imperfect Data: Towards Efficient Knowledge Distillation of Autoregressive Language Models for Text-to-SQL" [2024-10] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2410.11371)]

- "LR-SQL: A Supervised Fine-Tuning Method for Text2SQL Tasks under Low-Resource Scenarios" [2024-10] [[paper](https://arxiv.org/abs/2410.11457)]

- "MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation" [2024-10] [[paper](https://arxiv.org/abs/2410.12916)]

- "Learning Metadata-Agnostic Representations for Text-to-SQL In-Context Example Selection" [2024-10] [[paper](https://arxiv.org/abs/2410.14049)]

- "An Actor-Critic Approach to Boosting Text-to-SQL Large Language Model" [2024-10] [[paper](https://arxiv.org/abs/2410.22082)]

- "RSL-SQL: Robust Schema Linking in Text-to-SQL Generation" [2024-10] [[paper](https://arxiv.org/abs/2411.00073)]

- "KeyInst: Keyword Instruction for Improving SQL Formulation in Text-to-SQL" [2024-10] [[paper](https://arxiv.org/abs/2411.00788)]

- "Grounding Natural Language to SQL Translation with Data-Based Self-Explanations" [2024-11] [[paper](https://arxiv.org/abs/2411.02948)]

- "PDC & DM-SFT: A Road for LLM SQL Bug-Fix Enhancing" [2024-11] [[paper](https://arxiv.org/abs/2411.06767)]

- "XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL" [2024-11] [[paper](https://arxiv.org/abs/2411.08599)]

- "Leveraging Prior Experience: An Expandable Auxiliary Knowledge Base for Text-to-SQL" [2024-11] [[paper](https://arxiv.org/abs/2411.13244)]

- "Text-to-SQL Calibration: No Need to Ask -- Just Rescale Model Probabilities" [2024-11] [[paper](https://arxiv.org/abs/2411.16742)]

- "SecureSQL: Evaluating Data Leakage of Large Language Models as Natural Language Interfaces to Databases" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.346/)]

- "A Survey of Large Language Model-Based Generative AI for Text-to-SQL: Benchmarks, Applications, Use Cases, and Challenges" [2024-12] [[paper](https://arxiv.org/abs/2412.05208)]

- "Cooperative SQL Generation for Segmented Databases By Using Multi-functional LLM Agents" [2024-12] [[paper](https://arxiv.org/abs/2412.05850)]

- "ROUTE: Robust Multitask Tuning and Collaboration for Text-to-SQL" [2024-12] [ICLR 2025] [[paper](https://arxiv.org/abs/2412.10138)]

- "Solid-SQL: Enhanced Schema-linking based In-context Learning for Robust Text-to-SQL" [2024-12] [[paper](https://arxiv.org/abs/2412.12522)]

- "Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types" [2024-12] [[paper](https://arxiv.org/abs/2412.17867)]

- "A Study of In-Context-Learning-Based Text-to-SQL Errors" [2025-01] [[paper](https://arxiv.org/abs/2501.09310)]

- "Confidence Estimation for Error Detection in Text-to-SQL Systems" [2025-01] [[paper](https://arxiv.org/abs/2501.09527)]

- "Reliable Text-to-SQL with Adaptive Abstention" [2025-01] [[paper](https://arxiv.org/abs/2501.10858)]

- "Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL" [2025-01] [[paper](https://arxiv.org/abs/2501.12372)]

- "Text-to-SQL based on Large Language Models and Database Keyword Search" [2025-01] [[paper](https://arxiv.org/abs/2501.13594)]

- "MCTS-SQL: An Effective Framework for Text-to-SQL with Monte Carlo Tree Search" [2025-01] [[paper](https://arxiv.org/abs/2501.16607)]

- "Extractive Schema Linking for Text-to-SQL" [2025-01] [[paper](https://arxiv.org/abs/2501.17174)]

- "ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Format Restriction, and Column Exploration" [2025-02] [[paper](https://arxiv.org/abs/2502.00675)]

- "PSM-SQL: Progressive Schema Learning with Multi-granularity Semantics for Text-to-SQL" [2025-02] [[paper](https://arxiv.org/abs/2502.05237)]

- "Rationalization Models for Text-to-SQL" [2025-02] [[paper](https://arxiv.org/abs/2502.06759)]

- "BASE-SQL: A powerful open source Text-To-SQL baseline approach" [2025-02] [[paper](https://arxiv.org/abs/2502.10739)]

- "MultiTEND: A Multilingual Benchmark for Natural Language to NoSQL Query Translation" [2025-02] [[paper](https://arxiv.org/abs/2502.11022)]

- "Bridging the Gap: Enabling Natural Language Queries for NoSQL Databases through Text-to-NoSQL Translation" [2025-02] [[paper](https://arxiv.org/abs/2502.11201)]

- "SAFE-SQL: Self-Augmented In-Context Learning with Fine-grained Example Selection for Text-to-SQL" [2025-02] [[paper](https://arxiv.org/abs/2502.11438)]

- "Uncovering the Impact of Chain-of-Thought Reasoning for Direct Preference Optimization: Lessons from Text-to-SQL" [2025-02] [[paper](https://arxiv.org/abs/2502.11656)]

- "SQL-o1: A Self-Reward Heuristic Dynamic Search Method for Text-to-SQL" [2025-02] [[paper](https://arxiv.org/abs/2502.11741)]

- "Knapsack Optimization-based Schema Linking for LLM-based Text-to-SQL Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.12911)]

- "STaR-SQL: Self-Taught Reasoner for Text-to-SQL" [2025-02] [[paper](https://arxiv.org/abs/2502.13550)]

- "Bridging the Gap: Transforming Natural Language Questions into SQL Queries via Abstract Query Pattern and Contextual Schema Markup" [2025-02] [[paper](https://arxiv.org/abs/2502.14682)]

- "OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignment" [2025-02] [[paper](https://arxiv.org/abs/2502.14913)]

- "SQLong: Enhanced NL2SQL for Longer Contexts with LLMs" [2025-02] [[paper](https://arxiv.org/abs/2502.16747)]

- "OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale" [2025-03] [[paper](https://arxiv.org/abs/2503.02240)]

- "DB-Explore: Automated Database Exploration and Instruction Synthesis for Text-to-SQL" [2025-03] [[paper](https://arxiv.org/abs/2503.04959)]

- "SQLCritic: Correcting Text-to-SQL Generation via Clause-wise Critic" [2025-03] [[paper](https://arxiv.org/abs/2503.07996)]

- "TinySQL: A Progressive Text-to-SQL Dataset for Mechanistic Interpretability Research" [2025-03] [[paper](https://arxiv.org/abs/2503.12730)]

- "Feather-SQL: A Lightweight NL2SQL Framework with Dual-Model Collaboration Paradigm for Small Language Models" [2025-03] [[paper](https://arxiv.org/abs/2503.17811)]

- "LinkAlign: Scalable Schema Linking for Real-World Large-Scale Multi-Database Text-to-SQL" [2025-03] [[paper](https://arxiv.org/abs/2503.18596)]

- "ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback" [2025-03] [[paper](https://arxiv.org/abs/2503.19988)]

- "EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing" [2025-03] [[paper](https://arxiv.org/abs/2503.22402)]

- "Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL" [2025-03] [[paper](https://arxiv.org/abs/2503.23157)]

- "Distill-C: Enhanced NL2SQL via Distilled Customization with LLMs" [2025-03] [[paper](https://arxiv.org/abs/2504.00048)]

- "Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards" [2025-05] [[paper](https://arxiv.org/abs/2505.04671)]

- "ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL" [2025-05] [[paper](https://arxiv.org/abs/2505.12768)]

- "CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning" [2025-05] [[paper](https://arxiv.org/abs/2505.13271)]

- "SQLForge: Synthesizing Reliable and Diverse Data to Enhance Text-to-SQL Reasoning in LLMs" [2025-05] [[paper](https://arxiv.org/abs/2505.13725)]

- "Cheaper, Better, Faster, Stronger: Robust Text-to-SQL without Chain-of-Thought or Fine-Tuning" [2025-05] [[paper](https://arxiv.org/abs/2505.14174)]

- "JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling" [2025-05] [[paper](https://arxiv.org/abs/2505.14305)]

- "UNJOIN: Enhancing Multi-Table Text-to-SQL Generation via Schema Simplification" [2025-05] [[paper](https://arxiv.org/abs/2505.18122)]

- "SchemaGraphSQL: Efficient Schema Linking with Pathfinding Graph Algorithms for Text-to-SQL on Large-Scale Databases" [2025-05] [[paper](https://arxiv.org/abs/2505.18363)]

- "DCG-SQL: Enhancing In-Context Learning for Text-to-SQL with Deep Contextual Schema Link Graph" [2025-05] [[paper](https://arxiv.org/abs/2505.19956)]

- "Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL" [2025-05] [[paper](https://arxiv.org/abs/2505.20315)]

### Program Proof

- "Baldur: Whole-Proof Generation and Repair with Large Language Models" [2023-03] [FSE 2023] [[paper](https://arxiv.org/abs/2303.04910)]

- "An In-Context Learning Agent for Formal Theorem-Proving" [2023-10] [[paper](https://arxiv.org/abs/2310.04353)]

- "Towards General Loop Invariant Generation: A Benchmark of Programs with Memory Manipulation" [2023-11] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2311.10483)]

- "Towards AI-Assisted Synthesis of Verified Dafny Methods" [2024-02] [FSE 2024] [[paper](https://arxiv.org/abs/2402.00247)]

- "Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming" [2024-05] [[paper](https://arxiv.org/abs/2405.01787)]

- "Laurel: Generating Dafny Assertions Using Large Language Models" [2024-05] [[paper](https://arxiv.org/abs/2405.16792)]

- "AutoVerus: Automated Proof Generation for Rust Code" [2024-09] [[paper](https://arxiv.org/abs/2409.13082)]

- "Proof Automation with Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.14274)]

- "Automated Proof Generation for Rust Code via Self-Evolution" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.15756)]

- "CoqPilot, a plugin for LLM-based generation of proofs" [2024-10] [[paper](https://arxiv.org/abs/2410.19605)]

- "dafny-annotator: AI-Assisted Verification of Dafny Programs" [2024-11] [[paper](https://arxiv.org/abs/2411.15143)]

- "AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement" [2024-12] [[paper](https://arxiv.org/abs/2412.06176)]

- "Enhancing Automated Loop Invariant Generation for Complex Programs with Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.10483)]

- "Rango: Adaptive Retrieval-Augmented Proving for Automated Software Verification" [2024-12] [[paper](https://arxiv.org/abs/2412.14063)]

- "Dafny as Verification-Aware Intermediate Language for Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.06283)]

- "Next Steps in LLM-Supported Java Verification" [2025-02] [[paper](https://arxiv.org/abs/2502.01573)]

- "Verifying LLM-Generated Code in the Context of Software Verification with Ada/SPARK" [2025-02] [[paper](https://arxiv.org/abs/2502.07728)]

- "RAG-Verus: Repository-Level Program Verification with LLMs using Retrieval Augmented Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.05344)]

- "FormalSpecCpp: A Dataset of C++ Formal Specifications created using LLMs" [2025-02] [[paper](https://arxiv.org/abs/2502.15217)]

- "Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference" [2025-02] [[paper](https://arxiv.org/abs/2503.04779)]

- "Can LLMs Formally Reason as Abstract Interpreters for Program Analysis?" [2025-03] [[paper](https://arxiv.org/abs/2503.12686)]

- "Can LLMs Enable Verification in Mainstream Programming?" [2025-03] [[paper](https://arxiv.org/abs/2503.14183)]

### Test Generation

- "Unit Test Case Generation with Transformers and Focal Context" [2020-09] [AST@ICSE 2022] [[paper](https://arxiv.org/abs/2009.05617)]

- "An Empirical Evaluation of Using Large Language Models for Automated Unit Test Generation" [2023-02] [IEEE TSE] [[paper](https://arxiv.org/abs/2302.06527)]

- "A3Test: Assertion-Augmented Automated Test Case Generation" [2023-02] [[paper](https://arxiv.org/abs/2302.10352)]

- "Learning Deep Semantics for Test Completion" [2023-02] [ICSE 2023] [[paper](https://arxiv.org/abs/2302.10166)]

- "Using Large Language Models to Generate JUnit Tests: An Empirical Study" [2023-04] [EASE 2024] [[paper](https://arxiv.org/abs/2305.00418)]

- "CodaMosa: Escaping Coverage Plateaus in Test Generation with Pre-Trained Large Language Models" [2023-05] [ICSE 2023] [[paper](https://dl.acm.org/doi/abs/10.1109/ICSE48619.2023.00085)]

- "No More Manual Tests? Evaluating and Improving ChatGPT for Unit Test Generation" [2023-05] [[paper](https://arxiv.org/abs/2305.04207)]

- "ChatUniTest: a ChatGPT-based automated unit test generation tool" [2023-05] [[paper](https://arxiv.org/abs/2305.04764)]

- "ChatGPT vs SBST: A Comparative Assessment of Unit Test Suite Generation" [2023-07] [[paper](https://arxiv.org/abs/2307.00588)]

- "Can Large Language Models Write Good Property-Based Tests?" [2023-07] [[paper](https://arxiv.org/abs/2307.04346)]

- "Domain Adaptation for Deep Unit Test Case Generation" [2023-08] [[paper](https://arxiv.org/abs/2308.08033)]

- "Effective Test Generation Using Pre-trained Large Language Models and Mutation Testing" [2023-08] [[paper](https://arxiv.org/abs/2308.16557)]

- "How well does LLM generate security tests?" [2023-10] [[paper](https://arxiv.org/abs/2310.00710)]

- "Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation" [2023-10] [[paper](https://arxiv.org/abs/2310.02368)]

- "An initial investigation of ChatGPT unit test generation capability" [2023-10] [SAST 2023] [[paper](https://dl.acm.org/doi/abs/10.1145/3624032.3624035)]

- "CoverUp: Coverage-Guided LLM-Based Test Generation" [2024-03] [[paper](https://arxiv.org/abs/2403.16218)]

- "Enhancing LLM-based Test Generation for Hard-to-Cover Branches via Program Analysis" [2024-04] [[paper](https://arxiv.org/abs/2404.04966)]

- "Large Language Models for Mobile GUI Text Input Generation: An Empirical Study" [2024-04] [[paper](https://arxiv.org/abs/2404.08948)]

- "Test Code Generation for Telecom Software Systems using Two-Stage Generative Model" [2024-04] [[paper](https://arxiv.org/abs/2404.09249)]

- "LLM-Powered Test Case Generation for Detecting Tricky Bugs" [2024-04] [[paper](https://arxiv.org/abs/2404.10304)]

- "Generating Test Scenarios from NL Requirements using Retrieval-Augmented LLMs: An Industrial Study" [2024-04] [[paper](https://arxiv.org/abs/2404.12772)]

- "Large Language Models as Test Case Generators: Performance Evaluation and Enhancement" [2024-04] [[paper](https://arxiv.org/abs/2404.13340)]

- "Leveraging Large Language Models for Automated Web-Form-Test Generation: An Empirical Study" [2024-05] [[paper](https://arxiv.org/abs/2405.09965)]

- "DLLens: Testing Deep Learning Libraries via LLM-aided Synthesis" [2024-06] [[paper](https://arxiv.org/abs/2406.07944)]

- "Exploring Fuzzing as Data Augmentation for Neural Test Generation" [2024-06] [[paper](https://arxiv.org/abs/2406.08665)]

- "Mokav: Execution-driven Differential Testing with LLMs" [2024-06] [[paper](https://arxiv.org/abs/2406.10375)]

- "SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.12952)]

- "CasModaTest: A Cascaded and Model-agnostic Self-directed Framework for Unit Test Generation" [2024-06] [[paper](https://arxiv.org/abs/2406.15743)]

- "An Empirical Study of Unit Test Generation with Large Language Models" [2024-06] [[paper](https://arxiv.org/abs/2406.18181)]

- "Large-scale, Independent and Comprehensive study of the power of LLMs for test case generation" [2024-06] [[paper](https://arxiv.org/abs/2407.00225)]

- "Augmenting LLMs to Repair Obsolete Test Cases with Static Collector and Neural Reranker" [2024-07] [[paper](https://arxiv.org/abs/2407.03625)]

- "Harnessing the Power of LLMs: Automating Unit Test Generation for High-Performance Computing" [2024-07] [[paper](https://arxiv.org/abs/2407.05202)]

- "An LLM-based Readability Measurement for Unit Tests' Context-aware Inputs" [2024-07] [[paper](https://arxiv.org/abs/2407.21369)]

- "A System for Automated Unit Test Generation Using Large Language Models and Assessment of Generated Test Suites" [2024-08] [[paper](https://arxiv.org/abs/2408.07846)]

- "Leveraging Large Language Models for Enhancing the Understandability of Generated Unit Tests" [2024-08] [[paper](https://arxiv.org/abs/2408.11710)]

- "Multi-language Unit Test Generation using LLMs" [2024-09] [[paper](https://arxiv.org/abs/2409.03093)]

- "Exploring the Integration of Large Language Models in Industrial Test Maintenance Processes" [2024-09] [[paper](https://arxiv.org/abs/2409.06416)]

- "Python Symbolic Execution with LLM-powered Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.09271)]

- "Rethinking the Influence of Source Code on Test Case Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.09464)]

- "On the Effectiveness of LLMs for Manual Test Verifications" [2024-09] [[paper](https://arxiv.org/abs/2409.12405)]

- "Retrieval-Augmented Test Generation: How Far Are We?" [2024-09] [[paper](https://arxiv.org/abs/2409.12682)]

- "Context-Enhanced LLM-Based Framework for Automatic Test Refactoring" [2024-09] [[paper](https://arxiv.org/abs/2409.16739)]

- "TestBench: Evaluating Class-Level Test Case Generation Capability of Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.17561)]

- "Advancing Bug Detection in Fastjson2 with Large Language Models Driven Unit Test Generation" [2024-10] [[paper](https://arxiv.org/abs/2410.09414)]

- "Test smells in LLM-Generated Unit Tests" [2024-10] [[paper](https://arxiv.org/abs/2410.10628)]

- "LLM-based Unit Test Generation via Property Retrieval" [2024-10] [[paper](https://arxiv.org/abs/2410.13542)]

- "Disrupting Test Development with AI Assistants" [2024-11] [[paper](https://arxiv.org/abs/2411.02328)]

- "Parameter-Efficient Fine-Tuning of Large Language Models for Unit Test Generation: An Empirical Study" [2024-11] [[paper](https://arxiv.org/abs/2411.02462)]

- "VALTEST: Automated Validation of Language Model Generated Test Cases" [2024-11] [[paper](https://arxiv.org/abs/2411.08254)]

- "REACCEPT: Automated Co-evolution of Production and Test Code Based on Dynamic Validation and Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.11033)]

- "What You See Is What You Get: Attention-based Self-guided Automatic Unit Test Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.00828)]

- "System Test Case Design from Requirements Specifications: Insights and Challenges of Using ChatGPT" [2024-12] [[paper](https://arxiv.org/abs/2412.03693)]

- "TDD-Bench Verified: Can LLMs Generate Tests for Issues Before They Get Resolved?" [2024-12] [[paper](https://arxiv.org/abs/2412.02883)]

- "CPP-UT-Bench: Can LLMs Write Complex Unit Tests in C++?" [2024-12] [[paper](https://arxiv.org/abs/2412.02735)]

- "Design choices made by LLM-based test generators prevent them from finding bugs" [2024-12] [[paper](https://arxiv.org/abs/2412.14137)]

- "A Large-scale Empirical Study on Fine-tuning Large Language Models for Unit Testing" [2024-12] [[paper](https://arxiv.org/abs/2412.16620)]

- "Improving the Readability of Automatically Generated Tests using Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.18843)]

- "The Potential of LLMs in Automating Software Testing: From Generation to Reporting" [2024-12] [[paper](https://arxiv.org/abs/2501.00217)]

- "The Prompt Alchemist: Automated LLM-Tailored Prompt Optimization for Test Case Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.01329)]

- "Enhancing LLM's Ability to Generate More Repository-Aware Unit Tests Through Precise Contextual Information Injection" [2025-01] [[paper](https://arxiv.org/abs/2501.07425)]

- "Test Wars: A Comparative Study of SBST, Symbolic Execution, and LLM-Based Approaches to Unit Test Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.10200)]

- "Can LLM Generate Regression Tests for Software Commits?" [2025-01] [[paper](https://arxiv.org/abs/2501.11086)]

- "Boundary Value Test Input Generation Using Prompt Engineering with LLMs: Fault Detection and Coverage Analysis" [2025-01] [[paper](https://arxiv.org/abs/2501.14465)]

- "A Systematic Approach for Assessing Large Language Models' Test Case Generation Capability" [2025-02] [[paper](https://arxiv.org/abs/2502.02866)]

- "ProjectTest: A Project-level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms" [2025-02] [[paper](https://arxiv.org/abs/2502.06556)]

- "CLOVER: A Test Case Generation Benchmark with Coverage, Long-Context, and Verification" [2025-02] [[paper](https://arxiv.org/abs/2502.08806)]

- "LLM-based Unit Test Generation for Dynamically-Typed Programs" [2025-03] [[paper](https://arxiv.org/abs/2503.14000)]

- "LLM Test Generation via Iterative Hybrid Program Analysis" [2025-03] [[paper](https://arxiv.org/abs/2503.13580)]

- "TestForge: Feedback-Driven, Agentic Test Suite Generation" [2025-03] [[paper](https://arxiv.org/abs/2503.14713)]

- "Unify and Triumph: Polyglot, Diverse, and Self-Consistent Generation of Unit Tests with LLMs" [2025-03] [[paper](https://arxiv.org/abs/2503.16144)]

### Oracle Generation

- "Generating Accurate Assert Statements for Unit Test Cases using Pretrained Transformers" [2020-09] [[paper](https://arxiv.org/abs/2009.05634)]

- "TOGA: A Neural Method for Test Oracle Generation" [2021-09] [ICSE 2022] [[paper](https://arxiv.org/abs/2109.09262)]

- "TOGLL: Correct and Strong Test Oracle Generation with LLMs" [2024-05] [[paper](https://arxiv.org/abs/2405.03786)]

- "Test Oracle Automation in the era of LLMs" [2024-05] [[paper](https://arxiv.org/abs/2405.12766)]

- "Beyond Code Generation: Assessing Code LLM Maturity with Postconditions" [2024-07] [[paper](https://arxiv.org/abs/2407.14118)]

- "Chat-like Asserts Prediction with the Support of Large Language Model" [2024-07] [[paper](https://arxiv.org/abs/2407.21429)]

- "Do LLMs generate test oracles that capture the actual or the expected program behaviour?" [2024-10] [[paper](https://arxiv.org/abs/2410.21136)]

- "Generating executable oracles to check conformance of client code to requirements of JDK Javadocs using LLMs" [2024-11] [[paper](https://arxiv.org/abs/2411.01789)]

- "Automatically Write Code Checker: An LLM-based Approach with Logic-guided API Retrieval and Case by Case Iteration" [2024-11] [[paper](https://arxiv.org/abs/2411.06796)]

- "ASSERTIFY: Utilizing Large Language Models to Generate Assertions for Production Code" [2024-11] [[paper](https://arxiv.org/abs/2411.16927)]

- "DeCon: Detecting Incorrect Assertions via Postconditions Generated by a Large Language Model" [2025-01] [[paper](https://arxiv.org/abs/2501.02901)]

- "AugmenTest: Enhancing Tests with LLM-Driven Oracles" [2025-01] [[paper](https://arxiv.org/abs/2501.17461)]

- "AsserT5: Test Assertion Generation Using a Fine-Tuned Code Language Model" [2025-02] [[paper](https://arxiv.org/abs/2502.02708)]

- "Improving Retrieval-Augmented Deep Assertion Generation via Joint Training" [2025-02] [[paper](https://arxiv.org/abs/2502.10696)]

- "Improving Deep Assertion Generation via Fine-Tuning Retrieval-Augmented Pre-trained Language Models" [2025-02] [[paper](https://arxiv.org/abs/2502.16071)]

### Mutation Testing

- "μBERT: Mutation Testing using Pre-Trained Language Models" [2022-03] [[paper](https://arxiv.org/abs/2203.03289)]

- "Efficient Mutation Testing via Pre-Trained Language Models" [2023-01] [[paper](https://arxiv.org/abs/2301.03543)]

- "LLMorpheus: Mutation Testing using Large Language Models" [2024-04] [[paper](https://arxiv.org/abs/2404.09952)]

- "An Exploratory Study on Using Large Language Models for Mutation Testing" [2024-06] [[paper](https://arxiv.org/abs/2406.09843)]

- "Fine-Tuning LLMs for Code Mutation: A New Era of Cyber Threats" [2024-10] [[paper](https://arxiv.org/abs/2410.22293)]

- "Simulink Mutation Testing using CodeBERT" [2025-01] [[paper](https://arxiv.org/abs/2501.07553)]

- "Mutation-Guided LLM-based Test Generation at Meta" [2025-01] [[paper](https://arxiv.org/abs/2501.12862)]

### Fuzz Testing

- "Large Language Models are Zero-Shot Fuzzers: Fuzzing Deep-Learning Libraries via Large Language Models" [2022-12] [[paper](https://arxiv.org/abs/2212.14834)]

- "Fuzz4All: Universal Fuzzing with Large Language Models" [2023-08] [[paper](https://arxiv.org/abs/2308.04748)]

- "WhiteFox: White-Box Compiler Fuzzing Empowered by Large Language Models" [2023-10] [[paper](https://arxiv.org/abs/2310.15991)]

- "LLAMAFUZZ: Large Language Model Enhanced Greybox Fuzzing" [2024-06] [[paper](https://arxiv.org/abs/2406.07714)]

- "FuzzCoder: Byte-level Fuzzing Test via Large Language Model" [2024-09] [[paper](https://arxiv.org/abs/2409.01944)]

- "ISC4DGF: Enhancing Directed Grey-box Fuzzing with LLM-Driven Initial Seed Corpus Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.14329)]

- "Large Language Models Based JSON Parser Fuzzing for Bug Discovery and Behavioral Analysis" [2024-10] [[paper](https://arxiv.org/abs/2410.21806)]

- "Fixing Security Vulnerabilities with AI in OSS-Fuzz" [2024-11] [[paper](https://arxiv.org/abs/2411.03346)]

- "A Code Knowledge Graph-Enhanced System for LLM-Based Fuzz Driver Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.11532)]

- "Harnessing Large Language Models for Seed Generation in Greybox Fuzzing" [2024-11] [[paper](https://arxiv.org/abs/2411.18143)]

- "Large Language Model assisted Hybrid Fuzzing" [2024-12] [[paper](https://arxiv.org/abs/2412.15931)]

- "Your Fix Is My Exploit: Enabling Comprehensive DL Library API Fuzzing with Large Language Models" [2025-01] [[paper](https://arxiv.org/abs/2501.04312)]

### Vulnerability Detection

- "VulDeePecker: A Deep Learning-Based System for Vulnerability Detection" [2018-01] [NDSS 2018] [[paper](https://arxiv.org/abs/1801.01681)]

- "DeepBugs: A Learning Approach to Name-based Bug Detection" [2018-04] [Proc. ACM Program. Lang.] [[paper](https://arxiv.org/abs/1805.11683)]

- "Automated Vulnerability Detection in Source Code Using Deep Representation Learning" [2018-07] [ICMLA 2018] [[paper](https://arxiv.org/abs/1807.04320)]

- "SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities" [2018-07] [IEEE TDSC] [[paper](https://arxiv.org/abs/1807.06756)]

- "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks" [2019-09] [NeurIPS 2019] [[paper](https://arxiv.org/abs/1909.03496)]

- "Improving bug detection via context-based code representation learning and attention-based neural networks" [2019-10] [Proc. ACM Program. Lang.] [[paper](https://dl.acm.org/doi/10.1145/3360588)]

- "Global Relational Models of Source Code" [2019-12] [ICLR 2020] [[paper](https://openreview.net/forum?id=B1lnbRNtwr)]

- "VulDeeLocator: A Deep Learning-based Fine-grained Vulnerability Detector" [2020-01] [IEEE TDSC] [[paper](https://arxiv.org/abs/2001.02350)]

- "Deep Learning based Vulnerability Detection: Are We There Yet?" [2020-09] [IEEE TSE] [[paper](https://arxiv.org/abs/2009.07235)]

- "Security Vulnerability Detection Using Deep Learning Natural Language Processing" [2021-05] [INFOCOM Workshops 2021] [[paper](https://arxiv.org/abs/2105.02388)]

- "Self-Supervised Bug Detection and Repair" [2021-05] [NeurIPS 2021] [[paper](https://arxiv.org/abs/2105.12787)]

- "Vulnerability Detection with Fine-grained Interpretations" [2021-06] [ESEC/SIGSOFT FSE 2021] [[paper](https://arxiv.org/abs/2106.10478)]

- "ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection" [2021-10] [ICSE Companion 2022] [[paper](https://arxiv.org/abs/2110.07317)]

- "VUDENC: Vulnerability Detection with Deep Learning on a Natural Codebase for Python" [2022-01] [Inf. Softw. Technol] [[paper](https://arxiv.org/abs/2201.08441)]

- "Transformer-Based Language Models for Software Vulnerability Detection" [222-04] [ACSAC 2022] [[paper](https://arxiv.org/abs/2204.03214)]

- "LineVul: A Transformer-based Line-Level Vulnerability Prediction" [2022-05] [MSR 2022] [[paper](https://ieeexplore.ieee.org/document/9796256)]

- "VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection" [2022-05] [IJCNN 2022] [[paper](https://arxiv.org/abs/2205.12424)]

- "Open Science in Software Engineering: A Study on Deep Learning-Based Vulnerability Detection" [2022-09] [IEEE TSE] [[paper](https://ieeexplore.ieee.org/document/9894099)]

- "An Empirical Study of Deep Learning Models for Vulnerability Detection" [2022-12] [ICSE 2023] [[paper](https://arxiv.org/abs/2212.08109)]

- "CSGVD: A deep learning approach combining sequence and graph embedding for source code vulnerability detection" [2023-01] [J. Syst. Softw.] [[paper](https://www.sciencedirect.com/science/article/pii/S0164121223000183)]

- "Benchmarking Software Vulnerability Detection Techniques: A Survey" [2023-03] [[paper](https://arxiv.org/abs/2303.16362)]

- "Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?" [2023-05] [[paper](https://arxiv.org/abs/2306.01754)]

- "A Survey on Automated Software Vulnerability Detection Using Machine Learning and Deep Learning" [2023-06] [[paper](https://arxiv.org/abs/2306.11673)]

- "Limits of Machine Learning for Automatic Vulnerability Detection" [2023-06] [[paper](https://arxiv.org/abs/2306.17193)]

- "Evaluating Instruction-Tuned Large Language Models on Code Comprehension and Generation" [2023-08] [[paper](https://arxiv.org/abs/2308.01240)]

- "Prompt-Enhanced Software Vulnerability Detection Using ChatGPT" [2023-08] [[paper](https://arxiv.org/abs/2308.12697)]

- "Towards Causal Deep Learning for Vulnerability Detection" [2023-10] [[paper](https://arxiv.org/abs/2310.07958)]

- "Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities" [2023-11] [[paper](https://arxiv.org/abs/2311.16169)]

- "How Far Have We Gone in Vulnerability Detection Using Large Language Models" [2023-11] [[paper](https://arxiv.org/abs/2311.12420)]

- "Can Large Language Models Identify And Reason About Security Vulnerabilities? Not Yet" [2023-12] [[paper](https://arxiv.org/abs/2312.12575)]

- "LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning" [2024-01] [[paper](https://arxiv.org/abs/2401.16185)]

- "Security Code Review by LLMs: A Deep Dive into Responses" [2024-01] [[paper](https://arxiv.org/abs/2401.16310)]

- "LLMDFA: Analyzing Dataflow in Code with Large Language Models" [2024-02] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2402.10754)]

- "Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities" [2024-02] [[paper](https://arxiv.org/abs/2402.17230)]

- "Multi-role Consensus through LLMs Discussions for Vulnerability Detection" [2024-03] [[paper](https://arxiv.org/abs/2403.14274)]

- "A Comprehensive Study of the Capabilities of Large Language Models for Vulnerability Detection" [2024-03] [[paper](https://arxiv.org/abs/2403.17218)]

- "Vulnerability Detection with Code Language Models: How Far Are We?" [2024-03] [[paper](https://arxiv.org/abs/2403.18624)]

- "Multitask-based Evaluation of Open-Source LLM on Software Vulnerability" [2024-04] [[paper](https://arxiv.org/abs/2404.02056)]

- "Large Language Model for Vulnerability Detection and Repair: Literature Review and Roadmap" [2024-04] [[paper](https://arxiv.org/abs/2404.02525)]

- "Pros and Cons! Evaluating ChatGPT on Software Vulnerability" [2024-04] [[paper](https://arxiv.org/abs/2404.03994)]

- "VulEval: Towards Repository-Level Evaluation of Software Vulnerability Detection" [2024-04] [[paper](https://arxiv.org/abs/2404.15596)]

- "DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection" [2024-05] [[paper](https://arxiv.org/abs/2405.01202)]

- "Bridging the Gap: A Study of AI-based Vulnerability Management between Industry and Academia" [2024-05] [[paper](https://arxiv.org/abs/2405.02435)]

- "Bridge and Hint: Extending Pre-trained Language Models for Long-Range Code" [2024-05] [[paper](https://arxiv.org/abs/2405.11233)]

- "Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study" [2024-05] [[paper](https://arxiv.org/abs/2405.15614)]

- "LLM-Assisted Static Analysis for Detecting Security Vulnerabilities" [2024-05] [[paper](https://arxiv.org/abs/2405.17238)]

- "Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning" [2024-06] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2406.03718)]

- "Security Vulnerability Detection with Multitask Self-Instructed Fine-Tuning of Large Language Models" [2024-06] [[paper](https://arxiv.org/abs/2406.05892)]

- "M2CVD: Multi-Model Collaboration for Code Vulnerability Detection" [2024-06] [[paper](https://arxiv.org/abs/2406.05940)]

- "Towards Effectively Detecting and Explaining Vulnerabilities Using Large Language Models" [2024-06] [[paper](https://arxiv.org/abs/2406.09701)]

- "Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG" [2024-06] [[paper](https://arxiv.org/abs/2406.11147)]

- "Bug In the Code Stack: Can LLMs Find Bugs in Large Python Code Stacks" [2024-06] [[paper](https://arxiv.org/abs/2406.15325)]

- "Supporting Cross-language Cross-project Bug Localization Using Pre-trained Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.02732)]

- "ALPINE: An adaptive language-agnostic pruning method for language models for code" [2024-07] [[paper](https://arxiv.org/abs/2407.04147)]

- "SCoPE: Evaluating LLMs for Software Vulnerability Detection" [2024-07] [[paper](https://arxiv.org/abs/2407.14372)]

- "Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection" [2024-07] [[paper](https://arxiv.org/abs/2407.16235)]

- "Code Structure-Aware through Line-level Semantic Learning for Code Vulnerability Detection" [2024-07] [[paper](https://arxiv.org/abs/2407.18877)]

- "A Study of Using Multimodal LLMs for Non-Crash Functional Bug Detection in Android Apps" [2024-07] [[paper](https://arxiv.org/abs/2407.19053)]

- "EaTVul: ChatGPT-based Evasion Attack Against Software Vulnerability Detection" [2024-07] [[paper](https://arxiv.org/abs/2407.19216)]

- "Evaluating Large Language Models in Detecting Test Smells" [2024-07] [[paper](https://arxiv.org/abs/2407.19261)]

- "Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models" [2024-07] [[paper](https://arxiv.org/abs/2408.00197)]

- "A Qualitative Study on Using ChatGPT for Software Security: Perception vs. Practicality" [2024-08] [[paper](https://arxiv.org/abs/2408.00435)]

- "Large Language Models for Secure Code Assessment: A Multi-Language Empirical Study" [2024-08] [[paper](https://arxiv.org/abs/2408.06428)]

- "VulCatch: Enhancing Binary Vulnerability Detection through CodeT5 Decompilation and KAN Advanced Feature Extraction" [2024-08] [[paper](https://arxiv.org/abs/2408.07181)]

- "Impact of Large Language Models of Code on Fault Localization" [2024-08] [[paper](https://arxiv.org/abs/2408.09657)]

- "Better Debugging: Combining Static Analysis and LLMs for Explainable Crashing Fault Localization" [2024-08] [[paper](https://arxiv.org/abs/2408.12070)]

- "Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques" [2024-09] [[paper](https://arxiv.org/abs/2409.01001)]

- "CLNX: Bridging Code and Natural Language for C/C++ Vulnerability-Contributing Commits Identification" [2024-09] [[paper](https://arxiv.org/abs/2409.07407)]

- "Code Vulnerability Detection: A Comparative Analysis of Emerging Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.10490)]

- "Program Slicing in the Era of Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.12369)]

- "Generating API Parameter Security Rules with LLM for API Misuse Detection" [2024-09] [[paper](https://arxiv.org/abs/2409.09288)]

- "Enhancing Fault Localization Through Ordered Code Analysis with LLM Agents and Self-Reflection" [2024-09] [[paper](https://arxiv.org/abs/2409.13642)]

- "Comparing Unidirectional, Bidirectional, and Word2vec Models for Discovering Vulnerabilities in Compiled Lifted Code" [2024-09] [[paper](https://arxiv.org/abs/2409.17513)]

- "Enhancing Pre-Trained Language Models for Vulnerability Detection via Semantic-Preserving Data Augmentation" [2024-10] [[paper](https://arxiv.org/abs/2410.00249)]

- "StagedVulBERT: Multi-Granular Vulnerability Detection with a Novel Pre-trained Code Model" [2024-10] [[paper](https://arxiv.org/abs/2410.05766)]

- "Understanding the AI-powered Binary Code Similarity Detection" [2024-10] [[paper](https://arxiv.org/abs/2410.07537)]

- "RealVul: Can We Detect Vulnerabilities in Web Applications with LLM?" [2024-10] [[paper](https://arxiv.org/abs/2410.07573)]

- "Just-In-Time Software Defect Prediction via Bi-modal Change Representation Learning" [2024-10] [[paper](https://arxiv.org/abs/2410.12107)]

- "DFEPT: Data Flow Embedding for Enhancing Pre-Trained Model Based Vulnerability Detection" [2024-10] [[paper](https://arxiv.org/abs/2410.18479)]

- "Utilizing Precise and Complete Code Context to Guide LLM in Automatic False Positive Mitigation" [2024-11] [[paper](https://arxiv.org/abs/2411.03079)]

- "Smart-LLaMA: Two-Stage Post-Training of Large Language Models for Smart Contract Vulnerability Detection and Explanation" [2024-11] [[paper](https://arxiv.org/abs/2411.06221)]

- "FlexFL: Flexible and Effective Fault Localization with Open-Source Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.10714)]

- "Breaking the Cycle of Recurring Failures: Applying Generative AI to Root Cause Analysis in Legacy Banking Systems" [2024-11] [[paper](https://arxiv.org/abs/2411.13017)]

- "Are Large Language Models Memorizing Bug Benchmarks?" [2024-11] [[paper](https://arxiv.org/abs/2411.13323)]

- "An Empirical Study of Vulnerability Detection using Federated Learning" [2024-11] [[paper](https://arxiv.org/abs/2411.16099)]

- "Fault Localization from the Semantic Code Search Perspective" [2024-11] [[paper](https://arxiv.org/abs/2411.17230)]

- "Applying Contrastive Learning to Code Vulnerability Type Classification" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.666/)]

- "Enhancing IR-based Fault Localization using Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.03754)]

- "Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection" [2024-12] [[paper](https://arxiv.org/abs/2412.12039)]

- "A Large Language Model Approach to Identify Flakiness in C++ Projects" [2024-12] [[paper](https://arxiv.org/abs/2412.12340)]

- "A Comprehensive Evaluation of Parameter-Efficient Fine-Tuning on Method-Level Code Smell Detection" [2024-12] [[paper](https://arxiv.org/abs/2412.13801)]

- "Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection & Repair in the IDE" [2024-12] [[paper](https://arxiv.org/abs/2412.14306)]

- "Large Language Models and Code Security: A Systematic Literature Review" [2024-12] [[paper](https://arxiv.org/abs/2412.15004)]

- "Vulnerability Detection in Popular Programming Languages with Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.15905)]

- "Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study" [2024-12] [[paper](https://arxiv.org/abs/2412.18260)]

- "The Impact of Input Order Bias on Large Language Models for Software Fault Localization" [2024-12] [[paper](https://arxiv.org/abs/2412.18750)]

- "CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection" [2025-01] [[paper](https://arxiv.org/abs/2501.04510)]

- "Automating the Detection of Code Vulnerabilities by Analyzing GitHub Issues" [2025-01] [[paper](https://arxiv.org/abs/2501.05258)]

- "Improved IR-based Bug Localization with Intelligent Relevance Feedback" [2025-01] [[paper](https://arxiv.org/abs/2501.10542)]

- "Assessing Large Language Models in Comprehending and Verifying Concurrent Programs across Memory Models" [2025-01] [[paper](https://arxiv.org/abs/2501.14326)]

- "RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing" [2025-01] [[paper](https://arxiv.org/abs/2501.18160)]

- "Streamlining Security Vulnerability Triage with Large Language Models" [2025-01] [[paper](https://arxiv.org/abs/2501.18908)]

- "COSMosFL: Ensemble of Small Language Models for Fault Localisation" [2025-02] [[paper](https://arxiv.org/abs/2502.02908)]

- "Large Language Models for In-File Vulnerability Localization Can Be "Lost in the End"" [2025-02] [[paper](https://arxiv.org/abs/2502.06898)]

- "Where's the Bug? Attention Probing for Scalable Fault Localization" [2025-02] [[paper](https://arxiv.org/abs/2502.13966)]

- "Bridging Bug Localization and Issue Fixing: A Hierarchical Localization Framework Leveraging Large Language Models" [2025-02] [[paper](https://arxiv.org/abs/2502.15292)]

- "A Contemporary Survey of Large Language Model Assisted Program Analysis" [2025-02] [[paper](https://arxiv.org/abs/2502.18474)]

- "Beyond Natural Language Perplexity: Detecting Dead Code Poisoning in Code Generation Datasets" [2025-02] [[paper](https://arxiv.org/abs/2502.20246)]

- "LocAgent: Graph-Guided LLM Agents for Code Localization" [2025-03] [[paper](https://arxiv.org/abs/2503.09089)]

- "HALURust: Exploiting Hallucinations of Large Language Models to Detect Vulnerabilities in Rust" [2025-03] [[paper](https://arxiv.org/abs/2503.10793)]

- "CoSIL: Software Issue Localization via LLM-Driven Code Repository Graph Searching" [2025-03] [[paper](https://arxiv.org/abs/2503.22424)]

- "How Accurately Do Large Language Models Understand Code?" [2025-04] [[paper](https://arxiv.org/abs/2504.04372)]

- "Program Semantic Inequivalence Game with Large Language Models" [2025-05] [[paper](https://arxiv.org/abs/2505.03818)]

- "A Preliminary Study of Large Language Models for Multilingual Vulnerability Detection" [2025-05] [[paper](https://arxiv.org/abs/2505.07376)]

- "Learning to Focus: Context Extraction for Efficient Code Vulnerability Detection with Language Models" [2025-05] [[paper](https://arxiv.org/abs/2505.17460)]

### Malicious Code Detection

- "I-MAD: Interpretable Malware Detector Using Galaxy Transformer", 2019-09, Comput. Secur. 2021, [[paper](https://arxiv.org/abs/1909.06865)]

- "Malbert: A novel pre-training method for malware detection", 2021-09, Comput. Secur. 2021, [[paper](https://www.sciencedirect.com/science/article/pii/S0167404821002820)]

- "Single-Shot Black-Box Adversarial Attacks Against Malware Detectors: A Causal Language Model Approach", 2021-12, ISI 2021, [[paper](https://arxiv.org/abs/2112.01724)]

- "M2VMapper: Malware-to-Vulnerability mapping for Android using text processing", 2021-12, Expert Syst. Appl. 2022, [[paper](https://www.sciencedirect.com/science/article/pii/S0957417421016572)]

- "An Ensemble of Pre-trained Transformer Models For Imbalanced Multiclass Malware Classification", 2021-12, Comput. Secur. 2022, [[paper](https://arxiv.org/abs/2112.13236)]

- "Static Malware Detection Using Stacked BiLSTM and GPT-2", 2022-05, IEEE Access 2022, [[paper](https://ieeexplore.ieee.org/document/9785789)]

- "APT Malicious Sample Organization Traceability Based on Text Transformer Model", 2022-07, PRML 2022, [[paper](https://ieeexplore.ieee.org/document/9882232)]

- "Self-Supervised Vision Transformers for Malware Detection", 2022-08, IEEE Access 2022, [[paper](https://arxiv.org/abs/2208.07049)]

- "A Survey of Recent Advances in Deep Learning Models for Detecting Malware in Desktop and Mobile Platforms", 2022-09, ACM Computing Surveys, [[paper](https://dl.acm.org/doi/abs/10.1145/3638240)]

- "Malicious Source Code Detection Using Transformer", 2022-09, [[paper](https://arxiv.org/abs/2209.07957)]

- "MalBERTv2: Code Aware BERT-Based Model for Malware Identification" [2023-03] [Big Data Cogn. Comput. 2023] [[paper](https://www.mdpi.com/2504-2289/7/2/60)]

- "GPThreats-3: Is Automatic Malware Generation a Threat?" [2023-05] [SPW 2023] [[paper](https://ieeexplore.ieee.org/abstract/document/10188649)]

- "GitHub Copilot: A Threat to High School Security? Exploring GitHub Copilot's Proficiency in Generating Malware from Simple User Prompts" [2023-08] [ETNCC 2023] [[paper](https://ieeexplore.ieee.org/abstract/document/10284976)]

- "An Attacker’s Dream? Exploring the Capabilities of ChatGPT for Developing Malware" [2023-08] [CSET 2023] [[paper](https://dl.acm.org/doi/abs/10.1145/3607505.3607513)]

- "Malicious code detection in android: the role of sequence characteristics and disassembling methods" [2023-12] [Int. J. Inf. Sec. 2023] [[paper](https://arxiv.org/abs/2312.01113)]

- "Prompt Engineering-assisted Malware Dynamic Analysis Using GPT-4" [2023-12] [[paper](https://arxiv.org/abs/2312.08317)]

- "Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models" [2024-03] [[paper](https://arxiv.org/abs/2403.12196)]

- "AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering" [2024-04] [[paper](https://arxiv.org/abs/2404.18816)]

- "Tactics, Techniques, and Procedures (TTPs) in Interpreted Malware: A Zero-Shot Generation with Large Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.08532)]

- "DetectBERT: Towards Full App-Level Representation Learning to Detect Android Malware" [2024-08] [[paper](https://arxiv.org/abs/2408.16353)]

- "PackageIntel: Leveraging Large Language Models for Automated Intelligence Extraction in Package Ecosystems" [2024-09] [[paper](https://arxiv.org/abs/2409.15049)]

- "AgentDroid: A Multi-Agent Framework for Detecting Fraudulent Android Applications" [2025-03] [[paper](https://arxiv.org/abs/2503.12163)]

- "Large Language Model (LLM) for Software Security: Code Analysis, Malware Analysis, Reverse Engineering" [2025-04] [[paper](https://arxiv.org/abs/2504.07137)]

### Compiler Optimization

- "Learning Performance-Improving Code Edits" [2023-06] [ICLR 2024 Spotlight] [[paper](https://arxiv.org/abs/2302.07867)]

- "Large Language Models for Compiler Optimization" [2023-09] [[paper](https://arxiv.org/abs/2309.07062)]

- "Refining Decompiled C Code with Large Language Models" [2023-10] [[paper](https://arxiv.org/abs/2310.06530)]

- "Priority Sampling of Large Language Models for Compilers" [2024-02] [[paper](https://arxiv.org/abs/2402.18734)]

- "Should AI Optimize Your Code? A Comparative Study of Current Large Language Models Versus Classical Optimizing Compilers" [2024-06] [[paper](https://arxiv.org/abs/2406.12146)]

- "Iterative or Innovative? A Problem-Oriented Perspective for Code Optimization" [2024-06] [[paper](https://arxiv.org/abs/2406.11935)]

- "Meta Large Language Model Compiler: Foundation Models of Compiler Optimization" [2024-06] [[paper](https://arxiv.org/abs/2407.02524)]

- "ViC: Virtual Compiler Is All You Need For Assembly Code Search" [2024-08] [[paper](https://arxiv.org/abs/2408.06385)]

- "Search-Based LLMs for Code Optimization" [2024-08] [[paper](https://arxiv.org/abs/2408.12159)]

- "E-code: Mastering Efficient Code Generation through Pretrained Models and Expert Encoder Group" [2024-08] [[paper](https://arxiv.org/abs/2408.12948)]

- "Large Language Models for Energy-Efficient Code: Emerging Results and Future Directions" [2024-10] [[paper](https://arxiv.org/abs/2410.09241)]

- "Introducing Compiler Semantics into Large Language Models as Programming Language Translators: A Case Study of C to x86 Assembly" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.55/)]

- "Towards LLM-based optimization compilers. Can LLMs learn how to apply a single peephole optimization? Reasoning is all LLMs need!" [2024-12] [[paper](https://arxiv.org/abs/2412.12163)]

- "Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation" [2024-12] [[paper](https://arxiv.org/abs/2412.16135)]

- "Finding Missed Code Size Optimizations in Compilers using LLMs" [2024-12] [[paper](https://arxiv.org/abs/2501.00655)]

- "Optimizing Code Runtime Performance through Context-Aware Retrieval-Augmented Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.16692)]

- "VecTrans: LLM Transformation Framework for Better Auto-vectorization on High-performance CPU" [2025-03] [[paper](https://arxiv.org/abs/2503.19449)]

- "Improving Assembly Code Performance with Large Language Models via Reinforcement Learning" [2025-05] [[paper](https://arxiv.org/abs/2505.11480)]

- "Autocomp: LLM-Driven Code Optimization for Tensor Accelerators" [2025-05] [[paper](https://arxiv.org/abs/2505.18574)]

### Binary Analysis and Decompilation

- "Using recurrent neural networks for decompilation" [2018-03] [SANER 2018] [[paper](https://ieeexplore.ieee.org/document/8330222)]

- "Evolving Exact Decompilation" [2018] [[paper](https://eschulte.github.io/data/bed.pdf)]

- "Towards Neural Decompilation" [2019-05] [[paper](https://arxiv.org/abs/1905.08325)]

- "Coda: An End-to-End Neural Program Decompiler" [2019-06] [NeurIPS 2019] [[paper](https://arxiv.org/abs/1906.12029)]

- "N-Bref : A High-fidelity Decompiler Exploiting Programming Structures" [2020-09] [[paper](https://openreview.net/forum?id=6GkL6qM3LV)]

- "Neutron: an attention-based neural decompiler" [2021-03] [Cybersecurity 2021] [[paper](https://cybersecurity.springeropen.com/articles/10.1186/s42400-021-00070-0)]

- "Beyond the C: Retargetable Decompilation using Neural Machine Translation" [2022-12] [[paper](https://arxiv.org/abs/2212.08950)]

- "Boosting Neural Networks to Decompile Optimized Binaries" [2023-01] [ACSAC 2022] [[paper](https://arxiv.org/abs/2301.00969)]

- "SLaDe: A Portable Small Language Model Decompiler for Optimized Assembly" [2023-05] [[paper](https://arxiv.org/abs/2305.12520)]

- "Nova: Generative Language Models for Assembly Code with Hierarchical Attention and Contrastive Learning" [2023-11] [ICLR 2025] [[paper](https://arxiv.org/abs/2311.13721)]

- "CodeArt: Better Code Models by Attention Regularization When Symbols Are Lacking" [2024-11] [[paper](https://arxiv.org/abs/2402.11842)]

- "LLM4Decompile: Decompiling Binary Code with Large Language Models" [2024-03] [EMNLP 2024] [[paper](https://arxiv.org/abs/2403.05286)]

- "WaDec: Decompile WebAssembly Using Large Language Model" [2024-06] [[paper](https://arxiv.org/abs/2406.11346)]

- "MAD: Move AI Decompiler to Improve Transparency and Auditability on Non-Open-Source Blockchain Smart Contract" [2024-10] [[paper](https://arxiv.org/abs/2410.15275)]

- "Source Code Foundation Models are Transferable Binary Analysis Knowledge Bases" [2024-11] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.19581)]

- "A Progressive Transformer for Unifying Binary Code Embedding and Knowledge Transfer" [2024-12] [[paper](https://arxiv.org/abs/2412.11177)]

- "Augmenting Smart Contract Decompiler Output through Fine-grained Dependency Analysis and LLM-facilitated Semantic Recovery" [2025-01] [[paper](https://arxiv.org/abs/2501.08670)]

- "Idioms: Neural Decompilation With Joint Code and Type Prediction" [2025-02] [[paper](https://arxiv.org/abs/2502.04536)]

- "Can Large Language Models Understand Intermediate Representations?" [2025-02] [[paper](https://arxiv.org/abs/2502.06854)]

- "On the Role of Pre-trained Embeddings in Binary Code Analysis" [2025-02] [[paper](https://arxiv.org/abs/2502.08682)]

- "Control Flow-Augmented Decompiler based on Large Language Model" [2025-03] [[paper](https://arxiv.org/abs/2503.07215)]

- "ASMA-Tune: Unlocking LLMs' Assembly Code Comprehension via Structural-Semantic Instruction Tuning" [2025-03] [[paper](https://arxiv.org/abs/2503.11617)]

- "BinMetric: A Comprehensive Binary Analysis Benchmark for Large Language Models" [2025-05] [[paper](https://arxiv.org/abs/2505.07360)]

### Commit Message Generation

- "Automated Commit Message Generation with Large Language Models: An Empirical Study and Beyond" [2024-04] [[paper](https://arxiv.org/abs/2404.14824)]

- "Towards Realistic Evaluation of Commit Message Generation by Matching Online and Offline Settings" [2024-10] [[paper](https://arxiv.org/abs/2410.12046)]

- "Optimization is Better than Generation: Optimizing Commit Message Leveraging Human-written Commit Message" [2025-01] [[paper](https://arxiv.org/abs/2501.09861)]

- "An Empirical Study on Commit Message Generation using LLMs via In-Context Learning" [2025-02] [[paper](https://arxiv.org/abs/2502.18904)]

- "Consider What Humans Consider: Optimizing Commit Message Leveraging Contexts Considered By Human" [2025-03] [[paper](https://arxiv.org/abs/2503.11960)]

### Code Review

- "Using Pre-Trained Models to Boost Code Review Automation" [2022-01] [ICSE 2022] [[paper](https://arxiv.org/abs/2201.06850)]

- "AUGER: Automatically Generating Review Comments with Pre-training Models" [2022-08] [ESEC/FSE 2022] [[paper](https://arxiv.org/abs/2208.08014)]

- "Automatic Code Review by Learning the Structure Information of Code Graph" [2023-02] [Sensors] [[paper](https://www.mdpi.com/1424-8220/23/5/2551)]

- "LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning" [2023-08] [ISSRE 2023] [[paper](https://arxiv.org/abs/2308.11148)]

- "CodeAgent: Autonomous Communicative Agents for Code Review" [2024-02] [EMNLP 2024] [[paper](https://arxiv.org/abs/2402.02172)]

- "AI-powered Code Review with LLMs: Early Results" [2024-04] [[paper](https://arxiv.org/abs/2404.18496)]

- "AI-Assisted Assessment of Coding Practices in Modern Code Review" [2024-05] [[paper](https://arxiv.org/abs/2405.13565)]

- "A GPT-based Code Review System for Programming Language Learning" [2024-07] [[paper](https://arxiv.org/abs/2407.04722)]

- "LLM Critics Help Catch LLM Bugs" [2024-06] [[paper](https://arxiv.org/abs/2407.00215)]

- "Exploring the Capabilities of LLMs for Code Change Related Tasks" [2024-07] [[paper](https://arxiv.org/abs/2407.02824)]

- "Evaluating Language Models for Generating and Judging Programming Feedback" [2024-07] [[paper](https://arxiv.org/abs/2407.04873)]

- "Can LLMs Replace Manual Annotation of Software Engineering Artifacts?" [2024-08] [[paper](https://arxiv.org/abs/2408.05534)]

- "Leveraging Reviewer Experience in Code Review Comment Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.10959)]

- "CRScore: Grounding Automated Evaluation of Code Review Comments in Code Claims and Smells" [2024-09] [[paper](https://arxiv.org/abs/2409.19801)]

- "Enhancing Code Annotation Reliability: Generative AI's Role in Comment Quality Assessment Models" [2024-10] [[paper](https://arxiv.org/abs/2410.22323)]

- "Knowledge-Guided Prompt Learning for Request Quality Assurance in Public Code Review" [2024-10] [[paper](https://arxiv.org/abs/2410.21673)]

- "Impact of LLM-based Review Comment Generation in Practice: A Mixed Open-/Closed-source User Study" [2024-11] [[paper](https://arxiv.org/abs/2411.07091)]

- "Prompting and Fine-tuning Large Language Models for Automated Code Review Comment Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.10129)]

- "Deep Learning-based Code Reviews: A Paradigm Shift or a Double-Edged Sword?" [2024-11] [[paper](https://arxiv.org/abs/2411.11401)]

- "Redefining Crowdsourced Test Report Prioritization: An Innovative Approach with Large Language Model" [2024-11] [[paper](https://arxiv.org/abs/2411.17045)]

- "Exploring the Potential of Llama Models in Automated Code Refinement: A Replication Study" [2024-12] [[paper](https://arxiv.org/abs/2412.02789)]

- "Automated Code Review In Practice" [2024-12] [[paper](https://arxiv.org/abs/2412.18531)]

- "Code Review Automation Via Multi-task Federated LLM -- An Empirical Study" [2024-12] [[paper](https://arxiv.org/abs/2412.15676)]

- "DeepCRCEval: Revisiting the Evaluation of Code Review Comment Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.18291)]

- "Distilling Desired Comments for Enhanced Code Review with Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.20340)]

- "Deep Assessment of Code Review Generation Approaches: Beyond Lexical Similarity" [2025-01] [[paper](https://arxiv.org/abs/2501.05176)]

- "Debugging Without Error Messages: How LLM Prompting Strategy Affects Programming Error Explanation Effectiveness" [2025-01] [[paper](https://arxiv.org/abs/2501.05706)]

- "I Can Find You in Seconds! Leveraging Large Language Models for Code Authorship Attribution" [2025-01] [[paper](https://arxiv.org/abs/2501.08165)]

- "Code Change Intention, Development Artifact and History Vulnerability: Putting Them Together for Vulnerability Fix Detection by LLM" [2025-01] [[paper](https://arxiv.org/abs/2501.14983)]

- "BitsAI-CR: Automated Code Review via LLM in Practice" [2025-01] [[paper](https://arxiv.org/abs/2501.15134)]

- "Large Language Model Critics for Execution-Free Evaluation of Code Changes" [2025-01] [[paper](https://arxiv.org/abs/2501.16655)]

- "Too Noisy To Learn: Enhancing Data Quality for Code Review C" [2025-02] [[paper](https://arxiv.org/abs/2502.02757)]

- "Harnessing Large Language Models for Curated Code Reviews" [2025-02] [[paper](https://arxiv.org/abs/2502.03425)]

- "Combining Large Language Models with Static Analyzers for Code Review Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.06633)]

- "Automating Code Review: A Systematic Literature Review" [2025-03] [[paper](https://arxiv.org/abs/2503.09510)]

- "Towards Practical Defect-Focused Automated Code Review" [2025-05] [[paper](https://arxiv.org/abs/2505.17928)]

### Log Analysis

- "LogStamp: Automatic Online Log Parsing Based on Sequence Labelling" [2022-08] [[paper](https://arxiv.org/abs/2208.10282)]

- "Log Parsing with Prompt-based Few-shot Learning" [2023-02] [ICSE 2023] [[paper](https://arxiv.org/abs/2302.07435)]

- "Log Parsing: How Far Can ChatGPT Go?" [2023-06] [ASE 2023] [[paper](https://arxiv.org/abs/2306.01590)]

- "LogPrompt: Prompt Engineering Towards Zero-Shot and Interpretable Log Analysis" [2023-08] [[paper](https://arxiv.org/abs/2308.07610)]

- "LogGPT: Exploring ChatGPT for Log-Based Anomaly Detection" [2023-09] [[paper](https://arxiv.org/abs/2309.01189)]

- "An Assessment of ChatGPT on Log Data" [2023-09] [[paper](https://arxiv.org/abs/2309.07938)]

- "LILAC: Log Parsing using LLMs with Adaptive Parsing Cache" [2023-10] [[paper](https://arxiv.org/abs/2310.01796)]

- "LLMParser: An Exploratory Study on Using Large Language Models for Log Parsing" [2024-04] [[paper](https://arxiv.org/abs/2404.18001)]

- "On the Influence of Data Resampling for Deep Learning-Based Log Anomaly Detection: Insights and Recommendations" [2024-05] [[paper](https://arxiv.org/abs/2405.03489)]

- "Log Parsing with Self-Generated In-Context Learning and Self-Correction" [2024-06] [[paper](https://arxiv.org/abs/2406.03376)]

- "Stronger, Faster, and Cheaper Log Parsing with LLMs" [2024-06] [[paper](https://arxiv.org/abs/2406.06156)]

- "ULog: Unsupervised Log Parsing with Large Language Models through Log Contrastive Units" [2024-06] [[paper](https://arxiv.org/abs/2406.07174)]

- "Anomaly Detection on Unstable Logs with GPT Models" [2024-06] [[paper](https://arxiv.org/abs/2406.07467)]

- "LogParser-LLM: Advancing Efficient Log Parsing with Large Language Models" [2024-08] [KDD 2024] [[paper](https://arxiv.org/abs/2408.13727)]

- "LUK: Empowering Log Understanding with Expert Knowledge from Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.01909)]

- "A Comparative Study on Large Language Models for Log Parsing" [2024-09] [[paper](https://arxiv.org/abs/2409.02474)]

- "What Information Contributes to Log-based Anomaly Detection? Insights from a Configurable Transformer-Based Approach" [2024-10] [[paper](https://arxiv.org/abs/2409.20503)]

- "LogLM: From Task-based to Instruction-based Automated Log Analysis" [2024-10] [[paper](https://arxiv.org/abs/2410.09352)]

- "LogLLM: Log-based Anomaly Detection Using Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.08561)]

- "Adapting Large Language Models to Log Analysis with Interpretable Domain Knowledge" [2024-12] [[paper](https://arxiv.org/abs/2412.01377)]

- "Chatting with Logs: An exploratory study on Finetuning LLMs for LogQL" [2024-12] [[paper](https://arxiv.org/abs/2412.03612)]

- "LogBabylon: A Unified Framework for Cross-Log File Integration and Analysis" [2024-12] [[paper](https://arxiv.org/abs/2412.12364)]

- "AdaptiveLog: An Adaptive Log Analysis Framework with the Collaboration of Large and Small Language Model" [2025-01] [[paper](https://arxiv.org/abs/2501.11031)]

- "Explaining GitHub Actions Failures with Large Language Models: Challenges, Insights, and Limitations" [2025-01] [[paper](https://arxiv.org/abs/2501.16495)]

### Software Configuration

- "Configuration Validation with Large Language Models" [2023-10] [[paper](https://arxiv.org/abs/2310.09690)]

- "CloudEval-YAML: A Practical Benchmark for Cloud Configuration Generation" [2023-11] [[paper](https://arxiv.org/abs/2401.06786)]

- "Can LLMs Configure Software Tools" [2023-12] [[paper](https://arxiv.org/abs/2312.06121)]

- "LuaTaint: A Static Analysis System for Web Configuration Interface Vulnerability of Internet of Things Devices" [2024-02] [IOT] [[paper](https://arxiv.org/abs/2402.16043)]

- "LLM-Based Misconfiguration Detection for AWS Serverless Computing" [2024-11] [[paper](https://arxiv.org/abs/2411.00642)]

- "Leveraging LLM Agents for Translating Network Configurations" [2025-01] [[paper](https://arxiv.org/abs/2501.08760)]

- "Raiders of the Lost Dependency: Fixing Dependency Conflicts in Python using LLMs" [2025-01] [[paper](https://arxiv.org/abs/2501.16191)]

- "Enabling Autonomic Microservice Management through Self-Learning Agents" [2025-01] [[paper](https://arxiv.org/abs/2501.19056)]

- "LLMSecConfig: An LLM-Based Approach for Fixing Software Container Misconfigurations" [2025-02] [[paper](https://arxiv.org/abs/2502.02009)]

- "An LLM-based Agent for Reliable Docker Environment Configuration" [2025-02] [[paper](https://arxiv.org/abs/2502.13681)]

- "Automated Benchmark Generation for Repository-Level Coding Tasks" [2025-03] [[paper](https://arxiv.org/abs/2503.07701)]

- "BYOS: Knowledge-driven Large Language Models Bring Your Own Operating System More Excellent" [2025-03] [[paper](https://arxiv.org/abs/2503.09663)]

- "EnvBench: A Benchmark for Automated Environment Setup" [2025-03] [[paper](https://arxiv.org/abs/2503.14443)]

### Code QA & Reasoning

- "DialogAgent: An Auto-engagement Agent for Code Question Answering Data Production" [2024-12] [[paper](https://arxiv.org/abs/2412.08069)]

- "Unveiling the Magic of Code Reasoning through Hypothesis Decomposition and Amendment" [2025-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2502.13170)]

- "CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models" [2025-03] [[paper](https://arxiv.org/abs/2503.16167)]

- "Evaluating the Generalization Capabilities of Large Language Models on Code Reasoning" [2025-04] [[paper](https://arxiv.org/abs/2504.05518)]

- "The Code Barrier: What LLMs Actually Understand?" [2025-04] [[paper](https://arxiv.org/abs/2504.10557)]

- "Can Large Language Models Predict Parallel Code Performance?" [2025-05] [[paper](https://arxiv.org/abs/2505.03988)]

- "LongCodeBench: Evaluating Coding LLMs at 1M Context Windows" [2025-05] [[paper](https://arxiv.org/abs/2505.07897)]

- "Do Code LLMs Do Static Analysis?" [2025-05] [[paper](https://arxiv.org/abs/2505.12118)]

- "Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Reasoning" [2025-05] [[paper](https://arxiv.org/abs/2505.13353)]

- "MARCO: Meta-Reflection with Cross-Referencing for Code Reasoning" [2025-05] [[paper](https://arxiv.org/abs/2505.17481)]

### Software Modeling

- "Towards using Few-Shot Prompt Learning for Automating Model Completion" [2022-12] [[paper](https://arxiv.org/abs/2212.03404)]

- "Model Generation from Requirements with LLMs: an Exploratory Study" [2024-04] [[paper](https://arxiv.org/abs/2404.06371)]

- "How LLMs Aid in UML Modeling: An Exploratory Study with Novice Analysts" [2024-04] [[paper](https://arxiv.org/abs/2404.17739)]

- "Leveraging Large Language Models for Software Model Completion: Results from Industrial and Public Datasets" [2024-06] [[paper](https://arxiv.org/abs/2406.17651)]

- "Studying and Benchmarking Large Language Models For Log Level Suggestion" [2024-10] [[paper](https://arxiv.org/abs/2410.08499)]

- "A Model Is Not Built By A Single Prompt: LLM-Based Domain Modeling With Question Decomposition" [2024-10] [[paper](https://arxiv.org/abs/2410.09854)]

- "On the Utility of Domain Modeling Assistance with Large Language Models" [2024-10] [[paper](https://arxiv.org/abs/2410.12577)]

- "On the use of Large Language Models in Model-Driven Engineering" [2024-10] [[paper](https://arxiv.org/abs/2410.17370)]

- "LLM as a code generator in Agile Model Driven Development" [2024-10] [[paper](https://arxiv.org/abs/2410.18489)]

- "Assessing UML Models by ChatGPT: Implications for Education" [2024-12] [[paper](https://arxiv.org/abs/2412.17200)]

### Requirement Engineering

- "A Transformer-based Approach for Abstractive Summarization of Requirements from Obligations in Software Engineering Contracts" [2023-09] [RE 2023] [[paper](https://ieeexplore.ieee.org/document/10260954)]

- "Advancing Requirements Engineering through Generative AI: Assessing the Role of LLMs" [2023-10] [[paper](https://arxiv.org/abs/2310.13976)]

- "Requirements Engineering using Generative AI: Prompts and Prompting Patterns" [2023-11] [[paper](https://arxiv.org/abs/2311.03832)]

- "Prioritizing Software Requirements Using Large Language Models" [2024-04] [[paper](https://arxiv.org/abs/2405.01564)]

- "Lessons from the Use of Natural Language Inference (NLI) in Requirements Engineering Tasks" [2024-04] [[paper](https://arxiv.org/abs/2405.05135)]

- "Enhancing Legal Compliance and Regulation Analysis with Large Language Models" [2024-04] [[paper](https://arxiv.org/abs/2404.17522)]

- "MARE: Multi-Agents Collaboration Framework for Requirements Engineering" [2024-05] [[paper](https://arxiv.org/abs/2405.03256)]

- "Natural Language Processing for Requirements Traceability" [2024-05] [[paper](https://arxiv.org/abs/2405.10845)]

- "Multilingual Crowd-Based Requirements Engineering Using Large Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.06505)]

- "From Specifications to Prompts: On the Future of Generative LLMs in Requirements Engineering" [2024-08] [[paper](https://arxiv.org/abs/2408.09127)]

- "Leveraging LLMs for the Quality Assurance of Software Requirements" [2024-08] [[paper](https://arxiv.org/abs/2408.10886)]

- "Generative AI for Requirements Engineering: A Systematic Literature Review" [2024-09] [[paper](https://arxiv.org/abs/2409.06741)]

- "A Fine-grained Sentiment Analysis of App Reviews using Large Language Models: An Evaluation Study" [2024-09] [[paper](https://arxiv.org/abs/2409.07162)]

- "Leveraging Large Language Models for Predicting Cost and Duration in Software Engineering Projects" [2024-09] [[paper](https://arxiv.org/abs/2409.09617)]

- "Privacy Policy Analysis through Prompt Engineering for LLMs" [2024-09] [[paper](https://arxiv.org/abs/2409.14879)]

- "Exploring Requirements Elicitation from App Store User Reviews Using Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.15473)]

- "LLM-Cure: LLM-based Competitor User Review Analysis for Feature Enhancement" [2024-09] [[paper](https://arxiv.org/abs/2409.15724)]

- "Automatic Instantiation of Assurance Cases from Patterns Using Large Language Models" [2024-10] [[paper](https://arxiv.org/abs/2410.05488)]

- "Whose fault is it anyway? SILC: Safe Integration of LLM-Generated Code" [2024-10] [[paper](https://arxiv.org/abs/2410.18703)]

- "Assured Automatic Programming via Large Language Models" [2024-10] [[paper](https://arxiv.org/abs/2410.18494)]

- "Does GenAI Make Usability Testing Obsolete?" [2024-11] [[paper](https://arxiv.org/abs/2411.00634)]

- "Exploring LLMs for Verifying Technical System Specifications Against Requirements" [2024-11] [[paper](https://arxiv.org/abs/2411.11582)]

- "Towards the LLM-Based Generation of Formal Specifications from Natural-Language Contracts: Early Experiments with Symboleo" [2024-11] [[paper](https://arxiv.org/abs/2411.15898)]

- "PassionNet: An Innovative Framework for Duplicate and Conflicting Requirements Identification" [2024-12] [[paper](https://arxiv.org/abs/2412.01657)]

- "Generative Language Models Potential for Requirement Engineering Applications: Insights into Current Strengths and Limitations" [2024-12] [[paper](https://arxiv.org/abs/2412.00959)]

- "LicenseGPT: A Fine-tuned Foundation Model for Publicly Available Dataset License Compliance" [2024-12] [[paper](https://arxiv.org/abs/2501.00106)]

- "On the Impact of Requirements Smells in Prompts: The Case of Automated Traceability" [2025-01] [[paper](https://arxiv.org/abs/2501.04810)]

- "Analysis of LLMs vs Human Experts in Requirements Engineering" [2025-01] [[paper](https://arxiv.org/abs/2501.19297)]

## 6. Analysis of AI-Generated Code

### Security and Vulnerabilities

- "You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion" [2021-08] [USENIX Security Symposium 2021] [[paper](https://www.usenix.org/conference/usenixsecurity21/presentation/schuster)]

- "Is GitHub's Copilot as Bad as Humans at Introducing Vulnerabilities in Code?" [2022-04] [Empir. Softw. Eng.] [[paper](https://arxiv.org/abs/2204.04741)]

- "Lost at C: A User Study on the Security Implications of Large Language Model Code Assistants" [2022-08] [USENIX Security Symposium 2023] [[paper](https://arxiv.org/abs/2208.09727)]

- "Do Users Write More Insecure Code with AI Assistants?" [2022-1] [CCS 2023] [[paper](https://arxiv.org/abs/2211.03622)]

- "Large Language Models for Code: Security Hardening and Adversarial Testing" [2023-02] [CCS 2023] [[paper](https://arxiv.org/abs/2302.05319)]

- "Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models" [2023-12] [[paper](https://arxiv.org/abs/2312.04724)]

- "CodeAttack: Revealing Safety Generalization Challenges of Large Language Models via Code Completion" [2024-03] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2403.07865)]

- "Just another copy and paste? Comparing the security vulnerabilities of ChatGPT generated code and StackOverflow answers" [2024-03] [[paper](https://arxiv.org/abs/2403.15600)]

- "DeVAIC: A Tool for Security Assessment of AI-generated Code" [2024-04] [[paper](https://arxiv.org/abs/2404.07548)]

- "CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models" [2024-04] [[paper](https://arxiv.org/abs/2404.13161)]

- "LLMs in Web-Development: Evaluating LLM-Generated PHP code unveiling vulnerabilities and limitations" [2024-04] [[paper](https://arxiv.org/abs/2404.14459)]

- "Do Neutral Prompts Produce Insecure Code? FormAI-v2 Dataset: Labelling Vulnerabilities in Code Generated by Large Language Models" [2024-04] [[paper](https://arxiv.org/abs/2404.18353)]

- "Codexity: Secure AI-assisted Code Generation" [2024-05] [[paper](https://arxiv.org/abs/2405.03927)]

- "Measuring Impacts of Poisoning on Model Parameters and Embeddings for Large Language Models of Code" [2024-05] [[paper](https://arxiv.org/abs/2405.11466)]

- "An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection" [2024-06] [[paper](https://arxiv.org/abs/2406.06822)]

- "Is Your AI-Generated Code Really Secure? Evaluating Large Language Models on Secure Code Generation with CodeSecEval" [2024-07] [[paper](https://arxiv.org/abs/2407.02395)]

- "Prompting Techniques for Secure Code Generation: A Systematic Investigation" [2024-07] [[paper](https://arxiv.org/abs/2407.07064)]

- "TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs" [2024-07] [[paper](https://arxiv.org/abs/2407.09164)]

- "MaPPing Your Model: Assessing the Impact of Adversarial Attacks on LLM-based Programming Assistants" [2024-07] [[paper](https://arxiv.org/abs/2407.11072)]

- "Eliminating Backdoors in Neural Code Models via Trigger Inversion" [2024-08] [[paper](https://arxiv.org/abs/2408.04683)]

- ""You still have to study" -- On the Security of LLM generated code" [2024-08] [[paper](https://arxiv.org/abs/2408.07106)]

- "How Well Do Large Language Models Serve as End-to-End Secure Code Producers?" [2024-08] [[paper](https://arxiv.org/abs/2408.10495)]

- "While GitHub Copilot Excels at Coding, Does It Ensure Responsible Output?" [2024-08] [[paper](https://arxiv.org/abs/2408.11006)]

- "PromSec: Prompt Optimization for Secure Generation of Functional Source Code with Large Language Models (LLMs)" [2024-09] [[paper](https://arxiv.org/abs/2409.12699)]

- "RMCBench: Benchmarking Large Language Models' Resistance to Malicious Code" [2024-09] [[paper](https://arxiv.org/abs/2409.15154)]

- "Artificial-Intelligence Generated Code Considered Harmful: A Road Map for Secure and High-Quality Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.19182)]

- "SecCoder: Towards Generalizable and Robust Secure Code Generation" [2024-10] [EMNLP 2024] [[paper](https://arxiv.org/abs/2410.01488)]

- "Demonstration Attack against In-Context Learning for Code Intelligence" [2024-10] [[paper](https://arxiv.org/abs/2410.02841)]

- "Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders" [2024-10] [[paper](https://arxiv.org/abs/2410.06462)]

- "SecCodePLT: A Unified Platform for Evaluating the Security of Code GenAI" [2024-10] [[paper](https://arxiv.org/abs/2410.11096)]

- "Security of Language Models for Code: A Systematic Literature Review" [2024-10] [[paper](https://arxiv.org/abs/2410.15631)]

- "RedCode: Risky Code Execution and Generation Benchmark for Code Agents" [2024-11] [[paper](https://arxiv.org/abs/2411.07781)]

- "ProSec: Fortifying Code LLMs with Proactive Security Alignment" [2024-11] [[paper](https://arxiv.org/abs/2411.12882)]

- "Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs" [2024-11] [[paper](https://arxiv.org/abs/2411.18216)]

- "SABER: Model-agnostic Backdoor Attack on Chain-of-Thought in Neural Code Generation" [2024-12] [[paper](https://arxiv.org/abs/2412.05829)]

- "CWEval: Outcome-driven Evaluation on Functionality and Security of LLM Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.08200)]

- "Security and Quality in LLM-Generated Code: A Multi-Language, Multi-Model Analysis" [2025-02] [[paper](https://arxiv.org/abs/2502.01853)]

- "Exploring the Security Threats of Knowledge Base Poisoning in Retrieval-Augmented Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.03233)]

- "Benchmarking Prompt Engineering Techniques for Secure Code Generation with GPT Models" [2025-02] [[paper](https://arxiv.org/abs/2502.06039)]

- "Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions" [2025-02] [[paper](https://arxiv.org/abs/2502.14202)]

- "XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants" [2025-03] [[paper](https://arxiv.org/abs/2503.14281)]

- "A Comprehensive Study of LLM Secure Code Generation" [2025-03] [[paper](https://arxiv.org/abs/2503.15554)]

- "Smoke and Mirrors: Jailbreaking LLM-based Code Generation via Implicit Malicious Prompts" [2025-03] [[paper](https://arxiv.org/abs/2503.17953)]

- "Give LLMs a Security Course: Securing Retrieval-Augmented Code Generation via Knowledge Injection" [2025-04] [[paper](https://arxiv.org/abs/2504.16429)]

- "Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective" [2025-05] [[paper](https://arxiv.org/abs/2505.10494)]

### Correctness

- "An Empirical Evaluation of GitHub Copilot's Code Suggestions" [2022-05] [MSR 2022] [[paper](https://ieeexplore.ieee.org/document/9796235)]

- "Large Language Models and Simple, Stupid Bugs" [2023-03] [MSR 2023] [[paper](https://arxiv.org/abs/2303.11455)]

- "Evaluating the Code Quality of AI-Assisted Code Generation Tools: An Empirical Study on GitHub Copilot, Amazon CodeWhisperer, and ChatGPT" [2023-04] [[paper](https://arxiv.org/abs/2304.10778)]

- "No Need to Lift a Finger Anymore? Assessing the Quality of Code Generation by ChatGPT" [2023-08] [[paper](https://arxiv.org/abs/2308.04838)]

- "The Counterfeit Conundrum: Can Code Language Models Grasp the Nuances of Their Incorrect Generations?" [2024-02] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.19475)]

- "Bugs in Large Language Models Generated Code: An Empirical Study" [2024-03] [[paper](https://arxiv.org/abs/2403.08937)]

- "ChatGPT Incorrectness Detection in Software Reviews" [2024-03] [[paper](https://arxiv.org/abs/2403.16347)]

- "Validating LLM-Generated Programs with Metamorphic Prompt Testing" [2024-06] [[paper](https://arxiv.org/abs/2406.06864)]

- "Where Do Large Language Models Fail When Generating Code?" [2024-06] [[paper](https://arxiv.org/abs/2406.08731)]

- "GitHub Copilot: the perfect Code compLeeter?" [2024-06] [[paper](https://arxiv.org/abs/2406.11326)]

- "What's Wrong with Your Code Generated by Large Language Models? An Extensive Study" [2024-07] [[paper](https://arxiv.org/abs/2407.06153)]

- "Uncovering Weaknesses in Neural Code Generation" [2024-07] [[paper](https://arxiv.org/abs/2407.09793)]

- "Understanding Defects in Generated Codes by Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.13372)]

- "CodeSift: An LLM-Based Reference-Less Framework for Automatic Code Validation" [2024-08] [[paper](https://arxiv.org/abs/2408.15630)]

- "Examination of Code generated by Large Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.16601)]

- "Fixing Code Generation Errors for Large Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.00676)]

- "Can OpenSource beat ChatGPT? -- A Comparative Study of Large Language Models for Text-to-Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.04164)]

- "Insights from Benchmarking Frontier Language Models on Web App Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.05177)]

- "Evaluating the Performance of Large Language Models in Competitive Programming: A Multi-Year, Multi-Grade Analysis" [2024-09] [[paper](https://arxiv.org/abs/2409.09054)]

- "A Case Study of Web App Coding with OpenAI Reasoning Models" [2024-09] [[paper](https://arxiv.org/abs/2409.13773)]

- "An evaluation of LLM code generation capabilities through graded exercises" [2024-10] [[paper](https://arxiv.org/abs/2410.16292)]

- "A Deep Dive Into Large Language Model Code Generation Mistakes: What and Why?" [2024-11] [[paper](https://arxiv.org/abs/2411.01414)]

- "Evaluating ChatGPT-3.5 Efficiency in Solving Coding Problems of Different Complexity Levels: An Empirical Analysis" [2024-11] [[paper](https://arxiv.org/abs/2411.07529)]

- "LLM4DS: Evaluating Large Language Models for Data Science Code Generation" [2024-11] [[paper](https://arxiv.org/abs/2411.11908)]

- "A Preliminary Study of Multilingual Code Language Models for Code Generation Task Using Translated Benchmarks" [2024-11] [[paper](https://arxiv.org/abs/2411.15470)]

- "Analyzing the Energy and Accuracy of LLMs in Software Development" [2024-11] [[paper](https://arxiv.org/abs/2412.00329)]

- "How Well Do LLMs Generate Code for Different Application Domains? Benchmark and Evaluation" [2024-12] [[paper](https://arxiv.org/abs/2412.18573)]

- "LLM-ProS: Analyzing Large Language Models' Performance in Competitive Problem Solving" [2025-02] [[paper](https://arxiv.org/abs/2502.04355)]

- "Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation" [2025-02] [[paper](https://arxiv.org/abs/2502.11620)]

- "Performance Evaluation of Large Language Models in Statistical Programming" [2025-02] [[paper](https://arxiv.org/abs/2502.13117)]

- "Unveiling Pitfalls: Understanding Why AI-driven Code Agents Fail at GitHub Issue Resolution" [2025-03] [[paper](https://arxiv.org/abs/2503.12374)]

- "Are "Solved Issues" in SWE-bench Really Solved Correctly? An Empirical Study" [2025-03] [[paper](https://arxiv.org/abs/2503.15223)]

### Hallucination

- "Exploring and Evaluating Hallucinations in LLM-Powered Code Generation" [2024-04] [[paper](https://arxiv.org/abs/2404.00971)]

- "CodeHalu: Code Hallucinations in LLMs Driven by Execution-based Verification" [2024-04] [[paper](https://arxiv.org/abs/2405.00253)]

- "We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs" [2024-06] [[paper](https://arxiv.org/abs/2406.10279)]

- "Code Hallucination" [2024-07] [[paper](https://arxiv.org/abs/2407.04831)]

- "On Mitigating Code LLM Hallucinations with API Documentation" [2024-07] [[paper](https://arxiv.org/abs/2407.09726)]

- "CodeMirage: Hallucinations in Code Generated by Large Language Models" [2024-08] [[paper](https://arxiv.org/abs/2408.08333)]

- "LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation" [2024-09] [[paper](https://arxiv.org/abs/2409.20550)]

- "Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code" [2024-10] [[paper](https://arxiv.org/abs/2410.09997)]

- "ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries" [2024-10] [[paper](https://arxiv.org/abs/2410.14748)]

- "Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges" [2025-04] [[paper](https://arxiv.org/abs/2504.20799)]

### Efficiency

- "EffiBench: Benchmarking the Efficiency of Automatically Generated Code" [2024-02] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2402.02037)]

- "Mercury: A Code Efficiency Benchmark for Code Large Language Models" [2024-02] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2402.07844)]

- "On Evaluating the Efficiency of Source Code Generated by LLMs" [2024-04] [[paper](https://arxiv.org/abs/2404.06041)]

- "A Controlled Experiment on the Energy Efficiency of the Source Code Generated by Code Llama" [2024-05] [[paper](https://arxiv.org/abs/2405.03616)]

- "From Effectiveness to Efficiency: Comparative Evaluation of Code Generated by LCGMs for Bilingual Programming Questions" [2024-06] [[paper](https://arxiv.org/abs/2406.00602)]

- "How Efficient is LLM-Generated Code? A Rigorous & High-Standard Benchmark" [2024-06] [ICLR 2025] [[paper](https://arxiv.org/abs/2406.06647)]

- "ECCO: Can We Improve Model-Generated Code Efficiency Without Sacrificing Functional Correctness?" [2024-07] [EMNLP 2024] [[paper](https://arxiv.org/abs/2407.14044)]

- "A Performance Study of LLM-Generated Code on Leetcode" [2024-07] [[paper](https://arxiv.org/abs/2407.21579)]

- "Evaluating Language Models for Efficient Code Generation" [2024-08] [[paper](https://arxiv.org/abs/2408.06450)]

- "Effi-Code: Unleashing Code Efficiency in Language Models" [2024-10] [[paper](https://arxiv.org/abs/2410.10209)]

- "Rethinking Code Refinement: Learning to Judge Code Efficiency" [2024-10] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2410.22375)]

- "Generating Energy-efficient code with LLMs" [2024-11] [[paper](https://arxiv.org/abs/2411.10599)]

- "An exploration of the effect of quantisation on energy consumption and inference time of StarCoder2" [2024-11] [[paper](https://arxiv.org/abs/2411.12758)]

- "ACECode: A Reinforcement Learning Framework for Aligning Code Efficiency and Correctness in Code Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.17264)]

- "AI-Powered, But Power-Hungry? Energy Efficiency of LLM-Generated Code" [2025-02] [[paper](https://arxiv.org/abs/2502.02412)]

- "Unveiling Inefficiencies in LLM-Generated Code: Toward a Comprehensive Taxonomy" [2025-03] [[paper](https://arxiv.org/abs/2503.06327)]

- "EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code" [2025-05] [[paper](https://arxiv.org/abs/2505.13004)]

- "Evaluating the Energy-Efficiency of the Code Generated by LLMs" [2025-05] [[paper](https://arxiv.org/abs/2505.20324)]

### Robustness

- "Beyond Accuracy: Evaluating Self-Consistency of Code Large Language Models with IdentityChain" [2023-10] [[paper](https://arxiv.org/abs/2310.14053)]

- "Do Large Code Models Understand Programming Concepts? A Black-box Approach" [2024-02] [ICML 2024] [[paper](https://arxiv.org/abs/2402.05980)]

- "Syntactic Robustness for LLM-based Code Generation" [2024-04] [[paper](https://arxiv.org/abs/2404.01535)]

- "NLPerturbator: Studying the Robustness of Code LLMs to Natural Language Variations" [2024-06] [[paper](https://arxiv.org/abs/2406.19783)]

- "An Empirical Study on Capability of Large Language Models in Understanding Code Semantics" [2024-07] [[paper](https://arxiv.org/abs/2407.03611)]

- "Comparing Robustness Against Adversarial Attacks in Code Generation: LLM-Generated vs. Human-Written" [2024-11] [[paper](https://arxiv.org/abs/2411.10565)]

- "On the Adversarial Robustness of Instruction-Tuned Large Language Models for Code" [2024-11] [[paper](https://arxiv.org/abs/2411.19508)]

- "What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models" [2024-12] [[paper](https://arxiv.org/abs/2412.08098)]

- "Enhancing the Robustness of LLM-Generated Code: Empirical Study and Framework" [2025-03] [[paper](https://arxiv.org/abs/2503.20197)]

- "CODECRASH: Stress Testing LLM Reasoning under Structural and Semantic Perturbations" [2025-04] [[paper](https://arxiv.org/abs/2504.14119)]

- "Success is in the Details: Evaluate and Enhance Details Sensitivity of Code LLMs through Counterfactuals" [2025-05] [[paper](https://arxiv.org/abs/2505.14597)]

### Interpretability

- "A Critical Study of What Code-LLMs (Do Not) Learn" [2024-06] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2406.11930)]

- "Looking into Black Box Code Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.04868)]

- "DeepCodeProbe: Towards Understanding What Models Trained on Code Learn" [2024-07] [[paper](https://arxiv.org/abs/2407.08890)]

- "Towards More Trustworthy and Interpretable LLMs for Code through Syntax-Grounded Explanations" [2024-07] [[paper](https://arxiv.org/abs/2407.08983)]

- "Exploring Coding Spot: Understanding Parametric Contributions to LLM Coding Performance" [2024-12] [[paper](https://arxiv.org/abs/2412.07113)]

- "Correctness Assessment of Code Generated by Large Language Models Using Internal Representations" [2025-01] [[paper](https://arxiv.org/abs/2501.12934)]

- "CodeSCM: Causal Analysis for Multi-Modal Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.05150)]

- "Mechanistic Understanding of Language Models in Syntactic Code Completion" [2025-02] [[paper](https://arxiv.org/abs/2502.18499)]

- "On Explaining (Large) Language Models For Code Using Global Code-Based Explanations" [2025-03] [[paper](https://arxiv.org/abs/2503.16771)]

### API Usage

- "How and Why LLMs Use Deprecated APIs in Code Completion? An Empirical Study" [2024-06] [[paper](https://arxiv.org/abs/2406.09834)]

- "Is ChatGPT a Good Software Librarian? An Exploratory Study on the Use of ChatGPT for Software Library Recommendations" [2024-08] [[paper](https://arxiv.org/abs/2408.05128)]

- "A Systematic Evaluation of Large Code Models in API Suggestion: When, Which, and How" [2024-09] [[paper](https://arxiv.org/abs/2409.13178)]

- "AutoAPIEval: A Framework for Automated Evaluation of LLMs in API-Oriented Code Generation" [2024-09] [[paper](https://arxiv.org/abs/2409.15228)]

- "ADC: Enhancing Function Calling Via Adversarial Datasets and Code Line-Level Feedback" [2024-12] [[paper](https://arxiv.org/abs/2412.17754)]

- "CODESYNC: Synchronizing Large Language Models with Dynamic Code Evolution at Scale" [2025-02] [[paper](https://arxiv.org/abs/2502.16645)]

- "RustEvo^2: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation" [2025-03] [[paper](https://arxiv.org/abs/2503.16922)]

- "Identifying and Mitigating API Misuse in Large Language Models" [2025-03] [[paper](https://arxiv.org/abs/2503.22821)]

### Privacy

- "Does Your Neural Code Completion Model Use My Code? A Membership Inference Approach" [2024-04] [[paper](https://arxiv.org/abs/2404.14296)]

- "CodeCipher: Learning to Obfuscate Source Code Against LLMs" [2024-10] [[paper](https://arxiv.org/abs/2410.05797)]

- "Decoding Secret Memorization in Code LLMs Through Token-Level Characterization" [2024-10] [[paper](https://arxiv.org/abs/2410.08858)]

- "When Fine-Tuning LLMs Meets Data Privacy: An Empirical Study of Federated Learning in LLM-Based Program Repair" [2024-12] [[paper](https://arxiv.org/abs/2412.01072)]

- "CallNavi: A Study and Challenge on Function Calling Routing and Invocation in Large Language Models" [2025-01] [[paper](https://arxiv.org/abs/2501.05255)]

- "Mitigating Sensitive Information Leakage in LLMs4Code through Machine Unlearning" [2025-02] [[paper](https://arxiv.org/abs/2502.05739)]

- "Towards Mitigating API Hallucination in Code Generated by LLMs with Hierarchical Dependency Aware" [2025-05] [[paper](https://arxiv.org/abs/2505.05057)]

### Bias

- "Exploring Multi-Lingual Bias of Large Code Models in Code Generation" [2024-04] [[paper](https://arxiv.org/abs/2404.19368)]

- "Mitigating Gender Bias in Code Large Language Models via Model Editing" [2024-10] [[paper](https://arxiv.org/abs/2410.07820)]

- "Bias Unveiled: Investigating Social Bias in LLM-Generated Code" [2024-11] [[paper](https://arxiv.org/abs/2411.10351)]

- "FairCode: Evaluating Social Bias of LLMs in Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.05396)]

- "Unveiling Provider Bias in Large Language Models for Code Generation" [2025-01] [[paper](https://arxiv.org/abs/2501.07849)]

- "Addressing Popularity Bias in Third-Party Library Recommendations Using LLMs" [2025-01] [[paper](https://arxiv.org/abs/2501.10313)]

- "LLMs Love Python: A Study of LLMs' Bias for Programming Languages and Libraries" [2025-03] [[paper](https://arxiv.org/abs/2503.17181)]

### AI-Generated Code Detection

- "Who Wrote this Code? Watermarking for Code Generation" [2023-05] [ACL 2024] [[paper](https://arxiv.org/abs/2305.15060)]

- "Zero-Shot Detection of Machine-Generated Codes" [2023-10] [[paper](https://arxiv.org/abs/2310.05103)]

- "Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers" [2024-01] [ICSE 2025] [[paper](https://arxiv.org/abs/2401.06461)]

- "CodeIP: A Grammar-Guided Multi-Bit Watermark for Large Language Models of Code" [2024-04] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2404.15639)]

- "ChatGPT Code Detection: Techniques for Uncovering the Source of Code" [2024-05] [[paper](https://arxiv.org/abs/2405.15512)]

- "Uncovering LLM-Generated Code: A Zero-Shot Synthetic Code Detector via Code Rewriting" [2024-05] [[paper](https://arxiv.org/abs/2405.16133)]

- "Automatic Detection of LLM-generated Code: A Case Study of Claude 3 Haiku" [2024-09] [[paper](https://arxiv.org/abs/2409.01382)]

- "An Empirical Study on Automatically Detecting AI-Generated Source Code: How Far Are We?" [2024-11] [[paper](https://arxiv.org/abs/2411.04299)]

- "Distinguishing LLM-generated from Human-written Code by Contrastive Learning" [2024-11] [[paper](https://arxiv.org/abs/2411.04704)]

- "Is This You, LLM? Recognizing AI-written Programs with Multilingual Code Stylometry" [2024-12] [[paper](https://arxiv.org/abs/2412.14611)]

- "Investigating Efficacy of Perplexity in Detecting LLM-Generated Code" [2024-12] [[paper](https://arxiv.org/abs/2412.16525)]

- "AIGCodeSet: A New Annotated Dataset for AI Generated Code Detection" [2024-12] [[paper](https://arxiv.org/abs/2412.16594)]

- "CodeVision: Detecting LLM-Generated Code Using 2D Token Probability Maps and Vision Models" [2025-01] [[paper](https://arxiv.org/abs/2501.03288)]

- "Robust and Secure Code Watermarking for Large Language Models via ML/Crypto Codesign" [2025-02] [[paper](https://arxiv.org/abs/2502.02068)]

- "Detection of LLM-Generated Java Code Using Discretized Nested Bigrams" [2025-02] [[paper](https://arxiv.org/abs/2502.15740)]

- "Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features" [2025-02] [[paper](https://arxiv.org/abs/2502.17749)]

- "Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code" [2025-02] [[paper](https://arxiv.org/abs/2502.18851)]

- "CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings" [2025-03] [[paper](https://arxiv.org/abs/2503.13733)]

### Others

- "Code Membership Inference for Detecting Unauthorized Data Use in Code Pre-trained Language Models" [2023-12] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2312.07200)]

- "Testing the Effect of Code Documentation on Large Language Model Code Understanding" [2024-04] [[paper](https://arxiv.org/abs/2404.03114)]

- "Where Are Large Language Models for Code Generation on GitHub?" [2024-06] [[paper](https://arxiv.org/abs/2406.19544)]

- "Beyond Functional Correctness: Investigating Coding Style Inconsistencies in Large Language Models" [2024-06] [[paper](https://arxiv.org/abs/2407.00456)]

- "Benchmarking Language Model Creativity: A Case Study on Code Generation" [2024-07] [[paper](https://arxiv.org/abs/2407.09007)]

- "Beyond Correctness: Benchmarking Multi-dimensional Code Generation for Large Language Models" [2024-07] [[paper](https://arxiv.org/abs/2407.11470)]

- "Is Functional Correctness Enough to Evaluate Code Language Models? Exploring Diversity of Generated Codes" [2024-08] [[paper](https://arxiv.org/abs/2408.14504)]

- "Strategic Optimization and Challenges of Large Language Models in Object-Oriented Programming" [2024-08] [[paper](https://arxiv.org/abs/2408.14834)]

- "A Survey on Evaluating Large Language Models in Code Generation Tasks" [2024-08] [[paper](https://arxiv.org/abs/2408.16498)]

- "An exploratory analysis of Community-based Question-Answering Platforms and GPT-3-driven Generative AI: Is it the end of online community-based learning?" [2024-09] [[paper](https://arxiv.org/abs/2409.17473)]

- "Code Generation and Algorithmic Problem Solving Using Llama 3.1 405B" [2024-09] [[paper](https://arxiv.org/abs/2409.19027)]

- "Benchmarking ChatGPT, Codeium, and GitHub Copilot: A Comparative Study of AI-Driven Programming and Debugging Assistants" [2024-09] [[paper](https://arxiv.org/abs/2409.19922)]

- "Model Editing for LLMs4Code: How Far are We?" [2024-11] [[paper](https://arxiv.org/abs/2411.06638)]

- "An Empirical Study on LLM-based Agents for Automated Bug Fixing" [2024-11] [[paper](https://arxiv.org/abs/2411.10213)]

- "Precision or Peril: Evaluating Code Quality from Quantized Large Language Models" [2024-11] [[paper](https://arxiv.org/abs/2411.10656)]

- "Measuring Emergent Capabilities of LLMs for Software Engineering: How Far Are We?" [2024-11] [[paper](https://arxiv.org/abs/2411.17927)]

- "Addressing Data Leakage in HumanEval Using Combinatorial Test Design" [2024-12] [[paper](https://arxiv.org/abs/2412.01526)]

- "Less is More: Towards Green Code Large Language Models via Unified Structural Pruning" [2024-12] [[paper](https://arxiv.org/abs/2412.15921)]

- "How Propense Are Large Language Models at Producing Code Smells? A Benchmarking Study" [2024-12] [[paper](https://arxiv.org/abs/2412.18989)]

- "Deriving Coding-Specific Sub-Models from LLMs using Resource-Efficient Pruning" [2025-01] [[paper](https://arxiv.org/abs/2501.05248)]

- "Do LLMs Provide Links to Code Similar to what they Generate? A Study with Gemini and Bing CoPilot" [2025-01] [[paper](https://arxiv.org/abs/2501.12134)]

- "Comparing Human and LLM Generated Code: The Jury is Still Out!" [2025-01] [[paper](https://arxiv.org/abs/2501.16857)]

- "On the Possibility of Breaking Copyleft Licenses When Reusing Code Generated by ChatGPT" [2025-02] [[paper](https://arxiv.org/abs/2502.05023)]

- "CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation" [2025-02] [[paper](https://arxiv.org/abs/2502.19166)]

- "How Diversely Can Language Models Solve Problems? Exploring the Algorithmic Diversity of Model-Generated Code" [2025-03] [[paper](https://arxiv.org/abs/2503.00691)]

- "Memorize or Generalize? Evaluating LLM Code Generation with Evolved Questions" [2025-03] [[paper](https://arxiv.org/abs/2503.02296)]

- "Human or LLM? A Comparative Study on Accessible Code Generation Capability" [2025-03] [[paper](https://arxiv.org/abs/2503.15885)]

- "Automated Harmfulness Testing for Code Large Language Models" [2025-03] [[paper](https://arxiv.org/abs/2503.16740)]

- "Code Red! On the Harmfulness of Applying Off-the-shelf Large Language Models to Programming Tasks" [2025-04] [[paper](https://arxiv.org/abs/2504.01850)]

- "Rethinking Repetition Problems of LLMs in Code Generation" [2025-05] [[paper](https://arxiv.org/abs/2505.10402)]

- "Style2Code: A Style-Controllable Code Generation Framework with Dual-Modal Contrastive Representation Learning" [2025-05] [[paper](https://arxiv.org/abs/2505.19442)]

## 7. Human-LLM Interaction

- "Expectation vs. Experience: Evaluating the Usability of Code Generation Tools Powered by Large Language Models" [2022-04] [CHI EA 2022] [[paper](https://dl.acm.org/doi/abs/10.1145/3491101.3519665)]

- "Grounded Copilot: How Programmers Interact with Code-Generating Models" [2022-06] [OOPSLA 2023] [[paper](https://arxiv.org/abs/2206.15000)]

- "Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming" [2022-10] [[paper](https://arxiv.org/abs/2210.14306)]

- "The Impact of AI on Developer Productivity: Evidence from GitHub Copilot" [2023-02] [[paper](https://arxiv.org/abs/2302.06590)]

- "The Programmer's Assistant: Conversational Interaction with a Large Language Model for Software Development" [2023-02] [IUI 2023] [[paper](https://arxiv.org/abs/2302.07080)]

- ""It's Weird That it Knows What I Want": Usability and Interactions with Copilot for Novice Programmers" [2023-04] [ACM TCHI] [[paper](https://arxiv.org/abs/2304.02491)]

- "DevGPT: Studying Developer-ChatGPT Conversations" [2023-08] [[paper](https://arxiv.org/abs/2309.03914)]

- "How Do Analysts Understand and Verify AI-Assisted Data Analyses?" [2023-09] [[paper](https://arxiv.org/abs/2309.10947)]

- "How Novices Use LLM-Based Code Generators to Solve CS1 Coding Tasks in a Self-Paced Learning Environment" [2023-09] [Koli Calling 2023] [[paper](https://arxiv.org/abs/2309.14049)]

- "Conversational Challenges in AI-Powered Data Science: Obstacles, Needs, and Design Opportunities" [2023-10] [[paper](https://arxiv.org/abs/2310.16164)]

- "The RealHumanEval: Evaluating Large Language Models' Abilities to Support Programmers" [2024-04] [[paper](https://arxiv.org/abs/2404.02806)]

- "Unlocking Adaptive User Experience with Generative AI" [2024-04] [[paper](https://arxiv.org/abs/2404.05442)]

- "BISCUIT: Scaffolding LLM-Generated Code with Ephemeral UIs in Computational Notebooks" [2024-04] [[paper](https://arxiv.org/abs/2404.07387)]

- "How far are AI-powered programming assistants from meeting developers' needs?" [2024-04] [[paper](https://arxiv.org/abs/2404.12000)]

- "Beyond Code Generation: An Observational Study of ChatGPT Usage in Software Engineering Practice" [2024-04] [[paper](https://arxiv.org/abs/2404.14901)]

- "The GPT Surprise: Offering Large Language Model Chat in a Massive Coding Class Reduced Engagement but Increased Adopters Exam Performances" [2024-04] [[paper](https://arxiv.org/abs/2407.09975)]

- "amplified.dev: a living document that begins to sketch a vision for a future where developers are amplified, not automated" [2024-05] [[paper](https://amplified.dev)]

- "Sketch Then Generate: Providing Incremental User Feedback and Guiding LLM Code Generation through Language-Oriented Code Sketches" [2024-05] [[paper](https://arxiv.org/abs/2405.03998)]

- "Using AI Assistants in Software Development: A Qualitative Study on Security Practices and Concerns" [2024-05] [[paper](https://arxiv.org/abs/2405.06371)]

- "Full Line Code Completion: Bringing AI to Desktop" [2024-05] [[paper](https://arxiv.org/abs/2405.08704)]

- "Developers' Perceptions on the Impact of ChatGPT in Software Development: A Survey" [2024-05] [[paper](https://arxiv.org/abs/2405.12195)]

- "A Transformer-Based Approach for Smart Invocation of Automatic Code Completion" [2024-05] [[paper](https://arxiv.org/abs/2405.14753)]

- "A Study on Developer Behaviors for Validating and Repairing LLM-Generated Code Using Eye Tracking and IDE Actions" [2024-05] [[paper](https://arxiv.org/abs/2405.16081)]

- "Analyzing Chat Protocols of Novice Programmers Solving Introductory Programming Tasks with ChatGPT" [2024-05] [[paper](https://arxiv.org/abs/2405.19132)]

- "Benchmarking the Communication Competence of Code Generation for LLMs and LLM Agent" [2024-05] [[paper](https://arxiv.org/abs/2406.00215)]

- "Learning Task Decomposition to Assist Humans in Competitive Programming" [2024-06] [ACL 2024] [[paper](https://arxiv.org/abs/2406.04604)]

- "Hints-In-Browser: Benchmarking Language Models for Programming Feedback Generation" [2024-07] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.05053)]

- "Impact of AI-tooling on the Engineering Workspace" [2024-06] [[paper](https://arxiv.org/abs/2406.07683)]

- "Using AI-Based Coding Assistants in Practice: State of Affairs, Perceptions, and Ways Forward" [2024-06] [[paper](https://arxiv.org/abs/2406.07765)]

- "Instruct, Not Assist: LLM-based Multi-Turn Planning and Hierarchical Questioning for Socratic Code Debugging" [2024-06] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2406.11709)]

- "Transforming Software Development: Evaluating the Efficiency and Challenges of GitHub Copilot in Real-World Projects" [2024-06] [[paper](https://arxiv.org/abs/2406.17910)]

- "Let the Code LLM Edit Itself When You Edit the Code" [2024-07] [ICLR 2025] [[paper](https://arxiv.org/abs/2407.03157)]

- "Enhancing Computer Programming Education with LLMs: A Study on Effective Prompt Engineering for Python Code Generation" [2024-07] [[paper](https://arxiv.org/abs/2407.05437)]

- "How Novice Programmers Use and Experience ChatGPT when Solving Programming Exercises in an Introductory Course" [2024-07] [[paper](https://arxiv.org/abs/2407.20792)]

- "Can Developers Prompt? A Controlled Experiment for Code Documentation Generation" [2024-08] [[paper](https://arxiv.org/abs/2408.00686)]

- "The Impact of Generative AI-Powered Code Generation Tools on Software Engineer Hiring: Recruiters' Experiences, Perceptions, and Strategies" [2024-09] [[paper](https://arxiv.org/abs/2409.00875)]

- "Investigating the Role of Cultural Values in Adopting Large Language Models for Software Engineering" [2024-09] [[paper](https://arxiv.org/abs/2409.05055)]

- "The Impact of Large Language Models on Open-source Innovation: Evidence from GitHub Copilot" [2024-09] [[paper](https://arxiv.org/abs/2409.08379)]

- ""I Don't Use AI for Everything": Exploring Utility, Attitude, and Responsibility of AI-empowered Tools in Software Development" [2024-09] [[paper](https://arxiv.org/abs/2409.13343)]

- "Harnessing the Potential of Gen-AI Coding Assistants in Public Sector Software Development" [2024-09] [[paper](https://arxiv.org/abs/2409.17434)]

- "Understanding the Human-LLM Dynamic: A Literature Survey of LLM Use in Programming Tasks" [2024-10] [[paper](https://arxiv.org/abs/2410.01026)]

- "Code-Survey: An LLM-Driven Methodology for Analyzing Large-Scale Codebases" [2024-10] [[paper](https://arxiv.org/abs/2410.01837)]

- "The potential of LLM-generated reports in DevSecOps" [2024-10] [[paper](https://arxiv.org/abs/2410.01899)]

- "The Impact of Generative AI on Collaborative Open-Source Software Development: Evidence from GitHub Copilot" [2024-10] [[paper](https://arxiv.org/abs/2410.02091)]

- "Exploring the Design Space of Cognitive Engagement Techniques with AI-Generated Code for Enhanced Learning" [2024-10] [[paper](https://arxiv.org/abs/2410.08922)]

- "One Step at a Time: Combining LLMs and Static Analysis to Generate Next-Step Hints for Programming Tasks" [2024-10] [[paper](https://arxiv.org/abs/2410.09268)]

- "How much does AI impact development speed? An enterprise-based randomized controlled trial" [2024-10] [[paper](https://arxiv.org/abs/2410.12944)]

- "Understanding the Effect of Algorithm Transparency of Model Explanations in Text-to-SQL Semantic Parsing" [2024-10] [[paper](https://arxiv.org/abs/2410.16283)]

- "Dear Diary: A randomized controlled trial of Generative AI coding tools in the workplace" [2024-10] [[paper](https://arxiv.org/abs/2410.18334)]

- "LLMs are Imperfect, Then What? An Empirical Study on LLM Failures in Software Engineering" [2024-11] [[paper](https://arxiv.org/abs/2411.09916)]

- "Human-In-the-Loop Software Development Agents" [2024-11] [[paper](https://arxiv.org/abs/2411.12924)]

- "Amplifying human performance in combinatorial competitive programming" [2024-11] [[paper](https://arxiv.org/abs/2411.19744)]

- "Why Do Developers Engage with ChatGPT in Issue-Tracker? Investigating Usage and Reliance on ChatGPT-Generated Code" [2024-12] [[paper](https://arxiv.org/abs/2412.06757)]

- "Examining the Use and Impact of an AI Code Assistant on Developer Productivity and Experience in the Enterprise" [2024-12] [[paper](https://arxiv.org/abs/2412.06603)]

- "Human and Machine: How Software Engineers Perceive and Engage with AI-Assisted Code Reviews Compared to Their Peers" [2025-01] [[paper](https://arxiv.org/abs/2501.02092)]

- "Towards Decoding Developer Cognition in the Age of AI Assistants" [2025-01] [[paper](https://arxiv.org/abs/2501.02684)]

- "Simulated Interactive Debugging" [2025-01] [[paper](https://arxiv.org/abs/2501.09694)]

- "Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution" [2025-01] [MSR 2025] [[paper](https://arxiv.org/abs/2501.11709)]

- "Experience with GitHub Copilot for Developer Productivity at Zoominfo" [2025-01] [[paper](https://arxiv.org/abs/2501.13282)]

- "Analysis of Student-LLM Interaction in a Software Engineering Project" [2025-02] [[paper](https://arxiv.org/abs/2502.01273)]

- "Understanding User Mental Models in AI-Driven Code Completion Tools: Insights from an Elicitation Study" [2025-02] [[paper](https://arxiv.org/abs/2502.02194)]

- "CodeA11y: Making AI Coding Assistants Useful for Accessible Web Development" [2025-02] [[paper](https://arxiv.org/abs/2502.10884)]

- "Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors" [2025-02] [[paper](https://arxiv.org/abs/2502.13311)]

- "From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions" [2025-02] [[paper](https://arxiv.org/abs/2502.13791)]

- "How Scientists Use Large Language Models to Program" [2025-02] [[paper](https://arxiv.org/abs/2502.17348)]

- "Do Comments and Expertise Still Matter? An Experiment on Programmers' Adoption of AI-Generated JavaScript Code" [2025-03] [[paper](https://arxiv.org/abs/2503.11453)]

- "Prompting LLMs for Code Editing: Struggles and Remedies" [2025-04] [[paper](https://arxiv.org/abs/2504.20196)]

## 8. Datasets

### 8.1 Pretraining

1. **CodeSearchNet**: "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [2019-09] [[paper](https://arxiv.org/abs/1909.09436)] [[repo](https://github.com/github/CodeSearchNet)] [[data](https://huggingface.co/datasets/code_search_net)]

2. **The Pile**: "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" [2020-12], [[paper](https://arxiv.org/abs/2101.00027)] [[data](https://pile.eleuther.ai/)]

3. **CodeParrot**, 2022-02, [[data](https://huggingface.co/datasets/codeparrot/github-code)]

4. **The Stack**: "The Stack: 3 TB of permissively licensed source code" [2022-11] [[paper](https://arxiv.org/abs/2211.15533)] [[data](https://huggingface.co/datasets/bigcode/the-stack)]

5. **ROOTS**: "The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset" [2023-03] [NeurIPS 2022 Datasets and Benchmarks Track] [[paper](https://arxiv.org/abs/2303.03915)] [[data](https://huggingface.co/datasets?search=bigscience-data/roots)]

6. **The Stack v2**: "StarCoder 2 and The Stack v2: The Next Generation" [2024-02] [[paper](https://arxiv.org/abs/2402.19173)] [[data](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)]

7. **The Heap**: "The Heap: A Contamination-Free Multilingual Code Dataset for Evaluating Large Language Models" [2025-01] [[paper](https://arxiv.org/abs/2501.09653)]

8. "Large Language Models are Qualified Benchmark Builders: Rebuilding Pre-Training Datasets for Advancing Code Intelligence Tasks" [2025-04] [[paper](https://arxiv.org/abs/2504.19444)]

### 8.2 Benchmarks

#### Integrated Benchmarks

- **CodeXGLUE**: "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation" [2021-02] [NeurIPS Datasets and Benchmarks 2021] [[paper](https://arxiv.org/abs/2102.04664)] [[repo](https://github.com/microsoft/CodeXGLUE)] [[data](https://huggingface.co/datasets?search=code_x_glue)]

- **CodefuseEval**: "CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model" [2023-10] [[paper](https://arxiv.org/abs/2310.06266)] [[repo](https://github.com/codefuse-ai/codefuse-evaluation)]

- **CodeScope**: "CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark for Evaluating LLMs on Code Understanding and Generation" [2023-11] [ACL 2024] [[paper](https://arxiv.org/abs/2311.08588)] [[repo](https://github.com/WeixiangYAN/CodeScope)]

- **LiveCodeBench**: "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code" [2024-03] [ICLR 2025] [[paper](https://arxiv.org/abs/2403.07974)] [[repo](https://github.com/LiveCodeBench/LiveCodeBench)]

- **CodeEditorBench**: "CodeEditorBench: Evaluating Code Editing Capability of Large Language Models" [2024-04] [[paper](https://arxiv.org/abs/2404.03543)] [[repo](https://github.com/CodeEditorBench/CodeEditorBench)]

- **Long Code Arena**: "Long Code Arena: a Set of Benchmarks for Long-Context Code Models" [2024-06] [[paper](https://arxiv.org/abs/2406.11612)] [[repo](https://github.com/JetBrains-Research/lca-baselines)]

- **CodeRAG-Bench**: "CodeRAG-Bench: Can Retrieval Augment Code Generation?" [2024-06] [[paper](https://arxiv.org/abs/2406.14497)] [[repo](https://github.com/code-rag-bench/code-rag-bench)]

- **LiveBench**: "LiveBench: A Challenging, Contamination-Free LLM Benchmark" [2024-06] [[paper](https://arxiv.org/abs/2406.19314)] [[repo](https://github.com/livebench/livebench)]

- **DebugEval**: "Enhancing the Code Debugging Ability of LLMs via Communicative Agent Based Data Refinement" [2024-08] [[paper](https://arxiv.org/abs/2408.05006)] [[repo](https://github.com/NEUIR/DebugEval)]

- **FullStack Bench**: "FullStack Bench: Evaluating LLMs as Full Stack Coder" [2024-11] [[paper](https://arxiv.org/abs/2412.00535)] [[data](https://github.com/bytedance/FullStackBench)]

- **StackEval**: "StackEval: Benchmarking LLMs in Coding Assistance" [2024-12] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2412.05288)] [[data](https://github.com/ProsusAI/stack-eval)]

- "How Should I Build A Benchmark?" [2025-01] [[paper](https://arxiv.org/abs/2501.10711)]

- **HLE**: "Humanity's Last Exam" [2025-01] [[paper](https://arxiv.org/abs/2501.14249)] [[data](https://github.com/centerforaisafety/hle)]

- **SuperGPQA**: "SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines" [2025-02] [[paper](https://arxiv.org/abs/2502.14739)]

- **BBEH**: "BIG-Bench Extra Hard" [2025-02] [[paper](https://arxiv.org/abs/2502.19187)] [[data](https://github.com/google-deepmind/bbeh)]

- "**CoCo-Bench**: A Comprehensive Code Benchmark For Multi-task Large Language Model Evaluation" [2025-04] [[paper](https://arxiv.org/abs/2504.20673)]

#### Evaluation Metrics

- "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis" [2020-09] [[paper](https://arxiv.org/abs/2009.10297)]

- "CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code" [2023-02] [EMNLP 2023] [[paper](https://arxiv.org/abs/2302.05527)]

- "Unsupervised Evaluation of Code LLMs with Round-Trip Correctness" [2024-02] [ICML 2024] [[paper](https://arxiv.org/abs/2402.08699)]

- "CodeJudge: Evaluating Code Generation with Large Language Models" [2024-10] [EMNLP 2024] [[paper](https://arxiv.org/abs/2410.02184)]

- "Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering" [2025-02] [[paper](https://arxiv.org/abs/2502.06193)]

- "Bridging LLM-Generated Code and Requirements: Reverse Generation technique and SBC Metric for Developer Insights" [2025-02] [[paper](https://arxiv.org/abs/2502.07835)]

- "Copilot Arena: A Platform for Code LLM Evaluation in the Wild" [2025-02] [[paper](https://arxiv.org/abs/2502.09328)]

- "MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation" [2025-02] [[paper](https://arxiv.org/abs/2502.12468)]

- "CodeCriticBench: A Holistic Code Critique Benchmark for Large Language Models" [2025-02] [[paper](https://arxiv.org/abs/2502.16614)]

- "Disproving Program Equivalence with LLMs" [2025-02] [[paper](https://arxiv.org/abs/2502.18473)]

- "Can Language Models Falsify? Evaluating Algorithmic Reasoning with Counterexample Creation" [2025-02] [[paper](https://arxiv.org/abs/2502.19414)]

- "CodeVisionary: An Agent-based Framework for Evaluating Large Language Models in Code Generation" [2025-04] [[paper](https://arxiv.org/abs/2504.13472)]

- "EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective" [2025-05] [[paper](https://arxiv.org/abs/2505.12185)]

- "Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation" [2025-05] [[paper](https://arxiv.org/abs/2505.16222)]

- "CODE-DITING: A Reasoning-Based Metric for Functional Alignment in Code Evaluation" [2025-05] [[paper](https://arxiv.org/abs/2505.19502)]

#### Program Synthesis

| Date    | Venue                            | Benchmark                                        | Size                 | Language                                                                         | Source                                                                                                                                                                                                                                                                                       |
| ------- | -------------------------------- | ------------------------------------------------ | -------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2018-02 | LREC 2018                        | NL2Bash                                          | 9305                 | Bash                                                                             | "NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System" [[paper](https://arxiv.org/abs/1802.08979)] [[data](https://github.com/TellinaTool/nl2bash)]                                                                                            |
| 2018-08 | EMNLP 2018                       | CONCODE                                          | 104K                 | Java                                                                             | "Mapping Language to Code in Programmatic Context" [[paper](https://arxiv.org/abs/1808.09588)] [[data](https://github.com/sriniiyer/concode)]                                                                                                                                                |
| 2019-10 | EMNLP-IJCNLP 2019                | JuICe                                            | 1.5M/3725 \*         | Python                                                                           | "JuICe: A Large Scale Distantly Supervised Dataset for Open Domain Context-based Code Generation" [[paper](https://arxiv.org/abs/1910.02216)] [[data](https://github.com/rajasagashe/juice)]                                                                                                 |
| 2021-05 | NeurIPS 2021                     | APPS                                             | 10000                | Python                                                                           | "Measuring Coding Challenge Competence With APPS" [[paper](https://arxiv.org/abs/2105.09938)] [[data](https://github.com/hendrycks/apps)]                                                                                                                                                    |
| 2021-07 | arXiv                            | HumanEval                                        | 164                  | Python                                                                           | "Evaluating Large Language Models Trained on Code" [[paper](https://arxiv.org/abs/2107.03374)] [[data](https://github.com/openai/human-eval)]                                                                                                                                                |
| 2021-08 | arXiv                            | MBPP/MathQA-Python                               | 974/23914            | Python                                                                           | "Program Synthesis with Large Language Models" [[paper](https://arxiv.org/abs/2108.07732)] [[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)] [[MathQA-Python](https://github.com/google/trax/blob/master/trax/examples/MathQA_Python_generation_notebook.ipynb)] |
| 2021-08 | ACL/IJCNLP 2021                  | PlotCoder                                        | 40797                | Python                                                                           | "PlotCoder: Hierarchical Decoding for Synthesizing Visualization Code in Programmatic Context" [[paper](https://aclanthology.org/2021.acl-long.169/)] [[data](https://github.com/jungyhuk/plotcoder)]                                                                                        |
| 2022-01 | arXiv                            | DSP                                              | 1119                 | Python                                                                           | "Training and Evaluating a Jupyter Notebook Data Science Assistant" [[paper](https://arxiv.org/abs/2201.12901)] [[data](https://github.com/microsoft/DataScienceProblems)]                                                                                                                   |
| 2022-02 | Science                          | CodeContests                                     | 13610                | C++, Python, Java                                                                | "Competition-Level Code Generation with AlphaCode" [[paper](https://arxiv.org/abs/2203.07814)] [[data](https://github.com/google-deepmind/code_contests)]                                                                                                                                    |
| 2022-03 | EACL 2023 Findings               | MCoNaLa                                          | 896                  | Python                                                                           | "MCoNaLa: A Benchmark for Code Generation from Multiple Natural Languages" [[paper](https://arxiv.org/abs/2203.08388)] [[data](https://github.com/zorazrw/multilingual-conala)]                                                                                                              |
| 2022-06 | arXiv                            | AixBench                                         | 336                  | Java                                                                             | "AixBench: A Code Generation Benchmark Dataset" [[paper](https://arxiv.org/abs/2206.13179)] [[data](https://github.com/aixcoder-plugin/nl2code-dataset)]                                                                                                                                     |
| 2022-08 | IEEE Trans. Software Engineering | MultiPL-E                                        |                      |                                                                                  | "MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation", [[paper](https://arxiv.org/abs/2208.08227)] [[data](https://github.com/nuprl/MultiPL-E)]                                                                                                             |
| 2022-10 | ICLR 2023                        | MBXP                                             | 12.4K                | Python, Java, JS, TypeScript, Go, C#, PHP, Ruby, Kotlin, C++, Perl, Scala, Swift | "Multi-lingual Evaluation of Code Generation Models" [[paper](https://arxiv.org/abs/2210.14868)] [[data](https://github.com/amazon-science/mxeval)]                                                                                                                                          |
| 2022-10 | ICLR 2023                        | Multilingual HumanEval                           | 1.9K                 | Python, Java, JS, TypeScript, Go, C#, PHP, Ruby, Kotlin, Perl, Scala, Swift      | "Multi-lingual Evaluation of Code Generation Models" [[paper](https://arxiv.org/abs/2210.14868)] [[data](https://github.com/amazon-science/mxeval)]                                                                                                                                          |
| 2022-10 | ICLR 2023                        | MathQA-X                                         | 5.6K                 | Python, Java, JS                                                                 | "Multi-lingual Evaluation of Code Generation Models" [[paper](https://arxiv.org/abs/2210.14868)] [[data](https://github.com/amazon-science/mxeval)]                                                                                                                                          |
| 2022-11 | arXiv                            | ExeDS                                            | 534                  | Python                                                                           | "Execution-based Evaluation for Data Science Code Generation Models" [[paper](https://arxiv.org/abs/2211.09374)] [[data](https://github.com/Jun-jie-Huang/ExeDS)]                                                                                                                            |
| 2022-11 | arXiv                            | DS-1000                                          | 1000                 | Python                                                                           | "DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation" [[paper](https://arxiv.org/abs/2211.11501)] [[data](https://github.com/xlang-ai/DS-1000)]                                                                                                                       |
| 2022-12 | arXiv                            | ODEX                                             | 945                  | Python                                                                           | "Execution-Based Evaluation for Open-Domain Code Generation" [[paper](https://arxiv.org/abs/2212.10481)] [[data](https://github.com/zorazrw/odex)]                                                                                                                                           |
| 2023-02 | arXiv                            | CoderEval                                        | 460                  | Python, Java                                                                     | "CoderEval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models" [[paper](https://arxiv.org/abs/2302.00288)] [[data](https://github.com/CoderEval/CoderEval)]                                                                                                        |
| 2023-03 | ACL 2024                         | xCodeEval                                        | 5.5M                 | C, C#, C++, Go, Java, JS, Kotlin, PHP, Python, Ruby, Rust                        | "XCodeEval: An Execution-based Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval" [[paper](https://arxiv.org/abs/2303.03004)] [[data](https://github.com/ntunlp/xCodeEval)]                                                         |
| 2023-03 | arXiv                            | HumanEval-X                                      | 820                  | Python, C++, Java, JS, Go                                                        | "CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X" [[paper](https://arxiv.org/abs/2303.17568)] [[data](https://hub.docker.com/r/codegeex/codegeex)]                                                                                            |
| 2023-05 | arXiv                            | HumanEval+                                       | 164                  | Python                                                                           | "Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation" [[paper](https://arxiv.org/abs/2305.01210)] [[data](https://github.com/evalplus/evalplus)]                                                                              |
| 2023-06 | ACL 2024 Findings                | StudentEval                                      | 1749 $^\dagger$      | Python                                                                           | "StudentEval: A Benchmark of Student-Written Prompts for Large Language Models of Code" [[paper](https://arxiv.org/abs/2306.04556)] [[data](https://huggingface.co/datasets/wellesley-easel/StudentEval)]                                                                                    |
| 2023-08 | ICLR 2024 Spotlight              | HumanEvalPack                                    | 984                  | Python, JS, Go, Java, C++, Rust                                                  | "OctoPack: Instruction Tuning Code Large Language Models" [[paper](https://arxiv.org/abs/2308.07124)] [[data](https://huggingface.co/datasets/bigcode/humanevalpack)]                                                                                                                        |
| 2023-06 | NeurIPS 2023                     | DotPrompts                                       | 10538 $^\ddagger$    | Java                                                                             | "Guiding Language Models of Code with Global Context using Monitors" [[paper](https://arxiv.org/abs/2306.10763)] [[data](https://github.com/microsoft/monitors4codegen)]                                                                                                                     |
| 2023-09 | arXiv                            | CodeApex                                         | 476                  | C++                                                                              | "CodeApex: A Bilingual Programming Evaluation Benchmark for Large Language Models" [[paper](https://arxiv.org/abs/2309.01940)] [[data](https://github.com/APEXLAB/CodeApex)]                                                                                                                 |
| 2023-09 | arXiv                            | VerilogEval                                      | 8645/156 $^\diamond$ | Verilog                                                                          | "VerilogEval: Evaluating Large Language Models for Verilog Code Generation" [[paper](https://arxiv.org/abs/2309.07544)] [[data](https://github.com/NVlabs/verilog-eval)]                                                                                                                     |
| 2023-11 | arXiv                            | ML-Bench                                         | 10040                | Bash                                                                             | "ML-Bench: Large Language Models Leverage Open-source Libraries for Machine Learning Tasks" [[paper](https://arxiv.org/abs/2311.09835)] [[data](https://ml-bench.github.io/)]                                                                                                                |
| 2023-12 | arXiv                            | TACO                                             | 26,433               | Python                                                                           | "TACO: Topics in Algorithmic COde generation dataset" [[paper](https://arxiv.org/abs/2312.14852)] [[data](https://github.com/FlagOpen/TACO)]                                                                                                                                                 |
| 2024-01 | EMNLP 2024 Findings              | PythonSaga                                       | 185                  | Python                                                                           | "PythonSaga: Redefining the Benchmark to Evaluate Code Generating LLMs" [[paper](https://arxiv.org/abs/2401.03855)] [[data](https://anonymous.4open.science/r/PythonSaga/README.md)]                                                                                                         |
| 2024-01 | HPDC                             | ParEval                                          | 420                  | C++, CUDA, HIP                                                                   | "Can Large Language Models Write Parallel Code?" [[paper](https://arxiv.org/abs/2401.12554)] [[data](https://github.com/parallelcodefoundry/ParEval)]                                                                                                                                        |
| 2024-02 | ACL 2024 Findings                | OOP                                              | 431                  | Python                                                                           | "OOP: Object-Oriented Programming Evaluation Benchmark for Large Language Models" [[paper](https://arxiv.org/abs/2401.06628)] [[data](https://github.com/alphadl/OOP-eval)]                                                                                                                  |
| 2024-02 | LREC-COLING 2024                 | HumanEval-XL                                     | 22080                | 23NL, 12PL                                                                       | "HumanEval-XL: A Multilingual Code Generation Benchmark for Cross-lingual Natural Language Generalization" [[paper](https://arxiv.org/abs/2402.16694)] [[data](https://github.com/FloatAI/humaneval-xl)]                                                                                     |
| 2024-04 | arXiv                            | USACO                                            | 307                  | Python                                                                           | "Can Language Models Solve Olympiad Programming?" [[paper](https://arxiv.org/abs/2404.10952)] [[data](https://github.com/princeton-nlp/USACO)]                                                                                                                                               |
| 2024-04 | LREC-COLING 2024                 | PECC                                             | 2396                 | Python                                                                           | "PECC: Problem Extraction and Coding Challenges" [[paper](https://arxiv.org/abs/2404.18766)] [[data](https://github.com/HallerPatrick/pecc)]                                                                                                                                                 |
| 2024-04 | arXiv                            | CodeGuard+                                       | 23                   | Python, C                                                                        | "Constrained Decoding for Secure Code Generation" [[paper](https://arxiv.org/abs/2405.00218)] [[data](https://github.com/Dynamite321/CodeGuardPlus)]                                                                                                                                         |
| 2024-05 | ACL 2024 Findings                | NaturalCodeBench                                 | 402                  | Python, Java                                                                     | "NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts" [[paper](https://arxiv.org/abs/2405.04520)] [[data](https://github.com/THUDM/NaturalCodeBench)]                                                                                              |
| 2024-05 | arXiv                            | MHPP                                             | 140                  | Python                                                                           | "MHPP: Exploring the Capabilities and Limitations of Language Models Beyond Basic Code Generation" [[paper](https://arxiv.org/abs/2405.11430)] [[repo](https://github.com/SparksofAGI/MHPP)]                                                                                                 |
| 2024-06 | arXiv                            | VHDL-Eval                                        | 202                  | VHDL                                                                             | "VHDL-Eval: A Framework for Evaluating Large Language Models in VHDL Code Generation" [[paper](https://arxiv.org/abs/2406.04379)]                                                                                                                                                            |
| 2024-06 | arXiv                            | AICoderEval                                      | 492                  | Python                                                                           | "AICoderEval: Improving AI Domain Code Generation of Large Language Models" [[paper](https://arxiv.org/abs/2406.04712)] [[data](https://huggingface.co/datasets/vixuowis/AICoderEval)]                                                                                                       |
| 2024-06 | arXiv                            | VersiCode                                        | 98,692               | Python                                                                           | "VersiCode: Towards Version-controllable Code Generation" [[paper](https://arxiv.org/abs/2406.07411)] [[data](https://github.com/wutong8023/VersiCode)]                                                                                                                                      |
| 2024-06 | IEEE AITest 2024                 | ScenEval                                         | 12,864               | Java                                                                             | "ScenEval: A Benchmark for Scenario-Based Evaluation of Code Generation" [[paper](https://arxiv.org/abs/2406.12635)]                                                                                                                                                                         |
| 2024-06 | ICLR 2025                        | BigCodeBench                                     | 1,140                | Python                                                                           | "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions" [[paper](https://arxiv.org/abs/2406.15877)] [[data](https://github.com/bigcode-project/bigcodebench)]                                                                                      |
| 2024-07 | arXiv                            | CodeUpdateArena                                  | 670                  | Python                                                                           | "CodeUpdateArena: Benchmarking Knowledge Editing on API Updates" [[paper](https://arxiv.org/abs/2407.06249)] [[data](https://github.com/leo-liuzy/CodeUpdateArena)]                                                                                                                          |
| 2024-07 | EMNLP 2024 Findings              | LBPP                                             | 161                  | Python                                                                           | "On Leakage of Code Generation Evaluation Datasets" [[paper](https://arxiv.org/abs/2407.07565)] [[data](https://huggingface.co/datasets/CohereForAI/lbpp)]                                                                                                                                   |
| 2024-07 | arXiv                            | NoviCode                                         | 150                  | Python                                                                           | "NoviCode: Generating Programs from Natural Language Utterances by Novices" [[paper](https://arxiv.org/abs/2407.10626)] [[data](https://github.com/BIU-NLP/novicode)]                                                                                                                        |
| 2024-07 | arXiv                            | Case2Code                                        | 1.3M                 | Python                                                                           | "Case2Code: Learning Inductive Reasoning with Synthetic Data" [[paper](https://arxiv.org/abs/2407.12504)] [[data](https://github.com/choosewhatulike/case2code)]                                                                                                                             |
| 2024-07 | NeurIPS 2024                     | SciCode                                          | 338                  | Python                                                                           | "SciCode: A Research Coding Benchmark Curated by Scientists" [[paper](https://arxiv.org/abs/2407.13168)] [[data](https://github.com/scicode-bench/SciCode)]                                                                                                                                  |
| 2024-07 | arXiv                            | auto-regression                                  | 460                  | Python                                                                           | "Generating Unseen Code Tests In Infinitum" [[paper](https://arxiv.org/abs/2407.19772)]                                                                                                                                                                                                      |
| 2024-07 | arXiv                            | WebApp1K                                         | 1000                 | JavaScript                                                                       | "WebApp1K: A Practical Code-Generation Benchmark for Web App Development" [[paper](https://arxiv.org/abs/2408.00019)] [[data](https://huggingface.co/datasets/onekq-ai/WebApp1K-React)]                                                                                                      |
| 2024-08 | ACL 2024 Findings                | CodeInsight                                      | 3409                 | Python                                                                           | "CodeInsight: A Curated Dataset of Practical Coding Solutions from Stack Overflow" [[paper](https://arxiv.org/abs/2409.16819)] [[data](https://github.com/NathanaelBeau/CodeInsight)]                                                                                                        |
| 2024-08 | arXiv                            | DomainEval                                       | 2454                 | Python                                                                           | "DOMAINEVAL: An Auto-Constructed Benchmark for Multi-Domain Code Generation" [[paper](https://arxiv.org/abs/2408.13204)] [[data](https://github.com/domaineval/DomainEval)]                                                                                                                  |
| 2024-09 | arXiv                            | ComplexCodeEval                                  | 7184/3897            | Python/Java                                                                      | "ComplexCodeEval: A Benchmark for Evaluating Large Code Models on More Complex Code" [[paper](https://arxiv.org/abs/2409.10280)] [[data](https://github.com/ComplexCodeEval/ComplexCodeEval)]                                                                                                |
| 2024-09 | ASE 2024                         | CoCoNote                                         | 58221                | Python Notebook                                                                  | "Contextualized Data-Wrangling Code Generation in Computational Notebooks" [[paper](https://arxiv.org/abs/2409.13551)] [[data](https://github.com/Jun-jie-Huang/CoCoNote)]                                                                                                                   |
| 2024-10 | arXiv                            | unnamed                                          | 77                   | Python                                                                           | "Evaluation of Code LLMs on Geospatial Code Generation" [[paper](https://arxiv.org/abs/2410.04617)] [[data](https://github.com/kraina-ai/geospatial-code-llms-dataset)]                                                                                                                      |
| 2024-10 | arXiv                            | mHumanEval                                       | 836,400              | 25PL, 204NL                                                                      | "mHumanEval -- A Multilingual Benchmark to Evaluate Large Language Models for Code Generation" [[paper](https://arxiv.org/abs/2410.15037)] [[data](https://github.com/mraihan-gmu/mHumanEval)]                                                                                               |
| 2024-10 | arXiv                            | FeatEng                                          | 103                  | Python                                                                           | "Can Models Help Us Create Better Models? Evaluating LLMs as Data Scientists" [[paper](https://arxiv.org/abs/2410.23331)] [[data](https://github.com/FeatEng/FeatEng)]                                                                                                                       |
| 2024-11 | arXiv                            | GitChameleon                                     | 116                  | Python                                                                           | "GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models" [[paper](https://arxiv.org/abs/2411.05830)] [[data](https://github.com/NizarIslah/GitChameleon)]                                                                                                      |
| 2024-11 | EMNLP 2024 Findings              | MBUPP                                            | 466                  | Python                                                                           | "One-to-many testing for code generation from (just) natural language" [[paper](https://aclanthology.org/2024.findings-emnlp.902/)] [[data](https://github.com/microsoft/prose-benchmarks/tree/main/MBUPP)]                                                                                  |
| 2024-11 | arXiv                            | LibEvolutionEval                                 | 34.7K                | Python                                                                           | "LibEvolutionEval: A Benchmark and Study for Version-Specific Code Generation" [[paper](https://arxiv.org/abs/2412.04478)]                                                                                                                                                                   |
| 2024-12 | arXiv                            | PandasPlotBench                                  | 175                  | Python                                                                           | "Drawing Pandas: A Benchmark for LLMs in Generating Plotting Code" [[paper](https://arxiv.org/abs/2412.02764)] [[data](https://huggingface.co/datasets/JetBrains-Research/plot_bench)]                                                                                                       |
| 2024-12 | arXiv                            | CodeArena                                        | 397                  | 44                                                                               | "Evaluating and Aligning CodeLLMs on Human Preference" [[paper](https://arxiv.org/abs/2412.05210)] [[data](https://github.com/QwenLM/Qwen2.5-Coder/tree/main/qwencoder-eval/instruct/CodeArena)]                                                                                             |
| 2024-12 | arXiv                            | OBFUSEVAL                                        | 1354                 | C                                                                                | "Unseen Horizons: Unveiling the Real Capability of LLM Code Generation Beyond the Familiar" [[paper](https://arxiv.org/abs/2412.08109)] [[data](https://github.com/zhangbuzhang/ObfusEval)]                                                                                                  |
| 2024-12 | arXiv                            | HumanEval Pro / MBPP Pro / BigCodeBench-Lite Pro | 164/378/57           | Python                                                                           | "HumanEval Pro and MBPP Pro: Evaluating Large Language Models on Self-invoking Code Generation" [[paper](https://arxiv.org/abs/2412.21199)] [[data](https://github.com/CodeEval-Pro/CodeEval-Pro/tree/main)]                                                                                 |
| 2025-01 | arXiv                            | CodeElo                                          | 387                  | -                                                                                | "CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings" [[paper](https://arxiv.org/abs/2501.01257)] [[data](https://codeelo-bench.github.io/)]                                                                                                   |
| 2025-02 | FSE 2025                         | COFFE                                            | 756                  | Python                                                                           | "COFFE: A Code Efficiency Benchmark for Code Generation" [[paper](https://arxiv.org/abs/2502.02827)] [[data](https://github.com/JohnnyPeng18/Coffe)]                                                                                                                                         |
| 2025-02 | arXiv                            | FVAPPS                                           | 4715                 | Python                                                                           | "Proving the Coding Interview: A Benchmark for Formally Verified Code Generation" [[paper](https://arxiv.org/abs/2502.05714)] [[data](https://huggingface.co/datasets/quinn-dougherty/fvapps)]                                                                                               |
| 2025-02 | arXiv                            | KernelBench                                      | 250                  | Python                                                                           | "KernelBench: Can LLMs Write Efficient GPU Kernels?" [[paper](https://arxiv.org/abs/2502.10517)]                                                                                                                                                                                             |
| 2025-02 | arXiv                            | BaxBench                                         | 392                  | Go, JS, PHP, Python, Ruby, Rust                                                  | "BaxBench: Can LLMs Generate Correct and Secure Backends?" [[paper](https://arxiv.org/abs/2502.11844)] [[data](https://baxbench.com/)]                                                                                                                                                       |
| 2025-02 | arXiv                            | DataSciBench                                     | 222                  | Python                                                                           | "DataSciBench: An LLM Agent Benchmark for Data Science" [[paper](https://arxiv.org/abs/2502.13897)] [[data](https://github.com/THUDM/DataSciBench)]                                                                                                                                          |
| 2025-02 | arXiv                            | TritonBench                                      | 184                  | Triton                                                                           | "TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators" [[paper](https://arxiv.org/abs/2502.14752)] [[data](https://github.com/thunlp/TritonBench)]                                                                                                    |
| 2025-02 | arXiv                            | Deep-Bench                                       | 520                  | Python                                                                           | "Deep-Bench: Deep Learning Benchmark Dataset for Code Generation" [[paper](https://arxiv.org/abs/2502.18726)] [[data](https://anonymous.4open.science/r/DL-Bench-D65E/README.md)]                                                                                                            |
| 2025-02 | arXiv                            | IndicEval-XL                                     | 6720                 | 12                                                                               | "IndicEval-XL: Bridging Linguistic Diversity in Code Generation Across Indic Languages" [[paper](https://arxiv.org/abs/2502.19067)] [[data](https://github.com/telekom/IndicEval-XL)]                                                                                                        |
| 2025-02 | arXiv                            | PseudoEval                                       | 1059                 | Python, C++, Rust                                                                | "Isolating Language-Coding from Problem-Solving: Benchmarking LLMs with PseudoEval" [[paper](https://arxiv.org/abs/2502.19149)] [[data](https://anonymous.4open.science/r/PseudocodeACL25-7B74/README.md)]                                                                                   |
| 2025-02 | arXiv                            | ProBench                                         | 790                  | C++, Java, Python                                                                | "ProBench: Benchmarking Large Language Models in Competitive Programming" [[paper](https://arxiv.org/abs/2502.20868)] [[data](https://github.com/YL-9/probench)]                                                                                                                             |
| 2025-03 | arXiv                            | DyCodeEval                                       | -                    | -                                                                                | "Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination" [[paper](https://arxiv.org/abs/2503.04149)]                                                                                                                                          |
| 2025-03 | arXiv                            | DynaCode                                         | 405                  | Python                                                                           | "DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation" [[paper](https://arxiv.org/abs/2503.10452)]                                                                                                                                    |
| 2025-03 | arXiv                            | BigO(Bench)                                      | 3105                 | Python                                                                           | "BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity?" [[paper](https://arxiv.org/abs/2503.15242)] [[data](https://github.com/facebookresearch/bigobench)]                                                                                                       |
| 2025-04 | arXiv                            | -                                                | 842K                 | Python                                                                           | "A Large-scale Class-level Benchmark Dataset for Code Generation with LLMs" [[paper](https://arxiv.org/abs/2504.15564)] [[data](https://anonymous.4open.science/r/class-level-benchmark-dataset-B132/README.md)]                                                                             |
| 2025-05 | arXiv                            | YABLoCo                                          | 215                  | C, C++                                                                           | "YABLoCo: Yet Another Benchmark for Long Context Code Generation" [[paper](https://arxiv.org/abs/2505.04406)] [[data](https://github.com/yabloco-codegen/yabloco-benchmark)]                                                                                                                 |
| 2025-05 | arXiv                            | OSS-Bench                                        | -                    | -                                                                                | "OSS-Bench: Benchmark Generator for Coding LLMs" [[paper](https://arxiv.org/abs/2505.12331)] [[data](https://github.com/oss-bench/oss-bench)]                                                                                                                                                |
| 2025-05 | arXiv                            | DS-Bench                                         | 1000                 | Python                                                                           | "DS-Bench: A Realistic Benchmark for Data Science Code Generation" [[paper](https://arxiv.org/abs/2505.15621)] [[data](https://github.com/ShuyinOuyang/DS_bench)]                                                                                                                            |

\* Automatically mined/human-annotated

$^\dagger$ 1749 prompts for 48 problems

$^\ddagger$ 10538 prompts for 1420 problems

$^\diamond$ Machine/human prompts

#### Visually Grounded Program Synthesis

| Date    | Venue               | Benchmark       | Size | Language                              | Source                                                                                                                                                                                                                                      |
| ------- | ------------------- | --------------- | ---- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024-04 | EMNLP 2024 Findings | MMCode          | 3548 | Python                                | "MMCode: Evaluating Multi-Modal Code Large Language Models with Visually Rich Programming Problems" [[paper](https://arxiv.org/abs/2404.09486)] [[data](https://github.com/happylkx/MMCode)]                                                |
| 2024-05 | arXiv               | Plot2Code       | 132  | Python                                | "Plot2Code: A Comprehensive Benchmark for Evaluating Multi-modal Large Language Models in Code Generation from Scientific Plots" [[paper](https://arxiv.org/abs/2405.07990)] [[data](https://huggingface.co/datasets/TencentARC/Plot2Code)] |
| 2024-06 | ICLR 2025           | ChartMimic      | 1000 | Python                                | "ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation" [[paper](https://arxiv.org/abs/2406.09961)] [[data](https://github.com/ChartMimic/ChartMimic)]                                                 |
| 2024-10 | arXiv               | HumanEval-V     | 108  | Python                                | "HumanEval-V: Evaluating Visual Understanding and Reasoning Abilities of Large Multimodal Models Through Coding Tasks" [[paper](https://arxiv.org/abs/2410.12381)] [[data](https://github.com/HumanEval-V/HumanEval-V-Benchmark)]           |
| 2024-10 | arXiv               | TurtleBench     | 260  | Python                                | "TurtleBench: A Visual Programming Benchmark in Turtle Geometry" [[paper](https://arxiv.org/abs/2411.00264)] [[data](https://github.com/sinaris76/TurtleBench)]                                                                             |
| 2024-12 | ICLR 2025           | BigDocs         | 7.5M | HTML, LaTeX, SVG, JSON, Markdown, etc | "BigDocs: An Open and Permissively-Licensed Dataset for Training Multimodal Models on Document and Code Tasks" [[paper](https://arxiv.org/abs/2412.04626)] [[data](https://bigdocs.github.io/)]                                             |
| 2024-12 | arXiv               | Visual SWEbench | 133  | Python                                | "CodeV: Issue Resolving with Visual Data" [[paper](https://arxiv.org/abs/2412.17315)] [[data](https://github.com/luolin101/CodeV)]                                                                                                          |
| 2025-02 | arXiv               | Code-Vision     | 438  | Python                                | "Code-Vision: Evaluating Multimodal LLMs Logic Understanding and Code Generation Capabilities" [[paper](https://arxiv.org/abs/2502.11829)] [[data](https://github.com/wanghanbinpanda/CodeVision)]                                          |

#### Code Reasoning and QA

| Date    | Venue               | Benchmark   | Size     | Language                            | Source                                                                                                                                                                                                                   |
| ------- | ------------------- | ----------- | -------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2021-09 | EMNLP 2021 Findings | CodeQA      | 120K/70K | Java/Python                         | "CodeQA: A Question Answering Dataset for Source Code Comprehension" [[paper](https://arxiv.org/abs/2109.08365)] [[data](https://github.com/jadecxliu/CodeQA)]                                                           |
| 2022-10 | NAACL 2022          | CS1QA       | 9237     | Python                              | "CS1QA: A Dataset for Assisting Code-based Question Answering in an Introductory Programming Course" [[paper](https://arxiv.org/abs/2210.14494)] [[data](https://github.com/cyoon47/CS1QA)]                              |
| 2023-09 | arXiv               | CodeApex    | 250      | C++                                 | "CodeApex: A Bilingual Programming Evaluation Benchmark for Large Language Models" [[paper](https://arxiv.org/abs/2309.01940)] [[data](https://github.com/APEXLAB/CodeApex)]                                             |
| 2024-01 | ICML 2024           | CRUXEval    | 800      | Python                              | "CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution" [[paper](https://arxiv.org/abs/2401.03065)] [[data](https://github.com/facebookresearch/cruxeval)]                                               |
| 2024-05 | arXiv               | PythonIO    | 2650     | Python                              | "Multiple-Choice Questions are Efficient and Robust LLM Evaluators" [[paper](https://arxiv.org/abs/2405.11966)] [[data](https://github.com/Geralt-Targaryen/MC-Evaluation)]                                              |
| 2024-05 | arXiv               | StaCCQA     | 270K     | Python                              | "Aligning LLMs through Multi-perspective User Preference Ranking-based Feedback for Programming Question Answering" [[paper](https://arxiv.org/abs/2406.00037)] [[data](https://anonymous.4open.science/r/PETCoQA-810A)] |
| 2024-06 | arXiv               | RepoQA      | 500      | Python, C++, Java, Rust, TypeScript | "RepoQA: Evaluating Long Context Code Understanding" [[paper](https://arxiv.org/abs/2406.06025)] [[data](https://github.com/evalplus/repoqa)]                                                                            |
| 2024-08 | arXiv               | CruxEval-X  | 12.6K    | 19                                  | "CRUXEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution" [[paper](https://arxiv.org/abs/2408.13001)] [[data](https://github.com/CRUXEVAL-X/cruxeval-x)]                                    |
| 2024-09 | arXiv               | SpecEval    | 204      | Java                                | "SpecEval: Evaluating Code Comprehension in Large Language Models via Program Specifications" [[paper](https://arxiv.org/abs/2409.12866)] [[data](https://sites.google.com/view/speceval/)]                              |
| 2024-10 | ICLR 2025           | CodeMMLU    | 19912    | 13                                  | "CodeMMLU: A Multi-Task Benchmark for Assessing Code Understanding Capabilities of CodeLLMs" [[paper](https://arxiv.org/abs/2410.01999)] [[data](https://github.com/FSoft-AI4Code/CodeMMLU)]                             |
| 2024-11 | arXiv               | unnamed     | 80232    | Python                              | "Leveraging Large Language Models in Code Question Answering: Baselines and Issues" [[paper](https://arxiv.org/abs/2411.03012)] [[data](https://github.com/IU-AES-AI4Code/CodeQuestionAnswering)]                        |
| 2024-11 | arXiv               | ScratchEval | 305      | Scratch                             | "ScratchEval: Are GPT-4o Smarter than My Child? Evaluating Large Multimodal Models with Visual Programming Challenges" [[paper](https://arxiv.org/abs/2411.18932)] [[data](https://github.com/HKBUNLP/ScratchEval)]      |
| 2024-12 | arXiv               | CodeRepoQA  | 585,687  | Python, Java, TS, JS, Go            | "CodeRepoQA: A Large-scale Benchmark for Software Engineering Question Answering" [[paper](https://arxiv.org/abs/2412.14764)] [[data](https://github.com/kinesiatricssxilm14/CodeRepoQA)]                                |
| 2025-01 | arXiv               | CoReQA      | 1,563    | Python, Java, Go, TS                | "CoReQA: Uncovering Potentials of Language Models in Code Repository Question Answering" [[paper](https://arxiv.org/abs/2501.03447)]                                                                                     |
| 2025-02 | arXiv               | EquiBench   | 2400     | C, CUDA, x86-64, Python             | "EquiBench: Benchmarking Code Reasoning Capabilities of Large Language Models via Equivalence Checking" [[paper](https://arxiv.org/abs/2502.12466)]                                                                      |
| 2025-03 | arXiv               | LONGCODEU   | 3983     | Python                              | "LONGCODEU: Benchmarking Long-Context Language Models on Long Code Understanding" [[paper](https://arxiv.org/abs/2503.04359)]                                                                                            |

#### Text-to-SQL

- "Deep learning driven natural languages text to SQL query conversion: A survey", 2022-08, arXiv, [[paper](https://arxiv.org/abs/2208.04415)]
- "Recent Advances in Text-to-SQL: A Survey of What We Have and What We Expect", 2022-08, COLING 2022, [[paper](https://arxiv.org/abs/2208.10099)]
- "A Survey on Text-to-SQL Parsing: Concepts, Methods, and Future Directions", 2022-08, arXiv, [[paper](https://arxiv.org/abs/2208.13629)]
- "A survey on deep learning approaches for text-to-SQL", 2023-01, VLDB J., [[paper](https://link.springer.com/article/10.1007/s00778-022-00776-8)]

| Date    | Venue               | Benchmark        | Size       | Language | Source                                                                                                                                                                                                       |
| ------- | ------------------- | ---------------- | ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2017-08 | arXiv               | WikiSQL          | 80654      |          | "Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning" [[paper](https://arxiv.org/abs/1709.00103)] [[data](https://github.com/salesforce/WikiSQL)]                      |
| 2018-06 | CL 2018             | Advising         | 4570       |          | "Improving Text-to-SQL Evaluation Methodology" [[paper](https://arxiv.org/abs/1806.09029)] [[data](https://github.com/jkkummerfeld/text2sql-data/)]                                                          |
| 2018-09 | EMNLP 2018          | Spider           | 10181      |          | "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task" [[paper](https://arxiv.org/abs/1809.08887)] [[data](https://yale-lily.github.io/spider)]    |
| 2019-06 | ACL 2019            | SParC            | 12726      |          | "SParC: Cross-Domain Semantic Parsing in Context" [[paper](https://arxiv.org/abs/1906.02285)] [[data](https://yale-lily.github.io/sparc)]                                                                    |
| 2019-07 | WWW 2020            | MIMICSQL         | 10000      |          | "Text-to-SQL Generation for Question Answering on Electronic Medical Records" [[paper](https://arxiv.org/abs/1908.01839)] [[data](https://github.com/wangpinggl/TREQS)]                                      |
| 2019-09 | EMNLP-IJCNLP 2019   | CoSQL            | 15598      |          | "CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases" [[paper](https://arxiv.org/abs/1909.05378)] [[data](https://yale-lily.github.io/cosql)]        |
| 2020-05 | LREC 2020           | Criteria-to-SQL  | 2003       |          | "Dataset and Enhanced Model for Eligibility Criteria-to-SQL Semantic Parsing" [[paper](https://aclanthology.org/2020.lrec-1.714/)] [[data](https://github.com/xiaojingyu92/Criteria2SQL)]                    |
| 2020-10 | EMNLP 2020 Findings | Squall           | 11276      |          | "On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries" [[paper](https://arxiv.org/abs/2010.11246)] [[data](https://github.com/tzshi/squall)]                                    |
| 2020-10 | NAACL-HLT 2021      | Spider-Realistic | 508        |          | "Structure-Grounded Pretraining for Text-to-SQL" [[paper](https://arxiv.org/abs/2010.12773)] [[data](https://zenodo.org/records/5205322)]                                                                    |
| 2021-06 | ACL/IJCNLP 2021     | Spider-Syn       | 8034       |          | "Towards Robustness of Text-to-SQL Models against Synonym Substitution" [[paper](https://arxiv.org/abs/2106.01065)] [[data](https://arxiv.org/abs/2106.01065)]                                               |
| 2021-06 | NLP4Prog 2021       | SEDE             | 12023      |          | "Text-to-SQL in the Wild: A Naturally-Occurring Dataset Based on Stack Exchange Data" [[paper](https://arxiv.org/abs/2106.05006)] [[data](https://github.com/hirupert/sede)]                                 |
| 2021-06 | ACL/IJCNLP 2021     | KaggleDBQA       | 400        |          | "KaggleDBQA: Realistic Evaluation of Text-to-SQL Parsers" [[paper](https://arxiv.org/abs/2106.11455)] [[data](https://github.com/chiahsuan156/KaggleDBQA)]                                                   |
| 2021-09 | EMNLP               | Spider-DK        | 535        |          | "Exploring Underexplored Limitations of Cross-Domain Text-to-SQL Generalization" [[paper](https://arxiv.org/abs/2109.05157)] [[data](https://github.com/ygan/Spider-DK)]                                     |
| 2022-05 | NAACL 2022 Findings | Spider-SS/CG     | 8034/45599 |          | "Measuring and Improving Compositional Generalization in Text-to-SQL via Component Alignment" [[paper](https://arxiv.org/abs/2205.02054)] [[data](https://github.com/ygan/SpiderSS-SpiderCG)]                |
| 2023-05 | arXiv               | BIRD             | 12751      |          | "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs" [[paper](https://arxiv.org/abs/2305.03111)] [[data](https://bird-bench.github.io/)]              |
| 2023-06 | ACL 2023            | XSemPLR          | 24.4K      |          | "XSemPLR: Cross-Lingual Semantic Parsing in Multiple Natural Languages and Meaning Representations" [[paper](https://arxiv.org/abs/2306.04085)] [[data](https://github.com/psunlpgroup/XSemPLR)]             |
| 2024-05 | ACL 2024 Findings   | EHR-SeqSQL       | 31669      |          | "EHR-SeqSQL : A Sequential Text-to-SQL Dataset For Interactively Exploring Electronic Health Records" [[paper](https://arxiv.org/abs/2406.00019)]                                                            |
| 2024-06 | NAACL 2024          | BookSQL          | 100K       |          | "BookSQL: A Large Scale Text-to-SQL Dataset for Accounting Domain" [[paper](https://arxiv.org/abs/2406.07860)] [[data](https://github.com/Exploration-Lab/BookSQL)]                                          |
| 2024-08 | ACL 2024 Findings   | MultiSQL         | 9257       |          | "MultiSQL: A Schema-Integrated Context-Dependent Text2SQL Dataset with Diverse SQL Operations" [[paper](https://aclanthology.org/2024.findings-acl.823/)] [[data](https://github.com/grandchicken/MultiSQL)] |
| 2024-09 | arXiv               | BEAVER           | 93         |          | "BEAVER: An Enterprise Benchmark for Text-to-SQL" [[paper](https://arxiv.org/abs/2409.02038)]                                                                                                                |
| 2024-10 | arXiv               | PRACTIQ          | 2812       |          | "PRACTIQ: A Practical Conversational Text-to-SQL dataset with Ambiguous and Unanswerable Queries" [[paper](https://arxiv.org/abs/2410.11076)]                                                                |
| 2024-10 | arXiv               | BIS              | 239        |          | "BIS: NL2SQL Service Evaluation Benchmark for Business Intelligence Scenarios" [[paper](https://arxiv.org/abs/2410.22925)] [[data](https://github.com/boracaglayan/bis-nl2sql)]                              |
| 2024-11 | ICLR 2025           | Spider 2.0       | 632        |          | "Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows" [[paper](https://arxiv.org/abs/2411.07763)] [[data](https://github.com/xlang-ai/Spider2)]                            |
| 2025-01 | arXiv               | Dialect2SQL      | 9428       |          | "Dialect2SQL: A Novel Text-to-SQL Dataset for Arabic Dialects with a Focus on Moroccan Darija" [[paper](https://arxiv.org/abs/2501.11498)]                                                                   |
| 2025-05 | arXiv               | LogicCat         | 4038       |          | "LogicCat: A Chain-of-Thought Text-to-SQL Benchmark for Multi-Domain Reasoning Challenges" [[paper](https://arxiv.org/abs/2505.18744)]                                                                       |
| 2025-05 | arXiv               | BiomedSQL        | 68,000     |          | "BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases" [[paper](https://arxiv.org/abs/2505.20321)] [[data](https://github.com/NIH-CARD/biomedsql)]                                  |

#### Code Translation

| Date    | Venue                                | Benchmark                | Size   | Language                                                  | Source                                                                                                                                                                                                              |
| ------- | ------------------------------------ | ------------------------ | ------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2020-06 | NeurIPS 2020                         | Transcoder GeeksforGeeks | 1.4K   | C++, Java, Python                                         | "Unsupervised Translation of Programming Languages" [[paper](https://arxiv.org/abs/2006.03511)] [[data](https://github.com/facebookresearch/TransCoder)]                                                            |
| 2021-02 | NeurIPS Datasets and Benchmarks 2021 | CodeTrans                | 11.8K  | Java, C#                                                  | "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation" [[paper](https://arxiv.org/abs/2102.04664)] [[data](https://huggingface.co/datasets/code_x_glue_cc_code_to_code_trans)]     |
| 2021-08 | ACL 2023 Findings                    | Avatar                   | 9515   | Java, Python                                              | "AVATAR: A Parallel Corpus for Java-Python Program Translation" [[paper](https://arxiv.org/abs/2108.11590)] [[data](https://github.com/wasiahmad/AVATAR)]                                                           |
| 2022-06 | AAAI 2022                            | CoST                     | 132K   | C++, Java, Python, C#, JS, PHP, C                         | "Multilingual Code Snippets Training for Program Translation" [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21434)] [[data](https://github.com/reddy-lab-code-research/MuST-CoST)]                      |
| 2022-06 | arXiv                                | XLCoST                   | 567K   | C++, Java, Python, C#, JS, PHP, C                         | "XLCoST: A Benchmark Dataset for Cross-lingual Code Intelligence" [[paper](https://arxiv.org/abs/2206.08474)] [[data](https://github.com/reddy-lab-code-research/XLCoST)]                                           |
| 2023-03 | arXiv                                | xCodeEval                | 5.6M   | C, C#, C++, Go, Java, JS, Kotlin, PHP, Python, Ruby, Rust | "xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval" [[paper](https://arxiv.org/abs/2303.03004)] [[data](https://github.com/ntunlp/xCodeEval)] |
| 2023-03 | arXiv                                | HumanEval-X              | 1640   | Python, C++, Java, JS, Go                                 | "CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X" [[paper](https://arxiv.org/abs/2303.17568)] [[data](https://github.com/THUDM/CodeGeeX)]                            |
| 2023-08 | arXiv                                | G-TransEval              | 4000   | C++, Java, C#, JS, Python                                 | "On the Evaluation of Neural Code Translation: Taxonomy and Benchmark" [[paper](https://arxiv.org/abs/2308.08961)] [[data](https://github.com/PolyEval/G-TransEval)]                                                |
| 2023-10 | arXiv                                | CodeTransOcean           | 270.5K | 45                                                        | "CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation" [[paper](https://arxiv.org/abs/2310.04951)] [[data](https://github.com/WeixiangYAN/CodeTransOcean)]                                   |
| 2024-11 | arXiv                                | Classeval-T              | 94     | Python, Java, C++                                         | "Escalating LLM-based Code Translation Benchmarking into the Class-level Era" [[paper](https://arxiv.org/abs/2411.06145)]                                                                                           |
| 2024-11 | arXiv                                | RustRepoTrans            | 375    | C++, Java, Python, Rust                                   | "Repository-level Code Translation Benchmark Targeting Rust" [[paper](https://arxiv.org/abs/2411.13990)] [[data](https://github.com/TrustedGPT/RustRepoTrans)]                                                      |
| 2024-12 | arXiv                                | RepoTransBench           | 100    | Python, Java                                              | "RepoTransBench: A Real-World Benchmark for Repository-Level Code Translation" [[paper](https://arxiv.org/abs/2412.17744)]                                                                                          |
| 2025-01 | arXiv                                | TransRepo-Bench          | 13     | Java, C#                                                  | "Skeleton-Guided-Translation: A Benchmarking Framework for Code Repository Translation with Fine-Grained Quality Evaluation" [[paper](https://arxiv.org/abs/2501.16050)]                                            |
| 2025-04 | arXiv                                | CRUST-Bench              | 100    | C, Rust                                                   | "CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation" [[paper](https://arxiv.org/abs/2504.15254)] [[data](https://github.com/anirudhkhatry/CRUST-bench)]                                        |

#### Program Repair

- "Neural Program Repair: Systems, Challenges and Solutions", 2022-02, Internetware 2022, [[paper](https://arxiv.org/abs/2202.10868)]
- "A Survey of Learning-based Automated Program Repair", 2023-01, arXiv, [[paper](https://arxiv.org/abs/2301.03270)]
- "A Survey on Automated Program Repair Techniques", 2023-03, arXiv, [[paper](https://arxiv.org/abs/2303.18184)]

| Date    | Venue                            | Benchmark           | Size      | Language                                                  | Source                                                                                                                                                                                                                    |
| ------- | -------------------------------- | ------------------- | --------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2014-07 | ISSTA 2014                       | Defects4J           | 357       | Java                                                      | "Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies for Java Programs" [[paper](https://dl.acm.org/doi/10.1145/2610384.2628055)] [[data](https://github.com/rjust/defects4j)]                  |
| 2015-12 | IEEE Trans. Software Engineering | ManyBugs/IntroClass | 185/998   | C                                                         | "The ManyBugs and IntroClass Benchmarks for Automated Repair of C Programs" [[paper](https://ieeexplore.ieee.org/document/7153570)] [[data](https://repairbenchmarks.cs.umass.edu/)]                                      |
| 2016-11 | FSE 2016                         | BugAID              | 105K      | JS                                                        | "Discovering Bug Patterns in JavaScript" [[paper](https://dl.acm.org/doi/10.1145/2950290.2950308)] [[data](https://salt.ece.ubc.ca/software/bugaid/)]                                                                     |
| 2017-02 | AAAI 2017                        | DeepFix             | 6971      | C                                                         | "DeepFix: Fixing Common C Language Errors by Deep Learning" [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10742)] [[data](https://bitbucket.org/iiscseal/deepfix/src/master/)]                                |
| 2017-05 | ICSE-C 2017                      | Codeflaws           | 3902      | C                                                         | "DeepFix: Fixing Common C Language Errors by Deep Learning" [[paper](https://dl.acm.org/doi/10.1109/ICSE-C.2017.76)] [[data](https://codeflaws.github.io/)]                                                               |
| 2017-10 | SPLASH 2017                      | QuixBugs            | 80        | Java, Python                                              | "QuixBugs: a multi-lingual program repair benchmark set based on the quixey challenge" [[paper](https://dl.acm.org/doi/10.1145/3135932.3135941)] [[data](https://github.com/jkoppel/QuixBugs)]                            |
| 2018-05 | MSR 2018                         | Bugs.jar            | 1158      | Java                                                      | "Bugs.jar: a large-scale, diverse dataset of real-world Java bugs" [[paper](https://dl.acm.org/doi/10.1145/3196398.3196473)] [[data](https://github.com/bugs-dot-jar/bugs-dot-jar)]                                       |
| 2018-12 | ACM Trans. Softw. Eng. Methodol. | BFP                 | 124K      | Java                                                      | "An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation" [[paper](https://arxiv.org/abs/1812.08693)] [[data](https://sites.google.com/view/learning-fixes)]                         |
| 2019-01 | SANER 2019                       | Bears               | 251       | Java                                                      | "Bears: An Extensible Java Bug Benchmark for Automatic Program Repair Studies" [[paper](https://arxiv.org/abs/1901.06024)] [[data](https://github.com/bears-bugs/bears-benchmark)]                                        |
| 2019-01 | ICSE 2019                        | unnamed             | 21.8K \*  | Java                                                      | "On Learning Meaningful Code Changes via Neural Machine Translation" [[paper](https://arxiv.org/abs/1901.09102)] [[data](https://sites.google.com/view/learning-codechanges)]                                             |
| 2019-04 | ICST 2019                        | BugsJS              | 453       | JS                                                        | "BugsJS: a Benchmark of JavaScript Bugs" [[paper](https://ieeexplore.ieee.org/document/8730197)] [[data](https://bugsjs.github.io/)]                                                                                      |
| 2019-05 | ICSE 2019                        | BugSwarm            | 1827/1264 | Java/Python                                               | "BugSwarm: mining and continuously growing a dataset of reproducible failures and fixes" [[paper](https://dl.acm.org/doi/10.1109/ICSE.2019.00048)] [[data](https://www.bugswarm.org/)]                                    |
| 2019-05 | ICSE 2019                        | CPatMiner           | 17K \*    | Java                                                      | "Graph-based mining of in-the-wild, fine-grained, semantic code change patterns" [[paper](https://dl.acm.org/doi/10.1109/ICSE.2019.00089)] [[data](https://nguyenhoan.github.io/CPatMiner/)]                              |
| 2019-05 | MSR 2020                         | ManySStuBs4J        | 154K      | Java                                                      | "How Often Do Single-Statement Bugs Occur? The ManySStuBs4J Dataset" [[paper](https://arxiv.org/abs/1905.13334)] [[data](https://github.com/mast-group/mineSStuBs)]                                                       |
| 2019-11 | ASE 2019                         | Refactory           | 1783      | Python                                                    | "Re-factoring based program repair applied to programming assignments" [[paper](https://dl.acm.org/doi/10.1109/ASE.2019.00044)] [[data](https://github.com/githubhuyang/refactory)]                                       |
| 2020-07 | ISSTA 2020                       | CoCoNut             | 24M       | Java, Python, C, JS                                       | "CoCoNuT: combining context-aware neural translation models using ensemble for program repair" [[paper](https://dl.acm.org/doi/10.1145/3395363.3397369)] [[data](https://github.com/lin-tan/CoCoNut-Artifact)]            |
| 2020-10 | Inf. Softw. Technol.             | Review4Repair       | 58021     | Java                                                      | "Review4Repair: Code Review Aided Automatic Program Repairing" [[paper](https://arxiv.org/abs/2010.01544)] [[data](https://github.com/Review4Repair/Review4Repair)]                                                       |
| 2020-11 | ESEC/FSE 2020                    | BugsInPy            | 493       | Python                                                    | "BugsInPy: A Database of Existing Bugs in Python Programs to Enable Controlled Testing and Debugging Studies" [[paper](https://dl.acm.org/doi/abs/10.1145/3368089.3417943)] [[data](https://github.com/soarsmu/BugsInPy)] |
| 2021-07 | ICML 2021                        | TFix                | 105K      | JS                                                        | "TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer" [[paper](https://proceedings.mlr.press/v139/berabi21a.html)] [[data](https://github.com/eth-sri/TFix)]                                              |
| 2021-08 | arXiv                            | Megadiff            | 663K \*   | Java                                                      | "Megadiff: A Dataset of 600k Java Source Code Changes Categorized by Diff Size" [[paper](https://arxiv.org/abs/2108.04631)] [[data](https://github.com/ASSERT-KTH/megadiff)]                                              |
| 2022-01 | SSB/TSSB                         | MSR 2022            | 9M/3M     | Python                                                    | "TSSB-3M: Mining single statement bugs at massive scale" [[paper](https://arxiv.org/abs/2201.12046)] [[data](https://cedricrupb.github.io/TSSB3M/)]                                                                       |
| 2022-10 | MSR 2022                         | FixJS               | 324K      | JS                                                        | "FixJS: a dataset of bug-fixing JavaScript commits" [[paper](https://dl.acm.org/doi/abs/10.1145/3524842.3528480)] [[data](https://github.com/AAI-USZ/FixJS)]                                                              |
| 2022-11 | ESEC/FSE 2022                    | TypeBugs            | 93        | Python                                                    | "PyTER: Effective Program Repair for Python Type Errors" [[paper](https://dl.acm.org/doi/abs/10.1145/3540250.3549130)] [[data](https://github.com/kupl/PyTER)]                                                            |
| 2023-03 | arXiv                            | xCodeEval           | 4.7M      | C, C#, C++, Go, Java, JS, Kotlin, PHP, Python, Ruby, Rust | "xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval" [[paper](https://arxiv.org/abs/2303.03004)] [[data](https://github.com/ntunlp/xCodeEval)]       |
| 2023-04 | arXiv                            | RunBugRun           | 450K      | C, C++, Java, Python, JS, Ruby, Go, PHP                   | "RunBugRun -- An Executable Dataset for Automated Program Repair" [[paper](https://arxiv.org/abs/2304.01102)] [[data](https://github.com/giganticode/run_bug_run)]                                                        |
| 2023-08 | arXiv                            | HumanEvalPack       | 984       | Python, JS, Go, Java, C++, Rust                           | "OctoPack: Instruction Tuning Code Large Language Models" [[paper](https://arxiv.org/abs/2308.07124)] [[data](https://huggingface.co/datasets/bigcode/humanevalpack)]                                                     |
| 2024-01 | arXiv                            | DebugBench          | 4253      | C++, Java, Python                                         | "DebugBench: Evaluating Debugging Capability of Large Language Models" [[paper](https://arxiv.org/abs/2401.04621)] [[data](https://github.com/thunlp/DebugBench)]                                                         |
| 2024-11 | arXiv                            | MdEval              | 3513      | 18                                                        | "MdEval: Massively Multilingual Code Debugging" [[paper](https://arxiv.org/abs/2411.02310)]                                                                                                                               |
| 2025-01 | arXiv                            | unnamed             | 48,398    | Python                                                    | "Suggesting Code Edits in Interactive Machine Learning Notebooks Using Large Language Models" [[paper](https://arxiv.org/abs/2501.09745)] [[data](https://zenodo.org/records/14281690)]                                   |

\* These are code-change datasest, and only a subset therein concerns bug fixing.

#### Code Summarization

- "A Survey of Automatic Source Code Summarization", 2022-02, Symmetry, [[paper](https://www.mdpi.com/2073-8994/14/3/471)]

| Date    | Venue       | Benchmark     | Size    | Language                        | Source                                                                                                                                                                                                                             |
| ------- | ----------- | ------------- | ------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2016-08 | ACL 2016    | CODE-NN       | 66K/32K | C#/SQL                          | "Summarizing Source Code using a Neural Attention Model" [[paper](https://aclanthology.org/P16-1195/)] [[data](https://github.com/sriniiyer/codenn)]                                                                               |
| 2017-07 | IJCNLP 2017 | unnamed       | 150K    | Python                          | "A parallel corpus of Python functions and documentation strings for automated code documentation and code generation" [[paper](https://arxiv.org/abs/1707.02275)] [[data](https://github.com/EdinburghNLP/code-docstring-corpus)] |
| 2018-05 | ICPC 2018   | DeepCom       | 588K    | Java                            | "Deep code comment generation" [[paper](https://dl.acm.org/doi/10.1145/3196321.3196334)] [[data](https://github.com/xing-hu/DeepCom)]                                                                                              |
| 2018-07 | IJCAI 2018  | TL-CodeSum    | 411K    | Java                            | "Summarizing Source Code with Transferred API Knowledge" [[paper](https://www.ijcai.org/proceedings/2018/314)] [[data](https://github.com/xing-hu/TL-CodeSum)]                                                                     |
| 2018-11 | ASE 2018    | unnamed       | 109K    | Python                          | "Improving Automatic Source Code Summarization via Deep Reinforcement Learning" [[paper](https://arxiv.org/abs/1811.07234)] [[data](https://github.com/wanyao1992/code_summarization_public)]                                      |
| 2019-09 | arxiv       | CodeSearchNet | 2.3M    | Go, JS, Python, PHP, Java, Ruby | "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [[paper](https://arxiv.org/abs/1909.09436)] [[data](https://github.com/github/CodeSearchNet)]                                                              |
| 2023-08 | arXiv       | HumanEvalPack | 984     | Python, JS, Go, Java, C++, Rust | "OctoPack: Instruction Tuning Code Large Language Models" [[paper](https://arxiv.org/abs/2308.07124)] [[data](https://huggingface.co/datasets/bigcode/humanevalpack)]                                                              |

#### Defect/Vulnerability Detection

- "Benchmarking Software Vulnerability Detection Techniques: A Survey", 2023-03, arXiv, [[paper](https://arxiv.org/abs/2303.16362)]

| Date    | Venue                        | Benchmark      | Size  | Language                    | Source                                                                                                                                                                                                                           |
| ------- | ---------------------------- | -------------- | ----- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2018-01 | NDSS 2018                    | CGD            | 62K   | C, C++                      | "VulDeePecker: A Deep Learning-Based System for Vulnerability Detection" [[paper](https://arxiv.org/abs/1801.01681)] [[data](https://github.com/CGCL-codes/VulDeePecker)]                                                        |
| 2018-04 | IEEE Trans. Ind. Informatics | unnamed        | 32988 | C, C++                      | "Cross-Project Transfer Representation Learning for Vulnerable Function Discovery" [[paper](https://ieeexplore.ieee.org/document/8329207)] [[data](https://github.com/DanielLin1986/TransferRepresentationLearning)]             |
| 2018-07 | ICMLA 2018                   | Draper VDISC   | 12.8M | C, C++                      | "Automated Vulnerability Detection in Source Code Using Deep Representation Learning" [[paper](https://arxiv.org/abs/1807.04320)] [[data](https://osf.io/d45bw/)]                                                                |
| 2018-07 | IEEE TDSC                    | SySeVR         | 15591 | C, C++                      | "SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities" [[paper](https://arxiv.org/abs/1807.06756)] [[data](https://github.com/SySeVR/SySeVR)]                                                          |
| 2019-02 | MSR 2019                     | unnamed        | 624   | Java                        | "A Manually-Curated Dataset of Fixes to Vulnerabilities of Open-Source Software" [[paper](https://arxiv.org/abs/1902.02595)] [[data](https://github.com/SAP/project-kb/tree/main/MSR2019)]                                       |
| 2019-09 | NeurIPS 2019                 | Devign         | 49K   | C                           | "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks" [[paper](https://arxiv.org/abs/1909.03496)] [[data](https://sites.google.com/view/devign)]                |
| 2019-11 | IEEE TDSC                    | unnamed        | 170K  | C, C++                      | "Software Vulnerability Discovery via Learning Multi-Domain Knowledge Bases" [[paper](https://ieeexplore.ieee.org/document/8906156)] [[data](https://github.com/DanielLin1986/RepresentationsLearningFromMulti_domain)]          |
| 2019-12 | ICLR 2020                    | GREAT          | 2.8M  | Python                      | "Global Relational Models of Source Code" [[paper](https://openreview.net/forum?id=B1lnbRNtwr)] [[data](https://zenodo.org/records/3954944)]                                                                                     |
| 2020-01 | IEEE TDSC                    | MVD            | 182K  | C, C++                      | "μVulDeePecker: A Deep Learning-Based System for Multiclass Vulnerability Detection" [[paper](https://arxiv.org/abs/2001.02334)] [[data](https://github.com/muVulDeePecker/muVulDeePecker)]                                      |
| 2020-02 | ICICS 2019                   | unnamed        | 1471  | C                           | "Deep Learning-Based Vulnerable Function Detection: A Benchmark" [[paper](https://link.springer.com/chapter/10.1007/978-3-030-41579-2_13)] [[data](https://github.com/DanielLin1986/Function-level-Vulnerability-Detection)]     |
| 2020-09 | IEEE Trans. Software Eng.    | ReVeal         | 18K   | C                           | "Deep Learning based Vulnerability Detection: Are We There Yet?" [[paper](https://arxiv.org/abs/2009.07235)] [[data](https://bit.ly/3bX30ai)]                                                                                    |
| 2020-09 | MSR 2020                     | Big-Vul        | 265K  | C, C++                      | "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries" [[paper](https://dl.acm.org/doi/10.1145/3379597.3387501)] [[data](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset)]                     |
| 2021-02 | ICSE (SEIP) 2021             | D2A            | 1.3M  | C, C++                      | "D2A: A Dataset Built for AI-Based Vulnerability Detection Methods Using Differential Analysis" [[paper](https://arxiv.org/abs/2102.07995)] [[data](https://github.com/ibm/D2A)]                                                 |
| 2021-05 | NeurIPS 2021                 | PyPIBugs       | 2374  | Python                      | "Self-Supervised Bug Detection and Repair" [[paper](https://arxiv.org/abs/2105.12787)] [[data](https://www.microsoft.com/en-us/download/103554)]                                                                                 |
| 2021-07 | In PROMISE 2021              | CVEfixes       | 5495  | 27                          | "CVEfixes: Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software" [[paper](https://arxiv.org/abs/2107.08760)] [[data](https://zenodo.org/records/7029359)]                                           |
| 2021-08 | ESEC/FSE 2021                | CrossVul       | 27476 | 40+                         | "CrossVul: a cross-language vulnerability dataset with commit data" [[paper](https://dl.acm.org/doi/10.1145/3468264.3473122)] [[data](https://zenodo.org/records/4734050)]                                                       |
| 2023-04 | RAID 2023                    | DiverseVul     | 349K  | C, C++                      | "DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection" [[paper](https://arxiv.org/abs/2304.00409)] [[data](https://github.com/wagner-group/diversevul)]                              |
| 2023-06 | arXiv                        | VulnPatchPairs | 26K   | C                           | "Limits of Machine Learning for Automatic Vulnerability Detection" [[paper](https://arxiv.org/abs/2306.17193)] [[data](https://github.com/niklasrisse/LimitsOfML4Vuln)]                                                          |
| 2023-11 | arXiv                        | VulBench       | 455   | C                           | "How Far Have We Gone in Vulnerability Detection Using Large Language Models" [[paper](https://arxiv.org/abs/2311.12420)] [[data](https://anonymous.4open.science/r/VulBench-EA6F/)]                                             |
| 2024-03 | arXiv                        | PrimeVul       | 236K  | C/C++                       | "Vulnerability Detection with Code Language Models: How Far Are We?" [[paper](https://arxiv.org/abs/2403.18624)]                                                                                                                 |
| 2024-06 | arXiv                        | VulDetectBench | 1000  | C/C++                       | "VulDetectBench: Evaluating the Deep Capability of Vulnerability Detection with Large Language Models" [[paper](https://arxiv.org/abs/2406.07595)] [[data](https://github.com/Sweetaroo/VulDetectBench)]                         |
| 2024-08 | arXiv                        | CodeJudge-Eval | 1860  | Python                      | "CodeJudge-Eval: Can Large Language Models be Good Judges in Code Understanding?" [[paper](https://arxiv.org/abs/2408.10718)] [[data](https://github.com/CodeLLM-Research/CodeJudge-Eval)]                                       |
| 2024-11 | arXiv                        | CleanVul       | 11632 | Java, Python, JS, C#, C/C++ | "CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics" [[paper](https://arxiv.org/abs/2411.17274)] [[data](https://github.com/yikun-li/CleanVul)]                                     |
| 2025-03 | arXiv                        | CASTLE         | 250   | C                           | "CASTLE: Benchmarking Dataset for Static Code Analyzers and LLMs towards CWE Detection" [[paper](https://arxiv.org/abs/2503.09433)] [[data](https://github.com/CASTLE-Benchmark/Tests-C250)]                                     |
| 2025-03 | arXiv                        | DSDBench       | 1117  | Python                      | "Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors" [[paper](https://arxiv.org/abs/2503.22388)] [[data](https://github.com/KevinCL16/DSDBench)]                         |
| 2025-05 | arXiv                        | SecVulEval     | 25440 | C/C++                       | "SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection" [[paper](https://arxiv.org/abs/2505.19828)] [[data](https://github.com/basimbd/SecVulEval)]                                                         |
| 2025-05 | arXiv                        | SV-TrustEval-C | 3,337 | C                           | "SV-TrustEval-C: Evaluating Structure and Semantic Reasoning in Large Language Models for Source Code Vulnerability Analysis" [[paper](https://arxiv.org/abs/2505.20630)] [[data](https://github.com/Jackline97/SV-TrustEval-C)] |

#### Code Retrieval

- "Code Search: A Survey of Techniques for Finding Code", 2022-04, ICSME 2021, [[paper](ACM Comput. Surv)]
- "A Survey of Deep Code Search", 2023-05, arXiv, [[paper](https://arxiv.org/abs/2305.05959)]

| Date    | Venue                                | Benchmark            | Size      | Language                                                | Source                                                                                                                                                                                                                                              |
| ------- | ------------------------------------ | -------------------- | --------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2018-03 | WWW 2018                             | StaQC                | 148K/120K | Python/SQL                                              | "StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow" [[paper](https://arxiv.org/abs/1803.09371)] [[data](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset)]                                                   |
| 2018-05 | ICSE 2018                            | DeepCS               | 16.2M     | Java                                                    | "Deep Code Search" [[paper](https://dl.acm.org/doi/10.1145/3180155.3180167)] [[data](https://github.com/guxd/deep-code-search)]                                                                                                                     |
| 2018-05 | MSR 2018                             | CoNaLa               | 600K/2.9K | Python                                                  | "Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow" [[paper](https://arxiv.org/abs/1805.08949)] [[data](https://conala-corpus.github.io/)]                                                                               |
| 2019-08 | arXiv                                | unnamed              | 287       | Java                                                    | "Neural Code Search Evaluation Dataset" [[paper](https://arxiv.org/abs/1908.09804)] [[data](https://github.com/facebookresearch/Neural-Code-Search-Evaluation-Dataset)]                                                                             |
| 2019-09 | arXiv                                | CodeSearchNet        | 2.3M/99   | Go, PHP, JS, Python, Java, Ruby                         | "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [[paper](https://arxiv.org/abs/1909.09436)] [[data](https://github.com/github/CodeSearchNet)]                                                                               |
| 2020-02 | SANER 2020                           | CosBench             | 52        | Java                                                    | "Are the Code Snippets What We Are Searching for? A Benchmark and an Empirical Study on Code Search with Natural-Language Queries" [[paper](https://ieeexplore.ieee.org/document/9054840)] [[data](https://github.com/BASE-LAB-SJTU/CosBench/wiki)] |
| 2020-08 | arXiv                                | SO-DS                | 2.2K      | Python                                                  | "Neural Code Search Revisited: Enhancing Code Snippet Retrieval through Natural Language Intent" [[paper](https://arxiv.org/abs/2008.12193)] [[data](https://github.com/nokia/codesearch)]                                                          |
| 2020-10 | ACM Trans. Knowl. Discov. Data       | FB-Java              | 249K      | Java                                                    | "Deep Graph Matching and Searching for Semantic Code Retrieval" [[paper](https://arxiv.org/abs/2010.12908)] [[data](https://github.com/ryderling/DGMS)]                                                                                             |
| 2021-02 | NeurIPS Datasets and Benchmarks 2021 | AdvTest/WebQueryTest | 280K/1K   | Python                                                  | "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation" [[paper](https://arxiv.org/abs/2102.04664)] [[data]]                                                                                                        |
| 2021-05 | ACL/IJCNLP 2021                      | CoSQA                | 21K       | Python                                                  | "CoSQA: 20,000+ Web Queries for Code Search and Question Answering" [[paper](https://arxiv.org/abs/2105.13239)] [[data](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery)]                                        |
| 2024-03 | arXiv                                | ProCQA               | 5.2M      | C, C++, Java, Python, Ruby, Lisp, JS, C#, Go, Rust, PHP | "ProCQA: A Large-scale Community-based Programming Question Answering Dataset for Code Search" [[paper](https://arxiv.org/abs/2403.16702)] [[data](https://github.com/jordane95/procqa)]                                                            |
| 2024-06 | arXiv                                | CoSQA+               | 109K      | Python                                                  | "CoSQA+: Enhancing Code Search Dataset with Matching Code" [[paper](https://arxiv.org/abs/2406.11589)] [[data](https://github.com/DeepSoftwareAnalytics/CoSQA_Plus)]                                                                                |
| 2024-07 | arXiv                                | CoIR                 | ~2M       | 14                                                      | "CoIR: A Comprehensive Benchmark for Code Information Retrieval Models" [[paper](https://arxiv.org/abs/2407.02883)] [[data](https://github.com/CoIR-team/coir)]                                                                                     |
| 2024-08 | arXiv                                | SeqCoBench           | 14.5K     | Python                                                  | "What can Large Language Models Capture about Code Functional Equivalence?" [[paper](https://arxiv.org/abs/2408.11081)]                                                                                                                             |

#### Type Inference

| Date    | Venue         | Benchmark                    | Size      | Language   | Source                                                                                                                                                                                                                           |
| ------- | ------------- | ---------------------------- | --------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2019-12 | ESEC/FSE 2020 | TypeWriter OSS               | 208K      | Python     | "TypeWriter: Neural Type Prediction with Search-based Validation" [[paper](https://arxiv.org/abs/1912.03768)] [[data](http://software-lab.org/projects/TypeWriter/data.tar.gz)]                                                  |
| 2020-04 | PLDI 2020     | Typilus                      | 252K      | Python     | "Typilus: Neural Type Hints" [[paper](https://arxiv.org/abs/2004.10657)] [[data](https://github.com/typilus/typilus)]                                                                                                            |
| 2020-04 | ICLR 2020     | LambdaNet                    | 300 \*    | TypeScript | "LambdaNet: Probabilistic Type Inference using Graph Neural Networks" [[paper](https://arxiv.org/abs/2005.02161)] [[data](https://github.com/MrVPlusOne/LambdaNet)]                                                              |
| 2021-04 | MSR 2021      | ManyTypes4Py                 | 869K      | Python     | "ManyTypes4Py: A Benchmark Python Dataset for Machine Learning-based Type Inference" [[paper](https://arxiv.org/abs/2104.04706)] [[data](https://github.com/saltudelft/many-types-4-py-dataset)]                                 |
| 2022-10 | MSR 2022      | ManyTypes4TypeScript         | 9.1M      | TypeScript | "ManyTypes4TypeScript: a comprehensive TypeScript dataset for sequence-based type inference" [[paper](https://dl.acm.org/doi/10.1145/3524842.3528507)] [[data](https://huggingface.co/datasets/kevinjesse/ManyTypes4TypeScript)] |
| 2023-02 | ECOOP 2023    | TypeWeaver                   | 513 \*    | TypeScript | "Do Machine Learning Models Produce TypeScript Types That Type Check?" [[paper](https://arxiv.org/abs/2302.12163)] [[data](https://zenodo.org/records/7662708)]                                                                  |
| 2023-03 | ICLR 2023     | BetterTypes4Py/InferTypes4Py | 608K/4.6K | Python     | "TypeT5: Seq2seq Type Inference using Static Analysis" [[paper](https://arxiv.org/abs/2303.09564)] [[data](https://github.com/utopia-group/TypeT5)]                                                                              |
| 2023-05 | arXiv         | OpenTau                      | 744 \*    | TypeScript | "Type Prediction With Program Decomposition and Fill-in-the-Type Training" [[paper](https://arxiv.org/abs/2305.17145)] [[data](https://github.com/GammaTauAI/opentau)]                                                           |

\* These are project counts.

#### Commit Message Generation

- "On the Evaluation of Commit Message Generation Models: An Experimental Study", 2021-07, ICSME 2021, [[paper](https://arxiv.org/abs/2107.05373)]

| Date    | Venue                            | Benchmark       | Size       | Language                        | Source                                                                                                                                                                                                        |
| ------- | -------------------------------- | --------------- | ---------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2017-03 | ICPC 2017                        | unnamed         | 509K       | Java                            | "Towards Automatic Generation of Short Summaries of Commits" [[paper](https://arxiv.org/abs/1703.09603)] [[data](https://notredame.app.box.com/s/wghwpw46x41nu6iulm6qi8j42finuxni)]                           |
| 2017-04 | ACL 2017                         | CommitGen       | 153K       | Python, JS, C++, Java           | "A Neural Architecture for Generating Natural Language Descriptions from Source Code Changes" [[paper](https://arxiv.org/abs/1704.04856)] [[data](https://github.com/epochx/commitgen)]                       |
| 2017-08 | ASE 2017                         | CommitGen       | 32K/75K \* | Java                            | "Automatically Generating Commit Messages from Diffs using Neural Machine Translation" [[paper](https://arxiv.org/abs/1708.09492)] [[data](https://notredame.app.box.com/s/wghwpw46x41nu6iulm6qi8j42finuxni)] |
| 2018-09 | ASE 2018                         | NNGen           | 27K        | Java                            | "Neural-machine-translation-based commit message generation: how far are we?" [[paper](https://dl.acm.org/doi/10.1145/3238147.3238190)] [[data](https://github.com/Tbabm/nngen)]                              |
| 2019-05 | MSR 2019                         | PtrGNCMsg       | 64.9K      | Java                            | "Generating commit messages from diffs using pointer-generator network" [[paper](https://dl.acm.org/doi/10.1109/MSR.2019.00056)] [[data(https://zenodo.org/records/2593787)]]                                 |
| 2019-08 | IJCAI 2019                       | CoDiSum         | 90.7K      | Java                            | "Commit message generation for source code changes" [[paper](https://www.ijcai.org/proceedings/2019/552)] [[data](https://github.com/SoftWiser-group/CoDiSum)]                                                |
| 2019-12 | IEEE Trans. Software Eng.        | ATOM            | 160K       | Java                            | "ATOM: Commit Message Generation Based on Abstract Syntax Tree and Hybrid Ranking" [[paper](https://arxiv.org/abs/1912.02972)] [[data](https://zenodo.org/records/4077754#.X4K2b5MzZTY)]                      |
| 2021-05 | arXiv                            | CommitBERT      | 346K       | Python, PHP, Go, Java, JS, Ruby | "CommitBERT: Commit Message Generation Using Pre-Trained Programming Language Model" [[paper](https://arxiv.org/abs/2105.14242)] [[data](https://github.com/graykode/commit-autosuggestions)]                 |
| 2021-07 | ICSME 2021                       | MCMD            | 2.25M      | Java, C#, C++, Python, JS       | "On the Evaluation of Commit Message Generation Models: An Experimental Study" [[paper](https://arxiv.org/abs/2107.05373)] [[data](https://github.com/DeepSoftwareAnalytics/CommitMsgEmpirical)]              |
| 2021-07 | ACM Trans. Softw. Eng. Methodol. | CoRec           | 107K       | Java                            | "Context-aware Retrieval-based Deep Commit Message Generation" [[paper](https://dl.acm.org/doi/10.1145/3464689)] [[data](https://zenodo.org/records/3828107)]                                                 |
| 2023-07 | ASE 2023                         | ExGroFi         | 19263      | Java                            | "Delving into Commit-Issue Correlation to Enhance Commit Message Generation Models" [[paper](https://arxiv.org/abs/2308.00147)] [[data](https://zenodo.org/records/7885748#.ZFDT5IJBxD8)]                     |
| 2023-08 | ASE 2023                         | CommitChronicle | 10.7M      | 20                              | "From Commit Message Generation to History-Aware Commit Message Completion" [[paper](https://arxiv.org/abs/2308.07655)] [[data](https://github.com/JetBrains-Research/commit_message_generation)]             |

\* with/without verb-direct object filter

#### Repo-Level Coding

| Date    | Venue             | Benchmark        | Size                   | Language                       | Source                                                                                                                                                                                                                                         |
| ------- | ----------------- | ---------------- | ---------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023-03 | arXiv             | RepoEval         | 1600/1600/373 \*       | Python                         | "RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation" [[paper](https://arxiv.org/abs/2303.12570)] [[data](https://github.com/microsoft/CodeT/tree/main/RepoCoder)]                                          |
| 2023-06 | ICLR 2024         | RepoBench        | 890K/9M/43K $^\dagger$ | Python, Java                   | "RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems" [[paper](https://arxiv.org/abs/2306.03091)] [[data](https://github.com/Leolty/repobench)]                                                                              |
| 2023-06 | NeurIPS 2023      | PragmaticCode    | 880 \*\*               | Java                           | "Guiding Language Models of Code with Global Context using Monitors" [[paper](https://arxiv.org/abs/2306.10763)] [[data](https://github.com/microsoft/monitors4codegen)]                                                                       |
| 2023-06 | arXiv             | Stack-Repo       | 816K                   | Java                           | "RepoFusion: Training Code Models to Understand Your Repository" [[paper](https://arxiv.org/abs/2306.10998)] [[data](https://huggingface.co/RepoFusion)]                                                                                       |
| 2023-09 | ISMB 2024         | BioCoder         | 2269/460/460           | Python, Java                   | "BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models" [[paper](https://arxiv.org/abs/2308.16458)] [[data](https://huggingface.co/datasets/lilbillbiscuit/biocoder_public)]                                     |
| 2023-09 | arXiv             | CodePlan         | 645/21 $^\ddagger$     | C#/Python $^\ddagger$          | "CodePlan: Repository-level Coding using LLMs and Planning" [[paper](https://arxiv.org/abs/2309.12499)] [[data](https://aka.ms/CodePlan)]                                                                                                      |
| 2023-10 | arXiv             | SWE-Bench        | 2294                   | Python                         | "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" [[paper](https://arxiv.org/abs/2310.06770)] [[data](https://www.swebench.com/)]                                                                                             |
| 2023-10 | arXiv             | CrossCodeEval    | 9928                   | Python, Java, TypeScript, C#   | "CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion" [[paper](https://arxiv.org/abs/2310.11248)] [[data](https://crosscodeeval.github.io/)]                                                                    |
| 2024-03 | NeurIPS 2024      | EvoCodeBench     | 275                    | Python                         | "EvoCodeBench: An Evolving Code Generation Benchmark Aligned with Real-World Code Repositories" [[paper](https://arxiv.org/abs/2404.00599)] [[data](https://github.com/seketeam/EvoCodeBench)]                                                 |
| 2024-05 | ACL 2024 Findings | DevEval          | 1874                   | Python                         | "DevEval: A Manually-Annotated Code Generation Benchmark Aligned with Real-World Code Repositories" [[paper](https://arxiv.org/abs/2405.19856)] [[data](https://github.com/seketeam/DevEval)]                                                  |
| 2024-06 | arXiv             | JavaBench        | 389                    | Java                           | "Can AI Beat Undergraduates in Entry-level Java Assignments? Benchmarking Large Language Models on JavaBench" [[paper](https://arxiv.org/abs/2406.12902)] [[data](https://github.com/java-bench/JavaBench)]                                    |
| 2024-06 | arXiv             | HumanEvo         | 200/200                | Python/Java                    | "Towards more realistic evaluation of LLM-based code generation: an experimental study and beyond" [[paper](https://arxiv.org/abs/2406.06918)] [[data](https://github.com/DeepSoftwareAnalytics/EvoEval)]                                      |
| 2024-06 | arXiv             | RepoExec         | 355                    | Python                         | "REPOEXEC: Evaluate Code Generation with a Repository-Level Executable Benchmark" [[paper](https://arxiv.org/abs/2406.11927)]                                                                                                                  |
| 2024-06 | arXiv             | RES-Q            | 100                    | Python, JavaScript             | "RES-Q: Evaluating Code-Editing Large Language Model Systems at the Repository Scale" [[paper](https://arxiv.org/abs/2406.16801)] [[data](https://github.com/Qurrent-AI/RES-Q)]                                                                |
| 2024-08 | arXiv             | SWE-bench-java   | 91                     | Java                           | "SWE-bench-java: A GitHub Issue Resolving Benchmark for Java" [[paper](https://arxiv.org/abs/2408.14354)] [[data](https://github.com/multi-swe-bench/multi-swe-bench.github.io)]                                                               |
| 2024-10 | arXiv             | Codev-Bench      | 296                    | Python                         | "Codev-Bench: How Do LLMs Understand Developer-Centric Code Completion?" [[paper](https://arxiv.org/abs/2410.01353)] [[data](https://github.com/LingmaTongyi/Codev-Bench)]                                                                     |
| 2024-10 | ICLR 2025         | SWE-bench M      | 617                    | JavaScript                     | "SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?" [[paper](https://arxiv.org/abs/2410.03859)] [[data](https://www.swebench.com/multimodal)]                                                                         |
| 2024-10 | arXiv             | SWE-Bench+       | 548                    | Python                         | "SWE-Bench+: Enhanced Coding Benchmark for LLMs" [[paper](https://arxiv.org/abs/2410.06992)] [[data](https://zenodo.org/records/13879453)]                                                                                                     |
| 2024-10 | EMNLP 2024        | DA-Code          | 500                    | Python, Bash, SQL              | "DA-Code: Agent Data Science Code Generation Benchmark for Large Language Models" [[paper](https://arxiv.org/abs/2410.07331)] [[data](https://github.com/yiyihum/da-code)]                                                                     |
| 2024-10 | arXiv             | RepoCod          | 980                    | Python                         | "Can Language Models Replace Programmers? REPOCOD Says 'Not Yet'" [[paper](https://arxiv.org/abs/2410.21647)]                                                                                                                                  |
| 2024-10 | arXiv             | M2rc-Eval        | 5993 repos             | 18                             | "M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation" [[paper](https://arxiv.org/abs/2410.21157)] [[data](https://github.com/M2RC-Eval-Team/M2RC-Eval)]                                                              |
| 2024-11 | arXiv             | FAUN-Eval        | 300                    | Python, Java, JS, TS, Go       | "A Real-World Benchmark for Evaluating Fine-Grained Issue Solving Capabilities of Large Language Models" [[paper](https://arxiv.org/abs/2411.18019)] [[data](https://anonymous.4open.science/status/FAUN-Eval)]                                |
| 2024-12 | arXiv             | Commit0          | 54                     | Python                         | "Commit0: Library Generation from Scratch" [[paper](https://arxiv.org/abs/2412.01769)] [[data](https://github.com/commit-0/commit0)]                                                                                                           |
| 2024-12 | arXiv             | ExecRepoBench    | 1.2K                   | Python                         | "ExecRepoBench: Multi-level Executable Code Completion Evaluation" [[paper](https://arxiv.org/abs/2412.11990)] [[data](https://github.com/QwenLM/Qwen2.5-Coder/tree/main/qwencoder-eval/instruct/CodeArena)]                                   |
| 2025-01 | arXiv             | DI-Bench         | 581                    | Python, C#, Rust, JS           | "DI-BENCH: Benchmarking Large Language Models on Dependency Inference with Testable Repositories at Scale" [[paper](https://arxiv.org/abs/2501.13699)] [[data](https://github.com/Microsoft/DI-Bench)]                                         |
| 2025-02 | arXiv             | HackerRank-ASTRA | 65                     | frontend                       | "HackerRank-ASTRA: Evaluating Correctness & Consistency of Large Language Models on cross-domain multi-file project problems" [[paper](https://arxiv.org/abs/2502.00226)] [[data](https://huggingface.co/datasets/hackerrank/astra-benchmark)] |
| 2025-02 | arXiv             | SWE-Lancer       | 237                    | JS, TS                         | "SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?" [[paper](https://arxiv.org/abs/2502.12115)] [[data](https://github.com/openai/SWELancer-Benchmark)]                                            |
| 2025-03 | arXiv             | FEA-Bench        | 1401                   | Python                         | "FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation" [[paper](https://arxiv.org/abs/2503.06680)]                                                                                                |
| 2025-03 | arXiv             | DependEval       | 3.4K                   | 8                              | "DependEval: Benchmarking LLMs for Repository Dependency Understanding" [[paper](https://arxiv.org/abs/2503.06689)] [[data](https://github.com/ink7-sudo/DependEval)]                                                                          |
| 2025-03 | arXiv             | ProjectEval      | 20                     | Python                         | "ProjectEval: A Benchmark for Programming Agents Automated Evaluation on Project-Level Code Generation" [[paper](https://arxiv.org/abs/2503.07010)]                                                                                            |
| 2025-03 | arXiv             | RepoST-Eval      | 296                    | Python                         | "RepoST: Scalable Repository-Level Coding Environment Construction with Sandbox Testing" [[paper](https://arxiv.org/abs/2503.07358)] [[data](https://github.com/yiqingxyq/RepoST)]                                                             |
| 2025-04 | arXiv             | Multi-SWE-bench  | 1632                   | Java, JS, TS, Go, Rust, C, C++ | "Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving" [[paper](https://arxiv.org/abs/2504.02605)] [[data](https://github.com/multi-swe-bench/multi-swe-bench)]                                                                       |
| 2025-04 | arXiv             | SWE-PolyBench    | 2110                   | Java, JS, TS, Python           | "SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents" [[paper](https://arxiv.org/abs/2504.08703)] [[data](https://github.com/amazon-science/SWE-PolyBench)]                                             |
| 2025-04 | arXiv             | SecRepoBench     | 318                    | C/C++                          | "SecRepoBench: Benchmarking LLMs for Secure Code Generation in Real-World Repositories" [[paper](https://arxiv.org/abs/2504.21205)]                                                                                                            |
| 2025-05 | arXiv             | OmniGIRL         | 959                    | Python, JS, TS, Java           | "OmniGIRL: A Multilingual and Multimodal Benchmark for GitHub Issue Resolution" [[paper](https://arxiv.org/abs/2505.04606)] [[data](https://github.com/DeepSoftwareAnalytics/OmniGIRL)]                                                        |
| 2025-05 | arXiv             | SWE-Dev          | 14500                  | Python                         | "SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development" [[paper](https://arxiv.org/abs/2505.16975)] [[data](https://github.com/justLittleWhite/SWE-Dev)]                                                             |
| 2025-05 | arXiv             | SWE-rebench      | 21,000                 | Python                         | "SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents" [[paper](https://arxiv.org/abs/2505.20411)] [[data](https://huggingface.co/datasets/nebius/SWE-rebench)]                 |
| 2025-05 | arXiv             | AgentIssue-Bench | 50                     |                                | "Can Agents Fix Agent Issues?" [[paper](https://arxiv.org/abs/2505.20749)] [[data](https://github.com/alfin06/AgentIssue-Bench)]                                                                                                               |
| 2025-05 | arXiv             | GitGoodBench     | 900                    |                                | "GitGoodBench: A Novel Benchmark For Evaluating Agentic Performance On Git" [[paper](https://arxiv.org/abs/2505.22583)] [[data](https://github.com/JetBrains-Research/git-good-bench)]                                                         |

\*Line Completion/API Invocation Completion/Function Completion

$^\dagger$ Retrieval/Completion/Pipeline

\*\* File count

$^\ddagger$ Migration/Temporal Edit

#### Other tasks are coming soon!

## 9. Recommended Readings

30 papers as a primer on LLM.

|  Date   |         Keyword          | Paper                                                                                                                                                                                  | TL;DR                                                                                                                                |
| :-----: | :----------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 2014-09 |        Attention         | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)                                                                               | The original attention, proposed for encoder-decoder RNN                                                                             |
| 2015-08 |           BPE            | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)                                                                                        | Byte-pair encoding: split rare words into subword units                                                                              |
| 2017-06 |       Transformer        | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                                                                                                          | Replace LSTM with self-attention for long-range dependency and parallel training                                                     |
| 2017-10 | Mixed Precision Training | [Mixed Precision Training](https://arxiv.org/abs/1710.03740)                                                                                                                           | Store model weights in fp16 to save memory                                                                                           |
| 2018-04 |           GLUE           | [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)                                                              | A language understanding benchmark                                                                                                   |
| 2018-06 |           GPT            | [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | Pretraining-finetuning paradigm applied to Transformer decoder                                                                       |
| 2018-10 |           BERT           | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)                                                                   | Masked Language Modeling (MLM) applied to Transformer encoder for pretraining                                                        |
| 2019-02 |          GPT-2           | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)                                                  | GPT made larger (1.5B). They found language models implicitly learn about downstream tasks (such as translation) during pretraining. |
| 2019-05 |        SuperGLUE         | [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/abs/1905.00537)                                                                 | Another language understanding benchmark                                                                                             |
| 2019-07 |         RoBERTa          | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)                                                                                            | An optimized BERT                                                                                                                    |
| 2019-09 |       Megatron-LM        | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)                                                              | Model parallelism                                                                                                                    |
| 2019-10 |           ZeRO           | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)                                                                               | Memory-efficient distributed optimization                                                                                            |
| 2019-10 |            T5            | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)                                                                  | Transformer encoder-decoder pretrained with an MLM-like denoising objective                                                          |
| 2020-05 |          GPT-3           | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)                                                                                                              | By training an even larger version of GPT-2 (175B), they discovered a new learning paradigm: In-Context Learning (ICL)               |
| 2020-09 |           MMLU           | [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)                                                                                                 | A world-knowledge and complex reasoning benchmark                                                                                    |
| 2020-12 |           Pile           | [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)                                                                                   | A diverse pretraining dataset                                                                                                        |
| 2021-04 |           RoPE           | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)                                                                                      | Rotary position embedding                                                                                                            |
| 2021-06 |           LoRA           | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)                                                                                                 | Memory-efficient finetuning                                                                                                          |
| 2021-09 |           FLAN           | [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)                                                                                                   | Instruction-finetuning                                                                                                               |
| 2021-10 |            T0            | [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)                                                                                  | Also instruction finetuning, but applied to the much smaller T5                                                                      |
| 2021-12 |          Gopher          | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)                                                                         | A 280B LLM with comprehensive experiments                                                                                            |
| 2022-01 |           CoT            | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)                                                                              | Chain-of-Though reasoning                                                                                                            |
| 2022-03 |       InstructGPT        | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)                                                                                | GPT-3 instruction finetuned with RLHF (reinforcement learning from human feedback)                                                   |
| 2022-03 |        Chinchilla        | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)                                                                                                     | A smaller (70B) version of Gopher that's pretrained on more data                                                                     |
| 2022-04 |           PaLM           | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)                                                                                                      | The largest dense model ever (540B)                                                                                                  |
| 2022-05 |        0-shot CoT        | [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)                                                                                                      | Tell LLMs to think step by step, and they can actually do it                                                                         |
| 2022-06 |     Emergent Ability     | [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)                                                                                                        | A review on emergent abilities                                                                                                       |
| 2022-10 |           Flan           | [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)                                                                                                      | Consolidate all the existing instruction tuning datasets, and you get SOTA                                                           |
| 2022-11 |          BLOOM           | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)                                                                                    | The largest open-source dense LLM, trained on 46 languages, with detailed discussion about training and evaluation                   |
| 2022-12 |      Self-Instruct       | [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)                                                                           | Instruction tuning using LLM-generated data                                                                                          |

This list aims to provide the essential background for understanding current LLM technologies, and thus excludes more recent models such as [LLaMA](https://arxiv.org/abs/2302.13971), [GPT-4](https://arxiv.org/abs/2303.08774) or [PaLM 2](https://arxiv.org/abs/2305.10403). For comprehensive reviews on these more general topics, we refer to other sources such as [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM), [Awesome AIGC Tutorials](https://github.com/luban-agi/Awesome-AIGC-Tutorials), or for LLM applications in other specific domains: [Awesome Domain LLM](https://github.com/luban-agi/Awesome-Domain-LLM), [Awesome Tool Learning](https://github.com/luban-agi/Awesome-Tool-Learning#awesome-tool-learning), [Awesome-LLM-MT](https://github.com/hsing-wang/Awesome-LLM-MT), [Awesome Education LLM](https://github.com/Geralt-Targaryen/Awesome-Education-LLM).

## Citation

If you find this repo or our survey helpful, please consider citing us:

```
@article{zhang2024unifying,
   title={Unifying the Perspectives of {NLP} and Software Engineering: A Survey on Language Models for Code},
   author={Ziyin Zhang and Chaoyu Chen and Bingchang Liu and Cong Liao and Zi Gong and Hang Yu and Jianguo Li and Rui Wang},
   journal={Transactions on Machine Learning Research},
   issn={2835-8856},
   year={2024},
   url={https://openreview.net/forum?id=hkNnGqZnpa},
   note={}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=codefuse-ai/Awesome-Code-LLM&type=Date)](https://star-history.com/#codefuse-ai/Awesome-Code-LLM&Date)

## Join US

### Recruitment

<details>
<summary>English version</summary>
We are the AI Native team within the Platform Technology Business Group at Ant Group, dedicated to the intelligentization of Ant Group's platform engineering. Established for over three years, our team has played a pivotal role in supporting the intelligent operation and maintenance of Ant Group's cloud computing infrastructure. Our mission is to build algorithm services and platforms with a wide user base through world-class technological innovation and impact, supporting the implementation of internal and external products and businesses.
Embracing an innovation-driven ethos, our team not only supports business implementation but also propels technological influence. Over the past three years, we have published more than 20 papers at top conferences like ICLR, NeurIPS, KDD, and ACL. Our innovative business outcomes have earned us two Ant Technology's highest T-Star awards and one SuperMA award from Ant Group. Our open-source project CodeFuse has received 4K stars as of February 2024, and our models have been downloaded over 1.5 million times on Huggingface and Modelscope.

**We are on the lookout for top talents to join our vibrant team! If you're eager to develop your career in an environment filled with energy, innovation, and a culture of excellence, we welcome you to explore our career opportunities for both campus and experienced hires. Join us and be a part of creating the next milestone in the industry.**

**Campus Recruitment**: https://hrrecommend.antgroup.com/guide.html?code=8uoP5mlus5DqQYbE_EnqcE2FD5JZH21MwvMUIb9mb6X3osXPuBraG54SyM8GLn_7

**Experienced Hires**: https://talent.antgroup.com/off-campus-position?positionId=1933830

</details>

<details>
<summary>中文版</summary>
我们是平台技术事业群 AI Native 团队，负责蚂蚁蚂蚁集团平台工程的智能化，团队成立 3 年多以来，支持了蚂蚁集团云计算基础设施智能化运维的升级改造。团队的 Mission 是，通过世界级的技术创新和影响，构建有广泛用户的算法服务和平台，支撑内外部产品和业务落地。团队秉承创新基因，在支撑业务落地的同时，推动技术影响。3 年以来在 ICLR、NeurIPS、KDD、ACL 等顶会发表论文 20 余篇，创新业务结果获得两次蚂蚁技术最高奖 T-Star，1 次蚂蚁集团最高奖 SuperMA。开源项目 CodeFuse 获得 4K 点赞(2024 年 2 月)，Huggingface 和 modelscope 上模型累积下载量超过 150 万次。

**我们正在寻找行业中的佼佼者加入我们的团队！如果您希望在一个充满活力、创新和卓越文化的环境中发展您的职业生涯，欢迎您查看我们的社招&校招机会，加入我们，一起创造下一个行业里程碑。**

**校招**：https://hrrecommend.antgroup.com/guide.html?code=8uoP5mlus5DqQYbE_EnqcE2FD5JZH21MwvMUIb9mb6X3osXPuBraG54SyM8GLn_7

**社招**：https://talent.antgroup.com/off-campus-position?positionId=1933830

</details>

### Contact Us (WeChat)

永久企业微信交流群：

<p align='center'>
<img src='imgs/wechat.PNG' style='width: 40%; '>
</p>
