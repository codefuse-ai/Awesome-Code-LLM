# Awesome-Code-LLM

This is the repo for our [survey](https://arxiv.org/abs/2311.07989) - a comprehensive review of LLM researches for code. Works in each category are ordered chronologically. If you have a basic understanding of machine learning but are new to NLP, we also provide a list of recommended readings in [section 4](#4-recommended-readings).

<p align='center'>
<img src='imgs/overview.png' style='width: 80%; '>
</p>

## Table of Contents

1. [Surveys](#1-surveys)

2. [Models](#2-models)

   2.1 [Off-the-Shelf LLM](#21-off-the-shelf-llm)

   2.2 [Existing LLM Further Trained on Code](#22-existing-llm-further-trained-on-code)

   2.3 [General Pretraining on Code](#23-general-pretraining-on-code)

   - [Encoder](#encoder)
   - [Decoder](#decoder)
   - [Encoder-Decoder](#encoder-decoder)
   - [UniLM](#unilm)

   <!-- prettier ignore -->

   2.4 [Instruction Fine-Tuning on Code](#24-instruction-fine-tuning-on-code)

   2.5 [Reinforcement Learning on Code](#25-reinforcement-learning-on-code)

3. [Datasets](#3-datasets)

   3.1 [Pretraining](#31-pretraining)

   3.2 [Benchmarks](#32-benchmarks)

   - [Program Synthesis](#program-synthesis)
   - [Code Translation](#code-translation)
   - [Program Repair](#program-repair)
   - [Code Summarization](#code-summarization)
   - [Code Retrieval](#code-retrieval)
   - [Type Inference](#type-inference)
   - [Repo-Level Coding](#repo-level-coding)

4. [Recommended Readings](#4-recommended-readings)

5. [Citation](#citation)

## 1. Surveys

We list six recent surveys on similar topics. While they are all about language models for code, the first two focus on NLP side, and the later four focus on SE side.

1. "Large Language Models Meet NL2Code: A Survey", 2022-12, ACL 2023, [[paper](https://arxiv.org/abs/2212.09420)]

2. "A Survey on Pretrained Language Models for Neural Code Intelligence", 2022-12, arXiv, [[paper](https://arxiv.org/abs/2212.10079)]

3. "An Empirical Comparison of Pre-Trained Models of Source Code", 2023-02, ICSE 2023, [[paper](https://arxiv.org/abs/2302.04026)]

4. "Large Language Models for Software Engineering: A Systematic Literature Review", 2023-08, arXiv, [[paper](https://arxiv.org/abs/2308.10620)]

5. "Towards an Understanding of Large Language Models in Software Engineering Tasks", 2023-08, arXiv, [[paper](https://arxiv.org/abs/2308.11396)]

6. "Pitfalls in Language Models for Code Intelligence: A Taxonomy and Survey", 2023-10, arXiv, [[paper](https://arxiv.org/abs/2310.17903)]

## 2. Models

### 2.1 Off-the-Shelf LLM

These LLMs are not specifically trained for code, but have demonstrated varying coding capability.

1. **LaMDA**: "LaMDA: Language Models for Dialog Applications", 2022-01, arXiv, [[paper](https://arxiv.org/abs/2201.08239)]

2. **PaLM**: "PaLM: Scaling Language Modeling with Pathways", 2022-04, arXiv, [[paper](https://arxiv.org/abs/2204.02311)]

3. **GPT-NeoX**: "GPT-NeoX-20B: An Open-Source Autoregressive Language Model", 2022-04, ACL 2022 Workshop on Challenges & Perspectives in Creating Large Language Models, [[paper](https://arxiv.org/abs/2204.06745)] [[repo](https://github.com/EleutherAI/gpt-neox)]

4. **BLOOM**: "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model", 2022-11, arXiv, [[paper](https://arxiv.org/abs/2211.05100)] [[model](https://huggingface.co/models?search=bigscience/bloom)]

5. **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models", 2023-02, arXiv, [[paper](https://arxiv.org/abs/2302.13971)]

6. **GPT-4**: "GPT-4 Technical Report", 2023-03, arXiv, [[paper](https://arxiv.org/abs/2303.08774)]

7. **LLaMA 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023-07, arXiv, [[paper](https://arxiv.org/abs/2307.09288)] [[repo](https://github.com/facebookresearch/llama)]

8. **Phi-1.5**: "Textbooks Are All You Need II: phi-1.5 technical report", 2023-09, arXiv, [[paper](https://arxiv.org/abs/2309.05463)] [[model](https://huggingface.co/microsoft/phi-1_5)]

9. **Baichuan 2**: "Baichuan 2: Open Large-scale Language Models", 2023-09, arXiv, [[paper](https://arxiv.org/abs/2309.10305)] [[repo](https://github.com/baichuan-inc/Baichuan2)]

10. **Qwen**: "Qwen Technical Report", 2023-09, arXiv, [[paper](https://arxiv.org/abs/2309.16609)] [[repo](https://github.com/QwenLM/Qwen)]

### 2.2 Existing LLM Further Trained on Code

These models are general-purpose LLMs further pretrained on code-related data.

1. **Codex** (GPT-3): "Evaluating Large Language Models Trained on Code", 2021-07, arXiv, [[paper](https://arxiv.org/abs/2107.03374)]

2. **PaLM Coder** (PaLM): "PaLM: Scaling Language Modeling with Pathways", 2022-04, arXiv, [[paper](https://arxiv.org/abs/2204.02311)]

3. **Minerva** (PaLM): "Solving Quantitative Reasoning Problems with Language Models", 2022-06, arXiv, [[paper](https://arxiv.org/abs/2206.14858)]

4. **PaLM 2 \*** (PaLM 2): "PaLM 2 Technical Report", 2023-05, arXiv, [[paper](https://arxiv.org/abs/2305.10403)]

5. **Code LLaMA** (LLaMA 2): "Code Llama: Open Foundation Models for Code", 2023-08, arXiv, [[paper](https://arxiv.org/abs/2308.12950)] [[repo](https://github.com/facebookresearch/codellama)]

### 2.3 General Pretraining on Code

These models are Transformer encoders, decoders, and encoder-decoders pretrained from scratch using existing objectives for general language modeling.

<p align='center'>
<img src='imgs/model_detail.png' style='width: 90%; '>
</p>

#### Encoder

1. **CuBERT** (MLM + NSP): "Learning and Evaluating Contextual Embedding of Source Code", 2019-12, ICML 2020, [[paper](https://arxiv.org/abs/2001.00059)] [[repo](https://github.com/google-research/google-research/tree/master/cubert)]

2. **CodeBERT** (MLM + RTD): "CodeBERT: A Pre-Trained Model for Programming and Natural Languages", 2020-02, EMNLP findings 2020, [[paper](https://arxiv.org/abs/2002.08155)] [[repo](https://github.com/microsoft/CodeBERT)]

3. **GraphCodeBERT** (MLM + DFG Edge Prediction + DFG Node Alignment): "GraphCodeBERT: Pre-training Code Representations with Data Flow", 2020-09, ICLR 2021, [[paper](https://arxiv.org/abs/2009.08366)] [[repo](https://github.com/microsoft/CodeBERT)]

4. **SynCoBERT** (MLM + Identifier Prediction + AST Edge Prediction + Contrastive Learning): "SynCoBERT: Syntax-Guided Multi-Modal Contrastive Pre-Training for Code Representation", 2021-08, arXiv, [[paper](https://arxiv.org/abs/2108.04556)]

5. **DISCO** (MLM + Node Type MLM + Contrastive Learning): "Towards Learning (Dis)-Similarity of Source Code from Program Contrasts", 2021-q0, ACL 2022, [[paper](https://arxiv.org/abs/2110.03868)]

6. **Code-MVP** (MLM + Type Inference + Contrastive Learning): "CODE-MVP: Learning to Represent Source Code from Multiple Views with Contrastive Pre-Training", 2022-05, NAACL 2022 Technical Track, [[paper](https://arxiv.org/abs/2205.02029)]

#### Decoder

1. **GPT-C** (CLM): "IntelliCode Compose: Code Generation Using Transformer", 2020-05, ESEC/FSE 2020, [[paper](https://arxiv.org/abs/2005.08025)]

2. **CodeGPT** (CLM): "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation", 2021-02, NeurIPS Datasets and Benchmarks 2021, [[paper](https://arxiv.org/abs/2102.04664)] [[repo](https://github.com/microsoft/CodeXGLUE)]

3. **CodeParrot** (CLM), 2021-12, [[blog](https://huggingface.co/blog/codeparrot)]

4. **PolyCoder** (CLM): "A Systematic Evaluation of Large Language Models of Code", 2022-02, - DL4C@ICLR 2022, [[paper](https://arxiv.org/abs/2202.13169)] [[repo](https://github.com/VHellendoorn/Code-LMs)]

5. **CodeGen** (CLM): "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis", 2022-03, ICLR 2023, [[paper](https://arxiv.org/abs/2203.13474)] [[repo](https://github.com/salesforce/CodeGen)]

6. **InCoder** (Causal Masking): "InCoder: A Generative Model for Code Infilling and Synthesis", 2022-04, ICLR 2023, [[paper](https://arxiv.org/abs/2204.05999)] [[repo](https://github.com/dpfried/incoder)]

7. **PyCodeGPT** (CLM): "CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation", 2022-06, IJCAI-ECAI 2022, [[paper](https://arxiv.org/abs/2206.06888)] [[repo](https://github.com/microsoft/PyCodeGPT)]

8. **PanGu-Coder** (CLM): "PanGu-Coder: Program Synthesis with Function-Level Language Modeling", 2022-07, arxiv, [[paper](https://arxiv.org/abs/2207.11280)]

9. **SantaCoder** (FIM): "SantaCoder: don't reach for the stars!", 2023-01, arXiv, [[paper](https://arxiv.org/abs/2301.03988)] [[model](https://huggingface.co/bigcode/santacoder)]

10. **CodeGeeX** (CLM): "CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X", 2023-03, arxiv, [[paper](https://arxiv.org/abs/2303.17568)] [[repo](https://github.com/THUDM/CodeGeeX)]

11. **StarCoder** (FIM): "StarCoder: may the source be with you!", 2023-05, arXiv, [[paper](https://arxiv.org/abs/2305.06161)] [[model](https://huggingface.co/bigcode/starcoder)]

12. **Phi-1** (CLM): "Textbooks Are All You Need", 2023-06, arxiv, [[paper](https://arxiv.org/abs/2306.11644)] [[model](https://huggingface.co/microsoft/phi-1)]

13. **CodeFuse** (CLM): "CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model", 2023-10, arxiv, [[paper](https://arxiv.org/abs/2310.06266)] [[model](https://huggingface.co/codefuse-ai/CodeFuse-13B)]

14. **CodeShell** (CLM), 2023-10, [[repo](https://github.com/WisdomShell/codeshell)]

15. **DeepSeek Coder** (CLM), 2023-10, [[repo](https://github.com/deepseek-ai/DeepSeek-Coder)]

#### Encoder-Decoder

1. **PyMT5** (Span Corruption): "PyMT5: multi-mode translation of natural language and Python code with transformers", 2020-10, EMNLP 2020, [[paper](https://arxiv.org/abs/2010.03150)]

2. **Mastropaolo et al.** (MLM + Deobfuscation): "DOBF: A Deobfuscation Pre-Training Objective for Programming Languages", 2021-02, ICSE 2021, [[paper](https://arxiv.org/abs/2102.02017)] [[repo](https://github.com/antonio-mastropaolo/TransferLearning4Code)]

3. **DOBF** (Span Corruption): "Studying the Usage of Text-To-Text Transfer Transformer to Support Code-Related Tasks", 2021-02, NeurIPS 2021, [[paper](https://arxiv.org/abs/2102.07492)] [[repo](https://github.com/facebookresearch/CodeGen/blob/main/docs/dobf.md)]

4. **PLBART** (DAE): "Unified Pre-training for Program Understanding and Generation", 2021-03, NAACL 2021, [[paper](https://arxiv.org/abs/2103.06333)] [[repo](https://github.com/wasiahmad/PLBART)]

5. **CodeT5** (Span Corruption + Identifier Tagging + Masked Identifier Prediction + Text2Code + Code2Text): "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation", 2021-09, EMNLP 2021, [[paper](https://arxiv.org/abs/2109.00859)] [[repo](https://github.com/salesforce/CodeT5)]

6. **SPT-Code** (Span Corruption + NSP + Method Name Prediction): "SPT-Code: Sequence-to-Sequence Pre-Training for Learning Source Code Representations", 2022-01, ICSE 2022 Technical Track, [[paper](https://arxiv.org/abs/2201.01549)]

7. **AlphaCode** (MLM + CLM): "Competition-Level Code Generation with AlphaCode", 2022-02, Science, [[paper](https://arxiv.org/abs/2203.07814)] [[arxiv](https://deepmind.google/discover/blog/competitive-programming-with-alphacode/)]

8. **NatGen** (Code Naturalization): "NatGen: Generative pre-training by "Naturalizing" source code", 2022-06, ESEC/FSE 2022, [[paper](https://arxiv.org/abs/2206.07585)] [[repo](https://github.com/saikat107/NatGen)]

9. **CodeT5+** (Span Corruption + CLM + Text-Code Contrastive Learning + Text-Code Translation): "CodeT5+: Open Code Large Language Models for Code Understanding and Generation", 2023-05, arXiv, [[paper](https://arxiv.org/abs/2305.07922)] [[repo](https://github.com/salesforce/CodeT5)]

#### UniLM

1. **CugLM** (MLM + NSP + CLM): "Multi-task Learning based Pre-trained Language Model for Code Completion", 2020-12, ASE 2020, [[paper](https://arxiv.org/abs/2012.14631)]

2. **UniXcoder** (MLM + NSP + CLM + Span Corruption + Contrastive Learning + Code2Text): "UniXcoder: Unified Cross-Modal Pre-training for Code Representation", 2022-03, ACL 2022, [[paper](https://arxiv.org/abs/2203.03850)] [[repo](https://github.com/microsoft/CodeBERT)]

### 2.4 Instruction Fine-Tuning on Code

These models apply Instruction Fine-Tuning techniques to enhance the capacities of Code LLMs.

1. **WizardCoder** (StarCoder + Evol-Instruct): "WizardCoder: Empowering Code Large Language Models with Evol-Instruct", 2023-06, arXiv, [[paper](https://arxiv.org/abs/2306.08568)] [[repo](https://github.com/nlpxucan/WizardLM)]

2. **PanGu-Coder 2** (StarCoder + Evol-Instruct + RRTF): "PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback", 2023-07, arXiv, [[paper](https://arxiv.org/abs/2307.14936)]

3. **OctoCoder** (StarCoder) / **OctoGeeX** (CodeGeeX2): "OctoPack: Instruction Tuning Code Large Language Models", 2023-08, arXiv, [[paper](https://arxiv.org/abs/2308.07124)] [[repo](https://github.com/bigcode-project/octopack)]

4. **MFTCoder** (Code LLaMA): "MFTCoder: Boosting Code LLMs with Multitask Fine-Tuning", 2023-11, arXiv, [[paper](https://arxiv.org/abs/2311.02303)] [[repo](https://github.com/codefuse-ai/MFTCoder)]

### 2.5 Reinforcement Learning on Code

1. **CompCoder**: "Compilable Neural Code Generation with Compiler Feedback", 2022-03, ACL 2022, [[paper](https://arxiv.org/abs/2203.05132)]

2. **CodeRL**: "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning", 2022-07, NeurIPS 2022, [[paper](https://arxiv.org/abs/2207.01780)] [[repo](https://github.com/salesforce/CodeRL)]

3. **PPOCoder**: "Execution-based Code Generation using Deep Reinforcement Learning", 2023-01, TMLR 2023, [[paper](https://arxiv.org/abs/2301.13816)] [[repo](https://github.com/reddy-lab-code-research/PPOCoder)]

4. **RLTF**: "RLTF: Reinforcement Learning from Unit Test Feedback", 2023-07, arXiv, [[paper](https://arxiv.org/abs/2307.04349)] [[repo](https://github.com/Zyq-scut/RLTF)]

## 3. Datasets

### 3.1 Pretraining

1. **CodeSearchNet**: "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search", 2019-09, arXiv, [[paper](https://arxiv.org/abs/1909.09436)] [[repo](https://github.com/github/CodeSearchNet)] [[data](https://huggingface.co/datasets/code_search_net)]

2. **The Pile**: "The Pile: An 800GB Dataset of Diverse Text for Language Modeling", 2020-12, arXiv, [[paper](https://arxiv.org/abs/2101.00027)] [[data](https://pile.eleuther.ai/)]

3. **CodeParrot**, 2022-02, [[data](https://huggingface.co/datasets/codeparrot/github-code)]

4. **The Stack**: "The Stack: 3 TB of permissively licensed source code", 2022-11, arXiv, [[paper](https://arxiv.org/abs/2211.15533)] [[data](https://huggingface.co/datasets/bigcode/the-stack)]

5. **ROOTS**: "The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset", 2023-03, NeurIPS 2022 Datasets and Benchmarks Track, [[paper](https://arxiv.org/abs/2303.03915)] [[data](https://huggingface.co/datasets?search=bigscience-data/roots)]

### 3.2 Benchmarks

1. **CodeXGLUE**: "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation", 2021-02, NeurIPS Datasets and Benchmarks 2021, [[paper](https://arxiv.org/abs/2102.04664)] [[repo](https://github.com/microsoft/CodeXGLUE)] [[data](https://huggingface.co/datasets?search=code_x_glue)]

#### Program Synthesis

| Date    | Venue                            | Benchmark          | Size      | Language                        | Source                                                                                                                                                                                                                                                                                       |
| ------- | -------------------------------- | ------------------ | --------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2018-08 | EMNLP 2018                       | CONCODE            | 104K      | Java                            | "Mapping Language to Code in Programmatic Context" [[paper](https://arxiv.org/abs/1808.09588)] [[data](https://github.com/sriniiyer/concode)]                                                                                                                                                |
| 2021-05 | NeurIPS 2021                     | APPS               | 10000     | Python                          | "Measuring Coding Challenge Competence With APPS" [[paper](https://arxiv.org/abs/2105.09938)] [[data](https://github.com/hendrycks/apps)]                                                                                                                                                    |
| 2021-07 | arXiv                            | HumanEval          | 164       | Python                          | "Evaluating Large Language Models Trained on Code" [[paper](https://arxiv.org/abs/2107.03374)] [[data](https://github.com/openai/human-eval)]                                                                                                                                                |
| 2021-08 | arXiv                            | MBPP/MathQA-Python | 974/23914 | Python                          | "Program Synthesis with Large Language Models" [[paper](https://arxiv.org/abs/2108.07732)] [[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)] [[MathQA-Python](https://github.com/google/trax/blob/master/trax/examples/MathQA_Python_generation_notebook.ipynb)] |
| 2022-06 | arXiv                            | AixBench           | 336       | Java                            | "AixBench: A Code Generation Benchmark Dataset" [[paper](https://arxiv.org/abs/2206.13179)] [[data](https://github.com/aixcoder-plugin/nl2code-dataset)]                                                                                                                                     |
| 2022-08 | IEEE Trans. Software Engineering | MultiPL-E          |           |                                 | "MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation", [[paper](https://arxiv.org/abs/2208.08227)] [[data](https://github.com/nuprl/MultiPL-E)]                                                                                                             |
| 2022-11 | arXiv                            | DS-1000            | 1000      | Python                          | "DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation" [[paper](https://arxiv.org/abs/2211.11501)] [[data](https://github.com/xlang-ai/DS-1000)]                                                                                                                       |
| 2023-02 | arXiv                            | CoderEval          | 460       | Python, Java                    | "CoderEval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models" [[paper](https://arxiv.org/abs/2302.00288)] [[data](https://github.com/CoderEval/CoderEval)]                                                                                                        |
| 2023-03 | arXiv                            | HumanEval-X        | 820       | Python, C++, Java, JS, Go       | "CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X" [[paper](https://arxiv.org/abs/2303.17568)] [[data](https://hub.docker.com/r/codegeex/codegeex)]                                                                                            |
| 2023-05 | arXiv                            | HumanEval+         | 164       | Python                          | "Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation" [[paper](https://arxiv.org/abs/2305.01210)] [[data](https://github.com/evalplus/evalplus)]                                                                              |
| 2023-08 | arXiv                            | HumanEvalPack      | 984       | Python, JS, Go, Java, C++, Rust | "OctoPack: Instruction Tuning Code Large Language Models" [[paper](https://arxiv.org/abs/2308.07124)] [[data](https://huggingface.co/datasets/bigcode/humanevalpack)]                                                                                                                        |
| 2023-09 | arXiv                            | CodeApex           | 476       | C++                             | "CodeApex: A Bilingual Programming Evaluation Benchmark for Large Language Models" [[paper](https://arxiv.org/abs/2309.01940)] [[data](https://github.com/APEXLAB/CodeApex)]                                                                                                                 |

#### Code Translation

| Date    | Venue                                | Benchmark                | Size  | Language                          | Source                                                                                                                                                                                                          |
| ------- | ------------------------------------ | ------------------------ | ----- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2020-06 | NeurIPS 2020                         | Transcoder GeeksforGeeks | 1.4K  | C++, Java, Python                 | "Unsupervised Translation of Programming Languages" [[paper](https://arxiv.org/abs/2006.03511)] [[data](https://github.com/facebookresearch/TransCoder)]                                                        |
| 2021-02 | NeurIPS Datasets and Benchmarks 2021 | CodeTrans                | 11.8K | Java, C#                          | "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation" [[paper](https://arxiv.org/abs/2102.04664)] [[data](https://huggingface.co/datasets/code_x_glue_cc_code_to_code_trans)] |
| 2021-08 | ACL 2023 Findings                    | Avatar                   | 9515  | Java, Python                      | "AVATAR: A Parallel Corpus for Java-Python Program Translation" [[paper](https://arxiv.org/abs/2108.11590)] [[data](https://github.com/wasiahmad/AVATAR)]                                                       |
| 2022-06 | AAAI 2022                            | CoST                     | 132K  | C++, Java, Python, C#, JS, PHP, C | "Multilingual Code Snippets Training for Program Translation" [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21434)] [[data](https://github.com/reddy-lab-code-research/MuST-CoST)]                  |
| 2022-06 | arXiv                                | XLCoST                   | 567K  | C++, Java, Python, C#, JS, PHP, C | "XLCoST: A Benchmark Dataset for Cross-lingual Code Intelligence" [[paper](https://arxiv.org/abs/2206.08474)] [[data](https://github.com/reddy-lab-code-research/XLCoST)]                                       |
| 2023-03 | arXiv                                | HumanEval-X              | 1640  | Python, C++, Java, JS, Go         | "CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X" [[paper](https://arxiv.org/abs/2303.17568)] [[data](https://github.com/THUDM/CodeGeeX)]                        |
| 2023-08 | arXiv                                | G-TransEval              | 4000  | C++, Java, C#, JS, Python         | "On the Evaluation of Neural Code Translation: Taxonomy and Benchmark" [[paper](https://arxiv.org/abs/2308.08961)] [[data](https://github.com/PolyEval/G-TransEval)]                                            |

#### Program Repair

| Date    | Venue                            | Benchmark           | Size     | Language                        | Source                                                                                                                                                                                                                    |
| ------- | -------------------------------- | ------------------- | -------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2014-07 | ISSTA 2014                       | Defects4J           | 357      | Java                            | "Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies for Java Programs" [[paper](https://dl.acm.org/doi/10.1145/2610384.2628055)] [[data](https://github.com/rjust/defects4j)]                  |
| 2015-12 | IEEE Trans. Software Engineering | ManyBugs/IntroClass | 185/998  | C                               | "The ManyBugs and IntroClass Benchmarks for Automated Repair of C Programs" [[paper](https://ieeexplore.ieee.org/document/7153570)] [[data](https://repairbenchmarks.cs.umass.edu/)]                                      |
| 2016-11 | FSE 2016                         | BugAID              | 105K     | JS                              | "Discovering Bug Patterns in JavaScript" [[paper](https://dl.acm.org/doi/10.1145/2950290.2950308)] [[data](https://salt.ece.ubc.ca/software/bugaid/)]                                                                     |
| 2017-02 | AAAI 2017                        | DeepFix             | 6971     | C                               | "DeepFix: Fixing Common C Language Errors by Deep Learning" [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10742)] [[data](https://bitbucket.org/iiscseal/deepfix/src/master/)]                                |
| 2017-05 | ICSE-C 2017                      | Codeflaws           | 3902     | C                               | "DeepFix: Fixing Common C Language Errors by Deep Learning" [[paper](https://dl.acm.org/doi/10.1109/ICSE-C.2017.76)] [[data](https://codeflaws.github.io/)]                                                               |
| 2017-10 | SPLASH 2017                      | QuixBugs            | 80       | Java, Python                    | "QuixBugs: a multi-lingual program repair benchmark set based on the quixey challenge" [[paper](https://dl.acm.org/doi/10.1145/3135932.3135941)] [[data](https://github.com/jkoppel/QuixBugs)]                            |
| 2018-12 | ACM Trans. Softw. Eng. Methodol. | BFP                 | 124K     | Java                            | "An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation" [[paper](https://arxiv.org/abs/1812.08693)] [[data](https://sites.google.com/view/learning-fixes)]                         |
| 2019-01 | ICSE 2019                        | unnamed             | 21.8K \* | Java                            | "On Learning Meaningful Code Changes via Neural Machine Translation" [[paper](https://arxiv.org/abs/1901.09102)] [[data](https://sites.google.com/view/learning-codechanges)]                                             |
| 2019-05 | MSR 2020                         | ManySStuBs4J        | 154K     | Java                            | "How Often Do Single-Statement Bugs Occur? The ManySStuBs4J Dataset" [[paper](https://arxiv.org/abs/1905.13334)] [[data](https://github.com/mast-group/mineSStuBs)]                                                       |
| 2019-11 | ASE 2019                         | Refactory           | 1783     | Python                          | "Re-factoring based program repair applied to programming assignments" [[paper](https://dl.acm.org/doi/10.1109/ASE.2019.00044)] [[data](https://github.com/githubhuyang/refactory)]                                       |
| 2020-07 | ISSTA 2020                       | CoCoNut             | 24M      | Java, Python, C, JS             | "CoCoNuT: combining context-aware neural translation models using ensemble for program repair" [[paper](https://dl.acm.org/doi/10.1145/3395363.3397369)] [[data](https://github.com/lin-tan/CoCoNut-Artifact)]            |
| 2020-11 | ESEC/FSE 2020                    | BugsInPy            | 493      | Python                          | "BugsInPy: A Database of Existing Bugs in Python Programs to Enable Controlled Testing and Debugging Studies" [[paper](https://dl.acm.org/doi/abs/10.1145/3368089.3417943)] [[data](https://github.com/soarsmu/BugsInPy)] |
| 2021-07 | ICML 2021                        | TFix                | 105K     | JS                              | "TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer" [[paper](https://proceedings.mlr.press/v139/berabi21a.html)] [[data](https://github.com/eth-sri/TFix)]                                              |
| 2022-11 | ESEC/FSE 2022                    | TypeBugs            | 93       | Python                          | "PyTER: Effective Program Repair for Python Type Errors" [[paper](https://dl.acm.org/doi/abs/10.1145/3540250.3549130)] [[data](https://github.com/kupl/PyTER)]                                                            |
| 2023-08 | arXiv                            | HumanEvalPack       | 984      | Python, JS, Go, Java, C++, Rust | "OctoPack: Instruction Tuning Code Large Language Models" [[paper](https://arxiv.org/abs/2308.07124)] [[data](https://huggingface.co/datasets/bigcode/humanevalpack)]                                                     |

\* This is a code-change dataset, and only a subset therein concern bug fixing.

#### Code Summarization

| Date    | Venue       | Benchmark     | Size    | Language                        | Source                                                                                                                                                                                                                             |
| ------- | ----------- | ------------- | ------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2016-08 | ACL 2016    | CODE-NN       | 66K/32K | C#/SQL                          | "Summarizing Source Code using a Neural Attention Model" [[paper](https://aclanthology.org/P16-1195/)] [[data](https://github.com/sriniiyer/codenn)]                                                                               |
| 2017-07 | IJCNLP 2017 | unnamed       | 150K    | Python                          | "A parallel corpus of Python functions and documentation strings for automated code documentation and code generation" [[paper](https://arxiv.org/abs/1707.02275)] [[data](https://github.com/EdinburghNLP/code-docstring-corpus)] |
| 2018-05 | ICPC 2018   | DeepCom       | 588K    | Java                            | "Deep code comment generation" [[paper](https://dl.acm.org/doi/10.1145/3196321.3196334)] [[data](https://github.com/xing-hu/DeepCom)]                                                                                              |
| 2018-07 | IJCAI 2018  | TL-CodeSum    | 411K    | Java                            | "Summarizing Source Code with Transferred API Knowledge" [[paper](https://www.ijcai.org/proceedings/2018/314)] [[data](https://github.com/xing-hu/TL-CodeSum)]                                                                     |
| 2019-09 | arxiv       | CodeSearchNet | 2.3M    | Go, JS, Python, PHP, Java, Ruby | "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [[paper](https://arxiv.org/abs/1909.09436)] [[data](https://github.com/github/CodeSearchNet)]                                                              |
| 2023-08 | arXiv       | HumanEvalPack | 984     | Python, JS, Go, Java, C++, Rust | "OctoPack: Instruction Tuning Code Large Language Models" [[paper](https://arxiv.org/abs/2308.07124)] [[data](https://huggingface.co/datasets/bigcode/humanevalpack)]                                                              |

#### Code Retrieval

| Date    | Venue                                | Benchmark            | Size      | Language                        | Source                                                                                                                                                                                                                                              |
| ------- | ------------------------------------ | -------------------- | --------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2018-03 | WWW 2018                             | StaQC                | 148K/120K | Python/SQL                      | "StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow" [[paper](https://arxiv.org/abs/1803.09371)] [[data](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset)]                                                   |
| 2018-05 | ICSE 2018                            | DeepCS               | 18.2M     | Java                            | "Deep Code Search" [[paper](https://dl.acm.org/doi/10.1145/3180155.3180167)] [[data](https://github.com/guxd/deep-code-search)]                                                                                                                     |
| 2018-05 | MSR 2018                             | CoNaLa               | 600K/2.9K | Python                          | "Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow" [[paper](https://arxiv.org/abs/1805.08949)] [[data](https://conala-corpus.github.io/)]                                                                               |
| 2019-08 | arXiv                                | unnamed              | 287       | Java                            | "Neural Code Search Evaluation Dataset" [[paper](https://arxiv.org/abs/1908.09804)] [[data](https://github.com/facebookresearch/Neural-Code-Search-Evaluation-Dataset)]                                                                             |
| 2019-09 | arXiv                                | CodeSearchNet        | 2.3M/99   | Go, PHP, JS, Python, Java, Ruby | "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [[paper](https://arxiv.org/abs/1909.09436)] [[data](https://github.com/github/CodeSearchNet)]                                                                               |
| 2020-02 | SANER 2020                           | CosBench             | 52        | Java                            | "Are the Code Snippets What We Are Searching for? A Benchmark and an Empirical Study on Code Search with Natural-Language Queries" [[paper](https://ieeexplore.ieee.org/document/9054840)] [[data](https://github.com/BASE-LAB-SJTU/CosBench/wiki)] |
| 2020-08 | arXiv                                | SO-DS                | 2.2K      | Python                          | "Neural Code Search Revisited: Enhancing Code Snippet Retrieval through Natural Language Intent" [[paper](https://arxiv.org/abs/2008.12193)] [[data](https://github.com/nokia/codesearch)]                                                          |
| 2020-10 | ACM Trans. Knowl. Discov. Data       | FB-Java              | 249K      | Java                            | "Deep Graph Matching and Searching for Semantic Code Retrieval" [[paper](https://arxiv.org/abs/2010.12908)] [[data](https://github.com/ryderling/DGMS)]                                                                                             |
| 2021-02 | NeurIPS Datasets and Benchmarks 2021 | AdvTest/WebQueryTest | 280K/1K   | Python                          | "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation" [[paper](https://arxiv.org/abs/2102.04664)] [[data]]                                                                                                        |
| 2021-05 | ACL/IJCNLP 2021                      | CoSQA                | 21K       | Python                          | "CoSQA: 20,000+ Web Queries for Code Search and Question Answering" [[paper](https://arxiv.org/abs/2105.13239)] [[data](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery)]                                        |

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

#### Repo-Level Coding

| Date    | Venue | Benchmark     | Size                   | Language                     | Source                                                                                                                                                                                                |
| ------- | ----- | ------------- | ---------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023-03 | arXiv | RepoEval      | 1600/1600/373 \*       | Python                       | "RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation" [[paper](https://arxiv.org/abs/2303.12570)] [[data](https://github.com/microsoft/CodeT/tree/main/RepoCoder)] |
| 2023-06 | arXiv | RepoBench     | 890K/9M/43K $^\dagger$ | Python, Java                 | "RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems" [[paper](https://arxiv.org/abs/2306.03091)] [[data](https://github.com/Leolty/repobench)]                                     |
| 2023-06 | arXiv | Stack-Repo    | 816K                   | Java                         | "RepoFusion: Training Code Models to Understand Your Repository" [[paper](https://arxiv.org/abs/2306.10998)] [[data](https://huggingface.co/RepoFusion)]                                              |
| 2023-09 | arXiv | CodePlan      | 645/21 $^\ddagger$     | C#/Python $^\ddagger$        | "CodePlan: Repository-level Coding using LLMs and Planning" [[paper](https://arxiv.org/abs/2309.12499)] [[data](https://aka.ms/CodePlan)] \*\*                                                        |
| 2023-10 | arXiv | SWE-Bench     | 2294                   | Python                       | "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" [[paper](https://arxiv.org/abs/2310.06770)] [[data](https://www.swebench.com/)]                                                    |
| 2023-10 | arXiv | CrossCodeEval | 9928                   | Python, Java, TypeScript, C# | "CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion" [[paper](https://arxiv.org/abs/2310.11248)] [[data](https://crosscodeeval.github.io/)]                           |

\*Line Completion/API Invocation Completion/Function Completion

$^\dagger$ Retrieval/Completion/Pipeline

$^\ddagger$ Migration/Temporal Edit

\*\* This is the link given in the paper, but we are unable to access it at the time of writing.

#### Other tasks are coming soon!

## 4. Recommended Readings

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
| 2019-05 |        SuperGLUE         | [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/abs/1905.00537)                                                                 | Another langauge understanding benchmark                                                                                             |
| 2019-07 |         RoBERTa          | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)                                                                                            | An optimized BERT                                                                                                                    |
| 2019-09 |       Megatron-LM        | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)                                                              | Model parallelism                                                                                                                    |
| 2019-10 |           ZeRO           | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)                                                                               | Memory-efficient distributed optimization                                                                                            |
| 2019-10 |            T5            | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)                                                                  | Transformer encoder-decoder pretrained with an MLM-like denoising objective                                                          |
| 2020-05 |          GPT-3           | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)                                                                                                              | By training an even larger version of GPT-2 (175B), they discovered a new learning paradigm: In-Context Learning (ICL)               |
| 2020-09 |           MMLU           | [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)                                                                                                 | A world-knowledge and complex reasoning benchmark                                                                                    |
| 2020-12 |           Pile           | [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)                                                                                   | A diverse pretraining dataset                                                                                                        |
| 2021-06 |           LoRA           | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)                                                                                                 | Memory-efficient finetuning                                                                                                          |
| 2021-09 |           FLAN           | [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)                                                                                                   | Instruction-finetuning                                                                                                               |
| 2021-10 |            T0            | [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)                                                                                  | Also instruction finetuning, but applied to the much smaller T5                                                                      |
| 2021-12 |          Gopher          | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)                                                                         | A 280B LLM with comprehensive experiments                                                                                            |
| 2022-01 |           CoT            | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)                                                                              | Chain-of-Though reasoning                                                                                                            |
| 2022-03 |       InstructGPT        | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)                                                                                | GPT-3 instruction finetuned with RLHF (reinforcement learning from human feedback)                                                   |
| 2022-03 |        Chinchilla        | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)                                                                                                     | A smaller (70B) version of Gopher that's pretrained on more data                                                                     |
| 2022-04 |           PaLM           | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)                                                                                                      | The largest dense model ever (540B)                                                                                                  |
| 2022-05 |        0-shot CoT        | [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)                                                                                                      | Tell LLMs to think step by step, and they can actually do it                                                                         |
| 2022-06 |        BIG Bench         | [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615)                                                       | Another world-knowledge and complex reasoning benchmark                                                                              |
| 2022-06 |     Emergent Ability     | [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)                                                                                                        | A review on emergent abilities                                                                                                       |
| 2022-10 |           Flan           | [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)                                                                                                      | Consolidate all the existing instruction tuning datasets, and you get SOTA                                                           |
| 2022-11 |          BLOOM           | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)                                                                                    | The largest open-source LLM, trained on 46 languages, with detailed discussion about training and evaluation                         |
| 2022-12 |      Self-Instruct       | [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)                                                                           | Instruction tuning using LLM-generated data                                                                                          |

This list aims to provide the essential background for understanding current LLM technologies, and thus excludes more recent models such as [LLaMA](https://arxiv.org/abs/2302.13971), [GPT-4](https://arxiv.org/abs/2303.08774) or [PaLM 2](https://arxiv.org/abs/2305.10403). For comprehensive reviews on these more general topics, we refer to other sources such as [this paper](https://arxiv.org/abs/2303.18223) or these repositories: [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM), [Awesome AIGC Tutorials](https://github.com/luban-agi/Awesome-AIGC-Tutorials). And for specific domains: [Awesome Domain LLM](https://github.com/luban-agi/Awesome-Domain-LLM), [Awesome Tool Learning](https://github.com/luban-agi/Awesome-Tool-Learning#awesome-tool-learning), [Awesome-LLM-MT](https://github.com/hsing-wang/Awesome-LLM-MT).

## Citation

If you find this repo or our survey helpful, please consider citing us:

```
@misc{zhang2023survey,
      title={A Survey on Language Models for Code},
      author={Ziyin Zhang and Chaoyu Chen and Bingchang Liu and Cong Liao and Zi Gong and Hang Yu and Jianguo Li and Rui Wang},
      year={2023},
      eprint={2311.07989},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
