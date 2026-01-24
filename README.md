# ACM_TIST_submission
This repo consists of four folders inside the main folder. It also contains the p-nucleus sampling integrated AdvPrompter code:-
1) Situation-Driven-Adversarial-Attacks-main:-
    - The codes to attack the different LLMs: gpt-3.5-turbo-0125, phi-1.5, gpt-4, gemma-7b, Meta-Llama-3-8B, and the 4-bit quantized Llama-2 7B chat for the initial set of experiments.
    - [GPT-4 Judge](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety) code that we used in our research.
    - llama.out contains the collection of GPT-4 Judge outputs for the 4-bit quantized Llama-2 7B chat model on the initial set of human-readable adversarial full-prompts with situational context.
    - Contains the attack codes related to the few-shot chain-of-thought technique.

2) mul_adv:-
    - The codes that relate to how adversarial expressions generated with and without p-nucleus sampling integrated AdvPrompter perform in attacking an LLM when used in a full-prompt template.
    - The codes without any adversarial insertion.
    - Codes involve attacking several models.
    - GPT-4 Judge is now used with gpt-4o-mini.
    - Judge results with grok-4-1-fast-non-reasoning.
    - Detectability experiment for adversarial insertions.
  
3) Elo:-
   - Folder contains the code for Elo computation.

4) Baseline:-
   - Folder contains codes for the baseline experiment with AutoDan by Liu et al., TAP by Mehrotra et al., AdvPrompter by Paulus et al., and an ablation study (without MP).
