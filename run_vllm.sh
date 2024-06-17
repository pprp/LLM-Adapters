#!/bin/bash 

# ['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']

CUDA_VISIBLE_DEVICES=1 python evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset SVAMP \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-svamp.log

CUDA_VISIBLE_DEVICES=1 python evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset AddSub \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-addsub.log

CUDA_VISIBLE_DEVICES=1 python evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset MultiArith \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-multiarith.log

CUDA_VISIBLE_DEVICES=1 python evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset SingleEq \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-singleeq.log

CUDA_VISIBLE_DEVICES=1 python evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset gsm8k \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-gsm8k.log

CUDA_VISIBLE_DEVICES=1 python evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset AQuA \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-aqua.log

# ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset boolq \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-boolq.log

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset piqa \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-piqa.log

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset social_i_qa \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-social_i_qa.log

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset hellaswag \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-hellaswag.log

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset winogrande \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-winogrande.log

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset ARC-Challenge \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-ARC-Challenge.log

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset ARC-Easy \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-ARC-Easy.log

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate_vllm.py \
    --model LLaMA-7B \
    --dataset openbookqa \
    --batch_size 8 \
    --base_model '/aifs4su/mmdata/hf_download/llama-1-7b' > ./vllm_logs/llama-7b-openbookqa.log

