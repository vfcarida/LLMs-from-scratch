# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
# Code: https://github.com/rasbt/LLMs-from-scratch

import pytest
import torch
import torch.nn as nn
import tiktoken

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.grpo import (
    get_token_logprobs,
    compute_kl_penalty,
    generate_batch,
    GRPOTrainer
)

# Small config for testing
TEST_GPT_CONFIG = {
    "vocab_size": 100,
    "context_length": 32,
    "emb_dim": 64,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False
}

def test_get_token_logprobs():
    # Setup: batch=2, length=4, vocab=5
    B, L, V = 2, 4, 5
    logits = torch.randn(B, L, V)
    tokens = torch.randint(0, V, (B, L))
    prompt_len = 2
    
    logprobs = get_token_logprobs(logits, tokens, prompt_len)
    
    # Expected shape: [B, L - prompt_len] -> [2, 2]
    assert logprobs.shape == (2, 2)
    
    # Let's manually calculate for batch 0, completion index 0 (token index 2 in sequence)
    # Predicted logits for token index 2 is at logits[:, 1]
    expected_log_softmax = torch.log_softmax(logits[0, 1], dim=-1)
    expected_prob = expected_log_softmax[tokens[0, 2]]
    
    assert torch.allclose(logprobs[0, 0], expected_prob)

def test_compute_kl_penalty():
    logprobs = torch.tensor([-0.5, -1.0, -1.5])
    ref_logprobs = torch.tensor([-0.6, -0.8, -2.0])
    
    # Exact mode: logprobs - ref_logprobs
    kl_exact = compute_kl_penalty(logprobs, ref_logprobs, approx_mode="exact")
    assert torch.allclose(kl_exact, logprobs - ref_logprobs)
    
    # Schulman mode: exp(log_ref - log_policy) - (log_ref - log_policy) - 1
    kl_schulman = compute_kl_penalty(logprobs, ref_logprobs, approx_mode="schulman")
    log_ratio = ref_logprobs - logprobs
    expected = torch.exp(log_ratio) - log_ratio - 1.0
    assert torch.allclose(kl_schulman, expected)
    assert (kl_schulman >= 0.0).all()  # Schulman's estimator is non-negative

def test_generate_batch():
    model = GPTModel(TEST_GPT_CONFIG)
    model.eval()
    
    # Prompt shape: [2, 5]
    idx = torch.randint(0, 100, (2, 5))
    eos_id = 99
    pad_token_id = 0
    
    # Make the model generate eos_id on first step for the first prompt, but not the second
    # We mock or just test normal behavior:
    out = generate_batch(
        model=model,
        idx=idx,
        max_new_tokens=4,
        context_size=TEST_GPT_CONFIG["context_length"],
        temperature=0.0,
        eos_id=eos_id,
        pad_token_id=pad_token_id
    )
    
    # Expected output shape: [2, 9] (5 prompt tokens + 4 new tokens)
    assert out.shape == (2, 9)
    # Check that prompt prefix remains untouched
    assert torch.equal(out[:, :5], idx)

class MockTokenizer:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")
    def encode(self, text):
        return self.encoder.encode(text)
    def decode(self, tokens):
        return self.encoder.decode(tokens)

def test_grpo_trainer_step():
    torch.manual_seed(42)
    
    policy_model = GPTModel(TEST_GPT_CONFIG)
    ref_model = GPTModel(TEST_GPT_CONFIG)
    ref_model.load_state_dict(policy_model.state_dict())
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4)
    tokenizer = MockTokenizer()
    
    trainer = GRPOTrainer(
        policy_model=policy_model,
        reference_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        kl_coeff=0.01,
        clip_eps=0.2,
        group_size=4,
        max_completion_len=5,
        temperature=0.7,
        approx_kl_mode="schulman"
    )
    
    # Prompts: batch=2, prompt_len=3
    prompts = torch.randint(0, 100, (2, 3))
    
    # Dummy reward function: rewards sum of characters length, or random
    def reward_fn(completions):
        return torch.tensor([float(len(c)) for c in completions])
        
    metrics = trainer.train_step(prompts, reward_fn, epochs=1)
    
    assert "loss" in metrics
    assert "kl" in metrics
    assert "reward_mean" in metrics
    assert "advantage_mean" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["kl"], float)
    assert isinstance(metrics["reward_mean"], float)
    assert isinstance(metrics["advantage_mean"], float)
