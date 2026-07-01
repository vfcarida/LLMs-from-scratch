# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
# Code: https://github.com/rasbt/LLMs-from-scratch

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_token_logprobs(logits, tokens, prompt_len):
    """
    Compute log probabilities of generated completion tokens.
    
    Args:
        logits (torch.Tensor): Model outputs of shape [batch_size, seq_len, vocab_size].
        tokens (torch.Tensor): Token IDs of shape [batch_size, seq_len].
        prompt_len (int): Length of the prompt prefix.
        
    Returns:
        torch.Tensor: Log probabilities of shape [batch_size, completion_len].
    """
    # Shift logits and tokens to align predictions with targets
    # logits[:, t-1] predicts tokens[:, t]
    shift_logits = logits[:, prompt_len - 1 : -1, :]
    shift_tokens = tokens[:, prompt_len:]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = torch.gather(log_probs, dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)
    return token_logprobs

def compute_kl_penalty(logprobs, ref_logprobs, approx_mode="schulman"):
    """
    Compute token-level KL divergence between active policy and reference policy.
    
    Args:
        logprobs (torch.Tensor): Log probabilities from the active policy.
        ref_logprobs (torch.Tensor): Log probabilities from the reference policy.
        approx_mode (str): KL approximation mode ('schulman' or 'exact').
        
    Returns:
        torch.Tensor: Token-level KL divergence.
    """
    if approx_mode == "schulman":
        # Schulman's unbiased estimator of KL: exp(log_ref - log_policy) - (log_ref - log_policy) - 1
        # which is non-negative and has low variance.
        log_ratio = ref_logprobs - logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
    else:
        # Standard analytical KL: log_policy - log_ref
        kl = logprobs - ref_logprobs
    return kl

def generate_batch(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=None, eos_id=None, pad_token_id=50256):
    """
    Generate completions for a batch of prompts, handling EOS early stopping and padding.
    
    Args:
        model (nn.Module): The policy model.
        idx (torch.Tensor): Prompt token IDs of shape [batch_size, prompt_len].
        max_new_tokens (int): Maximum completion tokens to generate.
        context_size (int): Context size limit of the model.
        temperature (float): Temperature for scaling logits.
        top_k (int, optional): Top-k sampling limit.
        eos_id (int, optional): End-of-sequence token ID.
        pad_token_id (int): Padding token ID to use after EOS is hit.
        
    Returns:
        torch.Tensor: Generated sequences of shape [batch_size, prompt_len + generated_len].
    """
    batch_size = idx.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)
    
    for _ in range(max_new_tokens):
        if finished.all():
            break
            
        # Crop context window if needed
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Apply top-k filtering if specified
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)
            
        # Sample with temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        # If sequence is finished, append padding token instead of generated token
        if eos_id is not None:
            is_eos = (idx_next.squeeze(-1) == eos_id)
            idx_next = torch.where(finished.unsqueeze(-1), torch.tensor(pad_token_id, device=idx.device), idx_next)
            finished = finished | is_eos
            
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

class GRPOTrainer:
    """
    Trainer for Group Relative Policy Optimization (GRPO) from scratch.
    Implements rollouts, reward assignment, advantage computation, and policy updates.
    """
    def __init__(
        self,
        policy_model,
        reference_model,
        tokenizer,
        optimizer,
        device="cpu",
        kl_coeff=0.01,
        clip_eps=0.2,
        group_size=4,
        max_completion_len=64,
        temperature=0.7,
        top_k=None,
        approx_kl_mode="schulman",
        pad_token_id=50256,
        eos_id=50256,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model.eval()  # Keep reference model in eval mode
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps
        self.group_size = group_size
        self.max_completion_len = max_completion_len
        self.temperature = temperature
        self.top_k = top_k
        self.approx_kl_mode = approx_kl_mode
        self.pad_token_id = pad_token_id
        self.eos_id = eos_id
        
        # Ensure context length is retrieved from policy model config/attributes
        if hasattr(policy_model, "pos_emb") and policy_model.pos_emb is not None:
            self.context_size = policy_model.pos_emb.weight.shape[0]
        else:
            self.context_size = 1024

    def train_step(self, prompts, reward_fn, epochs=1):
        """
        Executes a single GRPO training step.
        
        Args:
            prompts (torch.Tensor): A batch of prompt token IDs of shape [batch_size, prompt_len].
            reward_fn (callable): A function that takes a list of generated strings and returns a tensor of rewards [batch_size * group_size].
            epochs (int): Number of optimization epochs per rollout batch.
            
        Returns:
            dict: Metrics containing mean loss, mean reward, advantages, and KL divergence.
        """
        self.policy_model.train()
        batch_size, prompt_len = prompts.shape
        total_samples = batch_size * self.group_size
        
        # 1. Rollout: generate group_size completions for each prompt
        # Repeat prompts to match group_size: [B, prompt_len] -> [B * G, prompt_len]
        prompts_expanded = prompts.repeat_interleave(self.group_size, dim=0).to(self.device)
        
        # Generate full sequences: [B * G, prompt_len + completion_len]
        seq_ids = generate_batch(
            model=self.policy_model,
            idx=prompts_expanded,
            max_new_tokens=self.max_completion_len,
            context_size=self.context_size,
            temperature=self.temperature,
            top_k=self.top_k,
            eos_id=self.eos_id,
            pad_token_id=self.pad_token_id
        )
        
        completion_ids = seq_ids[:, prompt_len:]
        
        # Decode completions to strings for the reward function
        completion_texts = []
        for i in range(total_samples):
            # Decode tokens excluding padding
            tokens_to_decode = [t for t in completion_ids[i].tolist() if t != self.pad_token_id]
            completion_texts.append(self.tokenizer.decode(tokens_to_decode))
            
        # 2. Reward assignment
        # reward_fn returns rewards tensor of shape [B * G]
        rewards = reward_fn(completion_texts)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards = rewards.to(self.device)
            
        # 3. Group relative advantage calculation
        # Reshape rewards to [B, G] to normalize within each prompt's group
        rewards_grouped = rewards.view(batch_size, self.group_size)
        mean_grouped = rewards_grouped.mean(dim=-1, keepdim=True)
        std_grouped = rewards_grouped.std(dim=-1, keepdim=True)
        
        # Use simple mean subtraction if std is extremely small or zero
        advantages_grouped = (rewards_grouped - mean_grouped) / (std_grouped + 1e-8)
        advantages = advantages_grouped.view(-1)  # Flatten back to [B * G]
        
        # 4. Reference model log-probabilities computation
        with torch.no_grad():
            ref_logits = self.reference_model(seq_ids)
            ref_logprobs = get_token_logprobs(ref_logits, seq_ids, prompt_len)
            
            # Cache old policy log-probabilities before starting optimization epochs
            old_logits = self.policy_model(seq_ids)
            old_logprobs = get_token_logprobs(old_logits, seq_ids, prompt_len)
            
        # 5. Optimization loop
        epoch_losses = []
        epoch_kls = []
        
        completion_mask = (completion_ids != self.pad_token_id).float()
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            
            # Active policy log-probabilities
            logits = self.policy_model(seq_ids)
            logprobs = get_token_logprobs(logits, seq_ids, prompt_len)
            
            # Ratio and clipped loss
            ratio = torch.exp(logprobs - old_logprobs)
            
            # Advantages are shape [B * G], we expand to [B * G, completion_len]
            advantages_expanded = advantages.unsqueeze(-1)
            
            surr1 = ratio * advantages_expanded
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_expanded
            policy_loss = -torch.min(surr1, surr2)
            
            # KL divergence penalty
            kl = compute_kl_penalty(logprobs, ref_logprobs, approx_mode=self.approx_kl_mode)
            
            # Total token loss
            token_loss = policy_loss + self.kl_coeff * kl
            
            # Mask out padding tokens
            masked_loss = token_loss * completion_mask
            
            # Average loss per sequence, then average over batch
            sequence_loss = masked_loss.sum(dim=-1) / (completion_mask.sum(dim=-1) + 1e-8)
            loss = sequence_loss.mean()
            
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            with torch.no_grad():
                mean_kl = (kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)
                epoch_kls.append(mean_kl.item())
                
        metrics = {
            "loss": sum(epoch_losses) / len(epoch_losses),
            "kl": sum(epoch_kls) / len(epoch_kls),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
        }
        return metrics
