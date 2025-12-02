"""
Speculative Decoding for Whisper
Implements speculative decoding using a small assistant model (distil-whisper)
to speed up inference with a larger main model.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class SpeculativeGreedyDecoder:
    """
    Greedy decoder with speculative decoding support.
    Uses a smaller assistant model to predict multiple tokens ahead,
    then validates them with the main model in parallel.
    """

    def __init__(
        self,
        temperature: float,
        eot: int,
        assistant_model: Optional[torch.nn.Module] = None,
        num_assistant_tokens: int = 5,
        use_speculative: bool = True
    ):
        """
        Args:
            temperature: Sampling temperature (0 for greedy)
            eot: End-of-text token ID
            assistant_model: Optional smaller Whisper model for speculation
            num_assistant_tokens: Number of tokens to predict ahead with assistant
            use_speculative: Whether to use speculative decoding
        """
        self.temperature = temperature
        self.eot = eot
        self.assistant_model = assistant_model
        self.num_assistant_tokens = num_assistant_tokens
        self.use_speculative = use_speculative and assistant_model is not None

        if self.use_speculative:
            logger.info(f"Speculative decoding enabled with {num_assistant_tokens} lookahead tokens")
        else:
            logger.info("Standard greedy decoding (no speculation)")

    def update(
        self,
        tokens: Tensor,
        logits: Tensor,
        sum_logprobs: Tensor,
        mel: Optional[Tensor] = None,
        main_model: Optional[torch.nn.Module] = None
    ) -> Tuple[Tensor, bool]:
        """
        Update tokens using speculative decoding if enabled.

        Args:
            tokens: Current token sequence [batch, seq_len]
            logits: Logits from main model [batch, vocab_size]
            sum_logprobs: Cumulative log probabilities [batch]
            mel: Mel spectrogram features [batch, n_ctx, n_state] (required for speculative)
            main_model: Main Whisper model (required for speculative)

        Returns:
            Updated tokens and completion flag
        """
        if not self.use_speculative or mel is None or main_model is None:
            # Fall back to standard greedy decoding
            return self._standard_update(tokens, logits, sum_logprobs)

        try:
            return self._speculative_update(tokens, logits, sum_logprobs, mel, main_model)
        except Exception as e:
            logger.warning(f"Speculative decoding failed: {e}, falling back to standard decoding")
            return self._standard_update(tokens, logits, sum_logprobs)

    def _standard_update(
        self,
        tokens: Tensor,
        logits: Tensor,
        sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Standard greedy decoding (original implementation)"""
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            from torch.distributions import Categorical
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def _speculative_update(
        self,
        tokens: Tensor,
        logits: Tensor,
        sum_logprobs: Tensor,
        mel: Tensor,
        main_model: torch.nn.Module
    ) -> Tuple[Tensor, bool]:
        """
        Speculative decoding update:
        1. Assistant model predicts N tokens ahead
        2. Main model validates all N+1 positions in parallel
        3. Accept validated tokens, reject rest
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Step 1: Generate candidate tokens with assistant model
        assistant_tokens = self._generate_assistant_candidates(tokens, mel)

        if assistant_tokens is None or assistant_tokens.shape[1] == 0:
            # Assistant failed, fall back to standard
            return self._standard_update(tokens, logits, sum_logprobs)

        # Step 2: Validate candidates with main model in parallel
        # Concatenate original tokens with assistant predictions
        candidate_tokens = torch.cat([tokens, assistant_tokens], dim=1)

        # Get main model logits for all positions at once
        with torch.no_grad():
            main_logits = main_model.logits(candidate_tokens, mel)

        # Step 3: Verify which tokens match main model's greedy choices
        accepted_count = 0
        new_tokens = tokens.clone()

        for i in range(assistant_tokens.shape[1]):
            # Position in the sequence we're checking
            pos = tokens.shape[1] + i

            # Main model's greedy choice at this position
            main_choice = main_logits[:, pos - 1, :].argmax(dim=-1)

            # Assistant's prediction
            assistant_choice = assistant_tokens[:, i]

            # Check if they match
            if (main_choice == assistant_choice).all():
                # Accept this token
                new_tokens = torch.cat([new_tokens, assistant_choice.unsqueeze(1)], dim=1)
                accepted_count += 1

                # Update log probabilities
                logprobs = F.log_softmax(main_logits[:, pos - 1, :].float(), dim=-1)
                current_logprobs = logprobs[torch.arange(batch_size), main_choice]
                sum_logprobs += current_logprobs * (new_tokens[:, -2] != self.eot)
            else:
                # Reject this and all subsequent tokens
                # Add main model's choice instead
                new_tokens = torch.cat([new_tokens, main_choice.unsqueeze(1)], dim=1)

                logprobs = F.log_softmax(main_logits[:, pos - 1, :].float(), dim=-1)
                current_logprobs = logprobs[torch.arange(batch_size), main_choice]
                sum_logprobs += current_logprobs * (new_tokens[:, -2] != self.eot)
                break

        # If all assistant tokens were accepted, add one more from main model
        if accepted_count == assistant_tokens.shape[1]:
            pos = new_tokens.shape[1]
            main_choice = main_logits[:, pos - 1, :].argmax(dim=-1)
            new_tokens = torch.cat([new_tokens, main_choice.unsqueeze(1)], dim=1)

            logprobs = F.log_softmax(main_logits[:, pos - 1, :].float(), dim=-1)
            current_logprobs = logprobs[torch.arange(batch_size), main_choice]
            sum_logprobs += current_logprobs * (new_tokens[:, -2] != self.eot)

        # Handle EOT
        new_tokens[new_tokens[:, -1] == self.eot, -1] = self.eot

        completed = (new_tokens[:, -1] == self.eot).all()

        if accepted_count > 0:
            logger.debug(f"Speculative: accepted {accepted_count}/{assistant_tokens.shape[1]} tokens")

        return new_tokens, completed

    def _generate_assistant_candidates(
        self,
        tokens: Tensor,
        mel: Tensor
    ) -> Optional[Tensor]:
        """
        Generate candidate tokens using the assistant model.

        Returns:
            Tensor of shape [batch, num_assistant_tokens] or None if failed
        """
        try:
            with torch.no_grad():
                candidate_tokens = []
                current_tokens = tokens.clone()

                for _ in range(self.num_assistant_tokens):
                    # Get logits from assistant model
                    logits = self.assistant_model.logits(current_tokens, mel)

                    # Greedy selection
                    next_token = logits[:, -1, :].argmax(dim=-1)

                    # Stop if EOT
                    if (next_token == self.eot).any():
                        break

                    candidate_tokens.append(next_token)
                    current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)

                if len(candidate_tokens) == 0:
                    return None

                return torch.stack(candidate_tokens, dim=1)

        except Exception as e:
            logger.debug(f"Assistant model inference failed: {e}")
            return None

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        """Finalize token sequence (same as standard decoder)"""
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()
