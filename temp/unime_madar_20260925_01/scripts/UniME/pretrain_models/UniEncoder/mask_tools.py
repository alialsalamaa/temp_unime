from typing import Sequence, Tuple

from einops import rearrange
import torch


def tokenizer(image: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    Tokenize a 3D image tensor into non-overlapping patches.
    """
    assert image.shape[2] % patch_size == 0 and image.shape[3] % patch_size == 0 and image.shape[4] % patch_size == 0
    return rearrange(
        image,
        'B num_modals (D d) (H h) (W w) -> B (num_modals D H W) d h w',
        d=patch_size,
        h=patch_size,
        w=patch_size,
    )


def _as_patch_tokens(raw_input: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Accept either raw volumes or already-tokenized patch tensors.
    """
    if raw_input.shape[2:] == (patch_size, patch_size, patch_size):
        return raw_input
    return tokenizer(raw_input, patch_size=patch_size)


def _validate_mask_args(total_tokens: int, num_modals: int, num_mask_modalities: int | None) -> None:
    if num_modals < 1:
        raise ValueError("num_modals must be positive")
    if total_tokens % num_modals != 0:
        raise ValueError("total token count must be divisible by num_modals")
    if num_mask_modalities is not None and num_mask_modalities < 0:
        raise ValueError("num_mask_modalities must be non-negative")
    if num_mask_modalities is not None and num_mask_modalities >= num_modals:
        raise ValueError("num_mask_modalities must be less than num_modals")
    if total_tokens < num_modals:
        raise ValueError("total token count must be >= num_modals")


def _token_modal_ids(total_tokens: int, num_modals: int, device: torch.device) -> torch.Tensor:
    tokens_per_modal = total_tokens // num_modals
    return torch.arange(total_tokens, device=device, dtype=torch.long) // tokens_per_modal


def _sample_modal_mask_torch(
    batch_size: int,
    num_modals: int,
    num_mask_modalities: int | None,
    device: torch.device,
    modality_mask_prob: float = 0.5,
) -> torch.Tensor:
    """
    Sample independent Bernoulli modality masks, conditioned on visible input.

    Section 3.2 conditions on at least one available modality. Enumerating the
    small modality set (four MRI channels) and sampling its conditional weights
    is equivalent to rejection sampling, without a data-dependent GPU loop.
    At p=0.5 each of the 15 nonempty available subsets is equally likely.
    ``num_mask_modalities`` is a legacy upper cap; values below K-1 add an
    extra condition and are not the paper's default masking distribution.
    """
    _validate_mask_args(num_modals, num_modals, num_mask_modalities)
    if not 0.0 <= modality_mask_prob < 1.0:
        raise ValueError("modality_mask_prob must be in [0, 1) to allow visible modalities")
    max_masked = num_modals - 1 if num_mask_modalities is None else num_mask_modalities
    subset_ids = torch.arange(2 ** num_modals, device=device)
    modality_bits = 2 ** torch.arange(num_modals, device=device)
    subsets = (subset_ids[:, None] & modality_bits[None, :]) != 0
    masked_counts = subsets.sum(dim=1)
    probabilities = (
        modality_mask_prob ** masked_counts
        * (1.0 - modality_mask_prob) ** (num_modals - masked_counts)
    )
    probabilities = probabilities.masked_fill(masked_counts > max_masked, 0.0)
    sampled_ids = torch.multinomial(probabilities, batch_size, replacement=True)
    return subsets[sampled_ids]


def _sample_keep_mask_torch(
    batch_size: int,
    total_tokens: int,
    num_modals: int,
    num_mask_modalities: int | None,
    use_patch_mask: bool,
    patch_mask_ratio: float,
    device: torch.device,
    modality_mask_prob: float = 0.5,
) -> torch.Tensor:
    """
    Sample token keep-mask with optional modality-level masking, using only torch ops.
    """
    _validate_mask_args(total_tokens, num_modals, num_mask_modalities)
    if not 0.0 <= patch_mask_ratio <= 1.0:
        raise ValueError("patch_mask_ratio must be in [0, 1]")
    token_modal_ids = _token_modal_ids(total_tokens, num_modals, device)
    masked_modal_mask = _sample_modal_mask_torch(
        batch_size=batch_size,
        num_modals=num_modals,
        num_mask_modalities=num_mask_modalities,
        device=device,
        modality_mask_prob=modality_mask_prob,
    )
    visible_token_mask = ~masked_modal_mask[:, token_modal_ids]

    if not use_patch_mask:
        return visible_token_mask

    # Equation 1 samples each patch independently, so the kept count fluctuates
    # around (1-q) * available_tokens rather than being fixed per sample.
    patch_keep_mask = torch.rand(batch_size, total_tokens, device=device) >= patch_mask_ratio
    return visible_token_mask & patch_keep_mask


def modal_mask_with_patch_mask(
    tokens_list: Sequence[int],
    num_modals: int = 4,
    num_mask_modalities: int | None = None,
    use_patch_mask: bool = True,
    patch_mask_ratio: float = 0.75,
    modality_mask_prob: float = 0.5,
) -> Tuple[list[int], list[int]]:
    """
    Compatibility wrapper that returns kept/masked token ids for a single sample.
    """
    token_values = torch.as_tensor(list(tokens_list), dtype=torch.long)
    keep_mask = _sample_keep_mask_torch(
        batch_size=1,
        total_tokens=token_values.numel(),
        num_modals=num_modals,
        num_mask_modalities=num_mask_modalities,
        use_patch_mask=use_patch_mask,
        patch_mask_ratio=patch_mask_ratio,
        device=token_values.device,
        modality_mask_prob=modality_mask_prob,
    )[0]
    sample_list = token_values[keep_mask].tolist()
    mask_list = token_values[~keep_mask].tolist()
    return sample_list, mask_list


def apply_mask(
    batch_size: int,
    raw_input: torch.Tensor,
    patch_size: int = 8,
    num_modals: int = 4,
    num_mask_modalities: int | None = None,
    use_patch_mask: bool = True,
    patch_mask_ratio: float = 0.75,
    crop_size: int = 96,
    modality_mask_prob: float = 0.5,
) -> torch.Tensor:
    """
    Apply modality masking + patch masking and return a binary-masked volume.
    """
    d_len = h_len = w_len = crop_size // patch_size
    total_tokens = int(d_len ** 3) * num_modals

    patch_tokens = _as_patch_tokens(raw_input, patch_size=patch_size)
    if patch_tokens.shape[0] == 1 and batch_size > 1:
        patch_tokens = patch_tokens.expand(batch_size, -1, -1, -1, -1)
    elif patch_tokens.shape[0] != batch_size:
        raise ValueError(
            f"batch_size={batch_size} does not match patch token batch={patch_tokens.shape[0]}"
        )
    if patch_tokens.shape[1] != total_tokens:
        raise ValueError(
            f"Expected {total_tokens} tokens for crop_size={crop_size}, patch_size={patch_size}, "
            f"num_modals={num_modals}; got {patch_tokens.shape[1]}"
        )

    keep_mask = _sample_keep_mask_torch(
        batch_size=batch_size,
        total_tokens=total_tokens,
        num_modals=num_modals,
        num_mask_modalities=num_mask_modalities,
        use_patch_mask=use_patch_mask,
        patch_mask_ratio=patch_mask_ratio,
        device=patch_tokens.device,
        modality_mask_prob=modality_mask_prob,
    )
    masked_patch_tokens = patch_tokens * keep_mask[:, :, None, None, None].to(dtype=patch_tokens.dtype)

    return rearrange(
        masked_patch_tokens,
        'B (num_modals D H W) d h w -> B num_modals (D d) (H h) (W w)',
        num_modals=num_modals,
        D=d_len,
        H=h_len,
        W=w_len,
    )
