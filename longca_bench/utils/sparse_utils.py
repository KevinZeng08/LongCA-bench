import torch

# ================ Utils for Block Sparse Attention ================


def generate_block_sparse_pattern(
    num_q_heads: int,
    num_kv_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    sparsity: float,
    mode: str = "per_kv_head",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a head-wise block sparse pattern, supporting both MHA and GQA semantics.

    The final returned mask is always of shape [1, num_q_heads, num_q_blocks, num_kv_blocks].

    Args:
        num_q_heads (int): Total number of query attention heads.
        num_kv_heads (int): Total number of key-value attention heads.
        num_q_blocks (int): Number of query blocks per head.
        num_kv_blocks (int): Number of key-value blocks per head.
        sparsity (float): The density ratio of connections.
        mode (str("per_q_head", "per_kv_head")):
            - "per_q_head": Each query head gets a unique random mask (for MHA).
            - "per_kv_head": Query heads in the same group share a mask (for GQA).
        device (str): The device to create tensors on.

    Returns:
        torch.Tensor: A boolean tensor mask of shape [1, num_q_heads, num_q_blocks, num_kv_blocks].
        torch.Tensor: A tensor containing the random scores used for selection,
                      shape is [1, num_mask_heads, num_q_blocks, num_kv_blocks],
                      where num_mask_heads is num_q_heads or num_kv_heads based on mode.
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    k = max(1, int(sparsity * num_kv_blocks))
    k = min(k, num_kv_blocks)

    if mode == "per_q_head":
        # Each Q head gets its own mask. This is equivalent to GQA where num_groups=num_q_heads.
        num_mask_heads = num_q_heads
    elif mode == "per_kv_head":
        # Masks are generated per KV head.
        num_mask_heads = num_kv_heads
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Create random scores based on the number of heads specified by the mode
    scores = torch.rand(num_mask_heads, num_q_blocks, num_kv_blocks, device=device)

    # 2. Get the indices of the top-k scoring key-value blocks
    _, topk_indices = torch.topk(scores, k, dim=-1)

    # 3. Create a boolean base mask initialized to all False
    base_mask = torch.zeros(
        num_mask_heads, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )

    # 4. Use scatter_ to efficiently set the corresponding positions to True
    base_mask.scatter_(2, topk_indices, True)

    # 5. Expand mask if generated at KV-head granularity for GQA
    if mode == "per_kv_head" and num_q_heads != num_kv_heads:
        num_groups = num_q_heads // num_kv_heads
        # Repeat the mask for each Q head in the group
        block_sparse_mask = torch.repeat_interleave(
            base_mask, repeats=num_groups, dim=0
        )
    else:
        block_sparse_mask = base_mask

    # 6. Add batch dimension
    block_sparse_mask = block_sparse_mask.unsqueeze(0)
    scores = scores.unsqueeze(0)

    return block_sparse_mask, scores

def get_sdpa_mask_from_block_sparse_mask(
    block_mask: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Converts a block-level sparse mask to an element-level boolean mask
    that is compatible with SDPA (scaled_dot_product_attention).

    Args:
        block_mask (torch.Tensor): The block mask of shape [H, num_q_blocks, num_k_blocks].
        seqlen_q (int): The full length of the query sequence.
        seqlen_k (int): The full length of the key/value sequence.
        block_size_q (int): The size of a Q block.
        block_size_k (int): The size of a K block.
        batch_size (int): The batch size.

    Returns:
        torch.Tensor: An SDPA-compatible mask of shape [B, H, S_q, S_k].
    """
    num_heads = block_mask.shape[1]
    device = block_mask.device

    # 1. Create a large 4D mask of the target shape, filled with False.
    #    This is our "canvas", where False means all positions are masked out by default.
    sdpa_mask = torch.zeros(
        (batch_size, num_heads, seqlen_q, seqlen_k), dtype=torch.bool, device=device
    )

    # 2. Efficiently find the coordinates (h, q_block, k_block) of all blocks to be activated.
    _, h_indices, qb_indices, kb_indices = torch.nonzero(block_mask, as_tuple=True)

    # 3. Iterate through all activated blocks.
    for h, qb, kb in zip(h_indices, qb_indices, kb_indices):
        # Calculate the start and end coordinates for this block in the element-level mask.
        q_start, q_end = qb * block_size_q, (qb + 1) * block_size_q
        k_start, k_end = kb * block_size_k, (kb + 1) * block_size_k

        # "Paint" the corresponding rectangular region on the canvas to True,
        # indicating that attention is allowed for these positions.
        sdpa_mask[:, h, q_start:q_end, k_start:k_end] = True

    return sdpa_mask
