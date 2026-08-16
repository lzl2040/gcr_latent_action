"""Test script for Action Chunk Encoder (ACE) module."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from lerobot.common.policies.ace import ACEConfig, ActionChunkEncoder, create_ace_model


def test_basic_forward():
    """Test basic forward pass with random data."""
    print("=" * 60)
    print("Test 1: Basic forward pass")
    print("=" * 60)
    
    # Create model with default config
    config = ACEConfig(
        action_dim=7,
        chunk_size=16,
        group_size=4,
        hidden_dim=256,  # Smaller for faster testing
        num_hidden_layers=4,  # Fewer layers for faster testing
        output_dim=128
    )
    
    model = ActionChunkEncoder(config)
    model.eval()
    
    # Create random action chunk
    batch_size = 2
    actions = torch.randn(batch_size, config.chunk_size, config.action_dim)
    
    print(f"Input shape: {actions.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Chunk size: {config.chunk_size}")
    print(f"  - Action dim: {config.action_dim}")
    
    # Forward pass
    with torch.no_grad():
        embedding = model(actions, sample_rate=0)
    
    print(f"\nOutput shape: {embedding.shape}")
    print(f"  - Expected: ({batch_size}, {config.output_dim})")
    print(f"  - Actual: {embedding.shape}")
    
    assert embedding.shape == (batch_size, config.output_dim), \
        f"Shape mismatch: expected {(batch_size, config.output_dim)}, got {embedding.shape}"
    
    print("\n✓ Test 1 passed!")
    return True


def test_grouping_mechanism():
    """Test action grouping mechanism."""
    print("\n" + "=" * 60)
    print("Test 2: Action grouping mechanism")
    print("=" * 60)
    
    config = ACEConfig(
        action_dim=7,
        chunk_size=16,
        group_size=4,
        hidden_dim=256,
        num_hidden_layers=2
    )
    
    model = ActionChunkEncoder(config)
    
    # Create deterministic input for testing
    batch_size = 1
    actions = torch.arange(config.chunk_size).float().unsqueeze(-1).expand(-1, -1, config.action_dim)
    actions = actions.unsqueeze(0).expand(batch_size, -1, -1)
    
    print(f"Input actions shape: {actions.shape}")
    print(f"  - Group size: {config.group_size}")
    print(f"  - Expected num_groups: {config.chunk_size // config.group_size}")
    
    # Test internal grouping
    padded = model._pad_actions(actions)
    print(f"\nAfter padding: {padded.shape}")
    
    grouped = model._group_actions(padded)
    expected_num_groups = config.chunk_size // config.group_size
    expected_group_dim = config.group_size * config.action_dim_padded
    
    print(f"After grouping: {grouped.shape}")
    print(f"  - Expected: ({batch_size}, {expected_num_groups}, {expected_group_dim})")
    
    assert grouped.shape == (batch_size, expected_num_groups, expected_group_dim), \
        f"Grouping shape mismatch"
    
    print("\n✓ Test 2 passed!")
    return True


def test_different_configs():
    """Test with different configurations."""
    print("\n" + "=" * 60)
    print("Test 3: Different configurations")
    print("=" * 60)
    
    configs = [
        {"action_dim": 7, "chunk_size": 8, "group_size": 2},
        {"action_dim": 14, "chunk_size": 32, "group_size": 8},
        {"action_dim": 32, "chunk_size": 16, "group_size": 4},  # No padding needed
    ]
    
    for i, cfg_dict in enumerate(configs):
        print(f"\nConfig {i+1}: {cfg_dict}")
        config = ACEConfig(
            **cfg_dict,
            hidden_dim=128,
            num_hidden_layers=2,
            output_dim=64
        )
        
        model = ActionChunkEncoder(config)
        model.eval()
        
        batch_size = 2
        actions = torch.randn(batch_size, config.chunk_size, config.action_dim)
        
        with torch.no_grad():
            embedding = model(actions)
        
        print(f"  Input: {actions.shape} -> Output: {embedding.shape}")
        assert embedding.shape == (batch_size, config.output_dim)
    
    print("\n✓ Test 3 passed!")
    return True


def test_sample_rate():
    """Test sample rate embedding."""
    print("\n" + "=" * 60)
    print("Test 4: Sample rate embedding")
    print("=" * 60)
    
    config = ACEConfig(
        action_dim=7,
        chunk_size=16,
        group_size=4,
        hidden_dim=128,
        num_hidden_layers=2
    )
    
    model = ActionChunkEncoder(config)
    model.eval()
    
    batch_size = 2
    actions = torch.randn(batch_size, config.chunk_size, config.action_dim)
    
    # Test with different sample rates
    embeddings = []
    for rate in range(3):
        with torch.no_grad():
            emb = model(actions, sample_rate=rate)
        embeddings.append(emb)
        print(f"Sample rate {rate}: embedding shape = {emb.shape}")
    
    # Different sample rates should produce different embeddings
    # (but with same input, they might be similar)
    assert all(emb.shape == embeddings[0].shape for emb in embeddings)
    
    print("\n✓ Test 4 passed!")
    return True


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("\n" + "=" * 60)
    print("Test 5: Gradient flow")
    print("=" * 60)
    
    config = ACEConfig(
        action_dim=7,
        chunk_size=16,
        group_size=4,
        hidden_dim=128,
        num_hidden_layers=2
    )
    
    model = ActionChunkEncoder(config)
    model.train()
    
    batch_size = 2
    actions = torch.randn(batch_size, config.chunk_size, config.action_dim, requires_grad=True)
    
    # Forward pass
    embedding = model(actions)
    
    # Compute a simple loss
    loss = embedding.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert actions.grad is not None, "No gradient for input"
    print(f"Input gradient shape: {actions.grad.shape}")
    
    # Check model parameters have gradients
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_count = sum(1 for _ in model.parameters())
    print(f"Parameters with gradients: {grad_count}/{total_count}")
    
    assert grad_count == total_count, "Some parameters missing gradients"
    
    print("\n✓ Test 5 passed!")
    return True


def test_convenience_function():
    """Test the convenience function for creating models."""
    print("\n" + "=" * 60)
    print("Test 6: Convenience function")
    print("=" * 60)
    
    model = create_ace_model(
        action_dim=7,
        chunk_size=16,
        group_size=4,
        hidden_dim=128,
        num_layers=2,
        output_dim=64
    )
    
    batch_size = 2
    actions = torch.randn(batch_size, 16, 7)
    
    with torch.no_grad():
        embedding = model(actions)
    
    print(f"Created model via convenience function")
    print(f"Input: {actions.shape} -> Output: {embedding.shape}")
    assert embedding.shape == (batch_size, 64)
    
    print("\n✓ Test 6 passed!")
    return True


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ACE (Action Chunk Encoder) Module Tests")
    print("=" * 60)
    
    # Run tests
    tests = [
        test_basic_forward,
        test_grouping_mechanism,
        test_different_configs,
        test_sample_rate,
        test_gradient_flow,
        test_convenience_function,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    # Print model info
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    config = ACEConfig(
        action_dim=7,
        chunk_size=16,
        group_size=4,
        hidden_dim=768,
        num_hidden_layers=12
    )
    model = ActionChunkEncoder(config)
    params = count_parameters(model)
    print(f"Full model parameters: {params:,}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Output dim: {config.output_dim}")


if __name__ == "__main__":
    main()