# Architecture-Invariant Preference Vector Transfer

## The Fundamental Challenge

You've identified the core weakness in naive cross-model preference vector transfer: **architectural variance breaks the assumption of shared latent geometry**.

### 1. Vector Extraction Architectural Dependencies

**Original Implementation Problems:**
```python
# NAIVE APPROACH - FAILS ACROSS ARCHITECTURES
extractor.intervention_layer = 16  # Assumes all models have same depth
vector = get_activation_difference(model, layer=16)  # Wrong!
```

**Why This Fails:**
- **LLaMA-2 vs LLaMA-3**: Different attention mechanisms (GQA vs MHA), layer organizations
- **Mistral vs Gemma**: Different normalization patterns (RMSNorm vs LayerNorm), activation functions
- **Hidden Dimensions**: 4096 (7B) ≠ 5120 (13B) ≠ 2048 (smaller models)
- **Layer Correspondence**: Layer 16/32 in Model A ≠ Layer 16/40 in Model B functionally

### 2. Architecture-Aware Solutions Implemented

#### A. Functional Layer Mapping
Instead of layer indices, map by functional role:

```python
# FUNCTIONAL MAPPING - ARCHITECTURE AGNOSTIC
functional_mapper = FunctionalLayerMapper()

# Extract from "representation processing" stage (60% depth)
repr_layer_mistral = functional_mapper.get_functional_layer(mistral_arch, "representation")  # Layer 19/32
repr_layer_gemma = functional_mapper.get_functional_layer(gemma_arch, "representation")    # Layer 15/28

# Vectors extracted from functionally equivalent layers
vector_mistral = extract_at_layer(mistral, repr_layer_mistral)
inject_at_layer(gemma, vector_adapted, repr_layer_gemma)
```

#### B. Multi-Point Extraction
Extract from multiple architectural components:

```python
# ROBUST MULTI-POINT EXTRACTION
extractor = MultiPointExtractor(model_name)
vectors = extractor.extract_multi_point_vectors(summary_pairs, [
    'feature_extraction',    # Early layers (10% depth)
    'representation',        # Middle layers (60% depth) 
    'decision_making'        # Late layers (90% depth)
])

# Component-specific extraction
attention_vector = extract_from_attention_output(model, layer)
mlp_vector = extract_from_mlp_output(model, layer)
residual_vector = extract_from_residual_stream(model, layer)
```

#### C. Dimension-Adaptive Extraction
Handle different hidden dimensions:

```python
# DIMENSION ADAPTATION
class DimensionAdaptiveExtractor:
    def adapt_dimension(self, vector: Tensor, target_dim: int) -> Tensor:
        current_dim = vector.shape[0]
        
        if current_dim > target_dim:
            # PCA or truncation
            return vector[:target_dim]  # Or use learned projection
        elif current_dim < target_dim:
            # Zero padding or interpolation
            padding = torch.zeros(target_dim - current_dim)
            return torch.cat([vector, padding])
        return vector
```

### 3. Architecture-Aware Injection

#### A. Intelligent Hook Placement
```python
# ARCHITECTURE-AWARE HOOK PLACEMENT
class ArchitectureAwareInjector:
    def find_injection_points(self, model_arch: ModelArchitecture) -> List[str]:
        # Find residual stream access points specific to this architecture
        if "llama" in model_arch.model_name.lower():
            return model_arch.residual_stream_layers  # LLaMA-specific hooks
        elif "mistral" in model_arch.model_name.lower():  
            return model_arch.attention_output_layers  # Mistral-specific hooks
        # ... architecture-specific logic
```

#### B. Compatibility-Based Injection Strategy
```python
# INJECTION STRATEGY BASED ON COMPATIBILITY
def create_injection_strategy(source_arch, target_arch, vector):
    compatibility = compute_compatibility_score(source_arch, target_arch)
    
    if compatibility['overall'] > 0.8:
        return DirectInjectionStrategy(vector)
    elif compatibility['dimension'] > 0.7:
        return ScaledInjectionStrategy(vector, scale=compatibility['dimension'])
    else:
        return AdaptedInjectionStrategy(vector, require_adapter=True)
```

### 4. Advanced Linear Adapter Training

#### A. Architecture-Aware Adapter Design
```python
class ArchitectureAwareAdapter(nn.Module):
    def __init__(self, source_arch, target_arch):
        super().__init__()
        
        # Design components based on architectural differences
        self.needs_dimension_transform = source_arch.hidden_size != target_arch.hidden_size
        self.needs_attention_adaptation = source_arch.attention_type != target_arch.attention_type
        self.needs_norm_adaptation = source_arch.layer_norm_type != target_arch.layer_norm_type
        
        # Build adapter network accordingly
        if self.needs_dimension_transform:
            if abs(source_arch.hidden_size - target_arch.hidden_size) < 0.2 * max(...):
                self.adapter = ResidualAdapter(...)  # Small difference
            else:
                self.adapter = LinearAdapter(...)     # Large difference
```

#### B. Multi-Scale Adaptation
```python
class MultiScaleAdapter(nn.Module):
    """Handles different frequency components of preference vectors"""
    def __init__(self, source_arch, target_arch, num_scales=3):
        # Create adapters for different representational scales
        self.scale_adapters = nn.ModuleList([
            LinearAdapter(source_arch.hidden_size, target_arch.hidden_size)
            for _ in range(num_scales)
        ])
        # Combine scale-specific adaptations
        self.combiner = nn.Linear(num_scales * target_arch.hidden_size, target_arch.hidden_size)
```

#### C. Training with Architectural Constraints
```python
def train_with_architectural_awareness(adapter, source_model, target_model, vectors):
    for epoch in range(num_epochs):
        for vector in vectors:
            # Measure effect in source model
            source_effect = measure_steering_effect(source_model, vector)
            
            # Transform vector through adapter
            adapted_vector = adapter(vector)
            
            # Compute loss based on effect preservation + architectural consistency
            effect_loss = compute_effect_preservation_loss(adapted_vector, source_effect)
            arch_loss = compute_architectural_consistency_loss(vector, adapted_vector, alignment)
            
            total_loss = effect_loss + 0.1 * arch_loss
            total_loss.backward()
```

### 5. Expected Failure Modes and Mitigations

#### A. Why Direct Transfer is Expected to Fail
1. **Attention Mechanism Differences**: 
   - LLaMA-2: Multi-head attention
   - LLaMA-3: Grouped query attention  
   - **Impact**: Different information flow patterns

2. **Normalization Pattern Differences**:
   - Mistral: RMSNorm 
   - Some models: LayerNorm
   - **Impact**: Different activation distributions

3. **Tokenizer Differences**:
   - SentencePiece vs Byte-level BPE
   - **Impact**: Different input representations

#### B. How Adapters Address These Issues
1. **Learned Linear Transformations**: 
   ```
   W * vector_source ≈ vector_target_space
   ```
   Where W learns the rotation/scaling between representational spaces

2. **Multi-Component Adaptation**:
   - Dimension transformation: Handle size differences
   - Normalization adaptation: Handle distribution differences  
   - Attention adaptation: Handle mechanism differences

3. **Architectural Constraints**:
   - Preserve relative magnitudes
   - Maintain semantic relationships
   - Respect target architecture's constraints

### 6. Implementation Strategy

#### Phase 1: Architecture Analysis
```python
analyzer = ArchitectureAnalyzer()
source_arch = analyzer.analyze_model_architecture("mistralai/Mistral-7B-v0.1")
target_arch = analyzer.analyze_model_architecture("google/gemma-7b")
compatibility = analyzer.compute_compatibility_score(source_arch, target_arch)
```

#### Phase 2: Robust Vector Extraction
```python
robust_extractor = RobustCrossModelExtractor()
vector_sets = robust_extractor.extract_robust_preference_vectors(
    source_model, summary_pairs, target_architectures=[target_model]
)
```

#### Phase 3: Architecture-Aware Adaptation
```python
adapter_trainer = ArchitectureAwareAdapterTrainer()
alignment = adapter_trainer.analyze_architectural_alignment(source_model, target_model)
adapter = adapter_trainer.create_architecture_aware_adapter(alignment)
```

#### Phase 4: Adaptive Injection
```python
injector = ArchitectureAwareInjector()
strategy = injector.create_injection_strategy(source_arch, target_arch, vector)
result = injector.inject_with_strategy(target_model, vector, strategy, prompt)
```

### 7. Research Implications

#### If Architecture-Aware Methods Succeed:
- **Linear Subspace Hypothesis**: Preference representations exist in linearly-related subspaces across architectures
- **Functional Equivalence**: Different architectures converge to similar functional organizations
- **Transferable Abstractions**: High-level concepts transcend architectural details

#### If They Partially Succeed:
- **Constrained Universality**: Some preference dimensions transfer better than others
- **Architecture Families**: Within-family transfer works better than across-family
- **Complexity Gradients**: Simpler preferences transfer better than complex ones

#### If They Fail:
- **Architecture Specificity**: Each model family develops unique representational schemes
- **Non-Linear Relationships**: Linear adapters insufficient; need more complex transformations
- **Fundamental Incompatibility**: No shared structure for high-level preferences

### 8. Updated Experimental Protocol

1. **Architecture Compatibility Screening**: Test compatibility scores before attempting transfer
2. **Multi-Point Vector Extraction**: Extract from functionally equivalent layers
3. **Graduated Adaptation**: Try direct → scaled → linear adapted → multi-scale adapted
4. **Component-Specific Testing**: Test attention vs MLP vs residual stream vectors separately
5. **Robustness Validation**: Test multiple vector types and extraction points

This architecture-aware approach transforms the research from "will naive transfer work?" (likely no) to "what architectural constraints allow preference transfer?" (much more scientifically valuable).