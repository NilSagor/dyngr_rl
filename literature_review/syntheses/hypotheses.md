# Research Hypotheses in Dynamic Graph Representation Learning

## 

## Hypothesis Categories

### 1. Architectural Hypothes

#### H1: Memory-Attention Complementarity

**Hypothesis**: Memory-augmented architectures and attention-based mechanisms 
are complementary approaches that can be combined for better dynamic graph representation learning.

**Supporting Evidence**:
- TAWRMAC combines memory with co-occurence attention
- Performance improvent over pure attention (DyGFormer) or pure memory (TGN)

**Testable Prediction**: A hybrid model combining TGN's memory module with DyGFormer's attention mechanism will outperform both individaul approaches.

**Experimental Design**:
- Baseline: TGN (memory only), DyGFormer (attention only)
- Hybrid: TGN + DyGFormer attention layers
- Evaluation: AP, AUC, MRR on Wikipeida, Reddit, MOOC datasets


#### H3: Frequency-Domain Effectiveness

**Hypotheses**: Frequency-domain approaches are particulary effective for dynamic graphs with periodic or quasi-periodic temporal patterns.

**Supporting Evidence**:
- FreeDyG shows 5-8% improvement on periodic datasets
- Signal processing theory supports frequency analysis for periodic signals

**Testable Prediction**: FreeDyG's performance advantage will correlate with the degree of periodicity in temporal patterns.

**Experimental Design**:

- Me



## Meta-Hypothese

### 

## Conclusion