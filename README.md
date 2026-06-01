# Shared Expert Pools for MoE Models Principle independently validated by UniPool (arXiv:2605.06665).
A phase-local expert-sharing and serving proposal for trillion-scale LLMs.

This repository explores a theoretical architecture for Mixture-of-Experts (MoE) language models based on phase-local shared expert pools.

The core idea is simple: expert capacity does not necessarily need to scale linearly with model depth. Instead of giving every MoE layer its own independent expert pool, adjacent layers within the same local processing phase may share a common pool of experts, while keeping their own attention, local dense processing, and routers.

The core premise has recently received independent empirical support from UniPool (arXiv:2605.06665), which shows that shared expert pools can match or outperform standard per-layer MoE baselines at smaller scales.

This repository extends that principle toward a trillion-scale, deployment-oriented design, including:

- phase-local expert pools instead of one global pool;
- layer-specific MLP routers and micro dense-FFN protection;
- a 1.15T / 235B-active reference architecture;
- trunk/expert GPU disaggregation;
- depth-aware expert dispatch;
- an estimated 1.5 GPUs per logical replica on a 72-GPU node.

No experimental validation has been performed for the specific architecture proposed here. This is a testable architecture and serving hypothesis, supported by parameter-count arithmetic, hardware-layout analysis, and the empirical motivation provided by UniPool.
