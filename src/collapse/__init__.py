"""Calibration collapse under synthetic self-training.

Package layout:
    config      -- dataclass config tree, YAML-backed
    utils       -- seeding, IO, logging, device/memory helpers
    data/       -- dataset registry + unified QA schema + seed-pool construction
    markers/    -- epistemic marker lexicons (en, bn) + extraction
    modeling/   -- model loading, generation with logprobs, SFT training
    eval/       -- grading, calibration, diversity, perplexity, probes, IVD
    filters/    -- collapse-mitigation baselines we intend to break (H4)
    analysis/   -- onset detection, matched-baseline, figures, tables
    loop        -- the generational self-training orchestrator
"""

__version__ = "0.1.0"
