"""
Guardrails package — implements the 6-layer guardrail pipeline described
in the platform spec:

    Layer 1  Input Guardrails    (backend.guardrails.input_guardrails)
    Layer 2  Query Control       (backend.guardrails.query_control)
    Layer 3  Retrieval Safety    (backend.guardrails.retrieval_safety)
    Layer 4  Context Control     (backend.guardrails.context_control)
    Layer 5  Generation Safety   (backend.guardrails.generation_safety)
    Layer 6  Output Safety       (backend.guardrails.output_safety)

``backend.guardrails.messages`` holds every standardised user-facing
refusal/block message so wording stays consistent across layers.

``backend.guardrails.pipeline`` orchestrates all six layers for a single
chat turn and produces the structured audit record.
"""
