"""
Standardised, user-facing guardrail response messages.

Centralised here so wording is consistent and can be audited/changed in
one place, per the "Response Message Standards" section of the spec.
"""

INPUT_TOO_LONG = (
    "Your query exceeds the maximum allowed length. Please shorten your "
    "question and try again."
)

TOXIC_INPUT = (
    "I'm unable to process this request. Please keep queries respectful "
    "and professional."
)

PROMPT_INJECTION_DETECTED = (
    "This type of input cannot be processed. Please ask a genuine question."
)

JAILBREAK_DETECTED = (
    "This type of input cannot be processed. Please ask a genuine question."
)

OUT_OF_SCOPE = (
    "I can help with questions related to the enterprise knowledge base. "
    "Please ask a relevant question."
)

NO_RELEVANT_DOCUMENTS = (
    "I couldn't find relevant information in the knowledge base for your query."
)

HALLUCINATION_DETECTED = (
    "I was unable to generate a reliable answer from the available documents. "
    "Please try rephrasing your question."
)

OUTPUT_BLOCKED_SAFETY = (
    "The generated response could not be returned due to safety filters. "
    "Please try a different question."
)
