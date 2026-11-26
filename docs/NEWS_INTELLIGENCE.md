# News intelligence v1

`GET /api/disclosures/{entry_id}/intelligence` works on existing local and
hosted workspace records, using the existing session isolation. No new database
schema, model invocation, remote API, credentials, or training is needed.

The response includes source metadata, model version, event cards, exact
source offsets, validated numbers/dates inside event evidence, the inferred
stage only when a valid stage span exists, conditional category-specific
implications, potentially affected stakeholder groups and verification tasks.
The UI renders it within both Analyze results and saved inbox details. The full
JSON is downloadable separately. The original event evidence and existing
brief download remain available.

This is an extractive intelligence layer, not a generative language model.
Evidence spans are validated against stored text, which does not prove the
publisher is authentic or the classification semantically correct. The
classifier's category grouping can combine same-category events. Historical
fixtures are paraphrases. Model scores are not calibrated probabilities.
Denied/terminated/unclear events do not activate positive event scenarios;
review results provide a verification prompt, not invented impact. The
playbook is explicitly hypothetical and requires analyst validation.

Validation: python -m unittest tests.test_intelligence -v; existing local and
hosted API tests also exercise the endpoint and browser isolation. The
existing CI runs the full suite and checks JavaScript syntax. Browser E2E,
independent expert review, live ingestion, market-context display and
independent model training/evaluation are separate gates.
