"""Text-to-SQL over DVF. See README.md."""

# Written into every prediction, so eval runs of different engines never mix.
# Bump it on any change to the prompt, the tools or the loop.
#   0.1  baseline
#   0.2  an answer is accepted only once the model has seen its final result
#   0.3  SQL the parser can't tokenize (unbalanced quote) goes back to the model
#        instead of crashing the question; never guess INSEE codes, read them
#        from dvf.communes
#   0.4  housing and living-environment views (rents, social housing, vacancy,
#        schools, transport, energy ratings, accessibility, long series, CDC
#        costs) allowed, with their conventions in the prompt
#   0.5  commune names resolved with the fuzzy search dvf.chercher_commune
#        (typos, accents, St/Saint, articles; ambiguous names -> clarify)
#        instead of an ILIKE example that matched several communes
ENGINE_VERSION = "0.5"
