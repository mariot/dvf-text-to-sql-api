"""Text-to-SQL over DVF. See README.md."""

# Written into every prediction, so eval runs of different engines never mix.
# Bump it on any change to the prompt, the tools or the loop.
#   0.1  baseline
#   0.2  an answer is accepted only once the model has seen its final result
#   0.3  SQL the parser can't tokenize (unbalanced quote) goes back to the model
#        instead of crashing the question; never guess INSEE codes, read them
#        from dvf.communes
ENGINE_VERSION = "0.3"
