"""Text-to-SQL over DVF. See README.md."""

# Written into every prediction, so eval runs of different engines never mix.
# Bump it on any change to the prompt, the tools or the loop.
#   0.1  baseline
#   0.2  an answer is accepted only once the model has seen its final result
ENGINE_VERSION = "0.2"
