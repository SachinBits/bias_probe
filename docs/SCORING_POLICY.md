# Scoring policy

Responses are scored by deterministic probe-aware rules first, with a fallback
judge only when the rules cannot settle the answer. Refusals and hedges are not
wrong answers; `unknown` and `unclear` are content labels when a probe defines
them; and `unscorable` means the scorer could not assign a label.

verdicts are probe-level signals for this model, snapshot, temperature and wording. Never claim universal bias.

Accuracy excludes refusals, hedges, and unscorable answers. Statistics use only
successful response records matching the current model, provider, temperature,
and max-token configuration. Superseded records remain visible in the report
but do not affect its metrics.

For trap probes, a significant accuracy drop is labelled `BIASED (pattern)`
only when the most common changed-variant error is the original variant's
famous answer. A significant drop caused by a different wrong label is
`FAILED (other error)`.
