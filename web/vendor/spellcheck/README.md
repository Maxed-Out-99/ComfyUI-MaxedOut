# Offline prompt spelling assets

This directory contains the browser-side dependencies for
`web/nodes/prompt_spellcheck.js`:

- `nspell.min.js` — browser bundle of `nspell` 2.1.5 and its `is-buffer`
  dependency.
- `en.aff` / `en.dic` — the US-English-compatible Hunspell data distributed
  by `dictionary-en` 4.0.0.

They are vendored so prompt spelling suggestions work locally without sending
prompt text or individual words over the network. See the adjacent license
files for attribution and redistribution terms.
