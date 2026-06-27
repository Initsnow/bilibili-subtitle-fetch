# ASR Subtitle Writing Rules

Use these rules when rewriting Bilibili subtitles into a Chinese tutorial, article, or polished long-form note.

## Core Principles

Correct ASR errors silently. Infer likely corrections from context without marking them:

- Homophone or near-homophone mistakes, such as `势力` to `实力` when the context requires capability.
- Confused common words, such as `反应` and `反映`.
- Domain terms, names, product names, commands, parameters, and technical vocabulary.
- Bad sentence breaks, bad word segmentation, and words merged because of speech pauses.
- Filler words and oral scaffolding, such as `嗯`, `啊`, `就是说`, `然后`, repeated openings, greetings, and needless transitions.

Stay faithful to the transcript. Do not add information, opinions, examples, definitions, or recommendations that the subtitles do not contain. If a passage is unclear, use the most reasonable local-context interpretation and do not expand beyond it.

Write concisely:

- Convert spoken language into written Chinese.
- Remove repetition and low-information phrasing.
- Keep every sentence informative.
- Express the full meaning with as few words as accuracy allows.

Maintain technical rigor:

- Preserve data, formulas, code, configuration, commands, file names, and parameter names exactly when they are clear.
- Preserve causal relationships, conditions, scope limits, and operation order.
- Do not soften precise constraints into vague wording.
- Keep procedural steps in the same sequence as the original explanation.

## Style

Use third-person or impersonal narration. Do not use `我`, `你`, or `大家`.

Use compact paragraphs. Avoid empty openings and generic endings, including `总的来说`, `希望对大家有帮助`, and similar filler.

Start directly with the subject. Do not write a separate introduction or conclusion unless the transcript itself contains substantive content that belongs there.

Prefer headings that expose the structure of the material, not decorative headings. Use lists only for real procedures, enumerations, comparisons, or checklists.

## Output Check

Before finalizing, verify that:

- Every claim is supported by the subtitles.
- Obvious ASR mistakes have been corrected.
- Technical details and step order remain intact.
- The text reads as written prose rather than subtitle fragments.
- No filler introduction, closing, or unsupported background has been added.
