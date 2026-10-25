# Writing guide

This page sets the rules for every page of these docs. The readers are
students first: astronomers who are new to the code, the catalogue or the
method. Experts read the same pages, so equations and the API stay one click
away.

## Voice

The rules follow ASD-STE100, Simplified Technical English, in spirit.

- Write sentences of at most 25 words, and instructions of at most 20.
- Put one idea in each sentence.
- Use the active voice and simple tenses.
- Write full sentences, with articles.
- Put a condition before the instruction it controls.
- Do not use marketing words, such as "powerful" or "seamless", or em dashes.
- State what the code does now. Why a change was made belongs in its pull
  request.
- State only measured numbers, or numbers computed from them with the
  computation shown.

## Terms

Use one word for one meaning, everywhere. The {doc}`glossary
<../reference/glossary>` defines each term; link a term with `{term}` where a
page first uses it.

| Use | For | Not |
|-----|-----|-----|
| spectrum | one observation of one star by one survey | input, data |
| survey | DESI, BOSS, LAMOST LRS or LAMOST MRS | instrument, input survey |
| label | one parameter or abundance of one star | target, value |
| label source | APOGEE or GALAH | label set, catalogue |
| prediction | the model's value μ | estimate, output |
| predicted scatter | s, before calibration | error, sigma |
| uncertainty | the reported spread, c s after calibration | error bar |
| calibration factor | c | correction |

## Pages

Each page has one job, and its section in the sidebar says which kind.

| Section | A page… |
|---------|---------|
| Start | gets a new reader running |
| The catalogue | describes the super-catalogue for people who use it |
| Guides | does one task, in numbered steps |
| How it works | explains one part of the method, with its equations |
| Reference | lists facts to look up: commands, settings, formats |
| Contribute | helps a student change the code |

- Open with one or two sentences that say what the page does for the reader.
- Give each section one concept. If a heading joins two concepts with "and",
  split the section.
- Give each fact one home. Link to it instead of repeating it.
- Leave defaults and setting names to the reference. An explanation names a
  setting only when the reader must change it.

## Figures

A figure shows one concept, the one its section explains.

- Draw it as SVG in `docs/figures/`, in the style of `multi-head.svg`. Draw
  ink with `currentColor`, and take role colours from the theme tokens with
  fallbacks, such as `var(--d-teal, #1D7468)`, so the figure follows both
  themes.
- Use colour for its role only: gold for a label, teal for the model, coral
  for a warning. Ink and grey draw everything else.
- Give the figure a `<title>` that describes it for a screen reader.
- Write a caption of one sentence that names the concept.
- Mark invented data: "illustrative" for a distribution, "example" for a
  value.

## Equations

Show the equation a section implements, with `$$ … $$`, and define each
symbol in the next sentence.

## Code

Give each code block one task. Show the command or call, then say what it
writes or returns.

## Citations

Cite a paper with `{cite:p}` and an entry in `docs/references.bib`.
