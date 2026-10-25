# Build the docs

This page shows how to build these docs, add a page, and add a figure or an
interactive explainer. The [writing guide](writing.md) sets how the pages
read.

## Step 1: Install the tools

```bash
pip install -e . --group docs
```

`--group` needs pip 25.1 or later. The API reference imports the package,
so the command installs it too.

## Step 2: Build the site

```bash
sphinx-build -W --keep-going docs docs/_build/html
```

The command writes the site to `docs/_build/html/`; open `index.html` in a
browser. `-W` turns every warning into an error, a malformed docstring
included, as the CI build does. CI builds the docs for every pull request
and publishes them from `main` to https://joshspeagle.com/dorothy/.

## Where the files are

| Path | Holds |
|------|-------|
| `docs/index.md` | The home page and the six section toctrees |
| `docs/start/`, `catalogue/`, `guides/`, `methods/`, `reference/`, `contribute/` | The pages of each section |
| `docs/reference/generated/` | The API pages that autosummary writes at build time; not in git |
| `docs/figures/` | SVG figures |
| `docs/_static/` | Logos, the favicon and explainer scripts |
| `docs/_themes/dorothy/` | The theme: `layout.html`, `home.html`, `theme.conf`, and `static/dorothy.css` and `static/dorothy.js` |
| `docs/references.bib` | The bibliography |
| `docs/conf.py` | The Sphinx configuration |
| `docs/dev/` | Development notes; the build excludes them |

## Add a page

1. Write the page in Markdown (MyST) in its section's folder, for example
   `docs/guides/my-task.md`. Start with a `#` title and a one- or
   two-sentence opener.
2. Add the page to its section's toctree in `docs/index.md`, in reading
   order:

   ````md
   ```{toctree}
   :caption: Guides
   :hidden:

   guides/train
   guides/my-task
   ```
   ````

3. Build the site and fix every warning.

The first page of a toctree is the section's landing page: the top bar links
to it.

## Add a figure

1. Draw the SVG in `docs/figures/`, in the style of `multi-head.svg`. Give it
   a `<title>` and `role="img"`.
2. Draw ink in `currentColor`. Take role colours from the theme tokens with
   fallbacks, such as `var(--d-teal, #1D7468)`.
3. Check the text size of the figure on a 360 px screen. If text renders
   under 11 px, also draw a narrow variant, `<name>-narrow.svg`, 360 units
   wide.
4. Include it with a caption of one sentence:

   ````md
   ::::{container} d-figure
   ```{raw} html
   :file: ../figures/name.svg
   :class: wide-only
   ```

   ```{raw} html
   :file: ../figures/name-narrow.svg
   :class: narrow-only
   ```

   :::{container} caption
   One sentence that states the concept.
   :::
   ::::
   ````

   Without a narrow variant, leave out the second block and the `:class:`
   line. At 600 px and less, the theme shows `narrow-only` and hides
   `wide-only`.

## Add an explainer

An explainer is an interactive figure on a How it works page. Each explainer
is one script in `docs/_static/explainers/`, such as `precision.js`. All
explainers share the styles in `docs/_static/explainers/explainer.css`.

1. Write the script without external libraries. It finds its container by
   its id, `d-<name>`, and builds the plot, the readout and the controls in
   it.
2. Draw the plot as an SVG with `role="img"` and an `aria-label`. Redraw it
   in a narrow layout at 600 px and less (`matchMedia`).
3. Put the readout in an element with `class="d-readout"` and
   `aria-live="polite"`.
4. Use native controls, which work from the keyboard. Use `<button>`
   elements, with `aria-pressed` for a toggle. Where the reader sets a
   number, use an `<input type="range">` slider with `aria-valuetext`.
5. If the explainer draws random numbers, seed them with `mulberry32`, as
   `calibration.js` does, so that every load is identical.
6. Take colours only from the `--d-*` theme tokens, through the classes of
   `explainer.css`.
7. Load the styles, the container and the script with a `{raw} html` block
   on the page:

   ````md
   ```{raw} html
   <link rel="stylesheet" href="../_static/explainers/explainer.css">
   <div class="d-lab" id="d-name">
   <p class="d-lab-h">One sentence that tells the reader what to do.</p>
   </div>
   <script src="../_static/explainers/name.js" defer></script>
   ```
   ````

   The paths are relative to the page. A page in a section folder starts
   them with `../`.
