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

1. Draw the SVG in `docs/figures/`, in the style of `multi-head.svg`: a
   `<title>`, `role="img"`, ink in `currentColor` and role colours from the
   theme tokens with fallbacks, such as `var(--d-teal, #1D7468)`.
2. If its text would render under 11 px on a 360 px screen, also draw a
   narrow variant, `<name>-narrow.svg`, 360 units wide.
3. Include it with a caption of one sentence:

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

An explainer is an interactive figure.

1. Write one self-contained JavaScript file in `docs/_static/explainers/`. It
   uses no external library, seeds its random numbers, takes colours from
   the `--d-*` tokens, has controls that work from the keyboard and reports
   its result in an element with `aria-live="polite"`.
2. Load it on its page with a `{raw} html` block that holds its container
   and the script:

   ````md
   ```{raw} html
   <div id="name-explainer"></div>
   <script src="../_static/explainers/name.js" defer></script>
   ```
   ````

   The path is relative to the page, so a page in a section folder starts
   it with `../`.
