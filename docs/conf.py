"""Sphinx configuration for the DOROTHY documentation."""

import dataclasses
from functools import partial
from importlib import metadata

import pybtex.plugin
import sphinxcontrib.bibtex.plugin
from docutils import nodes
from pybtex.richtext import Tag, Text
from pybtex.style.formatting.alpha import Style as AlphaStyle
from pybtex.style.template import (
    FieldIsMissing,
    field,
    first_of,
    optional,
    sentence,
    tag,
)
from pybtex.style.template import join as pybtex_join
from pydantic import BaseModel
from sphinx import addnodes
from sphinxcontrib.bibtex.style.referencing import BracketStyle, PersonStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle
from sphinxcontrib.bibtex.style.template import join, node

# -- Project information -----------------------------------------------------
# autodoc imports the installed package (pip install -e .)

project = "DOROTHY"
author = "the DOROTHY developers"
copyright = "2025-2026, the DOROTHY developers"
release = metadata.version("dorothy")

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

exclude_patterns = ["_build", "dev", "Thumbs.db", ".DS_Store"]
# The citation style classes below cannot be pickled into the cache, and
# names such as `type` (a builtin and a config field) are ambiguous references
suppress_warnings = ["config.cache", "ref.python"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Docstrings are Google style
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
# Attribute types appear in class signatures; copying annotations such as
# NDArray[np.bool_] into reST fields would turn "bool_" into a link target
napoleon_attr_annotations = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Types of undocumented parameters (e.g. dataclass fields) stay in the signature
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
# Document each object once, in the module that defines it (packages
# re-export names in __all__)
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}
autosummary_generate = True
autosummary_imported_members = False


# Citations: author-year in round brackets, as in astronomy journals. Long
# author lists end in the BibTeX "and others", shown as "et al.".

ET_AL = Text(" ", Tag("em", "et al."))


def _split_others(persons):
    if len(persons) > 1 and str(persons[-1]) == "others":
        return persons[:-1], True
    return persons, False


@node
def _names(children, data, role, other=None, **kwargs):
    try:
        persons = data["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, data["entry"]) from None
    persons, et_al = _split_others(list(persons))
    person = data["style"].person
    formatted = [person.style_plugin.format(p, person.abbreviate) for p in persons]
    text = join(other=other, **kwargs)[formatted].format_data(data)
    # join() already ends three or more names with `other` (et al.)
    if et_al and (other is None or len(persons) < 3):
        text = Text(text, ET_AL)
    return text


@node
def _author_or_editor_or_title(children, data, **kwargs):
    return first_of[
        optional[_names("author", **kwargs)],
        optional[_names("editor", **kwargs)],
        tag("em")[field("title")],
    ].format_data(data)


@dataclasses.dataclass
class _EtAlPersonStyle(PersonStyle):
    def names(self, role, full):
        return _names(
            role=role,
            sep=self.sep,
            sep2=self.sep2,
            last_sep=self.last_sep,
            other=None if full else self.other,
        )

    def author_or_editor_or_title(self, full):
        return _author_or_editor_or_title(
            sep=self.sep,
            sep2=self.sep2,
            last_sep=self.last_sep,
            other=None if full else self.other,
        )


@dataclasses.dataclass
class AstronomyReferenceStyle(AuthorYearReferenceStyle):
    person: PersonStyle = dataclasses.field(default_factory=_EtAlPersonStyle)
    bracket_parenthetical: BracketStyle = dataclasses.field(
        default_factory=partial(BracketStyle, left="(", right=")")
    )
    bracket_textual: BracketStyle = dataclasses.field(
        default_factory=partial(BracketStyle, left="(", right=")")
    )
    author_year_sep: str = " "


@node
def _bibliography_names(children, context, role, **kwargs):
    try:
        persons = context["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context["entry"]) from None
    persons, et_al = _split_others(list(persons))
    style = context["style"]
    formatted = [style.format_name(p, style.abbreviate_names) for p in persons]
    text = pybtex_join(**kwargs)[formatted].format_data(context)
    return Text(text, " et al.") if et_al else text


class AstronomyBibliographyStyle(AlphaStyle):
    def format_names(self, role, as_sentence=True):
        names = _bibliography_names(role, sep=", ", sep2=" and ", last_sep=", and ")
        return sentence[names] if as_sentence else names


sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "astronomy", AstronomyReferenceStyle
)
pybtex.plugin.register_plugin(
    "pybtex.style.formatting", "astronomy", AstronomyBibliographyStyle
)
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "astronomy"
bibtex_default_style = "astronomy"

# -- Options for HTML output -------------------------------------------------

html_theme = "dorothy"
html_theme_path = ["_themes"]
html_title = "DOROTHY"
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_theme_options = {"source_repository": "https://github.com/joshspeagle/dorothy"}
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# -- Configuration models ----------------------------------------------------
# Configuration fields are documented once, in their Field(description=...).
# This hook lists them in a table under each model's docstring.


def _field_table(app, what, name, obj, options, lines):
    if what != "class" or not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        return
    lines += [
        "",
        ".. list-table:: Fields",
        "   :header-rows: 1",
        "   :widths: 22 25 18 35",
        "",
        "   * - Name",
        "     - Type",
        "     - Default",
        "     - Description",
    ]
    for field_name, info in obj.model_fields.items():
        annotation = getattr(info.annotation, "__name__", None) or str(info.annotation)
        default = (
            "required"
            if info.is_required()
            else f"``{info.get_default(call_default_factory=True)!r}``"
        )
        lines += [
            f"   * - ``{field_name}``",
            f"     - ``{annotation.replace('typing.', '')}``",
            f"     - {default}",
            f"     - {info.description or ''}",
        ]


# -- Theme -------------------------------------------------------------------


def _page_context(app, pagename, templatename, context, doctree):
    """Give the templates the sections (the root toctrees) and the outline.

    The page's section is the toctree that lists its top-level ancestor. The
    home page (front matter ``layout: home``) uses ``home.html``.
    """
    env = app.env
    root = app.config.root_doc
    toctrees = list(env.master_doctree.findall(addnodes.toctree))
    context["dorothy_sections"] = [
        {"caption": t["caption"], "docname": t["entries"][0][1]} for t in toctrees
    ]
    parents = app.builder.relations
    top = pagename
    while top in parents and parents[top][0] not in (root, None):
        top = parents[top][0]
    context["dorothy_section_index"] = next(
        (k for k, t in enumerate(toctrees) if top in (d for _, d in t["entries"])),
        None,
    )
    toc = env.tocs.get(pagename)
    sections = toc.next_node(nodes.bullet_list) if toc else None
    context["dorothy_outline"] = sections is not None and len(sections) > 1
    if (context.get("meta") or {}).get("layout") == "home":
        return "home.html"


def setup(app):
    app.connect("autodoc-process-docstring", _field_table)
    app.add_js_file("dorothy.js", loading_method="defer")
    app.connect("html-page-context", _page_context)
