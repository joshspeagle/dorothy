# The model

The model maps the spectra of a star to a prediction μ and a log predicted
scatter ln s for every label. This page explains its parts: the encoders, the
combination, the trunk, the heads and the residual stream.

## Two models

DOROTHY has two models. Both output an array of shape `(2, n_parameters)` per
star, μ and ln s of every parameter, in {term}`normalized units`.

- {py:class}`~dorothy.models.mlp.MLP` reads one survey and predicts the
  labels of one label source. The `model` section configures it.
- {py:class}`~dorothy.models.multi_head_mlp.MultiHeadMLP` reads several
  surveys and predicts the labels of several label sources. The
  `multi_head_model` section configures it. Several surveys or several label
  sources need it.

Both are built from hidden blocks. A hidden block maps its input h to

$$
h \rightarrow \mathrm{drop}\left(\mathrm{act}\left(\mathrm{norm}(W h + b)\right)\right) .
$$

Here norm is a layer normalization {cite:p}`ba2016layernorm` or a batch
normalization. act is GELU {cite:p}`hendrycks2016gelu`, ReLU or SiLU, and drop
is an optional dropout.

The MLP flattens the `(3, n_wavelengths)` input and passes it through the
hidden blocks of `model.hidden_layers`. Its output head takes over the last
hidden block. Each of the two decoders of the head
([The label-error decoder](uncertainties.md#the-label-error-decoder)) has one
hidden block of that size, on the output of the block before. μ
therefore passes through exactly the configured layers.

## The multi-head model

The multi-head model has one encoder per survey, a shared trunk and one head
per label source.

::::{container} d-figure
```{raw} html
:file: ../figures/multi-head.svg
:class: wide-only
```

```{raw} html
:file: ../figures/multi-head-narrow.svg
:class: narrow-only
```

:::{container} caption
The multi-head model: one encoder per survey, shared layers, and one head per
label source.
:::
::::

## Encoders

One {py:class}`~dorothy.models.multi_head_mlp.SurveyEncoder` per survey
flattens that survey's `(3, n_wavelengths)` input. It passes it through the
hidden blocks of `encoder_hidden`, then through one more block, without
dropout, to a vector of `latent_dim` numbers. Each survey has its own grid
and resolution, and its encoder learns its own map. The layers after the
encoders therefore see vectors of one length from every survey.

## Combination

The combination joins the vectors $e_k$ of the surveys a star has into one
input for the trunk. The {term}`has_data` mask says which surveys each star
has. With $S$ the set of those surveys and $K$ the number of surveys of the
model,

$$
z_\text{concat} = \left(\tilde e_1, \ldots, \tilde e_K\right), \quad
\tilde e_k = \begin{cases} e_k & k \in S, \\ 0 & k \notin S, \end{cases}
$$

$$
z_\text{mean} = \frac{1}{|S|} \sum_{k \in S} e_k .
$$

`concat` places the vectors side by side, in the configured order of the
surveys, so $z$ has $K$ × `latent_dim` numbers. `mean` averages the vectors
of the surveys the star has, so $z$ has `latent_dim` numbers. Either way the
trunk sees an input of fixed size, whatever subset of surveys the star has.

```{raw} html
<link rel="stylesheet" href="../_static/explainers/explainer.css">
<div class="d-lab" id="d-combination">
<p class="d-lab-h">Choose the surveys a star has and the combination. See which encodings reach the trunk.</p>
</div>
<script src="../_static/explainers/explainer.js" defer></script>
<script src="../_static/explainers/combination.js" defer></script>
```

A survey that no star in a batch has is skipped. A model trained on several
surveys therefore predicts from any subset of them: `model(X, survey="boss")`
for the spectra of one survey, or a dict of surveys with their `has_data`
masks.

## Trunk

The {py:class}`~dorothy.models.multi_head_mlp.SharedTrunk` passes the
combined vector through the `trunk_hidden` layers, which form a
[residual stream](#residual-stream) unless `residual_stream` is false. Its
output describes the star whatever surveys observed it, and every head reads
it.

## Heads

One {py:class}`~dorothy.models.multi_head_mlp.OutputHead` per label source
maps the trunk output to that source's μ and ln s. Each head is a
{py:class}`~dorothy.models.output.LabelAwareOutput` with two decoders: a
label decoder for μ and an error decoder for ln s
([The label-error decoder](uncertainties.md#the-label-error-decoder)). Each
decoder has the `output_hidden` layers.
Each head reads only the label errors of its own label source.
`head_dropout` sets the dropout of each head.

## Residual stream

With `multi_head_model.residual_stream: true`, the trunk and the hidden
layers of the heads form a residual stream of width d = `latent_dim`.
The stream starts at the combined encodings. In `concat` mode with several
surveys, a linear map takes them from $K$ × `latent_dim` numbers to d;
otherwise they have d numbers already. Each entry h of `trunk_hidden` adds one
pre-norm residual block {cite:p}`he2016identity`,

$$
x \rightarrow x + W_2 \, \mathrm{act}\left(W_1 \, \mathrm{norm}(x)\right) .
$$

Here $W_1$ maps d numbers to h numbers and $W_2$ maps them back to d. A final
normalization closes the stream.

Each decoder of a head reads its own copy of the closed stream z. It adds one
residual block of hidden size h for each entry h of `output_hidden`. A
normalization and a linear map to one output per parameter follow. The
error decoder adds a linear map of the log label errors u to its copy:

$$
\mu = D_\mu(z), \qquad \ln s = D_s(z + W_u u),
$$

where $W_u$ has no bias. The residual block of $D_s$ makes ln s a nonlinear
function of u.

`residual_stream: false` builds plain hidden blocks instead: a trunk with a
128-number output, and error decoders that read u concatenated to their
input. The block count follows the layer sizes, not the parameter count.
Take one survey of 4506 wavelengths, 11 parameters and the default layer
sizes of the [Configuration](../reference/configuration.md) reference.
`count_parameters_by_component()` then reports 472,737 parameters for the residual trunk and head, and 316,513
for the plain ones. The encoder, with 14,503,168, holds most of the
parameters either way.

## Embeddings

{py:meth}`~dorothy.models.multi_head_mlp.MultiHeadMLP.get_embeddings` returns
the vector of a survey's encoder (`level="encoder"`) or the trunk output
(`level="trunk"`), which for a residual stream is the closed stream. Trunk
embeddings place stars from different surveys in one space, and
[anomaly detection](../guides/read-predictions.md#step-4-flag-unusual-spectra) uses
them. For the MLP, {py:meth}`~dorothy.models.mlp.MLP.get_embeddings` returns
the input of the output head unless `layer_index` names another layer.
