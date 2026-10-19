# Models

Both models are multilayer perceptrons. Each hidden layer is a linear map
followed by a normalization (`layernorm` {cite:p}`ba2016layernorm` or
`batchnorm`), an activation (`gelu` {cite:p}`hendrycks2016gelu`, `relu` or
`silu`) and, optionally, dropout. Both output an array of shape
`(2, n_parameters)` per star: the predicted value μ and the log-scatter ln s
of every parameter, in {term}`normalized units`.

Both end in a {py:class}`~dorothy.models.output.LabelAwareOutput`: a label
decoder predicts μ, and a parallel error decoder predicts ln s from the same
input and the reported label errors
([Scatter and label errors](training.md#scatter-and-label-errors)).

## MLP

{py:class}`~dorothy.models.mlp.MLP` reads one survey. It flattens the
`(3, n_wavelengths)` input and passes it through the hidden layers
(`model.hidden_layers`, by default 5000, 2000, 1000, 500, 200 and 100 units).
The two decoders take over the last hidden layer: each has one hidden layer
of that size on the output of the layer before. Configure it with the `model`
section.

## Multi-head MLP

{py:class}`~dorothy.models.multi_head_mlp.MultiHeadMLP` reads several surveys
and predicts for several label sources. Configure it with the
`multi_head_model` section.

```{raw} html
:file: ../figures/multi-head.svg
```

Encoders
: One {py:class}`~dorothy.models.multi_head_mlp.SurveyEncoder` per survey
  flattens that survey's spectrum and maps it through `encoder_hidden` layers
  to a vector of `latent_dim` numbers. The encoders absorb the differences
  between wavelength grids.

Combination
: `combination_mode: concat` places the surveys' vectors side by side, in a
  fixed order, with zeros for the surveys a star lacks. `mean` averages the
  vectors of the surveys the star has. Either way the next layer sees an input
  of fixed size, whatever subset of surveys the star has.

Trunk
: The {py:class}`~dorothy.models.multi_head_mlp.SharedTrunk` passes the
  combined vector through `trunk_hidden` layers, by default as a
  [residual stream](#residual-stream). Its output describes the star
  independently of the surveys and label sources.

Heads
: One {py:class}`~dorothy.models.multi_head_mlp.OutputHead` per label source
  maps the trunk output to that source's μ and ln s with its two decoders,
  each with `output_hidden` layers. Each head reads only its own source's
  label errors. `head_dropout` sets the dropout of each head.

The {term}`has_data` mask tells the model which surveys each star has. Only
the stars that have a survey use its encoder's output, and a survey that no
star in the batch has is skipped. A model trained on several surveys can therefore
predict from any one of them:
`model(X, survey="boss")` for a single-survey input, or a dict of surveys with
`has_data`.

## Residual stream

By default (`multi_head_model.residual_stream: true`) the trunk and the
hidden layers of the heads form a residual stream of width d = `latent_dim`.
`residual_stream: false` builds plain blocks instead: a trunk of hidden
blocks with a 128-number output, and heads whose error decoder concatenates
the log label errors to its input. Checkpoints saved before this option have
plain blocks.

The stream starts at the combined encodings. In `concat` mode with several
surveys, a linear map takes them from `n_surveys × latent_dim` numbers to d;
otherwise they have d numbers already. Each entry h of `trunk_hidden` adds one
pre-norm residual block {cite:p}`he2016identity`,

$$
x \leftarrow x + W_2 \, \mathrm{act}\left(W_1 \, \mathrm{norm}(x)\right),
$$

where W₁ maps d to h numbers and W₂ maps them back to d, and a final
normalization closes the stream. The normalization is the configured one,
`layernorm` by default. Every block reads and writes the same vector, so the
stream is one representation that can be probed at every depth.

Each head's two decoders read the closed stream. Each decoder adds one
residual block of hidden size h for each entry h of `output_hidden`, then a
normalization and a linear map to one output per parameter. The error
decoder adds a linear map of the log label errors u (see
[Scatter and label errors](training.md#scatter-and-label-errors)) to its copy
of the stream instead of concatenating u:

$$
\mu = D_\mu(z), \qquad \ln s = D_s(z + W_u u),
$$

where z is the closed stream and W_u has no bias. Its residual block makes
ln s a nonlinear function of u.

The block count follows the hidden layer sizes, not the parameter count. With
the defaults, the residual trunk and heads have more parameters than the
plain ones: 472,737 against 316,513 for one survey and one label source. The
survey encoders hold most of the parameters in either case.

## Embeddings

{py:meth}`~dorothy.models.multi_head_mlp.MultiHeadMLP.get_embeddings` returns
a survey encoder's vector (`level="encoder"`) or the trunk output
(`level="trunk"`), which for a residual stream is the closed stream. Trunk
embeddings place stars from different surveys in one space;
[anomaly detection](analysis.md#anomaly-detection) uses them. For the
MLP, {py:meth}`~dorothy.models.mlp.MLP.get_embeddings` returns the input of
the decoders by default.
