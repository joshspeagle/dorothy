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
  combined vector through `trunk_hidden` layers. Its output describes the star
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

## Embeddings

{py:meth}`~dorothy.models.multi_head_mlp.MultiHeadMLP.get_embeddings` returns
a survey encoder's vector (`level="encoder"`) or the trunk output
(`level="trunk"`). Trunk embeddings place stars from different surveys in one
space; [anomaly detection](analysis.md#anomaly-detection) uses them. For the
MLP, {py:meth}`~dorothy.models.mlp.MLP.get_embeddings` returns the input of
the decoders by default.
