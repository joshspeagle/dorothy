/* Explainer: combining the survey encodings (How it works / The model).

   The reader picks the surveys a star has and the combination mode. Each
   survey's encoder gives a vector, shown with 4 illustrative numbers in place
   of latent_dim. concat places the vectors side by side in the fixed survey
   order, with zeros for the surveys the star lacks; mean averages the vectors
   of the surveys it has. This is MultiHeadMLP._combine_encodings.
   Deterministic: no random numbers. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var root = document.getElementById("d-combination");
  if (!root) return;

  var SURVEYS = [
    { name: "BOSS", short: "BOSS", v: [0.9, 0.4, 0.3, 0.6] },
    { name: "DESI", short: "DESI", v: [0.7, 0.2, 0.5, 0.2] },
    { name: "LAMOST LRS", short: "LRS", v: [1.1, 0.6, 0.1, 0.4] },
    { name: "LAMOST MRS", short: "MRS", v: [0.8, 0.3, 0.4, 0.5] }
  ];
  var has = [true, true, false, false], mode = "concat", msg = "";

  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, attrs, parent, text) {
    var e = document.createElement(tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function arrow(svg, x1, y1, x2, y2, missing) {
    el("path", { class: missing ? "arrow-missing" : "arrow", d: "M" + x1 + " " + y1 + "L" + x2 + " " + y2 }, svg);
    var a = Math.atan2(y2 - y1, x2 - x1), c = Math.cos(a), s = Math.sin(a), bx = x2 - 7 * c, by = y2 - 7 * s;
    el("path", { class: missing ? "tip-missing" : "tip", d: "M" + x2 + " " + y2 + "L" + (bx - 3.2 * s) + " " + (by + 3.2 * c) + "L" + (bx + 3.2 * s) + " " + (by - 3.2 * c) + "Z" }, svg);
  }

  var plot = html("div", { class: "d-lab-plot" }, root);
  var svg = el("svg", { role: "img", "aria-label": "The surveys of one star, their encoder vectors, and the trunk input that the chosen combination makes from them." }, plot);
  var out = html("div", { class: "d-readout", "aria-live": "polite" }, root);
  function cell(key, cls) {
    var d = html("div", {}, out);
    html("span", { class: "k" }, d, key);
    return html("span", { class: cls }, d);
  }
  var sizeEl = cell("Trunk input", "big");
  var fromEl = cell("Made from", "big mid");
  var noteEl = cell("", "d-verdict ok");

  var ctl = html("div", { class: "d-lab-ctl wrap" }, root);
  var g1 = html("fieldset", { class: "d-toggles" }, ctl);
  html("legend", {}, g1, "Surveys the star has");
  var sButtons = SURVEYS.map(function (s, i) {
    var b = html("button", { class: "d-toggle", type: "button", "aria-pressed": String(has[i]) }, g1, s.name);
    b.addEventListener("click", function () {
      if (has[i] && has.filter(Boolean).length === 1) {
        msg = "A star needs at least one survey.";
      } else {
        has[i] = !has[i];
        msg = "";
      }
      update();
    });
    return b;
  });
  var g2 = html("fieldset", { class: "d-toggles" }, ctl);
  html("legend", {}, g2, "Combination");
  var mButtons = ["concat", "mean"].map(function (m) {
    var b = html("button", { class: "d-toggle", type: "button", "aria-pressed": String(m === mode) }, g2, m);
    b.addEventListener("click", function () { mode = m; msg = ""; update(); });
    return b;
  });

  var mq = window.matchMedia("(max-width: 600px)");
  function update() {
    var narrow = mq.matches;
    var W = narrow ? 340 : 680, RH = 40, T = 30, HH = T + 4 * RH + (narrow ? 46 : 30);
    var cw = narrow ? 27 : 44, ch = 22;
    var xName = 0, xEnc = narrow ? 46 : 96, xVec = narrow ? 94 : 168, xOut = narrow ? 220 : 440;
    svg.textContent = "";
    svg.setAttribute("viewBox", "0 0 " + W + " " + HH);
    el("text", { class: "head", x: xVec + 2 * cw, y: 16, "text-anchor": "middle" }, svg, "encodings");
    el("text", { class: "head", x: xOut + 2 * cw, y: 16, "text-anchor": "middle" }, svg, "trunk input");
    function vec(x, y, v, missing, zeros) {
      for (var j = 0; j < 4; j++) {
        var cls = missing ? (zeros ? "zero" : "missing") : "vec";
        el("rect", { class: cls, x: x + j * cw + 0.5, y: y + 0.5, width: cw - 1, height: ch - 1 }, svg);
        if (!missing || zeros) {
          el("text", { class: missing ? "tick" : "onfill", x: x + j * cw + cw / 2, y: y + 15, "text-anchor": "middle", "font-size": narrow ? 10 : 12 }, svg, missing ? "0" : v[j].toFixed(narrow ? 1 : 2));
        }
      }
    }
    var n = 0, mean = [0, 0, 0, 0];
    SURVEYS.forEach(function (s, i) {
      var y = T + i * RH, cy = y + ch / 2, missing = !has[i];
      el("text", { class: missing ? "tick" : "ink", x: xName, y: cy + 5 }, svg, narrow ? s.short : s.name);
      arrow(svg, xEnc - 10, cy, xEnc - 2, cy, missing);
      el("path", { class: missing ? "missing" : "model", d: "M" + xEnc + " " + (y - 4) + "L" + (xEnc + 40) + " " + (y + 4) + "V" + (y + ch - 4) + "L" + xEnc + " " + (y + ch + 4) + "Z" }, svg);
      arrow(svg, xEnc + 42, cy, xVec - 4, cy, missing);
      vec(xVec, y, s.v, missing, false);
      if (!missing) { n++; for (var j = 0; j < 4; j++) mean[j] += s.v[j]; }
      if (mode === "concat") {
        arrow(svg, xVec + 4 * cw + 6, cy, xOut - 6, cy, missing);
        vec(xOut, y, s.v, missing, true);
      }
    });
    if (mode === "mean") {
      mean = mean.map(function (x) { return x / n; });
      var my = T + 1.5 * RH;
      SURVEYS.forEach(function (s, i) {
        if (has[i]) arrow(svg, xVec + 4 * cw + 6, T + i * RH + ch / 2, xOut - 6, my + ch / 2, false);
      });
      vec(xOut, my, mean, false, false);
      el("text", { x: xOut + 2 * cw, y: my + ch + 18, "text-anchor": "middle" }, svg, "mean of " + n);
    } else {
      var bx = xOut + 4 * cw + (narrow ? 3 : 8), bw = narrow ? 4 : 5;
      el("path", { class: "ax", d: "M" + bx + " " + T + "h" + bw + "V" + (T + 3 * RH + ch) + "h-" + bw }, svg);
    }
    if (narrow) {
      el("text", { x: 0, y: HH - 18 }, svg, "4 numbers stand for the latent_dim");
      el("text", { x: 0, y: HH - 2 }, svg, "numbers of each vector (illustrative).");
    } else {
      el("text", { x: 0, y: HH - 4 }, svg, "4 numbers stand for the latent_dim numbers of each vector (illustrative).");
    }
    var missingNames = SURVEYS.filter(function (s, i) { return !has[i]; }).map(function (s) { return s.name; });
    if (mode === "concat") {
      sizeEl.textContent = "4 × 4 = 16";
      fromEl.textContent = n + (n === 1 ? " encoding, " : " encodings, ") + (4 - n) + (4 - n === 1 ? " zero block" : " zero blocks");
    } else {
      sizeEl.textContent = "4";
      fromEl.textContent = "mean of " + n + (n === 1 ? " encoding" : " encodings");
    }
    noteEl.textContent = msg;
    noteEl.className = "d-verdict" + (msg ? "" : " ok");
    sButtons.forEach(function (b, i) { b.setAttribute("aria-pressed", String(has[i])); });
    mButtons.forEach(function (b, i) { b.setAttribute("aria-pressed", String(["concat", "mean"][i] === mode)); });
    svg.setAttribute("aria-label", "The star has " + SURVEYS.filter(function (s, i) { return has[i]; }).map(function (s) { return s.name; }).join(", ") + ". " + (mode === "concat" ? "Concat gives 16 numbers" + (missingNames.length ? ", with zeros for " + missingNames.join(", ") : "") : "Mean averages " + n + " encodings into 4 numbers") + ".");
  }
  if (mq.addEventListener) mq.addEventListener("change", update);
  update();
})();
