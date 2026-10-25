/* Explainer: the precision encoding (How it works / Preprocessing).

   The reader sets the uncertainty sigma of one pixel, in normalized units,
   and sees rho = ln(1 + 1/sigma) on the curve. "Mask the pixel" shows the
   value of a masked pixel: rho = 0, the value that sigma -> infinity
   approaches, while the sigma channel itself would hold 0, the value of a
   perfect measurement. Deterministic: no random numbers. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var root = document.getElementById("d-precision");
  if (!root) return;

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
  function rho(sigma) { return Math.log1p(1 / sigma); }
  function fmt(v) {
    if (v >= 100) return v.toFixed(0);
    return v.toPrecision(3);
  }

  var LMIN = -3, LMAX = 3, RMAX = 7.5;
  var plot = html("div", { class: "d-lab-plot" }, root);
  var svg = el("svg", { role: "img", "aria-label": "The precision encoding rho = ln(1 + 1/sigma) against sigma on a logarithmic axis, with the current pixel marked." }, plot);
  var out = html("div", { class: "d-readout" }, root);
  function cell(key, cls) {
    var d = html("div", {}, out);
    html("span", { class: "k" }, d, key);
    return html("span", { class: cls }, d);
  }
  var sigmaEl = cell("σ (normalized units)", "big");
  var rhoEl = cell("ρ, the model input", "big");
  var rawEl = cell("σ channel without the encoding", "big mid");
  var noteEl = cell("", "d-verdict");
  var ctl = html("div", { class: "d-lab-ctl" }, root);
  var sl = html("div", { class: "d-slider" }, ctl);
  var lab = html("label", { for: "d-precision-sigma" }, sl, "Pixel uncertainty ");
  var num = html("span", { class: "num" }, lab);
  var slider = html("input", { id: "d-precision-sigma", type: "range", min: String(LMIN), max: String(LMAX), step: "0.01", value: "-1" }, sl);
  var scale = html("div", { class: "d-scale", "aria-hidden": "true" }, sl);
  ["0.001", "0.01", "0.1", "1", "10", "100", "1000"].forEach(function (s) { html("span", {}, scale, s); });
  var mask = html("button", { class: "d-toggle", type: "button", "aria-pressed": "false" }, ctl, "Mask the pixel");
  var status = html("p", { class: "d-sr", "aria-live": "polite" }, root);

  var mq = window.matchMedia("(max-width: 600px)"), P = {};
  function draw() {
    var narrow = mq.matches;
    svg.textContent = "";
    var W = narrow ? 340 : 680, HH = narrow ? 230 : 260, L = 34, R = 10, T = 16, B = 44;
    var PW = W - L - R, PH = HH - T - B;
    svg.setAttribute("viewBox", "0 0 " + W + " " + HH);
    function X(l) { return L + (l - LMIN) / (LMAX - LMIN) * PW; }
    function Y(r) { return T + PH - r / RMAX * PH; }
    el("line", { class: "ax", x1: L, x2: L + PW, y1: Y(0), y2: Y(0) }, svg);
    el("line", { class: "ax", x1: L, x2: L, y1: T, y2: Y(0) }, svg);
    for (var r = 0; r <= 7; r++) {
      el("line", { class: "ax", x1: L - 4, x2: L, y1: Y(r), y2: Y(r) }, svg);
      if (r % 2 === 0) el("text", { class: "tick", x: L - 7, y: Y(r) + 4, "text-anchor": "end" }, svg, String(r));
    }
    for (var l = LMIN; l <= LMAX; l++) {
      el("line", { class: "ax", x1: X(l), x2: X(l), y1: Y(0), y2: Y(0) + 6 }, svg);
      if (!narrow || l % 2 !== 0 || l === 0) {
        el("text", { class: "tick", x: X(l), y: Y(0) + 20, "text-anchor": "middle" }, svg, l < 0 ? "0." + "0".repeat(-l - 1) + "1" : String(Math.pow(10, l)));
      }
    }
    el("text", { class: "sym", x: L + PW, y: HH - 4, "text-anchor": "end" }, svg, "σ");
    el("text", { class: "sym", x: L + 6, y: T + 4 }, svg, "ρ");
    /* ln(1/sigma), the small-sigma asymptote */
    el("path", { class: "guide", d: "M" + X(LMIN) + " " + Y(-LMIN * Math.LN10) + "L" + X(0) + " " + Y(0) }, svg);
    el("text", { x: X(-0.55), y: Y(0.55 * Math.LN10) + 22 }, svg, "ln(1/σ)");
    var d = "";
    for (var k = 0; k <= 300; k++) {
      var lg = LMIN + (LMAX - LMIN) * k / 300;
      d += (d ? "L" : "M") + X(lg).toFixed(1) + " " + Y(rho(Math.pow(10, lg))).toFixed(1);
    }
    el("path", { class: "mline", d: d }, svg);
    el("text", { x: X(narrow ? 0.9 : 1.2), y: Y(rho(Math.pow(10, narrow ? 0.9 : 1.2))) - 30 }, svg, "ρ = ln(1 + 1/σ)");
    P = {
      X: X, Y: Y,
      drop: el("line", { class: "guide" }, svg),
      dot: el("circle", { class: "dot", r: 6 }, svg),
      tag: el("text", { class: "warn", "text-anchor": "start" }, svg)
    };
    update();
  }

  function update() {
    var l = +slider.value, sigma = Math.pow(10, l), masked = mask.getAttribute("aria-pressed") === "true";
    var value = masked ? 0 : rho(sigma);
    var x = P.X(l), y = P.Y(value);
    P.dot.setAttribute("cx", x.toFixed(1));
    P.dot.setAttribute("cy", y.toFixed(1));
    P.drop.setAttribute("x1", x.toFixed(1)); P.drop.setAttribute("x2", x.toFixed(1));
    P.drop.setAttribute("y1", y.toFixed(1)); P.drop.setAttribute("y2", P.Y(0).toFixed(1));
    P.tag.textContent = masked ? "masked: ρ = 0" : "";
    P.tag.setAttribute("x", (x + (l > 1.5 ? -110 : 10)).toFixed(1));
    P.tag.setAttribute("y", (P.Y(0) - 12).toFixed(1));
    num.textContent = "σ = " + fmt(sigma);
    sigmaEl.textContent = masked ? "none" : fmt(sigma);
    rhoEl.textContent = value.toFixed(2);
    rawEl.textContent = masked ? "0" : fmt(sigma);
    noteEl.textContent = masked ? "0 would read as a perfect measurement" : "";
    slider.disabled = masked;
    slider.setAttribute("aria-valuetext", "sigma = " + fmt(sigma));
    status.dataset.text = masked ? "Pixel masked: rho = 0." : "rho = " + value.toFixed(2) + ".";
  }

  function announce() { status.textContent = status.dataset.text; }
  slider.addEventListener("input", update);
  slider.addEventListener("change", announce);
  mask.addEventListener("click", function () {
    mask.setAttribute("aria-pressed", mask.getAttribute("aria-pressed") === "true" ? "false" : "true");
    update();
    announce();
  });
  if (mq.addEventListener) mq.addEventListener("change", draw);
  draw();
})();
