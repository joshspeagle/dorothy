/* Explainer: calibrate the uncertainties (How it works / Uncertainties).

   2,000 simulated validation stars have a predicted scatter 1.4 times too
   small, so their raw z-scores are 1.4 times a standard normal draw. The
   reader scales s by c and sees the histogram of z = r / (c s) against the
   standard normal density. "Fit c" sets c to the robust spread
   (P84 - P16) / 2 of the raw z-scores, as the trainer does. Seeded, so
   every load is identical. */
(function () {
  "use strict";
  var root = document.getElementById("d-calibration");
  if (!root) return;
  var E = window.dExplainer, el = E.el, html = E.html;

  function pdf(z) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }
  function pct(sorted, q) {
    var p = q * (sorted.length - 1), a = Math.floor(p), b = Math.ceil(p);
    return sorted[a] + (sorted[b] - sorted[a]) * (p - a);
  }

  /* simulated stars: a residual r = t g with any true scatter t and predicted
     scatter s = t / 1.4 has z = r / s = 1.4 g, so t need not be drawn */
  var rnd = E.mulberry32(2042);
  function gauss() { var u = 1 - rnd(), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  var N = 2000, zraw = new Float64Array(N);
  for (var i = 0; i < N; i++) zraw[i] = 1.4 * gauss();
  /* percentiles of the raw z-scores; those of z = zraw / c are these / c */
  var P16, P50, P84;
  (function () {
    var sorted = Array.from(zraw).sort(function (a, b) { return a - b; });
    P16 = pct(sorted, 0.16); P50 = pct(sorted, 0.5); P84 = pct(sorted, 0.84);
  })();
  var cFit = (P84 - P16) / 2;

  var C_MIN = 0.5, C_MAX = 2.5, ZMIN = -5, ZMAX = 5, BW = 0.2, NB = Math.round((ZMAX - ZMIN) / BW);
  /* counts of z = zraw / c in NB bins of width BW from ZMIN */
  function histogram(c) {
    var counts = new Array(NB).fill(0);
    for (var i = 0; i < N; i++) {
      var b = Math.floor((zraw[i] / c - ZMIN) / BW);
      if (b >= 0 && b < NB) counts[b]++;
    }
    return counts;
  }
  /* density axis: the tallest bar at the largest c, rounded up to 0.1 */
  var DMAX = Math.ceil(Math.max(Math.max.apply(null, histogram(C_MAX)) / (N * BW), pdf(0)) * 10) / 10;

  /* markup */
  var plot = html("div", { class: "d-lab-plot" }, root);
  var svg = el("svg", { role: "img", "aria-label": "Histogram of the z-scores of 2,000 simulated validation stars, with the standard normal density on top." }, plot);
  var out = html("div", { class: "d-readout" }, root);
  var spreadEl = E.cell(out, "Robust spread of z", "big");
  var verdictEl = E.cell(out, "Verdict", "d-verdict");
  var reportEl = E.cell(out, "Reported uncertainty", "big mid");
  var ctl = html("div", { class: "d-lab-ctl" }, root);
  var sl = html("div", { class: "d-slider" }, ctl);
  var lab = html("label", { for: "d-calibration-c" }, sl, "Calibration factor ");
  var num = html("span", { class: "num" }, lab);
  var slider = html("input", { id: "d-calibration-c", type: "range", min: String(C_MIN), max: String(C_MAX), step: "0.01", value: "1.00" }, sl);
  var scale = html("div", { class: "d-scale", "aria-hidden": "true" }, sl);
  for (var t = 0; t <= 4; t++) html("span", {}, scale, (C_MIN + t * (C_MAX - C_MIN) / 4).toFixed(1));
  var fit = html("button", { class: "d-btn", type: "button" }, ctl, "Fit c");
  var status = E.status(root);

  /* histogram */
  var mq = window.matchMedia("(max-width: 600px)"), H = {};
  function draw() {
    var narrow = mq.matches;
    svg.textContent = "";
    var W = narrow ? 340 : 680, HH = narrow ? 250 : 300, L = narrow ? 30 : 40, R = 8, T = 34, B = 40;
    var PW = W - L - R, PH = HH - T - B;
    svg.setAttribute("viewBox", "0 0 " + W + " " + HH);
    function X(z) { return L + (z - ZMIN) / (ZMAX - ZMIN) * PW; }
    function Y(p) { return T + PH - p / DMAX * PH; }
    for (var g = 1; g <= Math.round(DMAX * 10); g++) {
      el("line", { class: "ax", x1: L - 4, x2: L, y1: Y(g / 10), y2: Y(g / 10) }, svg);
      el("text", { class: "tick", x: L - 7, y: Y(g / 10) + 4, "text-anchor": "end" }, svg, (g / 10).toFixed(1));
    }
    var bars = [], gB = el("g", {}, svg);
    for (var b = 0; b < NB; b++) {
      bars.push(el("rect", { class: "bar", x: (X(ZMIN + b * BW) + 0.5).toFixed(1), width: Math.max(PW / NB - 1, 1).toFixed(1), y: Y(0), height: 0 }, gB));
    }
    var d = "";
    for (var z = ZMIN; z <= ZMAX + 1e-9; z += 0.05) d += (d ? "L" : "M") + X(z).toFixed(1) + " " + Y(pdf(z)).toFixed(1);
    el("path", { class: "curve", d: d }, svg);
    el("line", { class: "ax", x1: L, x2: L + PW, y1: Y(0), y2: Y(0) }, svg);
    el("line", { class: "ax", x1: L, x2: L, y1: T, y2: Y(0) }, svg);
    for (var k = ZMIN; k <= ZMAX; k++) {
      el("line", { class: "ax", x1: X(k), x2: X(k), y1: Y(0), y2: Y(0) + (k % 5 === 0 ? 7 : 4) }, svg);
      if (!narrow || Math.abs(k) % 2 === 1 || k === 0) {
        el("text", { class: "tick", x: X(k), y: Y(0) + 21, "text-anchor": "middle" }, svg, k < 0 ? "−" + (-k) : String(k));
      }
    }
    el("text", { class: "sym", x: L + PW, y: HH - 2, "text-anchor": "end" }, svg, "z");
    el("rect", { class: "bar", x: L + 6, y: 6, width: 14, height: 10 }, svg);
    el("text", { x: L + 26, y: 15 }, svg, "validation stars");
    el("line", { class: "curve", x1: L + (narrow ? 136 : 148), x2: L + (narrow ? 156 : 168), y1: 11, y2: 11 }, svg);
    el("text", { x: L + (narrow ? 162 : 174), y: 15 }, svg, "standard normal");
    var ebY = T + 6;
    el("path", { class: "guide", d: "M" + X(-1) + " " + (ebY + 10) + "V" + Y(0) + "M" + X(1) + " " + (ebY + 10) + "V" + Y(0) }, svg);
    H = {
      X: X, Y: Y, bars: bars, ebY: ebY,
      eb: el("path", { class: "mline" }, svg),
      dot: el("rect", { class: "dot", width: 8, height: 8, y: ebY - 4 }, svg),
      txt: el("text", { y: ebY - 10, "text-anchor": "middle" }, svg, "P16 to P84")
    };
    update();
  }

  function update() {
    var c = +slider.value, counts = histogram(c);
    for (var b = 0; b < NB; b++) {
      var y = H.Y(counts[b] / (N * BW));
      H.bars[b].setAttribute("y", y.toFixed(1));
      H.bars[b].setAttribute("height", (H.Y(0) - y).toFixed(1));
    }
    var p16 = P16 / c, p84 = P84 / c, med = P50 / c;
    var a = H.X(p16), bb = H.X(p84), e = H.ebY;
    H.eb.setAttribute("d", "M" + a + " " + e + "H" + bb + "M" + a + " " + (e - 6) + "v12M" + bb + " " + (e - 6) + "v12");
    H.dot.setAttribute("x", H.X(med) - 4);
    H.txt.setAttribute("x", H.X(med));
    var sp = (p84 - p16) / 2, word;
    if (Math.abs(sp - 1) <= 0.05) word = "calibrated";
    else word = sp > 1 ? "overconfident" : "underconfident";
    spreadEl.textContent = sp.toFixed(2);
    num.textContent = "c = " + c.toFixed(2);
    reportEl.textContent = c.toFixed(2) + " · s";
    verdictEl.textContent = word;
    verdictEl.className = "d-verdict" + (word === "calibrated" ? " ok" : "");
    slider.setAttribute("aria-valuetext", "c = " + c.toFixed(2));
    status.set("Robust spread of z " + sp.toFixed(2) + ", " + word + ".");
  }

  slider.addEventListener("input", update);
  slider.addEventListener("change", status.announce);
  fit.addEventListener("click", function () { slider.value = cFit.toFixed(2); update(); status.announce(); });
  mq.addEventListener("change", draw);
  draw();
})();
