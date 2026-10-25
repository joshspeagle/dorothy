/* Explainer: calibrate the uncertainties (How it works / Uncertainties).

   2,000 simulated validation stars have residuals r ~ N(0, t) with t drawn
   log-uniform in [0.03, 0.15]; the predicted scatter is s = t / 1.4, so the
   network is overconfident by 1.4. The reader scales s by c and sees the
   histogram of z = r / (c s) against the standard normal density. "Fit c"
   sets c to the robust spread (P84 - P16) / 2 of the raw z-scores, as the
   trainer does. Seeded, so every load is identical. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var root = document.getElementById("d-calibration");
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
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function pdf(z) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }
  function pct(sorted, q) {
    var p = q * (sorted.length - 1), a = Math.floor(p), b = Math.ceil(p);
    return sorted[a] + (sorted[b] - sorted[a]) * (p - a);
  }

  /* simulated stars */
  var rnd = mulberry32(2042);
  function gauss() { var u = 1 - rnd(), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  var N = 2000, zraw = new Float64Array(N), lo = Math.log(0.03), hi = Math.log(0.15);
  for (var i = 0; i < N; i++) {
    var t = Math.exp(lo + rnd() * (hi - lo));
    zraw[i] = t * gauss() / (t / 1.4);
  }
  var sorted = Array.from(zraw).sort(function (a, b) { return a - b; });
  var cFit = (pct(sorted, 0.84) - pct(sorted, 0.16)) / 2;

  /* markup */
  var plot = html("div", { class: "d-lab-plot" }, root);
  var svg = el("svg", { role: "img", "aria-label": "Histogram of the z-scores of 2,000 simulated validation stars, with the standard normal density on top." }, plot);
  var out = html("div", { class: "d-readout", "aria-live": "polite" }, root);
  function cell(key, cls) {
    var d = html("div", {}, out);
    html("span", { class: "k" }, d, key);
    return html("span", { class: cls }, d);
  }
  var spreadEl = cell("Robust spread of z", "big");
  var verdictEl = cell("Verdict", "d-verdict");
  var reportEl = cell("Reported uncertainty", "big mid");
  var ctl = html("div", { class: "d-lab-ctl" }, root);
  var sl = html("div", { class: "d-slider" }, ctl);
  var lab = html("label", { for: "d-calibration-c" }, sl, "Calibration factor ");
  var num = html("span", { class: "num" }, lab);
  var slider = html("input", { id: "d-calibration-c", type: "range", min: "0.5", max: "2.5", step: "0.01", value: "1.00" }, sl);
  var scale = html("div", { class: "d-scale", "aria-hidden": "true" }, sl);
  ["0.5", "1.0", "1.5", "2.0", "2.5"].forEach(function (s) { html("span", {}, scale, s); });
  var fit = html("button", { class: "d-btn", type: "button" }, ctl, "Fit c");

  /* histogram */
  var mq = window.matchMedia("(max-width: 600px)"), H = {};
  function draw() {
    var narrow = mq.matches;
    svg.textContent = "";
    var W = narrow ? 340 : 680, HH = narrow ? 250 : 300, L = narrow ? 30 : 40, R = 8, T = 34, B = 40;
    var PW = W - L - R, PH = HH - T - B, ZMIN = -5, ZMAX = 5, BW = 0.2, NB = 50, DMAX = 0.5;
    svg.setAttribute("viewBox", "0 0 " + W + " " + HH);
    function X(z) { return L + (z - ZMIN) / (ZMAX - ZMIN) * PW; }
    function Y(p) { return T + PH - p / DMAX * PH; }
    for (var g = 1; g <= 5; g++) {
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
    for (var k = -5; k <= 5; k++) {
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
    var ebY = T + 36;
    el("path", { class: "guide", d: "M" + X(-1) + " " + (ebY + 10) + "V" + Y(0) + "M" + X(1) + " " + (ebY + 10) + "V" + Y(0) }, svg);
    H = {
      X: X, Y: Y, bars: bars, NB: NB, BW: BW, ZMIN: ZMIN, DMAX: DMAX, ebY: ebY,
      eb: el("path", { class: "mline" }, svg),
      dot: el("rect", { class: "dot", width: 8, height: 8, y: ebY - 4 }, svg),
      txt: el("text", { y: ebY - 10, "text-anchor": "middle" }, svg, "P16 to P84")
    };
    update();
  }

  function update() {
    var c = +slider.value, counts = new Array(H.NB).fill(0);
    for (var i = 0; i < N; i++) {
      var bi = Math.floor((zraw[i] / c - H.ZMIN) / H.BW);
      if (bi >= 0 && bi < H.NB) counts[bi]++;
    }
    for (var b = 0; b < H.NB; b++) {
      var y = H.Y(Math.min(counts[b] / (N * H.BW), H.DMAX));
      H.bars[b].setAttribute("y", y.toFixed(1));
      H.bars[b].setAttribute("height", (H.Y(0) - y).toFixed(1));
    }
    var p16 = pct(sorted, 0.16) / c, p84 = pct(sorted, 0.84) / c, med = pct(sorted, 0.5) / c;
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
    slider.setAttribute("aria-valuetext", "c = " + c.toFixed(2) + ", robust spread " + sp.toFixed(2) + ", " + word);
  }

  slider.addEventListener("input", update);
  fit.addEventListener("click", function () { slider.value = cFit.toFixed(2); update(); });
  if (mq.addEventListener) mq.addEventListener("change", draw);
  draw();
})();
