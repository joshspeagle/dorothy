/* Helpers shared by the explainers of the How it works pages. A page loads
   this script before its explainer script. */
window.dExplainer = (function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function make(e, attrs, parent, text) {
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  /* an SVG element */
  function el(tag, attrs, parent, text) { return make(document.createElementNS(NS, tag), attrs, parent, text); }
  /* an HTML element */
  function html(tag, attrs, parent, text) { return make(document.createElement(tag), attrs, parent, text); }
  /* one key-value pair of a readout; returns the value span */
  function cell(out, key, cls) {
    var d = html("div", {}, out);
    html("span", { class: "k" }, d, key);
    return html("span", { class: cls }, d);
  }
  /* a seeded uniform generator on [0, 1) */
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  /* a visually hidden live region: set() stores the text, announce() reads it out */
  function status(root) {
    var node = html("p", { class: "d-sr", "aria-live": "polite" }, root), text = "";
    return {
      set: function (t) { text = t; },
      announce: function () { node.textContent = text; }
    };
  }
  return { el: el, html: html, cell: cell, mulberry32: mulberry32, status: status };
})();
