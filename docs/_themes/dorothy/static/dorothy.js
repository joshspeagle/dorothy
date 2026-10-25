/* Theme toggle, navigation toggles and the outline's current-section marker.
   The saved theme is applied before paint by an inline script in layout.html. */
(function () {
  var root = document.documentElement;

  function isDark() {
    var t = root.dataset.theme;
    return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function label(button) {
    button.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
  }
  document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
    label(button);
    button.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("dorothy-theme", next); } catch (e) {}
      label(button);
    });
  });

  /* Sidebar groups: the one holding the current page is open; the others open on click. */
  document.querySelectorAll(".side p.caption").forEach(function (caption) {
    var list = caption.nextElementSibling, text = caption.querySelector(".caption-text");
    var button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-expanded", list.querySelector(".current") ? "true" : "false");
    caption.replaceChild(button, text);
    button.appendChild(text);
    button.addEventListener("click", function () {
      button.setAttribute("aria-expanded", button.getAttribute("aria-expanded") === "true" ? "false" : "true");
    });
  });

  var menu = document.querySelector("[data-menu]"), nav = document.getElementById("sidebar");
  menu.addEventListener("click", function () {
    menu.setAttribute("aria-expanded", nav.classList.toggle("open") ? "true" : "false");
  });

  /* On this page: mark the last section whose heading has passed under the top bar. */
  var links = Array.prototype.slice.call(document.querySelectorAll(".toc > ul > li > ul > li > a"));
  if (!links.length) return;
  var targets = links.map(function (a) { return document.getElementById(decodeURIComponent(a.hash.slice(1))); });
  function mark() {
    var current = 0;
    targets.forEach(function (t, i) { if (t && t.getBoundingClientRect().top < 140) current = i; });
    if (innerHeight + scrollY >= document.body.scrollHeight - 2) current = links.length - 1;
    links.forEach(function (a, i) {
      a.classList.toggle("on", i === current);
      if (i === current) a.setAttribute("aria-current", "location"); else a.removeAttribute("aria-current");
    });
  }
  addEventListener("scroll", mark, { passive: true });
  mark();
})();
