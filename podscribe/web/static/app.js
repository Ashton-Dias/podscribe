"use strict";
const $app = document.getElementById("app");
const SVG_NS = "http://www.w3.org/2000/svg";

// ---------- helpers
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
}
const svg = (markup) => { const t = document.createElement("template"); t.innerHTML = markup.trim(); return t.content.firstChild; };
async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    method: opts.method || "GET",
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || data.detail || `Request failed (${res.status})`);
  return data;
}
const fmtDur = (s) => (s ? `${Math.round(s / 60)} min` : "");
const fmtClock = (sec) => { sec = Math.max(0, Math.floor(sec)); return `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`; };
function saveFile(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const STALL_MS = 6000; // no progress change for this long while transcribing -> show the working state instead of a frozen bar
const STAGE = { queued: "Preparing", finding: "Locating the feed", downloading: "Collecting the audio", transcribing: "Transcribing" };

// ---------- botanical figures (inline SVG, drawn in code; decorative)
const LEAF = "M0 0C14-11 38-12 58 0C38 12 14 11 0 0Z";
function heroFigure() {
  const pairs = [[149, 236, .95], [153, 202, .9], [156, 170, .84], [152, 140, .74], [149, 112, .66], [152, 86, .56], [156, 62, .46], [159, 42, .36]];
  const leaves = pairs.map(([x, y, s]) => `<g transform="translate(${x} ${y}) scale(${s})"><path class="lf" d="${LEAF}" transform="rotate(-32)"/><path class="rib" d="M3 0L50 0" transform="rotate(-32)"/><path class="lf" d="${LEAF}" transform="rotate(-148)"/><path class="rib" d="M3 0L50 0" transform="rotate(-148)"/></g>`).join("");
  const fig = h("figure", { class: "fig" });
  fig.append(svg(`<svg viewBox="0 0 300 332" role="img" aria-label="Botanical line illustration of a fern frond inside a circular specimen dish" focusable="false">
    <circle class="dish" cx="150" cy="150" r="142"/><circle class="ticks" cx="150" cy="150" r="146"/><circle class="ring" cx="150" cy="150" r="132"/>
    <g class="sprig"><path class="stem" d="M150 268C148 214 160 168 150 124C144 92 152 62 160 34"/>${leaves}</g>
    <path class="leader" d="M201 142L238 112L282 112"/><circle class="dot" cx="201" cy="142" r="2.500"/><text x="262" y="106">a</text>
    <path class="leader" d="M131 127L94 98L40 98"/><circle class="dot" cx="131" cy="127" r="2.500"/><text x="42" y="92">b</text>
    <path class="bar" d="M46 318H106M46 313V323M106 313V323"/><text x="114" y="322">0.5 mm</text></svg>`));
  fig.append(h("figcaption", {}, "Fig. 1 · A voice, in cross-section"));
  return fig;
}
function makeVine() {
  const y = (x) => 34 + 11 * Math.sin(x / 55), d = [];
  for (let x = 6; x <= 594; x += 6) d.push(`${d.length ? "L" : "M"}${x} ${y(x).toFixed(1)}`);
  const N = 13, leaves = [];
  for (let i = 0; i < N; i++) {
    const x = 30 + i * (540 / (N - 1)), ang = Math.atan2(11 / 55 * Math.cos(x / 55), 1) * 180 / Math.PI, side = i % 2 ? 1 : -1;
    const at = ((x - 6) / 588 * 100).toFixed(1), s = (1.15 + 0.3 * Math.sin(i * 1.3)).toFixed(2);
    leaves.push(`<g class="vine-leaf" style="--at:${at}" transform="translate(${x} ${y(x).toFixed(1)}) rotate(${(ang + side * 50).toFixed(1)}) scale(${s})"><path d="M0 0C6-6 17-6 27 0C17 6 6 6 0 0Z"/></g>`);
  }
  const el = svg(`<svg class="vine" viewBox="0 0 600 68" aria-hidden="true" focusable="false"><path class="vine-track" d="${d.join("")}"/><path class="vine-stem" pathLength="100" d="${d.join("")}"/>${leaves.join("")}</svg>`);
  return { el, set: (p) => {
    el.classList.toggle("indet", p == null);
    if (p == null) el.style.removeProperty("--p"); // let the .indet rule (full vine, breathing leaves) apply instead of a stale inline value
    else el.style.setProperty("--p", p);
  } };
}
const footLeaf = () => svg(`<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 22V9"/><path d="M12 13C8 13 5 10.500 4 6.500c4.500-.5 7.500 2 8 6.500Z"/><path d="M12 10c4 0 7-2.500 8-6.500-4.500-.5-7.500 2-8 6.500Z"/></svg>`);

// ---------- the one screen: form -> working (-> pick) -> done
let poll = null;
const stopPoll = () => { clearInterval(poll); poll = null; };
const measure = (label, value) => h("div", {}, h("dt", {}, label), h("span", { class: "leader" }), h("dd", {}, value));

function showForm(health, notice) {
  stopPoll();
  const url = h("input", { id: "url", type: "url", placeholder: "https://open.spotify.com/episode/...", required: true, autocomplete: "off", spellcheck: "false" });
  const model = h("select", { id: "model" }, health.models.map((m) => h("option", { value: m }, m)));
  model.value = health.default_model;
  const msg = h("div", { class: "notice" + (notice && notice.error ? " err" : ""), role: notice && notice.error ? "alert" : "status", hidden: !notice }, notice ? notice.text : "");
  const submit = h("button", { class: "primary", type: "submit" }, "Transcribe");
  $app.replaceChildren(h("div", { class: "intake" },
    h("div", { class: "copy" },
      h("div", { class: "eyebrow" }, "Intake"),
      h("h1", { class: "headline" }, "For the episodes that caught your attention, but you'd prefer to ", h("em", {}, "read"), "."),
      h("p", { class: "standfirst" }, "Paste a Spotify episode link. A timestamped transcript is composed on your Mac, ready for LLM-powered summary and study.")),
    heroFigure(),
    h("form", { class: "card", onsubmit: async (e) => {
      e.preventDefault(); submit.disabled = true; msg.hidden = true;
      try {
        const job = await api("/jobs", { method: "POST", body: { url: url.value.trim(), model: model.value } });
        follow(job.id, health);
      } catch (err) { msg.className = "notice err"; msg.setAttribute("role", "alert"); msg.textContent = err.message; msg.hidden = false; submit.disabled = false; }
    } },
      h("span", { class: "plate" }, "Specimen 001"),
      h("label", { for: "url" }, "Spotify episode link"), url,
      h("label", { for: "model" }, "Whisper model"), model,
      h("div", { style: "margin-top:clamp(16px,3vh,28px)" }, submit), msg)));
  url.focus();
}

function follow(jobId, health) {
  stopPoll();
  const title = h("div", { class: "title-plain", hidden: true });
  const stage = h("span", { class: "stage" }), clock = h("span", { class: "clock" }), pct = h("span", { class: "pct" });
  const vine = makeVine();
  const scale = h("div", { class: "scale", "aria-hidden": "true" }, ["0", "25", "50", "75", "100"].map((n) => h("span", {}, n)));
  const vinebox = h("div", { class: "vinebox" }, vine.el, scale);
  const note = h("p", { class: "note" });
  const picks = h("div", { class: "pick-list" });
  const cancel = h("button", { class: "quiet", onclick: async () => { cancel.disabled = true; try { await api(`/jobs/${jobId}/cancel`, { method: "POST" }); } catch (e) { /* the poll shows the outcome */ } } }, "Cancel");
  const eyebrow = h("div", { class: "eyebrow" }, "Observation");
  let lastP = null, lastMove = Date.now();
  $app.replaceChildren(h("div", { class: "card", "aria-live": "polite" },
    h("span", { class: "plate" }, "Specimen 001"), eyebrow, title,
    h("div", { class: "stagebar" }, stage, h("span", {}, clock, " ", pct)), vinebox, note, picks,
    h("div", { style: "margin-top:22px" }, cancel)));

  const tick = async () => {
    let j;
    try { j = await api(`/jobs/${jobId}`); }
    catch (e) { stopPoll(); return showForm(health, { error: true, text: "Lost the job (the server may have restarted). Try again." }); }
    if (j.status === "done") { stopPoll(); return finish(jobId, health); }
    if (j.status === "failed") { stopPoll(); return showForm(health, { error: true, text: j.error }); }
    if (j.status === "cancelled") { stopPoll(); return showForm(health, { text: "Cancelled. Nothing was kept." }); }
    title.textContent = j.title || ""; title.hidden = !j.title;
    clock.textContent = fmtClock(Date.now() / 1000 - j.created);
    if (j.status === "needs_choice") {
      eyebrow.textContent = "Identification"; stage.textContent = "Which specimen did you mean?";
      vinebox.hidden = true; pct.textContent = ""; note.textContent = `Not sure which episode matches “${j.spotify_title}”.`;
      if (!picks.childElementCount) picks.append(...j.candidates.map((c, i) => h("div", { class: "card pick" },
        h("span", { class: "letter" }, String.fromCharCode(65 + i)), h("strong", {}, c.title),
        h("div", { class: "muted" }, `${c.show} `, h("span", { class: "nw" }, `· ${c.date}`)), h("span", { class: "mono" }, `match ${c.score}%`), h("span", { class: "mono", title: c.feed_url }, c.feed_url),
        h("button", { onclick: async () => { picks.replaceChildren(); await api(`/jobs/${jobId}/choose`, { method: "POST", body: { index: i } }); } }, "Use this one"))));
      return;
    }
    eyebrow.textContent = "Observation"; vinebox.hidden = false; picks.replaceChildren();
    const p = j.progress == null ? null : Math.round(j.progress * 100);
    stage.textContent = STAGE[j.status] || j.status;
    pct.textContent = p != null && ["downloading", "transcribing"].includes(j.status) ? `${p}%` : "";
    let stalled = false;
    if (j.status === "transcribing" && p != null) {
      if (p !== lastP) { lastP = p; lastMove = Date.now(); }
      stalled = Date.now() - lastMove > STALL_MS;
    } else { lastP = null; lastMove = Date.now(); }
    note.textContent = stalled ? "Still working…" : (j.status === "transcribing" ? (j.message || "") : "");
    vine.set(stalled ? null : p);
  };
  poll = setInterval(tick, 1000); tick();
}

async function finish(jobId, health) {
  let r;
  try { r = await api(`/jobs/${jobId}/result`); }
  catch (e) { return showForm(health, { error: true, text: e.message }); }
  $app.replaceChildren(h("div", { class: "card narrow" },
    h("span", { class: "plate" }, "Plate I"),
    h("div", { class: "eyebrow" }, "Result"),
    h("h2", { class: "title-display" }, r.title),
    h("p", { class: "byline" }, `${r.show} · ${r.date}`),
    h("hr", { class: "rule" }),
    h("dl", { class: "measure" }, measure("Duration", fmtDur(r.duration)), measure("Words", r.words.toLocaleString()), measure("Tokens", `≈ ${r.tokens.toLocaleString()}`)),
    h("div", { class: "actions", style: "margin-top:clamp(16px,3vh,26px)" },
      h("button", { class: "primary", onclick: () => saveFile(`${r.filename_base}.md`, r.markdown, "text/markdown") }, "Download .md (for your LLM)"),
      h("button", { onclick: () => showForm(health) }, "New episode")),
    h("p", { class: "foot" }, footLeaf(), h("span", {}, "Nothing is saved. Download now; refreshing or closing this tab discards the transcript."))));
}

// ---------- start (resume a running job after a refresh; a finished transcript is gone by design)
(async () => {
  try {
    const [health, cur] = await Promise.all([api("/health"), api("/jobs/current")]);
    if (cur.job) follow(cur.job.id, health); else showForm(health);
  } catch (e) { $app.replaceChildren(h("p", { class: "notice err", role: "alert" }, e.message)); }
})();
