/*
 * app/frontend/app.js
 * ===================
 * Lógica do editor: upload -> classificação -> toggles de atributos ->
 * job de edição (polling) -> exibição dos resultados.
 *
 * Estado central:
 *   file        — foto escolhida
 *   origAttrs   — Set de atributos positivos da foto (classificados ou não)
 *   targetAttrs — Set de atributos desejados (toggles do usuário)
 * enable/disable = diferença entre os dois — o backend recebe as duas
 * listas completas e monta os vetores.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,
  attrNames: [],
  origAttrs: new Set(),
  targetAttrs: new Set(),
  classified: false,
  classifierAvailable: false,
  encoder: null,
  token: localStorage.getItem("dfm_token") || null,
  pollTimer: null,
};

// defaults por método — mesmos dos scripts CLI
const METHOD_DEFAULTS = {
  sdedit:    { s_id: 3.0, s_attr: 5.0, s_clip: 3.0 },
  inversion: { s_id: 1.0, s_attr: 3.0, s_clip: 1.0 },
};

const STRENGTH_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
const STRENGTH_DEFAULT = new Set([0.3, 0.5, 0.7]);

// ------------------------------------------------------------
// fetch com token opcional
// ------------------------------------------------------------

async function api(path, options = {}) {
  options.headers = options.headers || {};
  if (state.token) options.headers["X-API-Token"] = state.token;
  const resp = await fetch(path, options);
  if (resp.status === 401) {
    state.token = prompt("Este servidor exige um token de acesso (DFM_API_TOKEN):");
    if (state.token) {
      localStorage.setItem("dfm_token", state.token);
      return api(path, options);
    }
  }
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
}

// ------------------------------------------------------------
// health + bootstrap
// ------------------------------------------------------------

async function refreshHealth() {
  const el = $("health");
  try {
    const h = await api("/api/health");
    state.classifierAvailable = h.classifier_available;
    state.encoder = h.encoder;

    if (h.state === "ready") {
      el.textContent = `pronto — ${h.device}` + (h.encoder ? ` · ${h.encoder}` : "");
      el.className = "pill pill-ok";
    } else if (h.state === "loading" || h.state === "idle") {
      el.textContent = "carregando modelos…";
      el.className = "pill pill-wait";
      setTimeout(refreshHealth, 3000);
    } else {
      el.textContent = "erro: " + (h.error || (h.config_problems || []).join("; "));
      el.className = "pill pill-err";
      setTimeout(refreshHealth, 10000);
    }
  } catch (e) {
    el.textContent = "servidor inacessível";
    el.className = "pill pill-err";
    setTimeout(refreshHealth, 5000);
  }
}

async function loadAttributes() {
  const data = await api("/api/attributes");
  state.attrNames = data.attributes;
  const grid = $("attrs");
  grid.innerHTML = "";
  for (const name of state.attrNames) {
    const btn = document.createElement("button");
    btn.className = "attr";
    btn.textContent = name.replaceAll("_", " ");
    btn.title = name;
    btn.onclick = () => {
      if (state.targetAttrs.has(name)) state.targetAttrs.delete(name);
      else state.targetAttrs.add(name);
      renderAttrs();
    };
    btn.dataset.name = name;
    grid.appendChild(btn);
  }
}

function renderAttrs() {
  for (const btn of $("attrs").children) {
    const name = btn.dataset.name;
    const on = state.targetAttrs.has(name);
    btn.classList.toggle("on", on);
    btn.classList.toggle("changed", on !== state.origAttrs.has(name));
  }
  updateRunButton();
}

// ------------------------------------------------------------
// upload + classificação
// ------------------------------------------------------------

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.file = file;
  const img = $("preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("drop-hint").hidden = true;
  $("btn-classify").disabled = !state.classifierAvailable;
  $("classify-info").textContent = state.classifierAvailable
    ? "" : "classificador não configurado — marque os atributos à mão";
  state.classified = false;
  updateRunButton();
}

async function classify() {
  const btn = $("btn-classify");
  btn.disabled = true;
  $("classify-info").textContent = "detectando…";
  try {
    const fd = new FormData();
    fd.append("photo", state.file);
    const r = await api("/api/classify", { method: "POST", body: fd });
    state.origAttrs = new Set(r.attributes);
    state.targetAttrs = new Set(r.attributes);
    state.classified = true;
    $("btn-reset").disabled = false;
    $("classify-info").textContent = r.face_found
      ? `${r.attributes.length} atributos detectados`
      : "⚠ rosto não detectado — predição pode estar imprecisa";
    renderAttrs();
  } catch (e) {
    $("classify-info").textContent = "erro: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------
// parâmetros
// ------------------------------------------------------------

function currentMethod() {
  return document.querySelector("input[name=method]:checked").value;
}

function applyMethodDefaults() {
  const d = METHOD_DEFAULTS[currentMethod()];
  for (const k of ["s_id", "s_attr", "s_clip"]) {
    $(k).value = d[k];
    $(k + "_out").value = d[k];
  }
  $("strengths-box").hidden = currentMethod() !== "sdedit";
}

function buildStrengthChips() {
  const box = $("strengths");
  for (const s of STRENGTH_OPTIONS) {
    const chip = document.createElement("button");
    chip.className = "chip" + (STRENGTH_DEFAULT.has(s) ? " on" : "");
    chip.textContent = s;
    chip.dataset.value = s;
    chip.onclick = () => { chip.classList.toggle("on"); updateRunButton(); };
    box.appendChild(chip);
  }
}

function selectedStrengths() {
  return [...$("strengths").querySelectorAll(".chip.on")].map((c) => c.dataset.value);
}

function updateRunButton() {
  const hasEdit = state.file && (
    currentMethod() !== "sdedit" || selectedStrengths().length > 0
  );
  $("btn-run").disabled = !hasEdit;
}

// ------------------------------------------------------------
// revisão (âmbar/cinza) antes de gerar
// ------------------------------------------------------------

function renderReview() {
  for (const chip of $("review-grid").children) {
    const on = state.targetAttrs.has(chip.dataset.name);
    chip.classList.toggle("amber", on);
    chip.classList.toggle("grey", !on);
  }
}

function openReview() {
  const grid = $("review-grid");
  grid.innerHTML = "";
  for (const name of state.attrNames) {
    const chip = document.createElement("button");
    chip.className = "attr review-attr";
    chip.textContent = name.replaceAll("_", " ");
    chip.title = name;
    chip.dataset.name = name;
    chip.onclick = () => {
      if (state.targetAttrs.has(name)) state.targetAttrs.delete(name);
      else state.targetAttrs.add(name);
      renderReview();
      renderAttrs(); // mantém a grade de atributos (passo 2) em sincronia
    };
    grid.appendChild(chip);
  }
  renderReview();
  $("review-card").hidden = false;
  $("result-card").hidden = true;
  $("review-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeReview() {
  $("review-card").hidden = true;
}

// ------------------------------------------------------------
// job de edição + polling
// ------------------------------------------------------------

async function runEdit() {
  closeReview();
  clearTimeout(state.pollTimer);
  $("result-card").hidden = false;
  $("results").innerHTML = "";
  $("error").hidden = true;
  $("edit-summary").textContent = "";
  $("progress-box").hidden = false;
  $("progress-bar").style.width = "0%";
  $("progress-text").textContent = "enviando…";
  $("btn-run").disabled = true;

  const fd = new FormData();
  fd.append("photo", state.file);
  fd.append("method", currentMethod());
  fd.append("target_attrs", JSON.stringify([...state.targetAttrs]));
  // se o usuário nunca classificou, manda vazio -> o backend classifica
  // (ou falha com mensagem clara se não houver classificador)
  fd.append("orig_attrs", state.classified ? JSON.stringify([...state.origAttrs]) : "");
  fd.append("s_id", $("s_id").value);
  fd.append("s_attr", $("s_attr").value);
  fd.append("s_clip", $("s_clip").value);
  fd.append("strengths", selectedStrengths().join(","));
  fd.append("ddim_steps", $("ddim_steps").value);
  fd.append("seed", $("seed").value);

  try {
    const r = await api("/api/edit", { method: "POST", body: fd });
    poll(r.job_id);
  } catch (e) {
    showError(e.message);
  }
}

async function poll(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (job.status === "queued") {
      $("progress-text").textContent = "na fila…";
    } else if (job.status === "running") {
      $("progress-bar").style.width = (job.progress * 100).toFixed(1) + "%";
      $("progress-text").textContent =
        `${job.stage} — ${(job.progress * 100).toFixed(0)}%`;
    } else if (job.status === "done") {
      $("progress-bar").style.width = "100%";
      $("progress-box").hidden = true;
      showResult(job.result);
      $("btn-run").disabled = false;
      return;
    } else {
      showError(job.error || "falha desconhecida");
      return;
    }
    state.pollTimer = setTimeout(() => poll(jobId), 1500);
  } catch (e) {
    showError(e.message);
  }
}

function showResult(result) {
  const parts = [];
  if (result.enable.length) parts.push("+ " + result.enable.join(", "));
  if (result.disable.length) parts.push("− " + result.disable.join(", "));
  $("edit-summary").textContent = parts.length
    ? `Edições: ${parts.join("   ")}`
    : "Nenhum atributo alterado (reconstrução).";

  const box = $("results");
  box.innerHTML = "";
  for (const img of result.images) {
    const fig = document.createElement("figure");
    const el = document.createElement("img");
    el.src = img.url;
    el.loading = "lazy";
    const cap = document.createElement("figcaption");
    cap.textContent = img.label;
    fig.append(el, cap);
    box.appendChild(fig);
  }
}

function showError(msg) {
  $("progress-box").hidden = true;
  const el = $("error");
  el.textContent = "Erro: " + msg;
  el.hidden = false;
  $("btn-run").disabled = false;
}

// ------------------------------------------------------------
// wiring
// ------------------------------------------------------------

function wire() {
  const drop = $("drop");
  drop.onclick = () => $("file").click();
  $("file").onchange = (e) => setFile(e.target.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("dragover"); };
  drop.ondragleave = () => drop.classList.remove("dragover");
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    setFile(e.dataTransfer.files[0]);
  };

  $("btn-classify").onclick = classify;
  $("btn-reset").onclick = () => {
    state.targetAttrs = new Set(state.origAttrs);
    renderAttrs();
  };
  $("btn-clear").onclick = () => {
    state.targetAttrs.clear();
    renderAttrs();
  };
  $("btn-mark-orig").onclick = () => {
    state.origAttrs = new Set(state.targetAttrs);
    state.classified = true;
    $("btn-reset").disabled = false;
    $("mark-orig-info").textContent =
      `Seleção atual (${state.origAttrs.size} atributos) marcada como a foto original — ` +
      "continue ligando/desligando para definir o alvo da edição.";
    renderAttrs();
  };

  for (const radio of document.querySelectorAll("input[name=method]")) {
    radio.onchange = () => { applyMethodDefaults(); updateRunButton(); };
  }
  for (const k of ["s_id", "s_attr", "s_clip"]) {
    $(k).oninput = () => { $(k + "_out").value = $(k).value; };
  }
  $("btn-run").onclick = openReview;
  $("btn-confirm").onclick = runEdit;
  $("btn-back").onclick = closeReview;
}

wire();
buildStrengthChips();
applyMethodDefaults();
refreshHealth();
loadAttributes().catch((e) => console.error("attributes:", e));
