// SENSEGRID - live prediction UI. All results come from the backend .pkl pipeline; never faked.
const $ = id => document.getElementById(id);
const TOP_PIR = ["PIR_4", "PIR_47", "PIR_48", "PIR_40", "PIR_46"];
const ALL_PIR = Array.from({ length: 55 }, (_, i) => "PIR_" + (i + 1));
const STATUS_COLORS = { 0: "#059669", 1: "#0284c7", 3: "#e11d48" };
const PRESET_LABELS = { vacancy: "Vacancy (real test row)", stationary: "Stationary Human (real test row)", activity: "Other Activity (real test row)" };
let state = {};

function apiBase() { return $("apiBase").value.replace(/\/+$/, ""); }

const BOUNDS = {
  Temperature_F: { min: 0, max: 89, int: false },
  Hour: { min: 0, max: 23, int: true },
  Minute: { min: 0, max: 59, int: true },
  PIR_4: { min: 2611, max: 16383, int: true },
  PIR_47: { min: 2606, max: 16383, int: true },
  PIR_48: { min: 2606, max: 16383, int: true },
  PIR_40: { min: 2612, max: 16383, int: true },
  PIR_46: { min: 2606, max: 16383, int: true },
  OtherPIR: { min: 2602, max: 111602625, int: true }
};

function clampToBounds(key, raw) {
  const b = BOUNDS[key];
  let v = parseFloat(raw);
  if (!Number.isFinite(v)) v = b.min;
  if (b.int) v = Math.round(v);
  const clamped = Math.min(b.max, Math.max(b.min, v));
  return { value: clamped, wasClamped: clamped !== v || !Number.isFinite(parseFloat(raw)) };
}

function flashClamped(el) {
  el.classList.add("clamped");
  setTimeout(() => el.classList.remove("clamped"), 700);
}

function syncControls() {
  ["Temperature_F", "Hour", "Minute", ...TOP_PIR].forEach(key => {
    const el = $("s_" + key);
    if (el) el.value = clampToBounds(key, state[key] ?? BOUNDS[key].min).value;
  });
  const otherKeys = ALL_PIR.filter(k => !TOP_PIR.includes(k));
  const otherMean = otherKeys.reduce((acc, k) => acc + (state[k] ?? 0), 0) / otherKeys.length;
  $("s_OtherPIR").value = clampToBounds("OtherPIR", otherMean).value;
  $("s_DayOfWeek").value = state["DayOfWeek"] ?? 0;
}
function applyPreset(name) {
  state = { ...window.SITE_DATA.presets[name].features };
  $("predictResult").classList.add("hidden"); $("predictError").classList.add("hidden");
  syncControls();
}
function commitField(key) {
  const el = $("s_" + key);
  const { value, wasClamped } = clampToBounds(key, el.value);
  el.value = value;
  if (wasClamped) flashClamped(el);
  if (key === "OtherPIR") {
    ALL_PIR.forEach(k => { if (!TOP_PIR.includes(k)) state[k] = value; });
  } else {
    state[key] = value;
  }
}
function bindControls() {
  ["Temperature_F", "Hour", "Minute", ...TOP_PIR, "OtherPIR"].forEach(key => {
    const el = $("s_" + key);
    el.addEventListener("change", () => commitField(key));
    el.addEventListener("blur", () => commitField(key));
  });
  $("s_DayOfWeek").addEventListener("change", e => { state["DayOfWeek"] = parseInt(e.target.value, 10); });
  $("presetVacancy").addEventListener("click", () => applyPreset("vacancy"));
  $("presetStationary").addEventListener("click", () => applyPreset("stationary"));
  $("presetActivity").addEventListener("click", () => applyPreset("activity"));
  $("predictBtn").addEventListener("click", () => {
    ["Temperature_F", "Hour", "Minute", ...TOP_PIR].forEach(commitField);
    predict();
  });
}
async function checkBackend() {
  try {
    const r = await fetch(apiBase() + "/health", { signal: AbortSignal.timeout(4000) });
    const j = await r.json();
    $("backendStatus").textContent = j.model_loaded ? "● Backend online — model loaded" : "● Backend up, model NOT loaded";
    $("backendStatus").style.color = j.model_loaded ? "#059669" : "#e11d48";
  } catch {
    $("backendStatus").textContent = "● Backend offline — start it (see README), then Predict";
    $("backendStatus").style.color = "#e11d48";
  }
}
async function predict() {
  $("predictError").classList.add("hidden"); $("predictResult").classList.add("hidden");
  $("predictLoading").classList.remove("hidden");
  try {
    const r = await fetch(apiBase() + "/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: state }), signal: AbortSignal.timeout(15000)
    });
    const j = await r.json();
    if (j.error) throw new Error(j.error + (j.detail ? " — " + j.detail : ""));
    $("resultLabel").textContent = j.status;
    $("resultLabel").style.color = STATUS_COLORS[j.prediction] || "#f3ead9";
    $("resultConf").textContent = "confidence " + (j.confidence * 100).toFixed(1) + "%";
    $("probBars").innerHTML = j.probabilities.map(p =>
      `<div class="prob-row"><span>${p.status}</span><div class="prob-track"><div class="prob-fill" style="width:${(p.probability * 100).toFixed(1)}%;background:${STATUS_COLORS[p.label]}"></div></div><span>${(p.probability * 100).toFixed(1)}%</span></div>`).join("");
    $("resultReason").textContent = j.reason + ` (Mean PIR reading sent: ${Number(j.mean_pir).toLocaleString()} · Temperature: ${j.temperature_f} °F)`;
    $("predictResult").classList.remove("hidden");
  } catch (e) {
    $("predictError").textContent = "Prediction failed: " + e.message + " — check the backend is running at the URL above. No result is shown when the model cannot answer.";
    $("predictError").classList.remove("hidden");
  } finally { $("predictLoading").classList.add("hidden"); }
}
function renderImportance() {
  const feats = window.SITE_DATA.top_features, max = feats[0].importance;
  $("importanceChart").innerHTML = feats.map(f =>
    `<div class="imp-row"><span>${f.feature}</span><div class="imp-track"><div class="imp-fill" style="width:${(f.importance / max * 100).toFixed(1)}%"></div></div><span>${(f.importance * 100).toFixed(2)}%</span></div>`).join("");
}
bindControls(); applyPreset("vacancy"); renderImportance(); checkBackend();
