"use strict";

const hasGpuStatus = document.body.dataset.gpuStatus === "true";
const byId = (id) => document.getElementById(id);
const setText = (id, value) => { byId(id).textContent = value ?? "—"; };
const number = (value) => Number.isFinite(value) ? value.toLocaleString() : "—";
const duration = (seconds) => Number.isFinite(seconds) ? `${seconds.toFixed(2)} s` : "—";
const endpoints = [
  { path: "health", label: "Service health" },
  ...(hasGpuStatus ? [{ path: "gpu_status", label: "GPU availability and request activity" }] : []),
  { path: "model_info", label: "Model configuration and architecture" },
];
let refreshing = false;
let refreshTimer;
let lastHealthSuccess;

function element(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function badge(id, text, tone = "neutral") {
  setText(id, text);
  byId(id).dataset.tone = tone;
}

function renderRefreshState() {
  const enabled = byId("auto-refresh").checked;
  setText("refresh-state", enabled ? "Auto-refresh on · every 5 seconds" : "Auto-refresh paused · values are from the last check");
  setText("endpoint-refresh", enabled ? "Checked automatically every 5 seconds" : "Automatic checks paused");
}

for (const endpoint of endpoints) {
  const row = element("div", "endpoint");
  const heading = element("div", "endpoint-heading");
  const link = element("a", "", `/${endpoint.path} ↗`);
  link.href = `./${endpoint.path}`;
  const status = element("span", "badge", "Waiting");
  status.id = `endpoint-${endpoint.path}`;
  const note = element("p", "", endpoint.label);
  note.id = `endpoint-note-${endpoint.path}`;
  heading.append(link, status);
  row.append(heading, note);
  byId("endpoint-list").append(row);
}
setText("service-address", window.location.host + window.location.pathname);

function validGpuStatus(data) {
  const nonnegative = (value) => Number.isFinite(value) && value >= 0;
  return Number.isInteger(data.total_gpus) && nonnegative(data.total_gpus)
    && Number.isInteger(data.available_gpus) && nonnegative(data.available_gpus)
    && data.available_gpus <= data.total_gpus
    && data.gpu_stats && typeof data.gpu_stats === "object" && !Array.isArray(data.gpu_stats)
    && Object.values(data.gpu_stats).every((gpu) => gpu && typeof gpu.status === "string"
      && nonnegative(gpu.total_requests) && nonnegative(gpu.total_processing_time));
}

async function checkEndpoint(endpoint) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  const started = performance.now();
  try {
    const response = await fetch(`./${endpoint.path}`, { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("Invalid status response");
    if (endpoint.path === "health" && typeof data.status !== "string") throw new Error("Missing health status");
    if (endpoint.path === "gpu_status" && !validGpuStatus(data)) throw new Error("Invalid GPU status");
    if (endpoint.path === "model_info" && !data.model_path && !data.reward_model) throw new Error("Missing model information");
    return { data, ms: Math.round(performance.now() - started), status: response.status };
  } catch (error) {
    const message = error.name === "AbortError" ? "Timed out after 8 seconds" : error.message;
    return { error: message };
  } finally {
    clearTimeout(timeout);
  }
}

function renderGpu(result) {
  const list = byId("gpu-list");
  list.replaceChildren();
  if (!hasGpuStatus || result.error) {
    ["available-gpus", "finished-requests", "average-time"].forEach((id) => setText(id, "—"));
    setText("gpu-summary", hasGpuStatus ? "GPU telemetry unavailable" : "Not reported by this service");
    setText("worker-summary", "Availability and activity for each model replica.");
    badge("gpu-count", hasGpuStatus ? "Unavailable" : "Not reported", hasGpuStatus ? "warning" : "neutral");
    list.append(element("p", "empty-state", hasGpuStatus ? "Could not read GPU status. Metrics will return after a successful check." : "This baseline service does not expose a GPU pool. Service health and model details are available."));
    return;
  }
  const pool = result.data;
  const stats = Object.entries(pool.gpu_stats);
  const requests = stats.reduce((sum, [, gpu]) => sum + (gpu.total_requests || 0), 0);
  const processingTime = stats.reduce((sum, [, gpu]) => sum + (gpu.total_processing_time || 0), 0);
  const processing = stats.filter(([, gpu]) => gpu.status === "processing").length;
  setText("available-gpus", `${pool.available_gpus} / ${pool.total_gpus}`);
  setText("gpu-summary", `${processing} processing · ${pool.available_gpus} available`);
  setText("finished-requests", number(requests));
  setText("average-time", requests ? duration(processingTime / requests) : "—");
  setText("worker-summary", `${number(pool.max_workers)} worker threads · ${number(pool.total_gpus)} model replicas`);
  badge("gpu-count", `${pool.total_gpus} GPU${pool.total_gpus === 1 ? "" : "s"}`);
  if (!stats.length) list.append(element("p", "empty-state", "No GPU replicas reported by the service."));
  for (const [id, gpu] of stats) {
    const card = element("article", "gpu-card");
    const heading = element("div", "gpu-card-heading");
    const name = element("h3", "", `GPU ${id}`);
    name.append(element("span", "gpu-device", `cuda:${id}`));
    const state = element("span", "badge", gpu.status === "ready" ? "Ready" : gpu.status === "processing" ? "Processing" : "Unknown");
    state.dataset.tone = gpu.status === "ready" ? "good" : "warning";
    heading.append(name, state);
    const facts = element("dl", "gpu-facts");
    const values = [
      ["Finished requests", number(gpu.total_requests)],
      ["Avg. processing", gpu.total_requests ? duration(gpu.total_processing_time / gpu.total_requests) : "—"],
      ["Last request", (gpu.total_requests || gpu.status === "processing") && Number.isFinite(gpu.last_used) ? new Date(gpu.last_used * 1000).toLocaleTimeString() : "—"],
    ];
    for (const [label, value] of values) {
      const fact = element("div", "");
      fact.append(element("dt", "", label), element("dd", "", value));
      facts.append(fact);
    }
    card.append(heading, facts);
    list.append(card);
  }
}

function renderModel(result) {
  if (result.error) {
    badge("model-state", "Unavailable", "warning");
    setText("model-name", "Could not read model information.");
    ["model-architecture", "model-parameters", "model-type", "model-provider"].forEach((id) => setText(id, "—"));
    return;
  }
  const info = result.data;
  const config = info.config || {};
  const modelConfig = config.model_config || {};
  const architecture = info.model_architecture || {};
  badge("model-state", "Reported", "good");
  setText("model-name", info.model_path || config.model_path || modelConfig.model_name || info.reward_model);
  setText("model-architecture", architecture.model_class || "Not reported");
  setText("model-parameters", Number.isFinite(architecture.total_parameters) ? new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(architecture.total_parameters) : "Not reported");
  setText("model-type", info.experiment_config?.model?.model_type || info.reward_model || "Not reported");
  setText("model-provider", modelConfig.provider || modelConfig.vlm_provider || (hasGpuStatus ? "Local" : "Not reported"));
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  clearTimeout(refreshTimer);
  byId("refresh").disabled = true;
  setText("refresh-state", "Checking service…");
  try {
    const results = await Promise.all(endpoints.map(checkEndpoint));
    const checks = Object.fromEntries(endpoints.map((endpoint, index) => [endpoint.path, results[index]]));
    for (const [index, endpoint] of endpoints.entries()) {
      const result = results[index];
      const unhealthy = endpoint.path === "health" && result.data?.status !== "healthy";
      badge(`endpoint-${endpoint.path}`, result.error ? "Failed" : unhealthy ? "Unhealthy" : `HTTP ${result.status}`, result.error ? "error" : unhealthy ? "warning" : "good");
      setText(`endpoint-note-${endpoint.path}`, result.error || `${endpoint.label} · ${result.ms} ms`);
    }
    const health = checks.health;
    if (!health.error && health.data.status === "healthy") lastHealthSuccess = new Date();
    const failed = results.some((result) => result.error);
    let title, description, tone;
    if (health.error) {
      title = "Health check unavailable";
      description = `${health.error}. ${lastHealthSuccess ? `Last healthy response at ${lastHealthSuccess.toLocaleTimeString()}.` : "No healthy response received yet."}`;
      tone = "error";
    } else if (health.data.status !== "healthy") {
      title = "Service needs attention";
      description = `The health endpoint reports: ${health.data.status}.`;
      tone = "warning";
    } else if (failed) {
      title = "Service online · some checks failed";
      description = "Health endpoint responding. Some status details are unavailable.";
      tone = "warning";
    } else {
      title = "Service online";
      description = hasGpuStatus && checks.gpu_status.data.available_gpus === 0 ? "Health endpoint responding. All GPU replicas are currently in use." : "Health endpoint responding. Status checks are up to date.";
      tone = "good";
    }
    byId("service-banner").dataset.tone = tone;
    setText("service-state", title);
    setText("service-description", description);
    setText("health-latency", health.error ? "—" : `${health.ms} ms`);
    renderGpu(checks.gpu_status);
    renderModel(checks.model_info);
    setText("last-checked", `Last checked ${new Date().toLocaleTimeString()}`);
  } finally {
    refreshing = false;
    byId("refresh").disabled = false;
    renderRefreshState();
    if (byId("auto-refresh").checked) refreshTimer = setTimeout(refresh, 5000);
  }
}

byId("refresh").addEventListener("click", refresh);
byId("auto-refresh").addEventListener("change", () => {
  clearTimeout(refreshTimer);
  renderRefreshState();
  if (byId("auto-refresh").checked) refresh();
});
refresh();
