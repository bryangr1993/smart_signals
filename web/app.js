"use strict";

const $ = (id) => document.getElementById(id);
const steps = [...document.querySelectorAll("[data-step]")];
const sourceModes = [...document.querySelectorAll("[data-mode]")];
const geometryTolerance = 1e-12;
const activeStates = new Set(["loading", "running", "reconnecting", "awaiting_confirmation"]);
const states = {
  idle: ["Sin corrida activa", "neutral"],
  loading: ["Preparando", "warning"],
  running: ["En operación", "good"],
  completed: ["Completada", "good"],
  stopped: ["Detenida", "neutral"],
  interrupted: ["Interrumpida", "warning"],
  error: ["Error de corrida", "danger"],
  reconnecting: ["Reconectando", "warning"],
  awaiting_confirmation: ["Verifica el encuadre", "warning"],
};
const qualities = {
  valid: "Lectura disponible",
  ok: "Lectura disponible",
  degraded: "Lectura con advertencias",
  warming_up: "Estabilizando el seguimiento",
  stale: "Lectura desactualizada",
  offline: "Fuente sin conexión",
  roi_invalid: "Zona por verificar",
  invalid: "Lectura no disponible",
  unavailable: "Lectura no disponible",
};
const conditions = {
  unknown: "Sin clasificar",
  day_dry: "Día · seco",
  night_dry: "Noche · seco",
  day_rain: "Día · lluvia",
  night_rain: "Noche · lluvia",
};
const classLabels = {
  car: "Automóviles",
  motorcycle: "Motocicletas",
  bus: "Autobuses",
  truck: "Camiones",
  bicycle: "Bicicletas",
};
const integer = new Intl.NumberFormat("es-EC", {maximumFractionDigits: 0});
const decimal = new Intl.NumberFormat("es-EC", {maximumFractionDigits: 1});
const percent = new Intl.NumberFormat("es-EC", {style: "percent", maximumFractionDigits: 1});
const dayTime = new Intl.DateTimeFormat("es-EC", {dateStyle: "medium", timeStyle: "short"});
const state = {
  step: "source",
  mode: "library",
  preview: null,
  pendingProfile: null,
  roi: [],
  roiHistory: [],
  cursor: [0.5, 0.5],
  keyboard: false,
  drag: null,
  status: {state: "idle"},
  initialStatus: true,
  frameUrl: null,
  frameBusy: false,
  frameRequired: false,
  confirmFrame: false,
  sourceBusy: false,
  commandBusy: false,
  profiles: [],
  runs: [],
  runId: null,
  samples: [],
  sampleIndex: 0,
  sampleRoi: [],
  sampleImageReady: false,
  sampleRequest: 0,
  runsRequest: 0,
  annotationBusy: false,
  profileBusy: false,
};

function node(tag, value, className) {
  const element = document.createElement(tag);
  if (value !== undefined) element.textContent = value;
  if (className) element.className = className;
  return element;
}

async function request(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail ?? body?.message ?? body?.error;
    let message = "La aplicación no pudo completar la solicitud.";
    if (response.status >= 500) message = "La aplicación encontró un error. Vuelve a intentarlo.";
    else if (typeof detail === "string") message = detail;
    else if (response.status === 422) message = "Revisa los campos de la solicitud.";
    throw Object.assign(new Error(message), {status: response.status});
  }
  if (!body) throw new Error("La aplicación no devolvió una respuesta válida.");
  return body;
}

function post(path, data) {
  return request(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data),
  });
}

function errorText(error, fallback) {
  return error?.status && error.message ? error.message : fallback;
}

function announce(text = "", tone = "neutral") {
  const notice = $("notice");
  notice.textContent = text;
  notice.dataset.tone = tone;
  notice.hidden = !text;
}

function number(value, formatter = integer) {
  return Number.isFinite(value) ? formatter.format(value) : "—";
}

function timecode(seconds, fractions = false) {
  if (!Number.isFinite(seconds)) return "—";
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(whole / 60) % 60).padStart(2, "0");
  const secs = String(whole % 60).padStart(2, "0");
  const hours = whole >= 3600 ? String(Math.floor(whole / 3600)) + ":" : "";
  const tenth = fractions ? "." + String(Math.floor((seconds % 1) * 10 + 0.00001)).slice(-1) : "";
  return hours + minutes + ":" + secs + tenth;
}

function runTime(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : dayTime.format(date);
}

function setTheme(theme, save = false) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === "dark";
  const label = dark ? "Activar tema claro" : "Activar tema oscuro";
  $("theme-toggle").setAttribute("aria-label", label);
  $("theme-toggle").title = label;
  $("theme-icon").textContent = dark ? "☼" : "☾";
  document.querySelector('meta[name="theme-color"]').content = dark ? "#111819" : "#f3f5ee";
  if (save) {
    try { localStorage.setItem("smart-signals-theme", theme); } catch { /* Tema de la sesión. */ }
  }
  drawChart();
}

function showStep(step, focus = false) {
  state.step = step;
  announce();
  for (const button of steps) {
    if (button.dataset.step === step) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  }
  for (const panel of ["source", "zone", "operation"]) $("panel-" + panel).hidden = panel !== step;
  $("workbench").hidden = step === "results";
  $("results-panel").hidden = step !== "results";
  $("history-panel").hidden = step !== "operation";
  if (step === "results") refreshRuns();
  else renderViewer();
  if (step === "operation") {
    renderOperation();
    state.frameRequired = true;
    refreshFrame();
    requestAnimationFrame(drawChart);
  }
  if (focus) steps.find((button) => button.dataset.step === step)?.focus();
}

function setMode(mode) {
  state.mode = mode;
  for (const button of sourceModes) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }
  for (const name of ["library", "address", "upload"]) $("source-" + name).hidden = name !== mode;
}

async function refreshSources() {
  const select = $("library-select");
  try {
    const data = await request("/api/sources");
    const current = select.value;
    select.replaceChildren();
    const sources = Array.isArray(data?.sources) ? data.sources : [];
    for (const source of sources) {
      select.add(new Option(source.name ?? source.id ?? source.path, source.path));
    }
    if (!sources.length) select.add(new Option("No hay videos en la biblioteca", ""));
    if (sources.some((source) => source.path === current)) select.value = current;
    else if (sources.some((source) => source.path === data.default_path)) select.value = data.default_path;
  } catch {
    select.replaceChildren(new Option("Biblioteca no disponible", ""));
  }
}

function currentSource() {
  if (state.mode === "library") return $("library-select").value;
  return $("source-input").value.trim();
}

function sourceButton() {
  const active = activeStates.has(state.status.state);
  const preview = $("preview-button");
  const label = state.sourceBusy ? "Abriendo fuente…" : "Abrir vista previa";
  preview.disabled = state.sourceBusy || active;
  if (preview.textContent !== label) preview.textContent = label;
  $("source-lock").hidden = !active;
  $("load-profile").disabled = state.sourceBusy || active || !$("profile-select").value;
}

async function previewSource(source, savedConfig = null) {
  if (!source) throw new Error("Selecciona un video o escribe la ruta de la fuente.");
  const data = await post("/api/preview", {source});
  const separator = data.preview_url.includes("?") ? "&" : "?";
  const imageUrl = data.preview_url + separator + "v=" + Date.now();
  const image = new Image();
  image.src = imageUrl;
  try { await image.decode(); }
  catch { throw new Error("La fuente abrió, pero el cuadro de vista previa no se pudo mostrar."); }
  const sameSource = state.preview?.requestedSource === source;
  const sameShape = state.preview?.width === data.width && state.preview?.height === data.height;
  state.preview = {...data, imageUrl, requestedSource: source};
  state.pendingProfile = null;
  if (!sameSource || !sameShape) {
    state.roi = [];
    state.roiHistory = [];
  }
  if (savedConfig) applyConfig(savedConfig);
  $("video-info").textContent = sourceDescription(state.preview);
  if (!$("run-name").value.trim() || !sameSource) {
    $("run-name").value = savedConfig?.name || "";
  }
  updateRoiControls();
  showStep("zone", true);
}

async function openSource(event) {
  event.preventDefault();
  if (state.sourceBusy || activeStates.has(state.status.state)) return;
  state.sourceBusy = true;
  sourceButton();
  announce();
  try {
    let source = currentSource();
    if (state.mode === "upload") {
      const file = $("file-input").files[0];
      if (!file) throw new Error("Selecciona un video para subir.");
      const upload = new FormData();
      upload.append("file", file, file.name);
      const result = await request("/api/upload", {method: "POST", body: upload});
      source = result.path;
      $("source-input").value = source;
      setMode("address");
      refreshSources();
    }
    const pending = state.pendingProfile;
    const reuse = pending && state.mode === "address" && pending.source === publicAddress(source);
    await previewSource(source, reuse ? pending : null);
  } catch (error) {
    announce(error?.message === "Failed to fetch" ? "No se pudo abrir la fuente. Revisa la conexión con la aplicación." : error.message, "danger");
  } finally {
    state.sourceBusy = false;
    sourceButton();
  }
}

function sourceDescription(source) {
  if (!source) return "Fuente pendiente";
  const size = Number.isFinite(source.width) && Number.isFinite(source.height)
    ? source.width + " × " + source.height : "";
  const fps = source.fps_fallback ? "Frecuencia desconocida"
    : Number.isFinite(source.fps) && source.fps > 0 ? number(source.fps, decimal) + " fps" : "";
  const duration = Number.isFinite(source.duration_s) && source.duration_s > 0
    ? timecode(source.duration_s) : source.kind === "file" ? "Duración desconocida" : "";
  return [size, fps, duration].filter(Boolean).join("  ·  ") || "Fuente de video";
}

function renderViewer() {
  const operation = state.step === "operation";
  const editing = state.step === "zone" && Boolean(state.preview);
  const awaiting = state.status.state === "awaiting_confirmation";
  const frame = operation ? state.frameUrl : state.preview?.imageUrl;
  const source = operation ? state.status.source : state.preview;
  const showCheck = operation && awaiting && state.confirmFrame && Boolean(frame);
  $("viewer-heading").textContent = operation ? "Imagen de operación" : "Vista de la fuente";
  $("viewer-caption").textContent = operation
    ? (awaiting ? "Comprueba que la zona de conteo coincide con el encuadre recuperado." : "Último cuadro procesado de la corrida.")
    : (editing ? "Marca el área correspondiente a un acceso." : "Selecciona un video o conecta una cámara.");
  $("video-info").textContent = sourceDescription(source);
  const age = state.status.observation?.age_ms;
  $("frame-info").textContent = operation && Number.isFinite(age) && activeStates.has(state.status.state)
    ? number(age / 1000, decimal) + " s desde la lectura" : "";
  const viewerTag = $("viewer-tag");
  viewerTag.hidden = !source;
  viewerTag.textContent = operation ? "Operación" : (state.preview?.kind === "live" ? "En directo" : "Video");
  const aspect = source?.width > 0 && source?.height > 0 ? source.width / source.height : 16 / 9;
  $("view-surface").style.setProperty("--image-aspect", aspect);
  const image = $("preview-image");
  if (frame && image.getAttribute("src") !== frame) image.setAttribute("src", frame);
  image.hidden = !frame;
  image.alt = operation ? "Último cuadro de la corrida" : "Cuadro de la fuente seleccionada";
  $("empty-view").hidden = Boolean(frame);
  $("empty-title").textContent = operation ? "Sin imagen de operación" : "Sin vista previa";
  $("empty-detail").textContent = operation
    ? "El cuadro aparecerá cuando la fuente entregue una imagen."
    : "El primer cuadro de la fuente aparecerá aquí.";
  const canvas = $("roi-canvas");
  canvas.hidden = !editing && !showCheck;
  canvas.tabIndex = editing ? 0 : -1;
  if (editing) canvas.removeAttribute("aria-hidden");
  else canvas.setAttribute("aria-hidden", "true");
  if (editing || showCheck) requestAnimationFrame(drawRoi);
}

function copyRoi() {
  return state.roi.map((point) => point.slice());
}

function pushUndo(roi = copyRoi()) {
  state.roiHistory.push(roi);
  if (state.roiHistory.length > 40) state.roiHistory.shift();
}

function pointCross(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function onSegment(a, b, c) {
  return Math.abs(pointCross(a, b, c)) <= geometryTolerance && c.every((value, axis) => (
    value >= Math.min(a[axis], b[axis]) - geometryTolerance
    && value <= Math.max(a[axis], b[axis]) + geometryTolerance
  ));
}

function edgesCross(a, b, c, d) {
  const orientation = (p, q, r) => {
    const cross = pointCross(p, q, r);
    return Math.abs(cross) <= geometryTolerance ? 0 : Math.sign(cross);
  };
  const abc = orientation(a, b, c);
  const abd = orientation(a, b, d);
  const cda = orientation(c, d, a);
  const cdb = orientation(c, d, b);
  return (abc * abd < 0 && cda * cdb < 0)
    || (abc === 0 && onSegment(a, b, c)) || (abd === 0 && onSegment(a, b, d))
    || (cda === 0 && onSegment(c, d, a)) || (cdb === 0 && onSegment(c, d, b));
}

function roiProblem(roi = state.roi) {
  if (roi.length < 3) return "Añade al menos tres vértices.";
  let area = 0;
  for (let index = 0; index < roi.length; index += 1) {
    const a = roi[index];
    const b = roi[(index + 1) % roi.length];
    const c = roi[(index + 2) % roi.length];
    if (a[0] === b[0] && a[1] === b[1]) return "Separa los vértices superpuestos.";
    if (onSegment(a, b, c) || onSegment(b, c, a)) return "Ajusta los vértices para que los lados no se superpongan.";
    for (let other = index + 2; other < roi.length; other += 1) {
      if (index === 0 && other === roi.length - 1) continue;
      if (edgesCross(a, b, roi[other], roi[(other + 1) % roi.length])) {
        return "Ajusta los vértices para que los lados no se crucen.";
      }
    }
    area += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(area) <= geometryTolerance ? "Amplía el área del polígono." : "";
}

function updateRoiControls(rebuildFields = true) {
  const count = state.roi.length;
  const ready = Boolean(state.preview);
  const problem = roiProblem();
  const countText = count === 1 ? "1 vértice" : count + " vértices";
  $("zone-label").textContent = ready
    ? countText + (count >= 3 ? " · " + (problem || "Zona lista") : "")
    : "Abre una vista previa en Fuente.";
  $("zone-status").dataset.tone = !ready || count < 3 ? "neutral" : problem ? "warning" : "good";
  $("undo-roi").disabled = !ready || !state.roiHistory.length;
  $("clear-roi").disabled = !ready || !count;
  $("full-roi").disabled = !ready;
  $("start-run").disabled = !ready || Boolean(problem) || state.commandBusy || activeStates.has(state.status.state);
  $("save-profile").disabled = !ready || Boolean(problem);
  $("vertex-details").hidden = !ready;
  if (rebuildFields) renderVertexFields();
  drawRoi();
}

function renderVertexFields() {
  const container = $("vertex-fields");
  container.replaceChildren();
  state.roi.forEach((point, index) => {
    const row = node("div", undefined, "vertex-row field");
    row.append(node("span", String(index + 1)));
    point.forEach((value, axis) => {
      const name = axis === 0 ? "X" : "Y";
      const label = node("label", name + " %");
      const input = node("input");
      input.type = "number";
      input.min = "0";
      input.max = "100";
      input.step = "any";
      input.value = (value * 100).toFixed(2);
      input.setAttribute("aria-label", name + " del vértice " + (index + 1) + " en porcentaje");
      input.addEventListener("change", () => {
        if (!input.checkValidity() || input.value === "") {
          input.value = (state.roi[index][axis] * 100).toFixed(2);
          return;
        }
        pushUndo();
        state.roi[index][axis] = input.valueAsNumber / 100;
        updateRoiControls(false);
      });
      label.append(input);
      row.append(label);
    });
    container.append(row);
  });
}

function fittedCanvas(canvas) {
  const box = canvas.getBoundingClientRect();
  if (!box.width || !box.height) return null;
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(box.width * scale);
  const height = Math.round(box.height * scale);
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  return {ctx, width: box.width, height: box.height};
}

function paintPolygon(ctx, points, width, height, handles, invalid = false) {
  if (!points.length) return;
  const edge = invalid ? "#ffac91" : "#e8f77e";
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x * width, y * height);
    else ctx.lineTo(x * width, y * height);
  });
  if (points.length >= 3) {
    ctx.closePath();
    if (handles) {
      ctx.fillStyle = invalid ? "rgba(255, 172, 145, 0.17)" : "rgba(227, 242, 115, 0.17)";
      ctx.fill();
    }
  }
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(15, 26, 22, 0.88)";
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.strokeStyle = edge;
  ctx.lineWidth = 2.3;
  ctx.stroke();
  if (!handles) return;
  ctx.font = "bold 12px system-ui";
  points.forEach(([x, y], index) => {
    const left = x * width;
    const top = y * height;
    ctx.beginPath();
    ctx.arc(left, top, 5, 0, 2 * Math.PI);
    ctx.fillStyle = edge;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#19251c";
    ctx.stroke();
    const labelX = left < width - 22 ? left + 8 : left - 16;
    const labelY = top >= 17 ? top - 8 : top + 16;
    ctx.lineWidth = 3;
    ctx.strokeText(index + 1, labelX, labelY);
    ctx.fillText(index + 1, labelX, labelY);
  });
}

function drawRoi() {
  const canvas = $("roi-canvas");
  if (canvas.hidden) return;
  const layer = fittedCanvas(canvas);
  if (!layer) return;
  const {ctx, width, height} = layer;
  if (state.step !== "zone") {
    paintPolygon(ctx, pointsFrom(state.status.config?.roi), width, height, false);
    return;
  }
  paintPolygon(ctx, state.roi, width, height, true, state.roi.length >= 3 && Boolean(roiProblem()));
  if (state.keyboard && document.activeElement === canvas) {
    const [x, y] = [state.cursor[0] * width, state.cursor[1] * height];
    ctx.beginPath();
    ctx.moveTo(x - 10, y);
    ctx.lineTo(x + 10, y);
    ctx.moveTo(x, y - 10);
    ctx.lineTo(x, y + 10);
    ctx.strokeStyle = "#182223";
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function normalized(event) {
  const box = $("roi-canvas").getBoundingClientRect();
  const clamp = (value) => Math.min(1, Math.max(0, value));
  return [clamp((event.clientX - box.left) / box.width), clamp((event.clientY - box.top) / box.height)];
}

function addVertex(point) {
  if (state.roi.length >= 50) {
    announce("El polígono admite hasta 50 vértices.", "warning");
    return;
  }
  pushUndo();
  state.roi.push(point);
  updateRoiControls();
}

function undoRoi() {
  if (state.roiHistory.length) {
    state.roi = state.roiHistory.pop();
    updateRoiControls();
  }
}

function setRoi(points) {
  pushUndo();
  state.roi = points;
  updateRoiControls();
}

const roiCanvas = $("roi-canvas");
roiCanvas.addEventListener("pointerdown", (event) => {
  if (state.step !== "zone" || event.button !== 0 || state.drag) return;
  state.keyboard = false;
  const cursor = normalized(event);
  const {width, height} = roiCanvas.getBoundingClientRect();
  const radius = event.pointerType === "touch" ? 24 : 13;
  let nearest = -1;
  let distance = radius;
  state.roi.forEach(([x, y], index) => {
    const measured = Math.hypot((x - cursor[0]) * width, (y - cursor[1]) * height);
    if (measured < distance) {
      nearest = index;
      distance = measured;
    }
  });
  state.drag = {id: event.pointerId, index: nearest, before: copyRoi(), down: cursor};
  roiCanvas.setPointerCapture(event.pointerId);
});
roiCanvas.addEventListener("pointermove", (event) => {
  if (state.drag?.id !== event.pointerId || state.drag.index < 0) return;
  state.roi[state.drag.index] = normalized(event);
  updateRoiControls(false);
});
roiCanvas.addEventListener("pointerup", (event) => {
  const drag = state.drag;
  if (drag?.id !== event.pointerId) return;
  state.drag = null;
  if (drag.index < 0) {
    const point = normalized(event);
    if (Math.hypot(point[0] - drag.down[0], point[1] - drag.down[1]) < 0.03) addVertex(point);
  } else {
    const before = drag.before[drag.index];
    const after = state.roi[drag.index];
    if (before.some((value, axis) => value !== after[axis])) pushUndo(drag.before);
    updateRoiControls();
  }
});
roiCanvas.addEventListener("pointercancel", (event) => {
  if (state.drag?.id !== event.pointerId) return;
  state.roi = state.drag.before;
  state.drag = null;
  updateRoiControls();
});
roiCanvas.addEventListener("focus", drawRoi);
roiCanvas.addEventListener("blur", drawRoi);
roiCanvas.addEventListener("keydown", (event) => {
  const directions = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  const direction = directions[event.key];
  if (direction) {
    event.preventDefault();
    state.keyboard = true;
    const step = event.shiftKey ? 0.045 : 0.008;
    state.cursor = state.cursor.map((value, axis) => Math.max(0, Math.min(1, value + step * direction[axis])));
    drawRoi();
  } else if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    state.keyboard = true;
    addVertex(state.cursor.slice());
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoRoi();
  }
});

const configFields = {
  anchor: "anchor",
  model: "model",
  device: "device",
  image_size: "image-size",
  confidence: "confidence",
  sample_fps: "sample-fps",
  validation_interval_s: "validation-interval",
  crop_to_roi: "crop-to-roi",
  realtime: "realtime",
  tracking: "tracking",
  condition: "condition",
  save_annotated: "save-annotated",
};

function getConfig() {
  const config = {
    source: state.preview?.requestedSource || "",
    name: $("run-name").value.trim() || "Conteo vehicular",
    roi: copyRoi(),
  };
  for (const [key, id] of Object.entries(configFields)) {
    const field = $(id);
    if (field.type === "number") {
      if (!field.checkValidity() || !Number.isFinite(field.valueAsNumber)) {
        $("advanced").open = true;
        field.reportValidity();
        throw new Error("Revisa los valores de los ajustes avanzados.");
      }
      config[key] = field.valueAsNumber;
    } else if (field.type === "checkbox") config[key] = field.checked;
    else config[key] = field.value;
  }
  return config;
}

function pointsFrom(value) {
  if (!Array.isArray(value)) return [];
  const valid = value.every((point) => Array.isArray(point) && point.length === 2 && point.every((item) => (
    Number.isFinite(item) && item >= 0 && item <= 1
  )));
  if (!valid) return [];
  const points = value.map((point) => point.slice());
  if (points.length > 1 && points[0].every((value, axis) => value === points.at(-1)[axis])) points.pop();
  return points;
}

function applyConfig(config) {
  state.roi = pointsFrom(config.roi);
  state.roiHistory = [];
  $("run-name").value = config.name ?? "";
  $("crop-to-roi").checked = config.crop_to_roi === true;
  for (const [key, id] of Object.entries(configFields)) {
    const field = $(id);
    if (config[key] === undefined) continue;
    if (field.type === "checkbox") field.checked = Boolean(config[key]);
    else if (field.tagName !== "SELECT" || [...field.options].some((item) => item.value === config[key])) {
      field.value = config[key];
    }
  }
}

async function refreshProfiles() {
  try {
    const body = await request("/api/profiles");
    state.profiles = Array.isArray(body?.profiles) ? body.profiles : [];
    const select = $("profile-select");
    const selected = select.value;
    select.replaceChildren(new Option("Elegir perfil", ""));
    for (const profile of state.profiles) select.add(new Option(profile.name, String(profile.id)));
    if (state.profiles.some((profile) => String(profile.id) === selected)) select.value = selected;
    $("profile-control").hidden = !state.profiles.length;
    sourceButton();
  } catch {
    $("profile-control").hidden = true;
  }
}

async function loadProfile() {
  const profile = state.profiles.find((item) => String(item.id) === $("profile-select").value);
  if (!profile || state.sourceBusy || activeStates.has(state.status.state)) return;
  const entered = $("source-input").value.trim();
  const source = entered && publicAddress(entered) === profile.config?.source
    ? entered : profile.config?.source || currentSource();
  $("source-input").value = source;
  setMode("address");
  state.pendingProfile = profile.config;
  state.sourceBusy = true;
  sourceButton();
  try {
    await previewSource(source, profile.config);
  } catch (error) {
    announce(error?.message || "No se pudo abrir el perfil.", "danger");
  } finally {
    state.sourceBusy = false;
    sourceButton();
  }
}

function publicAddress(source) {
  const parts = /^([a-z][a-z0-9+.-]*):\/\/([^/?#]*)([^?#]*)/i.exec(source);
  if (!parts) return source;
  const [, protocol, authority, path] = parts;
  return protocol.toLowerCase() + "://" + authority.split("@").at(-1).toLowerCase() + path;
}

function openProfileDialog() {
  if (!state.preview || roiProblem()) return;
  $("profile-name").value = $("run-name").value.trim();
  $("profile-message").textContent = "";
  $("profile-dialog").showModal();
  $("profile-name").select();
}

async function saveProfile(event) {
  event.preventDefault();
  if (state.profileBusy) return;
  const name = $("profile-name").value.trim();
  const message = $("profile-message");
  if (!name) {
    message.textContent = "Escribe un nombre para el perfil.";
    message.dataset.tone = "danger";
    return;
  }
  state.profileBusy = true;
  $("confirm-profile").disabled = true;
  try {
    const config = getConfig();
    const containsCredentials = publicAddress(config.source) !== config.source;
    await post("/api/profiles", {name, config});
    $("profile-dialog").close();
    announce(containsCredentials
      ? "Perfil guardado. Al abrirlo, introduce de nuevo la URL de la cámara."
      : "Perfil guardado.", "good");
    refreshProfiles();
  } catch (error) {
    message.textContent = error?.message || "No se pudo guardar el perfil.";
    message.dataset.tone = "danger";
  } finally {
    state.profileBusy = false;
    $("confirm-profile").disabled = false;
  }
}

function clearFrame() {
  if (state.frameUrl) URL.revokeObjectURL(state.frameUrl);
  state.frameUrl = null;
  state.confirmFrame = false;
}

async function startRun() {
  if (state.commandBusy || activeStates.has(state.status.state) || !state.preview || roiProblem()) return;
  state.commandBusy = true;
  updateRoiControls(false);
  try {
    const config = getConfig();
    const response = await post("/api/start", config);
    clearFrame();
    state.status = {state: "loading", run_id: response.run_id, source: state.preview, config};
    state.runId = String(response.run_id);
    renderStatus();
    showStep("operation", true);
    refreshStatus();
  } catch (error) {
    announce(error?.message === "Failed to fetch" ? "No se pudo comunicar con la aplicación." : error.message, "danger");
  } finally {
    state.commandBusy = false;
    updateRoiControls(false);
  }
}

async function command(path, fallback) {
  if (state.commandBusy) return;
  state.commandBusy = true;
  renderOperation();
  announce();
  try {
    await post(path, {});
    await refreshStatus();
  } catch (error) {
    announce(errorText(error, fallback), "danger");
  } finally {
    state.commandBusy = false;
    renderOperation();
  }
}

function dataLine(label, value) {
  const pair = node("div");
  pair.append(node("dt", label), node("dd", value));
  return pair;
}

function renderOperation() {
  const status = state.status;
  const obs = status.observation || {};
  const message = {
    idle: "Inicia una corrida desde Zona.",
    loading: "Preparando el análisis de video.",
    running: "Conteo en la zona seleccionada.",
    completed: "Terminó el procesamiento del video.",
    stopped: "La corrida se detuvo.",
    error: "La corrida se interrumpió.",
    reconnecting: "Esperando que vuelva la fuente.",
    awaiting_confirmation: "Verifica el encuadre de la cámara antes de continuar.",
  };
  $("operation-message").textContent = status.message || message[status.state] || "";
  $("count").textContent = number(obs.vehicles_present);
  const historical = ["completed", "stopped", "interrupted"].includes(status.state) && Number.isFinite(obs.vehicles_present);
  $("observation-label").textContent = historical ? "Última lectura de la sesión" : (
    qualities[obs.status] || (Number.isFinite(obs.vehicles_present) ? "Última lectura" : "Sin lectura")
  );
  const warnings = Array.isArray(obs.warnings) ? obs.warnings : [];
  $("observation-warning").hidden = !warnings.length;
  $("observation-warning").textContent = warnings.join(" · ");
  const progress = $("progress");
  progress.hidden = !Number.isFinite(obs.progress);
  progress.value = Number.isFinite(obs.progress) ? Math.min(1, Math.max(0, obs.progress)) : 0;
  $("source-time").textContent = timecode(obs.source_time_s, true);
  $("processed").textContent = number(obs.processed_frames);
  $("inference").textContent = Number.isFinite(obs.inference_ms) ? number(obs.inference_ms, decimal) + " ms" : "—";
  $("processing-fps").textContent = Number.isFinite(obs.processing_fps) ? number(obs.processing_fps, decimal) + " fps" : "—";
  const classes = $("class-counts");
  const items = obs.by_class && Number.isFinite(obs.vehicles_present) ? Object.entries(obs.by_class) : [];
  classes.replaceChildren(...(items.length
    ? items.map(([name, count]) => dataLine(classLabels[name] || name, number(count)))
    : [dataLine("Sin desglose disponible", "—")]));
  $("stop-run").disabled = state.commandBusy || !activeStates.has(status.state);
  $("confirm-view").hidden = status.state !== "awaiting_confirmation";
  $("confirm-view").disabled = state.commandBusy || !state.confirmFrame;
}

function drawChart() {
  const canvas = $("history-chart");
  const surface = fittedCanvas(canvas);
  if (!surface) return;
  const {ctx, width, height} = surface;
  const raw = Array.isArray(state.status.history) ? state.status.history : [];
  const points = raw.filter((point) => Number.isFinite(point.t));
  const valid = points.filter((point) => Number.isFinite(point.count));
  $("chart-empty").hidden = valid.length > 0;
  if (!valid.length) return;
  const first = points[0].t;
  const last = points.at(-1).t;
  const max = Math.max(1, ...valid.map((point) => point.count));
  const interval = Math.max(1, Math.ceil(max / 4));
  const maximum = Math.ceil(max / interval) * interval;
  const left = 34;
  const right = width - 12;
  const top = 15;
  const bottom = height - 24;
  const span = last > first ? last - first : 1;
  const x = (t) => left + ((t - first) / span) * (right - left);
  const y = (count) => bottom - (count / maximum) * (bottom - top);
  const style = getComputedStyle(document.documentElement);
  ctx.font = "11px " + style.getPropertyValue("--mono");
  ctx.lineWidth = 1;
  ctx.strokeStyle = style.getPropertyValue("--stroke").trim();
  ctx.fillStyle = style.getPropertyValue("--secondary").trim();
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let value = 0; value <= maximum; value += interval) {
    const row = y(value);
    ctx.beginPath();
    ctx.moveTo(left, row);
    ctx.lineTo(right, row);
    ctx.stroke();
    ctx.fillText(String(value), left - 7, row);
  }
  ctx.textAlign = "left";
  ctx.fillText(timecode(first, span < 1), left, height - 9);
  if (last > first) {
    ctx.textAlign = "right";
    ctx.fillText(timecode(last, span < 1), right, height - 9);
  }
  ctx.strokeStyle = style.getPropertyValue("--focus").trim();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let drawing = false;
  for (const point of points) {
    if (!Number.isFinite(point.count)) {
      drawing = false;
      continue;
    }
    if (drawing) ctx.lineTo(x(point.t), y(point.count));
    else ctx.moveTo(x(point.t), y(point.count));
    drawing = true;
  }
  ctx.stroke();
  const newest = valid.at(-1);
  ctx.beginPath();
  ctx.arc(x(newest.t), y(newest.count), 3.5, 0, Math.PI * 2);
  ctx.fill();
}

function renderStatus() {
  const [label, tone] = states[state.status.state] || states.idle;
  $("session-label").textContent = label;
  $("session-badge").dataset.tone = tone;
  sourceButton();
  if (state.step === "zone") updateRoiControls(false);
  if (state.step === "operation") {
    renderOperation();
    renderViewer();
    drawChart();
  }
}

let statusBusy = false;
function setConnection(connected) {
  const connection = $("connection-label");
  const label = connected ? "Conectado" : "Sin conexión";
  if (connection.textContent === label) return;
  $("connection").dataset.tone = connected ? "good" : "danger";
  connection.textContent = label;
}

async function refreshStatus() {
  if (statusBusy) return;
  statusBusy = true;
  try {
    const before = state.status;
    const status = await request("/api/status", {signal: AbortSignal.timeout(3500)});
    setConnection(true);
    if (state.status !== before && state.status.state === "loading" && status.run_id !== state.status.run_id) return;
    if (before.run_id && status.run_id !== before.run_id) clearFrame();
    if (status.state !== before.state || status.run_id !== before.run_id) {
      state.frameRequired = true;
      if (status.state === "awaiting_confirmation") state.confirmFrame = false;
      if (state.step === "results" && !activeStates.has(status.state)) refreshRuns(true);
    }
    state.status = status;
    if (state.initialStatus) {
      state.initialStatus = false;
      if (activeStates.has(status.state) && state.step === "source") showStep("operation");
    }
    renderStatus();
  } catch {
    setConnection(false);
    if (state.step === "operation" && activeStates.has(state.status.state)) {
      $("count").textContent = "—";
      $("observation-label").textContent = "Sin conexión con la aplicación";
      $("class-counts").replaceChildren(dataLine("Sin desglose disponible", "—"));
    }
  } finally {
    statusBusy = false;
  }
}

async function refreshFrame() {
  const status = state.status;
  if (state.step !== "operation" || document.hidden || state.frameBusy || !status.run_id) return;
  if (!activeStates.has(status.state) && !state.frameRequired) return;
  const available = status.frame_available ?? (
    status.observation?.processed_frames > 0 || status.state === "awaiting_confirmation"
  );
  if (!available) return;
  state.frameBusy = true;
  try {
    const response = await fetch("/api/frame.jpg", {cache: "no-store", signal: AbortSignal.timeout(2200)});
    if (!response.ok) return;
    const image = await response.blob();
    if (state.step !== "operation" || state.status.run_id !== status.run_id) return;
    if (status.state !== state.status.state && state.status.state === "awaiting_confirmation") return;
    if (state.frameUrl) URL.revokeObjectURL(state.frameUrl);
    state.frameUrl = URL.createObjectURL(image);
    state.frameRequired = false;
    if (status.state === "awaiting_confirmation") {
      state.confirmFrame = true;
      renderOperation();
    }
    renderViewer();
  } catch {
    // Se conserva el último cuadro mientras llega el siguiente.
  } finally {
    state.frameBusy = false;
  }
}

function renderRunList() {
  const list = $("run-list");
  list.replaceChildren();
  if (!state.runs.length) {
    list.append(node("p", "Las corridas guardadas aparecerán aquí.", "muted"));
    return;
  }
  for (const run of state.runs) {
    const button = node("button", undefined, "run-button");
    button.type = "button";
    button.setAttribute("aria-pressed", String(String(run.id) === state.runId));
    const name = node("strong", run.name || "Corrida " + run.id);
    const detail = node("small", [runTime(run.created_at), states[run.state]?.[0]].filter(Boolean).join(" · "));
    button.append(name, detail);
    button.addEventListener("click", () => {
      state.runId = String(run.id);
      for (const item of list.querySelectorAll("button")) item.setAttribute("aria-pressed", String(item === button));
      openRun(run);
    });
    list.append(button);
  }
}

async function refreshRuns(keepEditor = false) {
  const generation = ++state.runsRequest;
  try {
    const data = await request("/api/runs");
    if (generation !== state.runsRequest || state.step !== "results") return;
    const previous = state.runs.find((run) => String(run.id) === state.runId);
    state.runs = Array.isArray(data?.runs) ? data.runs : [];
    const selected = state.runs.find((run) => String(run.id) === state.runId) ?? state.runs[0];
    state.runId = selected ? String(selected.id) : null;
    renderRunList();
    if (!keepEditor || !previous || selected?.id !== previous.id || selected.state !== previous.state) openRun(selected);
  } catch (error) {
    if (generation !== state.runsRequest || state.step !== "results") return;
    $("run-list").replaceChildren(node("p", errorText(error, "No se pudo abrir el historial de corridas."), "muted"));
  }
}

function renderRunSummary(run) {
  const summary = $("result-summary");
  summary.replaceChildren();
  const data = run.summary || {};
  const fields = [
    ["Cuadros analizados", data.processed_frames ?? data.frames_processed, integer],
    ["Lecturas válidas", data.valid_counts ?? data.observations ?? data.observation_count, integer],
    ["Máximo simultáneo", data.max_present ?? data.max_count ?? data.max_vehicles_present, integer],
    ["Presencia media", data.mean_present ?? data.mean_count ?? data.mean_vehicles_present, decimal],
  ];
  for (const [label, value, formatter] of fields) {
    if (Number.isFinite(value)) {
      const block = node("dl");
      block.append(dataLine(label, number(value, formatter)));
      summary.append(block);
    }
  }
}

async function openRun(run) {
  const generation = ++state.sampleRequest;
  const active = run && activeStates.has(run.state);
  state.samples = [];
  state.sampleIndex = 0;
  state.sampleRoi = [];
  state.sampleImageReady = false;
  state.annotationBusy = false;
  $("result-heading").textContent = run?.name || "Resultados";
  $("result-subtitle").textContent = run
    ? [runTime(run.created_at), states[run.state]?.[0]].filter(Boolean).join(" · ")
    : "Selecciona una corrida.";
  const download = $("download-run");
  download.hidden = !run || active;
  if (run) download.href = "/api/runs/" + encodeURIComponent(run.id) + "/download";
  $("result-summary").replaceChildren();
  $("validation-panel").hidden = !run;
  $("sample-editor").hidden = true;
  $("metrics-panel").hidden = true;
  $("sample-progress").hidden = true;
  $("validation-empty").hidden = false;
  $("validation-empty").textContent = active ? "La validación estará disponible al terminar la corrida." : "Cargando muestras…";
  if (!run) return;
  renderRunSummary(run);
  if (active) return;
  const path = "/api/runs/" + encodeURIComponent(run.id);
  const [samples, metrics] = await Promise.allSettled([request(path + "/samples"), request(path + "/metrics")]);
  if (generation !== state.sampleRequest || state.step !== "results") return;
  if (samples.status === "fulfilled") {
    const body = samples.value;
    state.samples = Array.isArray(body?.samples) ? body.samples : [];
    state.sampleRoi = pointsFrom(body?.roi ?? body?.config?.roi ?? run.config?.roi);
    const anchor = body?.anchor ?? body?.config?.anchor ?? run.config?.anchor;
    $("annotation-instructions").textContent = anchor === "bottom_center"
      ? "Cuenta los vehículos cuyo punto central inferior está dentro del polígono."
      : anchor === "center" ? "Cuenta los vehículos cuyo centro está dentro del polígono."
        : "Cuenta los vehículos dentro del polígono en el cuadro sin detecciones.";
    const firstPending = state.samples.findIndex((sample) => !Number.isInteger(sample.manual_count));
    state.sampleIndex = firstPending >= 0 ? firstPending : 0;
    renderSample();
  } else {
    $("validation-empty").textContent = errorText(samples.reason, "No se pudieron cargar las muestras.");
  }
  if (metrics.status === "fulfilled") renderMetrics(metrics.value?.metrics ?? metrics.value);
}

function drawSampleRoi() {
  const image = $("sample-image");
  const canvas = $("sample-roi");
  if ($("sample-editor").hidden || image.hidden || !state.sampleImageReady || !image.naturalWidth) {
    canvas.hidden = true;
    return;
  }
  const rect = image.getBoundingClientRect();
  const parent = $("sample-view").getBoundingClientRect();
  canvas.style.left = (rect.left - parent.left) + "px";
  canvas.style.top = (rect.top - parent.top) + "px";
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
  canvas.hidden = !state.sampleRoi.length;
  if (canvas.hidden) return;
  const layer = fittedCanvas(canvas);
  if (layer) paintPolygon(layer.ctx, state.sampleRoi, layer.width, layer.height, false);
}

function renderSample(clearMessage = true) {
  const sample = state.samples[state.sampleIndex];
  $("sample-editor").hidden = !sample;
  $("sample-progress").hidden = !sample;
  $("validation-empty").hidden = Boolean(sample);
  if (!sample) {
    $("validation-empty").textContent = "Esta corrida no tiene cuadros disponibles para validación.";
    return;
  }
  const labeled = state.samples.filter((item) => Number.isInteger(item.manual_count)).length;
  $("sample-progress").textContent = labeled + " de " + state.samples.length + " anotadas";
  $("sample-time").textContent = (state.sampleIndex + 1) + "/" + state.samples.length + " · " + timecode(sample.time_s, true);
  $("prev-sample").disabled = state.annotationBusy || state.sampleIndex === 0;
  $("next-sample").disabled = state.annotationBusy || state.sampleIndex === state.samples.length - 1;
  const image = $("sample-image");
  if (image.getAttribute("src") !== sample.image_url) {
    state.sampleImageReady = false;
    $("sample-roi").hidden = true;
    image.hidden = false;
    image.src = sample.image_url;
  } else if (image.complete && image.naturalWidth > 0) {
    state.sampleImageReady = true;
  }
  $("manual-count").value = Number.isInteger(sample.manual_count) ? sample.manual_count : "";
  $("sample-condition").value = conditions[sample.condition] ? sample.condition : "unknown";
  $("save-annotation").disabled = state.annotationBusy || !state.sampleImageReady;
  $("clear-annotation").hidden = !Number.isInteger(sample.manual_count);
  $("clear-annotation").disabled = state.annotationBusy;
  if (clearMessage) $("annotation-status").textContent = "";
  requestAnimationFrame(drawSampleRoi);
}

function moveSample(amount) {
  if (state.annotationBusy) return;
  state.sampleIndex = Math.max(0, Math.min(state.samples.length - 1, state.sampleIndex + amount));
  renderSample();
}

function renderMetrics(metrics) {
  $("metrics-panel").hidden = !metrics || (!state.samples.some((sample) => Number.isInteger(sample.manual_count)) && !(metrics.matched > 0));
  if ($("metrics-panel").hidden) return;
  const list = [
    ["Muestras comparadas", metrics.predicted_samples ?? metrics.matched, integer],
    ["Coincidencia exacta", metrics.exact_agreement, percent],
    ["Error medio absoluto", metrics.mae, decimal],
    ["Falsos vacíos", metrics.false_empty_rate, percent],
    ["Falsas presencias", metrics.false_occupied_rate, percent],
    ["Muestras anotadas", metrics.coverage, percent],
  ];
  if (metrics.prediction_coverage !== undefined) {
    list.push(["Anotaciones con lectura", metrics.prediction_coverage, percent]);
  }
  $("validation-metrics").replaceChildren(...list.map(([label, value, formatter]) => dataLine(label, number(value, formatter))));
  const groups = Array.isArray(metrics.by_condition)
    ? metrics.by_condition.map((group) => [group.condition, group])
    : Object.entries(metrics.by_condition || {});
  const container = $("condition-metrics");
  container.replaceChildren();
  if (!groups.length) return;
  const table = node("table", undefined, "condition-table");
  table.createCaption().textContent = "Resultados por condición";
  const header = table.createTHead().insertRow();
  for (const label of ["Condición", "Comparadas", "Coincidencia", "Error medio"]) {
    const th = node("th", label);
    th.scope = "col";
    header.append(th);
  }
  const body = table.createTBody();
  for (const [condition, values] of groups) {
    if (!values || typeof values !== "object") continue;
    const row = body.insertRow();
    for (const value of [conditions[condition] || condition, number(values.predicted_samples ?? values.matched), number(values.exact_agreement, percent), number(values.mae, decimal)]) {
      row.insertCell().textContent = value;
    }
  }
  container.append(table);
}

async function saveAnnotation(clear = false) {
  const sample = state.samples[state.sampleIndex];
  if (!sample || !state.runId || state.annotationBusy) return;
  const manual = $("manual-count");
  const count = clear ? null : manual.valueAsNumber;
  if (!clear && !state.sampleImageReady) return;
  if (!clear && (!Number.isInteger(count) || !manual.checkValidity())) {
    $("annotation-status").textContent = "Introduce un número entero entre 0 y 1.000.";
    $("annotation-status").dataset.tone = "danger";
    manual.focus();
    return;
  }
  const condition = $("sample-condition").value;
  const runId = state.runId;
  const generation = state.sampleRequest;
  state.annotationBusy = true;
  for (const id of ["save-annotation", "clear-annotation", "prev-sample", "next-sample"]) $(id).disabled = true;
  const message = $("annotation-status");
  message.dataset.tone = "neutral";
  message.textContent = "Guardando…";
  try {
    const data = await post("/api/runs/" + encodeURIComponent(runId) + "/annotations", {
      sample_id: sample.id, manual_count: count, condition,
    });
    if (generation !== state.sampleRequest) return;
    sample.manual_count = count;
    sample.condition = condition;
    renderMetrics(data?.metrics ?? data);
    message.textContent = clear ? "Se quitó el conteo manual." : "Conteo guardado.";
  } catch (error) {
    if (generation !== state.sampleRequest) return;
    message.dataset.tone = "danger";
    message.textContent = errorText(error, "No se pudo guardar el conteo.");
  } finally {
    if (generation === state.sampleRequest) {
      state.annotationBusy = false;
      $("prev-sample").disabled = state.sampleIndex === 0;
      $("next-sample").disabled = state.sampleIndex === state.samples.length - 1;
      $("save-annotation").disabled = !state.sampleImageReady;
      $("clear-annotation").disabled = false;
      $("clear-annotation").hidden = !Number.isInteger(sample.manual_count);
      if (clear && sample.manual_count === null) manual.value = "";
      $("sample-progress").textContent = state.samples.filter((item) => Number.isInteger(item.manual_count)).length
        + " de " + state.samples.length + " anotadas";
    }
  }
}

for (const button of steps) button.addEventListener("click", () => showStep(button.dataset.step));
for (const button of sourceModes) button.addEventListener("click", () => setMode(button.dataset.mode));
$("theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true));
$("source-form").addEventListener("submit", openSource);
$("profile-select").addEventListener("change", sourceButton);
$("load-profile").addEventListener("click", loadProfile);
$("undo-roi").addEventListener("click", undoRoi);
$("clear-roi").addEventListener("click", () => setRoi([]));
$("full-roi").addEventListener("click", () => setRoi([[0, 0], [1, 0], [1, 1], [0, 1]]));
$("start-run").addEventListener("click", startRun);
$("save-profile").addEventListener("click", openProfileDialog);
$("profile-form").addEventListener("submit", saveProfile);
$("cancel-profile").addEventListener("click", () => $("profile-dialog").close());
$("stop-run").addEventListener("click", () => command("/api/stop", "No se pudo detener la corrida."));
$("confirm-view").addEventListener("click", () => command("/api/confirm-view", "No se pudo confirmar el encuadre."));
$("open-results").addEventListener("click", () => showStep("results", true));
$("sample-image").addEventListener("load", () => {
  state.sampleImageReady = true;
  $("save-annotation").disabled = state.annotationBusy;
  drawSampleRoi();
});
$("sample-image").addEventListener("error", () => {
  state.sampleImageReady = false;
  $("save-annotation").disabled = true;
  $("sample-image").hidden = true;
  $("sample-roi").hidden = true;
  $("annotation-status").textContent = "No se pudo mostrar este cuadro.";
  $("annotation-status").dataset.tone = "danger";
});
$("prev-sample").addEventListener("click", () => moveSample(-1));
$("next-sample").addEventListener("click", () => moveSample(1));
$("annotation-form").addEventListener("submit", (event) => {
  event.preventDefault();
  saveAnnotation();
});
$("clear-annotation").addEventListener("click", () => saveAnnotation(true));

const resize = new ResizeObserver(() => {
  drawRoi();
  drawChart();
  drawSampleRoi();
});
for (const id of ["view-surface", "history-chart", "sample-view"]) resize.observe($(id));

let savedTheme = null;
try { savedTheme = localStorage.getItem("smart-signals-theme"); } catch { /* Tema predeterminado. */ }
setTheme(savedTheme === "light" ? "light" : "dark");
updateRoiControls();
refreshSources();
refreshProfiles();
refreshStatus();
window.setInterval(refreshStatus, 1000);
window.setInterval(refreshFrame, 400);
