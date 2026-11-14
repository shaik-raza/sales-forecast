import { apiRequest, showToast } from "./api.js";

const authScreen = document.querySelector("#auth-screen");
const dashboardShell = document.querySelector("#dashboard-shell");
const authForm = document.querySelector("#auth-form");
const authTitle = document.querySelector("#auth-title");
const authSubtitle = document.querySelector("#auth-subtitle");
const authSubmit = document.querySelector("#auth-submit");
const buttonLabel = authSubmit.querySelector(".button-label");
const toggleAuth = document.querySelector("#toggle-auth");
const loginTab = document.querySelector("#login-tab");
const signupTab = document.querySelector("#signup-tab");
const authPassword = document.querySelector("#auth-password");
const authStatus = document.querySelector("#auth-status");
const signoutButton = document.querySelector("#signout-button");
const themeToggle = document.querySelector("#theme-toggle");
const themeIcon = document.querySelector("#theme-icon");
const forecastForm = document.querySelector("#forecast-form");
const forecastStore = document.querySelector("#forecast-store");
const storeSummary = document.querySelector("#store-summary");
const forecastSubmit = document.querySelector("#forecast-submit");
const forecastValue = document.querySelector("#forecast-value");
let signupMode = false;
let selectedStore = null;
let dashboardLoaded = false;
let maeChart = null;
let predictionChart = null;
let historyChart = null;
let compareStoreDetails = null;
let historyRecords = [];
const MODEL_COLORS = {
  ensemble: "#b28cff",
  xgboost: "#79a8ff",
  lightgbm: "#70e6b0",
  catboost: "#ffbd72",
  random_forest: "#ff8e9e",
};

function applyTheme(theme) {
  const light = theme === "light";
  document.body.classList.toggle("light-theme", light);
  themeIcon.textContent = light ? "☀" : "☾";
  themeToggle.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
}

function updateAuthMode() {
  loginTab.classList.toggle("active", !signupMode);
  signupTab.classList.toggle("active", signupMode);
  authTitle.textContent = signupMode ? "Create your account" : "Welcome back";
  authSubtitle.textContent = signupMode
    ? "Create an account to access your forecasting workspace."
    : "Sign in to access your forecasting workspace.";
  buttonLabel.textContent = signupMode ? "Create account" : "Login";
  toggleAuth.textContent = signupMode ? "Sign in instead" : "Create an account";
  authPassword.autocomplete = signupMode ? "new-password" : "current-password";
}

function updateShell() {
  const signedIn = Boolean(localStorage.getItem("access_token"));
  authScreen.classList.toggle("hidden", signedIn);
  dashboardShell.classList.toggle("hidden", !signedIn);
  authStatus.textContent = signedIn ? "Signed in" : "Not signed in";
  document.querySelector(".status-dot").style.background = signedIn ? "#70e6b0" : "";
}

function showView() {
  const target = window.location.hash.replace("#", "") || "dashboard";
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.getAttribute("href") === `#${target}`);
  });
  document.querySelectorAll(".content-placeholder").forEach((view) => {
    view.classList.toggle("visible", view.id === `${target}-view`);
  });
  document.querySelector("#dashboard-view").style.display = target === "dashboard" ? "block" : "none";
  if (target === "dashboard" && localStorage.getItem("access_token") && !dashboardLoaded) {
    loadDashboard();
  }
  if (target === "history" && localStorage.getItem("access_token")) {
    loadHistory();
  }
}

function setLoading(loading) {
  authSubmit.classList.toggle("loading", loading);
  authSubmit.setAttribute("aria-busy", String(loading));
}

function setForecastLoading(loading) {
  forecastSubmit.classList.toggle("loading", loading);
  forecastSubmit.setAttribute("aria-busy", String(loading));
}

function renderStoreSummary(details) {
  selectedStore = details;
  storeSummary.innerHTML = `
    <span class="summary-icon">⌂</span>
    <div><strong>Store ${details.Store}</strong>
    <small>Type ${details.StoreType} · Assortment ${details.Assortment} · Competition ${details.CompetitionDistance.toLocaleString()} km</small></div>
  `;
}

async function loadStoreDetails() {
  const storeId = Number(forecastStore.value);
  if (!Number.isInteger(storeId) || storeId < 1 || storeId > 1115) {
    selectedStore = null;
    storeSummary.innerHTML = "<span class=\"summary-icon\">⌂</span><div><strong>Store details</strong><small>Enter a store ID from 1 to 1115.</small></div>";
    return;
  }
  try {
    renderStoreSummary(await apiRequest(`/stores/${storeId}`));
  } catch (error) {
    selectedStore = null;
    storeSummary.innerHTML = `<span class="summary-icon">!</span><div><strong>Store not found</strong><small>${error.message}</small></div>`;
    showToast(error.message);
  }
}

function isoWeek(date) {
  const target = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = target.getUTCDay() || 7;
  target.setUTCDate(target.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(target.getUTCFullYear(), 0, 1));
  return Math.ceil((((target - yearStart) / 86400000) + 1) / 7);
}

function forecastPayload() {
  const date = new Date(`${document.querySelector("#forecast-date").value}T00:00:00`);
  return {
    ...selectedStore,
    Date: document.querySelector("#forecast-date").value,
    DayOfWeek: ((date.getDay() + 6) % 7) + 1,
    Promo: Number(document.querySelector("#forecast-promo").checked),
    StateHoliday: Number(document.querySelector("#forecast-state-holiday").value),
    SchoolHoliday: Number(document.querySelector("#forecast-school-holiday").checked),
    Year: date.getFullYear(),
    Month: date.getMonth() + 1,
    Day: date.getDate(),
    WeekOfYear: isoWeek(date),
  };
}

function animateNumber(element, value, formatter) {
  const duration = 700;
  const start = performance.now();
  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - ((1 - progress) ** 3);
    element.textContent = formatter(value * eased);
    if (progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function metricValue(metrics, key) {
  return Number(metrics[key] || 0);
}

function renderDashboard(metrics, history) {
  const entries = Object.entries(metrics);
  const best = entries.reduce((winner, current) => (
    Number(current[1].RMSPE) < Number(winner[1].RMSPE) ? current : winner
  ));
  const cards = [
    { label: "Best model", value: best[0].replace("_", " "), caption: `Lowest RMSPE · ${Number(best[1].RMSPE).toFixed(2)}%`, text: true },
    { label: "Best model MAE", value: metricValue(best[1], "MAE"), caption: "Average error in euros", prefix: "€" },
    { label: "Best model MAPE", value: metricValue(best[1], "MAPE"), caption: "Mean percentage error", suffix: "%" },
    { label: "Best model R²", value: metricValue(best[1], "R2"), caption: "Explained variance", decimals: 4 },
    { label: "Your forecasts", value: history.length, caption: "Saved predictions", decimals: 0 },
  ];
  const kpis = document.querySelector("#dashboard-kpis");
  kpis.innerHTML = cards.map((card, index) => `
    <article class="glass-card kpi-card">
      <div class="kpi-label">${card.label}</div>
      <div class="kpi-value" id="kpi-${index}">${card.text ? card.value : "0"}</div>
      <div class="kpi-caption">${card.caption}</div>
    </article>
  `).join("");
  cards.forEach((card, index) => {
    if (!card.text) {
      const formatter = (value) => `${card.prefix || ""}${value.toFixed(card.decimals || 2)}${card.suffix || ""}`;
      animateNumber(document.querySelector(`#kpi-${index}`), card.value, formatter);
    }
  });

  const names = entries.map(([name]) => name.replace("_", " "));
  const values = entries.map(([, value]) => Number(value.MAE));
  if (maeChart) maeChart.destroy();
  maeChart = new Chart(document.querySelector("#mae-chart"), {
    type: "bar",
    data: {
      labels: names,
      datasets: [{
        data: values,
        backgroundColor: entries.map(([name]) => MODEL_COLORS[name] || "#79a8ff"),
        borderColor: entries.map(([name]) => MODEL_COLORS[name] || "#79a8ff"),
        borderWidth: 1,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#9da5c4" }, grid: { display: false } },
        y: { ticks: { color: "#9da5c4" }, grid: { color: "rgba(151, 164, 255, .1)" } },
      },
    },
  });

  document.querySelector(".history-loading").hidden = true;
  if (history.length === 0) {
    document.querySelector(".empty-state").hidden = false;
  } else {
    document.querySelector("#history-count").textContent = history.length;
    document.querySelector(".history-ready").hidden = false;
  }
}

async function loadDashboard() {
  dashboardLoaded = true;
  try {
    const [metrics, history] = await Promise.all([
      apiRequest("/models/metrics"),
      apiRequest("/history"),
    ]);
    renderDashboard(metrics, history);
  } catch (error) {
    dashboardLoaded = false;
    document.querySelector(".history-loading").hidden = true;
    showToast(error.message);
  }

  function comparePayload() {
    const dateText = document.querySelector("#compare-date").value;
    const date = new Date(`${dateText}T00:00:00`);
    return {
      ...compareStoreDetails,
      Date: dateText,
      DayOfWeek: ((date.getDay() + 6) % 7) + 1,
      Promo: Number(document.querySelector("#compare-promo").checked),
      StateHoliday: Number(document.querySelector("#compare-state-holiday").value),
      SchoolHoliday: Number(document.querySelector("#compare-school-holiday").checked),
      Year: date.getFullYear(),
      Month: date.getMonth() + 1,
      Day: date.getDate(),
      WeekOfYear: isoWeek(date),
    };
  }

  async function loadCompareStore() {
    const storeId = Number(document.querySelector("#compare-store").value);
    const summary = document.querySelector("#compare-store-summary");
    if (!Number.isInteger(storeId) || storeId < 1 || storeId > 1115) {
      compareStoreDetails = null;
      summary.innerHTML = "<span class=\"summary-icon\">⌂</span><div><strong>Store details</strong><small>Enter a store ID from 1 to 1115.</small></div>";
      return;
    }
    try {
      compareStoreDetails = await apiRequest(`/stores/${storeId}`);
      summary.innerHTML = `<span class="summary-icon">⌂</span><div><strong>Store ${compareStoreDetails.Store}</strong><small>Type ${compareStoreDetails.StoreType} · Assortment ${compareStoreDetails.Assortment} · Competition ${compareStoreDetails.CompetitionDistance.toLocaleString()} km</small></div>`;
    } catch (error) {
      compareStoreDetails = null;
      summary.innerHTML = `<span class="summary-icon">!</span><div><strong>Store not found</strong><small>${error.message}</small></div>`;
      showToast(error.message);
    }
  }

  function renderComparison(predictions, metrics) {
    const names = Object.keys(predictions);
    if (predictionChart) predictionChart.destroy();
    predictionChart = new Chart(document.querySelector("#prediction-chart"), {
      type: "bar",
      data: {
        labels: names.map((name) => name.replace("_", " ")),
        datasets: [{
          data: names.map((name) => predictions[name]),
          backgroundColor: names.map((name) => MODEL_COLORS[name]),
          borderColor: names.map((name) => MODEL_COLORS[name]),
          borderWidth: 1,
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#9da5c4" }, grid: { display: false } },
          y: { ticks: { color: "#9da5c4", callback: (value) => `€${Number(value).toLocaleString()}` }, grid: { color: "rgba(151, 164, 255, .1)" } },
        },
      },
    });

    const metricKeys = [
      ["MAE", "MAE", false],
      ["RMSE", "RMSE", false],
      ["MAPE", "MAPE", false],
      ["RMSPE", "RMSPE", false],
      ["R2", "R²", true],
    ];
    const modelNames = Object.keys(metrics);
    const bestValues = Object.fromEntries(metricKeys.map(([key, , higher]) => [
      key,
      Math[higher ? "max" : "min"](...modelNames.map((name) => Number(metrics[name][key]))),
    ]));
    document.querySelector("#metrics-table-body").innerHTML = modelNames.map((name) => `
      <tr>
        <td><span class="model-dot" style="background:${MODEL_COLORS[name]}"></span>${name.replace("_", " ")}</td>
        ${metricKeys.map(([key]) => `<td class="${Number(metrics[name][key]) === bestValues[key] ? "best-cell" : ""}">${Number(metrics[name][key]).toFixed(key === "R2" ? 4 : 2)}${["MAPE", "RMSPE"].includes(key) ? "%" : ""}</td>`).join("")}
      </tr>
    `).join("");
    const best = modelNames.reduce((winner, name) => Number(metrics[name].RMSPE) < Number(metrics[winner].RMSPE) ? name : winner, modelNames[0]);
    document.querySelector("#best-sentence").innerHTML = `Based on validation RMSPE, <strong>${best.replace("_", " ")}</strong> is the best model for this dataset.`;
    document.querySelector("#compare-output").hidden = false;
  }

  function filteredHistory() {
    const storeFilter = document.querySelector("#history-store-filter").value.trim();
    const modelFilter = document.querySelector("#history-model-filter").value;
    const sort = document.querySelector("#history-sort").value;
    const records = historyRecords.filter((record) => (
      (!storeFilter || String(record.store).includes(storeFilter))
      && (!modelFilter || record.model_used === modelFilter)
    ));
    records.sort((left, right) => {
      if (sort === "sales-desc") return right.predicted_sales - left.predicted_sales;
      if (sort === "sales-asc") return left.predicted_sales - right.predicted_sales;
      if (sort === "store-asc") return left.store - right.store;
      const difference = new Date(left.requested_at) - new Date(right.requested_at);
      return sort === "date-asc" ? difference : -difference;
    });
    return records;
  }

  function renderHistoryChart(records) {
    if (historyChart) historyChart.destroy();
    const chronological = [...records].sort(
      (left, right) => new Date(left.requested_at) - new Date(right.requested_at),
    );
    historyChart = new Chart(document.querySelector("#history-chart"), {
      type: "line",
      data: {
        labels: chronological.map((record) => new Date(record.requested_at).toLocaleDateString()),
        datasets: [{
          label: "Predicted sales",
          data: chronological.map((record) => record.predicted_sales),
          borderColor: "#79a8ff",
          backgroundColor: "rgba(121, 168, 255, .16)",
          pointBackgroundColor: chronological.map((record) => MODEL_COLORS[record.model_used] || "#79a8ff"),
          pointBorderColor: chronological.map((record) => MODEL_COLORS[record.model_used] || "#79a8ff"),
          pointRadius: 4,
          tension: .35,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#9da5c4", maxTicksLimit: 8 }, grid: { display: false } },
          y: { ticks: { color: "#9da5c4", callback: (value) => `€${Number(value).toLocaleString()}` }, grid: { color: "rgba(151, 164, 255, .1)" } },
        },
      },
    });
  }

  function renderHistory() {
    const records = filteredHistory();
    const empty = document.querySelector("#history-empty");
    const table = document.querySelector("#history-table-wrap");
    if (records.length === 0) {
      table.hidden = true;
      empty.hidden = false;
      if (historyChart) {
        historyChart.destroy();
        historyChart = null;
      }
      return;
    }
    table.hidden = false;
    empty.hidden = true;
    document.querySelector("#history-table-body").innerHTML = records.map((record) => `
      <tr>
        <td>${new Date(record.requested_at).toLocaleString()}</td>
        <td>${record.store}</td>
        <td><span class="model-dot" style="background:${MODEL_COLORS[record.model_used] || "#79a8ff"}"></span>${record.model_used.replace("_", " ")}</td>
        <td>€${Number(record.predicted_sales).toLocaleString("en-US", { maximumFractionDigits: 2 })}</td>
      </tr>
    `).join("");
    renderHistoryChart(records);
  }

  function populateHistoryModels() {
    const select = document.querySelector("#history-model-filter");
    const models = [...new Set(historyRecords.map((record) => record.model_used))].sort();
    select.innerHTML = '<option value="">All models</option>' + models.map((model) => (
      `<option value="${model}">${model.replace("_", " ")}</option>`
    )).join("");
  }

  function downloadHistoryCsv() {
    const records = filteredHistory();
    if (records.length === 0) {
      showToast("There is no history to download.");
      return;
    }
    const headers = ["requested_at", "store", "model_used", "predicted_sales"];
    const csv = [
      headers.join(","),
      ...records.map((record) => headers.map((header) => `"${String(record[header]).replaceAll('"', '""')}"`).join(",")),
    ].join("\r\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    link.download = "forecast-history.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  async function loadHistory() {
    try {
      historyRecords = await apiRequest("/history");
      populateHistoryModels();
      renderHistory();
    } catch (error) {
      showToast(error.message);
    }
  }
}

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!authForm.reportValidity()) return;

  const body = {
    email: document.querySelector("#auth-email").value.trim(),
    password: authPassword.value,
  };
  setLoading(true);
  try {
    if (signupMode) {
      await apiRequest("/signup", { method: "POST", body: JSON.stringify(body) });
      signupMode = false;
      updateAuthMode();
      authForm.reset();
      showToast("Account created. You can now sign in.");
      return;
    }

    const result = await apiRequest("/login", { method: "POST", body: JSON.stringify(body) });
    localStorage.setItem("access_token", result.access_token);
    updateShell();
    window.location.hash = "dashboard";
    dashboardLoaded = false;
    showToast("Signed in successfully.");
  } catch (error) {
    showToast(error.message);
  } finally {
    setLoading(false);
  }
});

toggleAuth.addEventListener("click", () => {
  signupMode = !signupMode;
  updateAuthMode();
});

loginTab.addEventListener("click", () => {
  signupMode = false;
  updateAuthMode();
});

signupTab.addEventListener("click", () => {
  signupMode = true;
  updateAuthMode();
});

forecastStore.addEventListener("change", loadStoreDetails);
forecastStore.addEventListener("blur", loadStoreDetails);

forecastForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!forecastForm.reportValidity()) return;
  if (!selectedStore || selectedStore.Store !== Number(forecastStore.value)) {
    await loadStoreDetails();
  }
  if (!selectedStore) return;
  setForecastLoading(true);
  try {
    const result = await apiRequest("/forecast", {
      method: "POST",
      body: JSON.stringify(forecastPayload()),
    });
    forecastValue.textContent = `€${Number(result.predicted_sales).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
    forecastValue.classList.remove("result-pop");
    void forecastValue.offsetWidth;
    forecastValue.classList.add("result-pop");
  } catch (error) {
    showToast(error.message);
  } finally {
    setForecastLoading(false);
  }
});

document.querySelector("#compare-store").addEventListener("change", loadCompareStore);
document.querySelector("#compare-store").addEventListener("blur", loadCompareStore);

document.querySelector("#compare-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form.reportValidity()) return;
  if (!compareStoreDetails || compareStoreDetails.Store !== Number(document.querySelector("#compare-store").value)) {
    await loadCompareStore();
  }
  if (!compareStoreDetails) return;

  const submit = document.querySelector("#compare-submit");
  submit.classList.add("loading");
  submit.setAttribute("aria-busy", "true");
  try {
    const [result, metrics] = await Promise.all([
      apiRequest("/models/compare", { method: "POST", body: JSON.stringify(comparePayload()) }),
      apiRequest("/models/metrics"),
    ]);
    renderComparison(result.predictions, metrics);
  } catch (error) {
    showToast(error.message);
  } finally {
    submit.classList.remove("loading");
    submit.setAttribute("aria-busy", "false");
  }
});

["history-store-filter", "history-model-filter", "history-sort"].forEach((id) => {
  document.querySelector(`#${id}`).addEventListener("input", renderHistory);
  document.querySelector(`#${id}`).addEventListener("change", renderHistory);
});
document.querySelector("#download-history").addEventListener("click", downloadHistoryCsv);

signoutButton.addEventListener("click", () => {
  localStorage.removeItem("access_token");
  dashboardLoaded = false;
  if (maeChart) {
    maeChart.destroy();
    maeChart = null;
  }
  if (predictionChart) {
    predictionChart.destroy();
    predictionChart = null;
  }
  if (historyChart) {
    historyChart.destroy();
    historyChart = null;
  }
  updateShell();
  window.location.hash = "login";
  showToast("Signed out.");
});

themeToggle.addEventListener("click", () => {
  const nextTheme = document.body.classList.contains("light-theme") ? "dark" : "light";
  localStorage.setItem("theme", nextTheme);
  applyTheme(nextTheme);
});

window.addEventListener("hashchange", showView);
applyTheme(localStorage.getItem("theme") || "dark");
updateAuthMode();
updateShell();
showView();
