const panel = document.querySelector("#market-panel");
const select = document.querySelector("#market-symbol");
const refresh = document.querySelector("#market-refresh");
const periodControls = document.querySelector("#market-period");
const external = document.querySelector("#market-external");
const exampleButton = document.querySelector("#market-example");
const recorded = document.querySelector("#market-recorded");
const source = document.querySelector("#market-source");
const output = document.querySelector("#market-display");
const status = document.querySelector("#market-status");
const canvas = document.querySelector("#market-chart");
const price = document.querySelector("#market-price");
const change = document.querySelector("#market-change");
const caption = document.querySelector("#market-time");
const buttons = [...document.querySelectorAll("[data-market-period]")];
const marketZone = "Asia/Kolkata";
const sessionDate = new Intl.DateTimeFormat("en-IN", {
  timeZone: marketZone, day: "numeric", month: "short", year: "numeric",
});
const shortDate = new Intl.DateTimeFormat("en-IN", {timeZone: marketZone, day: "numeric", month: "short"});
const shortTime = new Intl.DateTimeFormat("en-IN", {timeZone: marketZone, hour: "numeric", minute: "2-digit"});
const number = new Intl.NumberFormat("en-IN", {minimumFractionDigits: 2, maximumFractionDigits: 2});
const axis = new Intl.NumberFormat("en-IN", {maximumFractionDigits: 2});

let period = "1D";
let series;
let requestNumber = 0;
let lastStarted = 0;
let mode = "latest";
let enabled = true;

function setStatus(message, stale = false) {
  status.textContent = message;
  status.dataset.stale = String(stale);
}

const emptyStatus = {
  empty: "Upstox returned no candles for the selected period.",
  disabled: "This workspace opens current NSE charts on TradingView. Quotes may be delayed.",
  unavailable: "Upstox prices are currently unavailable.",
};

function updateExternalLink() {
  const instrument = select.selectedOptions[0];
  const symbol = instrument.dataset.tradingviewSymbol || instrument.value;
  external.href = "https://in.tradingview.com/symbols/NSE-" + encodeURIComponent(symbol) + "/";
  external.textContent = "Open " + instrument.textContent + " chart on TradingView";
}

function draw() {
  if (!series || output.hidden) return;
  const {width, height} = canvas.getBoundingClientRect();
  if (width < 80 || height < 60) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  const points = series.candles;
  const values = points.map((point) => point.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = Math.max((max - min) * 0.14, min * 0.0005);
  const low = min - padding;
  const high = max + padding;
  const right = width - 10;
  const top = 10;
  const bottom = height - 24;
  ctx.font = "11px system-ui";
  const left = Math.ceil(ctx.measureText(axis.format(high)).width) + 12;
  const x = (index) => points.length === 1 ? (left + right) / 2 : left + index * (right - left) / (points.length - 1);
  const y = (value) => bottom - (value - low) * (bottom - top) / (high - low);

  ctx.fillStyle = "#506779";
  ctx.strokeStyle = "#dbe5e0";
  ctx.textBaseline = "middle";
  for (let i = 0; i < 4; i += 1) {
    const value = high - i * (high - low) / 3;
    const row = y(value);
    ctx.fillText(axis.format(value), 0, row);
    ctx.beginPath();
    ctx.moveTo(left, row);
    ctx.lineTo(right, row);
    ctx.stroke();
  }

  ctx.textBaseline = "alphabetic";
  const indices = points.length > 2 && width > 380
    ? [0, Math.floor((points.length - 1) / 2), points.length - 1]
    : [...new Set([0, points.length - 1])];
  for (const index of indices) {
    const time = new Date(points[index].time);
    const text = (series.period === "1D" ? shortTime : shortDate).format(time);
    const align = Math.max(left, Math.min(right - ctx.measureText(text).width, x(index) - ctx.measureText(text).width / 2));
    ctx.fillText(text, align, height - 4);
  }

  const stroke = "#006354";
  if (points.length > 1) {
    ctx.beginPath();
    points.forEach((point, index) => ctx.lineTo(x(index), y(point.close)));
    ctx.lineTo(right, bottom);
    ctx.lineTo(left, bottom);
    ctx.closePath();
    const fill = ctx.createLinearGradient(0, top, 0, bottom);
    fill.addColorStop(0, "#d6f0e7");
    fill.addColorStop(1, "#f8fcfa");
    ctx.fillStyle = fill;
    ctx.fill();

    ctx.beginPath();
    points.forEach((point, index) => ctx.lineTo(x(index), y(point.close)));
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.fillStyle = stroke;
  ctx.arc(x(points.length - 1), y(points.at(-1).close), 3.2, 0, Math.PI * 2);
  ctx.fill();
}

function render(data) {
  series = data.candles.length ? data : undefined;
  source.textContent = data.source + (data.period === "1D" ? " · 5-minute candles" : " · Daily candles");
  output.hidden = !series;
  if (!series) {
    setStatus((emptyStatus[data.state] || emptyStatus.unavailable) + " Open the Recorded Infosys case to see a historical chart.", true);
    return;
  }
  const first = series.candles[0];
  const latest = series.candles.at(-1);
  const difference = (latest.close / first.close - 1) * 100;
  price.textContent = data.unit === "INR" ? "₹" + number.format(latest.close) : number.format(latest.close) + " pts";
  change.textContent = series.candles.length > 1
    ? (difference > 0 ? "+" : "") + number.format(difference) + "% over chart range"
    : "One candle";
  change.dataset.trend = difference > 0 ? "up" : difference < 0 ? "down" : "flat";
  const time = new Date(latest.time);
  caption.textContent = (data.state === "recorded" ? "Last recorded close · " : "Latest candle · ") + sessionDate.format(time)
    + (data.period === "1D" ? ", " + shortTime.format(time) : "") + " IST";
  canvas.setAttribute("aria-label", data.name + " closing price. " + caption.textContent + ". " + price.textContent + ". " + change.textContent);
  if (data.state === "recorded") {
    setStatus(enabled
      ? "Recorded Infosys prices from July 2024. Refresh prices to return to the selected instrument."
      : "Recorded Infosys prices from July 2024. Use TradingView for the selected instrument’s current chart.");
  } else {
    const checked = shortTime.format(new Date(data.checked_at)) + " IST";
    setStatus(data.state === "cached"
      ? "Showing cached candles. Last successful fetch " + checked + "."
      : "Checked " + checked + " · Refreshes every minute while this tab is visible.", data.state === "cached");
  }
  draw();
}

async function load(clear = false) {
  const firstCheck = requestNumber === 0;
  clear = clear || mode === "recorded";
  mode = "latest";
  recorded.hidden = true;
  exampleButton.setAttribute("aria-pressed", "false");
  exampleButton.disabled = false;
  buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.marketPeriod === period)));
  const sequence = ++requestNumber;
  lastStarted = Date.now();
  refresh.disabled = true;
  if (clear) {
    series = undefined;
    output.hidden = true;
    source.textContent = "Upstox" + (period === "1D" ? " · 5-minute candles" : " · Daily candles");
    setStatus("Loading candles…");
  }
  try {
    const response = await fetch("/api/market/" + encodeURIComponent(select.value) + "?period=" + period);
    if (sequence !== requestNumber) return;
    if (!response.ok) throw new Error("Cannot fetch market data");
    const data = await response.json();
    if (sequence === requestNumber) {
      enabled = data.state !== "disabled";
      periodControls.hidden = !enabled;
      refresh.hidden = !enabled;
      external.hidden = enabled;
      if (!enabled) updateExternalLink();
      render(data);
      if (firstCheck && !series) await showExample(emptyStatus[data.state] || emptyStatus.unavailable);
    }
  } catch {
    if (sequence === requestNumber) {
      if (firstCheck && !series) await showExample(emptyStatus.unavailable);
      else setStatus(series ? "Refresh failed. Showing the last chart." :
        "Market data is unavailable. Open the Recorded Infosys case for a historical chart.", true);
    }
  } finally {
    if (sequence === requestNumber) refresh.disabled = false;
  }
}

function renderComparison(data) {
  const title = document.createElement("h3");
  title.id = "recorded-heading";
  title.textContent = "Infosys earnings vs Nifty 50";
  const dates = document.createElement("p");
  const day = (value) => sessionDate.format(new Date(value));
  dates.textContent = data.title + " · " + day(data.filing_date) + " · Compared with the " + day(data.baseline_date) + " close.";
  dates.className = "small muted";

  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const label of ["Session", "Infosys", "Nifty 50", "Difference"]) {
    const heading = document.createElement("th");
    heading.scope = "col";
    heading.textContent = label;
    head.append(heading);
  }
  const signed = (value, unit) => (value > 0 ? "+" : "") + number.format(value) + unit;
  const body = table.createTBody();
  for (const item of data.comparisons.filter((entry) => entry.session === 1 || entry.session === 5)) {
    const row = body.insertRow();
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = "Day " + item.session + " · " + day(item.date);
    row.append(name);
    row.insertCell().textContent = signed(item.stock_pct, "%");
    row.insertCell().textContent = signed(item.index_pct, "%");
    row.insertCell().textContent = signed(item.difference_pp, " pp");
  }
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.tabIndex = 0;
  wrap.setAttribute("aria-label", "Recorded market comparison");
  wrap.append(table);
  const note = document.createElement("p");
  note.className = "small muted";
  note.textContent = "The release date is excluded. Other news and market moves may also affect these returns.";
  const links = document.createElement("div");
  links.className = "actions small";
  for (const [label, url] of Object.entries(data.source_links)) {
    const link = document.createElement("a");
    link.textContent = label;
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    links.append(link);
  }
  recorded.replaceChildren(title, dates, wrap, note, links);
  recorded.hidden = false;
}

async function showExample(reason = "") {
  const sequence = ++requestNumber;
  mode = "recorded";
  series = undefined;
  output.hidden = true;
  recorded.hidden = true;
  refresh.disabled = false;
  exampleButton.disabled = true;
  exampleButton.setAttribute("aria-pressed", "true");
  buttons.forEach((button) => button.setAttribute("aria-pressed", "false"));
  source.textContent = "Recorded Infosys example";
  setStatus("Loading the recorded chart…");
  try {
    const response = await fetch("/api/market-example");
    if (!response.ok) throw new Error("Recorded example unavailable");
    const data = await response.json();
    if (sequence !== requestNumber) return;
    render(data);
    renderComparison(data);
    if (reason) setStatus(reason + " Showing the recorded Infosys chart from July 2024.", true);
  } catch {
    if (sequence === requestNumber) setStatus("Could not load the recorded chart. Try again.", true);
  } finally {
    if (sequence === requestNumber) exampleButton.disabled = false;
  }
}

function refreshVisible() {
  if (mode === "latest" && enabled && !document.hidden && !panel.closest("[hidden]") && Date.now() - lastStarted >= 60_000) load();
}

buttons.forEach((button) => button.addEventListener("click", () => {
  if (button.dataset.marketPeriod === period && mode === "latest") return;
  period = button.dataset.marketPeriod;
  buttons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  load(true);
}));
select.addEventListener("change", () => {
  if (enabled) load(true);
  else updateExternalLink();
});
refresh.addEventListener("click", () => load());
exampleButton.addEventListener("click", () => showExample());
new ResizeObserver(draw).observe(canvas);
window.addEventListener("hashchange", refreshVisible);
document.addEventListener("visibilitychange", refreshVisible);
setInterval(refreshVisible, 5_000);
refreshVisible();
