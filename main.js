// No import needed for static HTML/JS/CSS

// Interval ID for the prediction polling loop
let pollInterval = null;

/**
 * Opens the translator modal.
 * Restarts the video feed (in case it stalled) and begins polling /prediction.
 */
function openDemo() {
  document.getElementById("modal").classList.add("open");

  // Set the video src — the browser will connect to the MJPEG stream.
  // Adding a timestamp prevents the browser from using a cached URL.
  const img = document.getElementById("videoFeed");

  // Poll the /prediction endpoint every 200ms to update the UI
  pollInterval = setInterval(fetchPrediction, 200);
}

/**
 * Closes the modal and stops the prediction polling loop.
 */
function closeDemo() {
  document.getElementById("modal").classList.remove("open");
  clearInterval(pollInterval);
  pollInterval = null;
}

/**
 * Closes the modal if the user clicks the dark overlay behind it
 * (but not if they click inside the modal itself).
 */
function handleOverlayClick(e) {
  if (e.target === document.getElementById("modal")) {
    closeDemo();
  }
}

/**
 * Fetches the latest prediction from Flask and updates the UI.
 * Called every 200ms while the modal is open.
 * Silently ignores errors (e.g. if the server isn't ready yet).
 */
async function fetchPrediction() {
  try {
    const API = "https://sac-hacks-vii.onrender.com"
    const res  = await fetch("${API}/prediction");
    const data = await res.json();
    updateUI(data);
  } catch (e) {
    // Server not ready or network issue — skip this tick silently
  }
}

/**
 * Updates all UI elements with the latest prediction data.
 *
 * @param {Object} data - Response from /prediction:
 *   { preview, conf, sentence, stable }
 */
function updateUI(data) {
  const letterEl    = document.getElementById("predLetter");
  const confEl      = document.getElementById("predConf");
  const fillEl      = document.getElementById("stabilityFill");
  const stableNumEl = document.getElementById("stableNum");
  const sentenceEl  = document.getElementById("sentenceText");

  // Update predicted letter and confidence text
  if (data.preview) {
    letterEl.textContent = data.preview;
    letterEl.style.color = "var(--sage-d)";
    confEl.textContent   = `${data.conf}% confidence`;
  } else {
    letterEl.textContent = "—";
    letterEl.style.color = "var(--sand)";
    confEl.textContent   = "waiting for hand...";
  }

  // Update stability progress bar width (0-100%)
  fillEl.style.width      = data.stable + "%";
  stableNumEl.textContent = data.stable + "%";

  // Update sentence display
  if (data.sentence) {
    sentenceEl.textContent = data.sentence;
    sentenceEl.classList.remove("empty");
  } else {
    sentenceEl.textContent = "Start signing...";
    sentenceEl.classList.add("empty");
  }
}

/**
 * Sends a command to a Flask POST endpoint.
 * Used by Space, Backspace, and Clear buttons.
 *
 * @param {string} cmd - "space", "backspace", or "clear"
 */
async function sendCmd(cmd) {
  await fetch("/" + cmd, { method: "POST" });
}

/**
 * Keyboard shortcuts so users don't have to click the buttons.
 * Only active while the modal is open.
 */
document.addEventListener("keydown", (e) => {
  // Do nothing if modal isn't open
  if (!document.getElementById("modal").classList.contains("open")) return;

  if (e.key === "Escape")     closeDemo();
  if (e.key === " ")          { e.preventDefault(); sendCmd("space"); }
  if (e.key === "Backspace")  sendCmd("backspace");
  if (e.key === "c")          sendCmd("clear");
});