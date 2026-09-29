// Face Attendance Client Logic

document.addEventListener("DOMContentLoaded", () => {
  initLiveFeed();
  loadStats();
  setInterval(loadStats, 3000); // Poll fresh attendance every 3 seconds
});

async function loadStats() {
  try {
    const res = await fetch("/api/attendance/today");
    if (!res.ok) return;
    const data = await res.json();

    const presentEl = document.getElementById("stat-present");
    const activeEl = document.getElementById("stat-active");
    const absentEl = document.getElementById("stat-absent");
    const logList = document.getElementById("kiosk-log-list");

    if (presentEl) presentEl.innerText = data.present_count;
    if (activeEl) activeEl.innerText = data.total_active_employees;
    if (absentEl) absentEl.innerText = data.absent_count;

    if (logList && data.logs) {
      renderRecentLogs(data.logs, logList);
    }
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

function renderRecentLogs(logs, container) {
  if (logs.length === 0) {
    container.innerHTML = `<div style="text-align:center; padding: 2rem; color: var(--text-muted); font-size: 0.9rem;">No attendance logged today yet.</div>`;
    return;
  }

  // Render top 15 logs
  container.innerHTML = logs.slice(0, 15).map(log => {
    const isCheckIn = log.event_type === "CHECK_IN";
    const tagClass = isCheckIn ? "tag-checkin" : "tag-checkout";
    const eventLabel = isCheckIn ? "Check-In" : "Check-Out";
    const timeStr = log.timestamp ? log.timestamp.split(" ")[1] : "";

    return `
      <div class="log-item">
        <div>
          <div class="log-emp-name">${escapeHtml(log.employee_name)}</div>
          <div class="log-emp-dept">${escapeHtml(log.department)} &bull; ID: ${escapeHtml(log.employee_id)}</div>
        </div>
        <div style="text-align: right;">
          <span class="log-tag ${tagClass}">${eventLabel}</span>
          <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">${timeStr}</div>
        </div>
      </div>
    `;
  }).join("");
}

function escapeHtml(text) {
  if (!text) return "";
  return text.toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function initLiveFeed() {
  const feed = document.getElementById("camera-stream");
  if (feed) {
    feed.src = "/api/video/feed";
  }
}

/* Enrollment Modal & Camera Capture */
let webCamStream = null;

function openEnrollModal() {
  const modal = document.getElementById("enroll-modal");
  if (modal) modal.classList.add("active");
}

function closeEnrollModal() {
  const modal = document.getElementById("enroll-modal");
  if (modal) modal.classList.remove("active");
  stopEnrollWebcam();
}

async function startEnrollWebcam() {
  const video = document.getElementById("enroll-video");
  const canvas = document.getElementById("enroll-canvas");
  const captureBtn = document.getElementById("btn-capture-snapshot");

  try {
    webCamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = webCamStream;
    video.style.display = "block";
    canvas.style.display = "none";
    if (captureBtn) captureBtn.style.display = "inline-flex";
  } catch (err) {
    alert("Could not access browser webcam: " + err.message);
  }
}

function stopEnrollWebcam() {
  if (webCamStream) {
    webCamStream.getTracks().forEach(track => track.stop());
    webCamStream = null;
  }
}

let capturedBlob = null;

function captureSnapshot() {
  const video = document.getElementById("enroll-video");
  const canvas = document.getElementById("enroll-canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  video.style.display = "none";
  canvas.style.display = "block";
  stopEnrollWebcam();

  canvas.toBlob((blob) => {
    capturedBlob = blob;
  }, "image/jpeg", 0.95);
}

async function submitEnrollment(event) {
  event.preventDefault();
  const empId = document.getElementById("enroll-id").value.trim();
  const name = document.getElementById("enroll-name").value.trim();
  const dept = document.getElementById("enroll-dept").value.trim();
  const fileInput = document.getElementById("enroll-file");
  const statusDiv = document.getElementById("enroll-status");

  if (!empId || !name) {
    alert("Please fill in both Employee ID and Full Name.");
    return;
  }

  let imageBlob = capturedBlob;
  if (!imageBlob && fileInput && fileInput.files.length > 0) {
    imageBlob = fileInput.files[0];
  }

  if (!imageBlob) {
    alert("Please capture a photo with the webcam or select an image file.");
    return;
  }

  const formData = new FormData();
  formData.append("employee_id", empId);
  formData.append("name", name);
  formData.append("department", dept || "General");
  formData.append("image", imageBlob, "enrollment.jpg");

  statusDiv.innerHTML = `<span style="color: var(--accent-cyan);">Extracting face embedding & enrolling...</span>`;

  try {
    const response = await fetch("/api/employees/enroll", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      statusDiv.innerHTML = `<span style="color: var(--accent-red);">${escapeHtml(data.detail || "Enrollment failed.")}</span>`;
      return;
    }

    statusDiv.innerHTML = `<span style="color: var(--accent-emerald);">Success! Enrolled ${escapeHtml(data.employee.name)}.</span>`;
    setTimeout(() => {
      closeEnrollModal();
      loadStats();
      if (typeof loadEmployeeTable === "function") {
        loadEmployeeTable();
      }
    }, 1500);

  } catch (err) {
    statusDiv.innerHTML = `<span style="color: var(--accent-red);">Network Error: ${err.message}</span>`;
  }
}
