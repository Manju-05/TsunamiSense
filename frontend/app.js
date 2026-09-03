/**
 * AI-Driven Tsunami Early Warning System - Frontend Logic
 * Connects UI inputs, Leaflet map, REST API endpoints, and benchmark viewers.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const magSlider = document.getElementById("magnitudeSlider");
  const magInput = document.getElementById("magnitudeInput");
  const magValDisplay = document.getElementById("magValDisplay");

  const depthSlider = document.getElementById("depthSlider");
  const depthInput = document.getElementById("depthInput");
  const depthValDisplay = document.getElementById("depthValDisplay");

  const latInput = document.getElementById("latitudeInput");
  const lonInput = document.getElementById("longitudeInput");
  const seismicForm = document.getElementById("seismicForm");
  const evaluateBtn = document.getElementById("evaluateBtn");
  const presetContainer = document.getElementById("presetButtonsContainer");

  // Output Elements
  const consensusBanner = document.getElementById("consensusBanner");
  const bannerRiskTier = document.getElementById("bannerRiskTier");
  const bannerTitle = document.getElementById("bannerTitle");
  const bannerDesc = document.getElementById("bannerDesc");
  const bannerIcon = document.getElementById("bannerIcon");
  const consensusProb = document.getElementById("consensusProb");

  const rfProbCircle = document.getElementById("rfProbCircle");
  const rfProbText = document.getElementById("rfProbText");
  const rfVerdict = document.getElementById("rfVerdict");

  const svmProbCircle = document.getElementById("svmProbCircle");
  const svmProbText = document.getElementById("svmProbText");
  const svmVerdict = document.getElementById("svmVerdict");

  const lrProbCircle = document.getElementById("lrProbCircle");
  const lrProbText = document.getElementById("lrProbText");
  const lrVerdict = document.getElementById("lrVerdict");

  const benchmarkTableBody = document.getElementById("benchmarkTableBody");
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // 1. Initialize Leaflet Map
  let map, marker, tsunamiZoneCircle;
  const initialLat = parseFloat(latInput.value) || -15.489;
  const initialLon = parseFloat(lonInput.value) || -172.095;

  function initMap() {
    map = L.map("seismicMap", {
      center: [initialLat, initialLon],
      zoom: 3,
      minZoom: 1,
      maxZoom: 12,
    });

    // 100% Free, Public OpenStreetMap / Ocean tile layer (No API Key Required)
    const osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 18,
    });

    const esriOceanLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}", {
      attribution: 'Tiles &copy; Esri, GEBCO, NOAA, National Geographic, DeLorme, HERE, Geonames.org',
      maxZoom: 13,
    });

    // Add default OpenStreetMap layer
    osmLayer.addTo(map);

    // Layer control so user can switch between OpenStreetMap and Ocean Bathymetry
    L.control.layers({
      "OpenStreetMap (Standard)": osmLayer,
      "Esri Ocean Bathymetry": esriOceanLayer
    }).addTo(map);

    // High-visibility glowing epicenter marker with pulsing beacon
    const seismicIcon = L.divIcon({
      className: "seismic-marker-container",
      html: `
        <div class="seismic-beacon"></div>
        <div class="seismic-core"></div>
      `,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });

    marker = L.marker([initialLat, initialLon], {
      icon: seismicIcon,
      draggable: true,
      zIndexOffset: 10000,
    }).addTo(map);

    // Attach permanent high-contrast tooltip
    marker.bindTooltip(`📍 Epicenter: ${initialLat.toFixed(2)}°, ${initialLon.toFixed(2)}°`, {
      permanent: true,
      direction: "top",
      offset: [0, -14],
      className: "epicenter-tooltip",
    });

    tsunamiZoneCircle = L.circle([initialLat, initialLon], {
      color: "#ef4444",
      fillColor: "#ef4444",
      fillOpacity: 0.12,
      radius: 200000,
      weight: 1.5,
    }).addTo(map);

    // Map click event
    map.on("click", (e) => {
      const wrapped = e.latlng.wrap();
      updateCoordinates(wrapped.lat, wrapped.lng, true);
      triggerPrediction();
    });

    // Marker drag event
    marker.on("dragend", (e) => {
      const wrapped = e.target.getLatLng().wrap();
      updateCoordinates(wrapped.lat, wrapped.lng, false);
      triggerPrediction();
    });
  }

  function updateCoordinates(lat, lon, panMap = true) {
    // Ensure bounds within physical coordinate system
    const boundedLat = Math.min(Math.max(parseFloat(lat), -90.0), 90.0);
    let boundedLon = parseFloat(lon);
    // Normalize longitude between -180 and 180
    while (boundedLon > 180) boundedLon -= 360;
    while (boundedLon < -180) boundedLon += 360;

    const roundedLat = parseFloat(boundedLat.toFixed(4));
    const roundedLon = parseFloat(boundedLon.toFixed(4));

    latInput.value = roundedLat;
    lonInput.value = roundedLon;

    // Update marker position and tooltip text
    marker.setLatLng([roundedLat, roundedLon]);
    marker.setTooltipContent(`📍 Epicenter: ${roundedLat}°, ${roundedLon}°`);
    tsunamiZoneCircle.setLatLng([roundedLat, roundedLon]);

    // Smoothly center the map view on the selected point
    if (panMap && map) {
      map.panTo([roundedLat, roundedLon], { animate: true, duration: 0.4 });
    }
  }

  // 2. Input Synchronization
  magSlider.addEventListener("input", (e) => {
    magInput.value = e.target.value;
    magValDisplay.textContent = e.target.value;
  });
  magInput.addEventListener("input", (e) => {
    magSlider.value = e.target.value;
    magValDisplay.textContent = e.target.value;
  });

  depthSlider.addEventListener("input", (e) => {
    depthInput.value = e.target.value;
    depthValDisplay.textContent = `${e.target.value} km`;
  });
  depthInput.addEventListener("input", (e) => {
    depthSlider.value = e.target.value;
    depthValDisplay.textContent = `${e.target.value} km`;
  });

  latInput.addEventListener("change", () => {
    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);
    if (!isNaN(lat) && !isNaN(lon)) {
      updateCoordinates(lat, lon, true);
      triggerPrediction();
    }
  });
  lonInput.addEventListener("change", () => {
    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);
    if (!isNaN(lat) && !isNaN(lon)) {
      updateCoordinates(lat, lon, true);
      triggerPrediction();
    }
  });

  // 3. Historical Case Studies Loading
  async function loadCaseStudies() {
    try {
      const res = await fetch("/api/case-studies");
      if (!res.ok) return;
      const caseStudies = await res.json();

      presetContainer.innerHTML = "";
      caseStudies.forEach((cs, idx) => {
        const btn = document.createElement("button");
        btn.className = `preset-btn ${idx === 0 ? "active" : ""}`;
        btn.textContent = cs.name.split(" ")[0] + " " + cs.name.split(" ")[1] + ` (M${cs.magnitude})`;
        btn.title = cs.description;

        btn.addEventListener("click", () => {
          document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");

          magSlider.value = cs.magnitude;
          magInput.value = cs.magnitude;
          magValDisplay.textContent = cs.magnitude;

          depthSlider.value = cs.depth;
          depthInput.value = cs.depth;
          depthValDisplay.textContent = `${cs.depth} km`;

          updateCoordinates(cs.latitude, cs.longitude);
          triggerPrediction();
        });

        presetContainer.appendChild(btn);
      });
    } catch (e) {
      console.warn("Could not load case studies:", e);
    }
  }

  // 4. Prediction Execution
  async function triggerPrediction() {
    const magnitude = parseFloat(magInput.value);
    const depth = parseFloat(depthInput.value);
    const latitude = parseFloat(latInput.value);
    const longitude = parseFloat(lonInput.value);

    evaluateBtn.disabled = true;
    evaluateBtn.innerHTML = '<span class="btn-icon">⏳</span> Computing Neural Consensus...';

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ magnitude, depth, latitude, longitude }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();
      renderPredictionOutput(data);
    } catch (error) {
      console.error("Error during prediction:", error);
    } finally {
      evaluateBtn.disabled = false;
      evaluateBtn.innerHTML = '<span class="btn-icon">⚡</span> Evaluate Tsunami Risk';
    }
  }

  function renderPredictionOutput(data) {
    const rf = data.models.random_forest;
    const svm = data.models.support_vector_machine;
    const lr = data.models.logistic_regression;

    // Random Forest (Champion)
    const rfPct = (rf.probability * 100).toFixed(1);
    rfProbText.textContent = `${rfPct}%`;
    rfProbCircle.style.setProperty("--prob", `${rfPct}%`);
    setVerdict(rfVerdict, rf.prediction);

    // SVM
    const svmPct = (svm.probability * 100).toFixed(1);
    svmProbText.textContent = `${svmPct}%`;
    svmProbCircle.style.setProperty("--prob", `${svmPct}%`);
    setVerdict(svmVerdict, svm.prediction);

    // Logistic Regression
    const lrPct = (lr.probability * 100).toFixed(1);
    lrProbText.textContent = `${lrPct}%`;
    lrProbCircle.style.setProperty("--prob", `${lrPct}%`);
    setVerdict(lrVerdict, lr.prediction);

    // Update Consensus Alert Banner
    const maxProb = Math.max(rf.probability, svm.probability, lr.probability);
    consensusProb.textContent = `${(maxProb * 100).toFixed(1)}%`;

    consensusBanner.className = "consensus-banner";
    if (data.risk_level === "CRITICAL") {
      consensusBanner.classList.add("alert-critical");
      bannerRiskTier.textContent = "CRITICAL TSUNAMI THREAT";
      bannerTitle.textContent = "HIGH TSUNAMI RISK - WARNING ISSUED";
      bannerDesc.textContent = "High-magnitude shallow event in subduction perimeter. Immediate coastal evacuation recommended.";
      bannerIcon.textContent = "🌊";
      tsunamiZoneCircle.setStyle({ color: "#ef4444", fillColor: "#ef4444" });
    } else if (data.risk_level === "ELEVATED") {
      consensusBanner.classList.add("alert-elevated");
      bannerRiskTier.textContent = "ELEVATED SEISMIC THREAT";
      bannerTitle.textContent = "MODERATE TSUNAMI RISK - ADVISORY";
      bannerDesc.textContent = "Moderate water displacement risk detected. Seismological monitoring advised.";
      bannerIcon.textContent = "⚠️";
      tsunamiZoneCircle.setStyle({ color: "#f59e0b", fillColor: "#f59e0b" });
    } else {
      consensusBanner.classList.add("alert-normal");
      bannerRiskTier.textContent = "NORMAL SEISMIC STATUS";
      bannerTitle.textContent = "LOW TSUNAMI RISK - NO WARNING";
      bannerDesc.textContent = "Minimal vertical seabed displacement probability. No significant tsunami waves expected.";
      bannerIcon.textContent = "🛡️";
      tsunamiZoneCircle.setStyle({ color: "#10b981", fillColor: "#10b981" });
    }
  }

  function setVerdict(element, pred) {
    if (pred === 1) {
      element.className = "verdict-badge verdict-tsunami";
      element.textContent = "TSUNAMI (1)";
    } else {
      element.className = "verdict-badge verdict-safe";
      element.textContent = "NO TSUNAMI (0)";
    }
  }

  // 5. Load Metrics Table
  async function loadMetricsTable() {
    try {
      const res = await fetch("/api/metrics");
      if (!res.ok) return;
      const data = await res.json();

      benchmarkTableBody.innerHTML = "";
      for (const [modelName, m] of Object.entries(data)) {
        const row = document.createElement("tr");
        const isBest = modelName.includes("Random Forest");
        row.innerHTML = `
          <td>${m.model_name || modelName} ${isBest ? '<span class="champion-badge" style="position:static;margin-left:8px;">★ Best</span>' : ""}</td>
          <td>${(m.accuracy * 100).toFixed(1)}%</td>
          <td>${(m.precision * 100).toFixed(1)}%</td>
          <td>${(m.recall * 100).toFixed(1)}%</td>
          <td>${m.f1_score.toFixed(3)}</td>
          <td>${m.f2_score.toFixed(3)}</td>
          <td>${m.auc_roc.toFixed(3)}</td>
          <td>${m.pr_auc.toFixed(3)}</td>
          <td>${m.mcc.toFixed(3)}</td>
        `;
        benchmarkTableBody.appendChild(row);
      }
    } catch (e) {
      console.warn("Could not load metrics summary:", e);
    }
  }

  // 6. Tabs Handling
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Form Submit
  seismicForm.addEventListener("submit", (e) => {
    e.preventDefault();
    triggerPrediction();
  });

  // Initialize
  initMap();
  loadCaseStudies();
  loadMetricsTable();
  triggerPrediction();
});
