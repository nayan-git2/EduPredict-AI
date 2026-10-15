// EduPredict AI - Student Performance ML Frontend Application
let radarChartInstance = null;
let whatIfChartInstance = null;
let importanceChartInstance = null;
let currentBatchData = [];
let modelPresets = [];
let debounceTimer = null;

const FEATURES = ['attendance', 'study_hours', 'assignment_score', 'internal_marks', 'previous_marks'];

// Document Ready Initialization
document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  setupInputSync();
  await loadModelInfo();
  setupEventListeners();

  // Run initial prediction
  triggerPrediction();

  // Run initial what-if chart
  runWhatIf('study_hours');
});

// Setup two-way data sync between Range Slider and Numeric Input
function setupInputSync() {
  FEATURES.forEach(feat => {
    const range = document.getElementById(`range_${feat}`);
    const num = document.getElementById(`num_${feat}`);
    if (!range || !num) return;

    range.addEventListener('input', (e) => {
      num.value = e.target.value;
      onInputChange();
    });

    num.addEventListener('input', (e) => {
      let val = parseFloat(e.target.value);
      const min = parseFloat(range.min);
      const max = parseFloat(range.max);
      if (isNaN(val)) val = min;
      if (val < min) val = min;
      if (val > max) val = max;
      range.value = val;
      onInputChange();
    });
  });
}

function onInputChange() {
  const autoPredict = document.getElementById('autoPredictToggle');
  if (autoPredict && autoPredict.checked) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      triggerPrediction();
    }, 150);
  }
}

// Fetch Model Info and Presets from Backend API
async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    const data = await res.json();
    if (data.presets) {
      modelPresets = data.presets;
      renderPresets(data.presets);
    }
    if (data.feature_importances) {
      renderFeatureImportanceChart(data.feature_importances);
    }
  } catch (err) {
    console.error('Failed to load model info:', err);
  }
}

// Render Presets
function renderPresets(presets) {
  const container = document.getElementById('presetPillsContainer');
  if (!container) return;

  container.innerHTML = '';
  presets.forEach(p => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'px-3 py-1 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-slate-700/80 text-xs font-semibold text-slate-200 transition-all active:scale-95 flex items-center gap-1.5';
    btn.innerHTML = `<span>${p.name}</span>`;
    btn.onclick = () => applyPreset(p.values);
    container.appendChild(btn);
  });
}

function applyPreset(values) {
  for (const [feat, val] of Object.entries(values)) {
    const range = document.getElementById(`range_${feat}`);
    const num = document.getElementById(`num_${feat}`);
    if (range && num) {
      range.value = val;
      num.value = val;
    }
  }
  triggerPrediction();
}

// Tab Switching
function switchTab(tabId) {
  const tabs = ['single', 'whatif', 'batch', 'model'];
  tabs.forEach(t => {
    const content = document.getElementById(`content-${t}`);
    const btn = document.getElementById(`tab-${t}`);
    if (content) {
      if (t === tabId) {
        content.classList.remove('hidden');
        content.classList.add('block');
      } else {
        content.classList.add('hidden');
        content.classList.remove('block');
      }
    }
    if (btn) {
      if (t === tabId) {
        btn.classList.add('tab-active', 'border-indigo-500', 'text-white');
        btn.classList.remove('border-transparent', 'text-slate-400');
      } else {
        btn.classList.remove('tab-active', 'border-indigo-500', 'text-white');
        btn.classList.add('border-transparent', 'text-slate-400');
      }
    }
  });

  if (tabId === 'whatif' && whatIfChartInstance) {
    whatIfChartInstance.resize();
  }
  if (tabId === 'model' && importanceChartInstance) {
    importanceChartInstance.resize();
  }
}

// Get current input values as a JS object
function getCurrentInputs() {
  const inputs = {};
  FEATURES.forEach(feat => {
    const num = document.getElementById(`num_${feat}`);
    inputs[feat] = num ? parseFloat(num.value) || 0 : 0;
  });
  return inputs;
}

// Execute Individual ML Prediction
async function triggerPrediction() {
  const inputs = getCurrentInputs();
  const btn = document.getElementById('predictBtn');
  if (btn) {
    btn.classList.add('opacity-80');
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs)
    });

    const data = await res.json();
    if (data.success) {
      updatePredictionDashboard(data);
    } else {
      console.error('Prediction failed:', data.error);
    }
  } catch (err) {
    console.error('API Error:', err);
  } finally {
    if (btn) {
      btn.classList.remove('opacity-80');
    }
  }
}

// Update the Result Dashboard
function updatePredictionDashboard(data) {
  const pred = data.prediction;
  const score = pred.score;

  // Animate score number
  const scoreEl = document.getElementById('predictedScoreVal');
  if (scoreEl) {
    scoreEl.textContent = score.toFixed(2);
  }

  // CLI Mirror elements
  const cliMarks = document.getElementById('cliPredictedMarks');
  if (cliMarks) cliMarks.textContent = score.toFixed(2);

  const cliCategory = document.getElementById('cliPerformanceCategory');
  const catText = pred.performance_category || pred.status;
  if (cliCategory) cliCategory.textContent = catText;

  const perfSubtitle = document.getElementById('performanceCategorySubtitle');
  if (perfSubtitle) perfSubtitle.textContent = catText;

  // Circular progress ring
  const circle = document.getElementById('scoreProgressCircle');
  if (circle) {
    const circumference = 314.159;
    const offset = circumference - (score / 100) * circumference;
    circle.style.strokeDashoffset = offset;

    // Color coordination based on grade
    let strokeColor = '#6366f1'; // indigo
    if (score >= 80) strokeColor = '#10b981'; // emerald
    else if (score >= 65) strokeColor = '#6366f1'; // indigo
    else if (score >= 50) strokeColor = '#f59e0b'; // amber
    else strokeColor = '#ef4444'; // rose

    circle.setAttribute('stroke', strokeColor);
  }

  // Badges
  const badgeGrade = document.getElementById('badgeGrade');
  if (badgeGrade) {
    badgeGrade.textContent = `Grade ${pred.grade}`;
    badgeGrade.className = `px-3.5 py-1 text-sm font-extrabold rounded-lg ${
      pred.grade.startsWith('A') ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
      pred.grade === 'B' ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' :
      pred.grade === 'C' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' :
      pred.grade === 'D' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
      'bg-rose-500/20 text-rose-300 border border-rose-500/30'
    }`;
  }

  const badgeStatus = document.getElementById('badgeStatus');
  if (badgeStatus) {
    badgeStatus.textContent = catText;
  }

  // GPA & Variance
  const gpaEl = document.getElementById('predictedGpa');
  if (gpaEl) gpaEl.textContent = `${pred.gpa.toFixed(1)} / 4.0`;

  const stdEl = document.getElementById('predictedStd');
  if (stdEl) stdEl.textContent = `±${pred.uncertainty_std.toFixed(2)}`;

  const ciEl = document.getElementById('confidenceIntervalVal');
  if (ciEl) ciEl.textContent = `[${pred.confidence_interval_95[0].toFixed(1)} - ${pred.confidence_interval_95[1].toFixed(1)}]`;

  // Feedback note
  const feedbackText = document.getElementById('feedbackText');
  if (feedbackText) feedbackText.textContent = pred.feedback;

  // Update Radar Chart
  updateRadarChart(data.inputs);

  // Update Marginal Gains List
  updateMarginalGains(data.marginal_impacts);

  // Update Recommendations
  updateRecommendations(data.recommendations);

  // Update Sample API response in Developer tab
  const apiSample = document.getElementById('apiSampleResponse');
  if (apiSample) {
    apiSample.textContent = JSON.stringify(data, null, 2);
  }
}

// Radar Chart (Student vs High-Performer Benchmark)
function updateRadarChart(inputs) {
  const ctx = document.getElementById('studentRadarChart');
  if (!ctx) return;

  // Normalized values (0 to 100)
  const studentData = [
    inputs.attendance,
    Math.min(100, (inputs.study_hours / 10) * 100),
    inputs.assignment_score,
    inputs.internal_marks,
    inputs.previous_marks
  ];

  const benchmarkData = [95, 80, 92, 90, 88];

  if (!radarChartInstance) {
    radarChartInstance = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Attendance', 'Study (scaled)', 'Assignments', 'Internals', 'Previous'],
        datasets: [
          {
            label: 'This Student',
            data: studentData,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.25)',
            borderWidth: 2,
            pointBackgroundColor: '#6366f1'
          },
          {
            label: 'Honor Benchmark',
            data: benchmarkData,
            borderColor: '#10b981',
            borderDash: [4, 4],
            backgroundColor: 'rgba(16, 185, 129, 0.05)',
            borderWidth: 1.5,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: { display: false, stepSize: 25 },
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
            pointLabels: {
              color: '#94a3b8',
              font: { size: 10, family: 'Plus Jakarta Sans' }
            }
          }
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#cbd5e1',
              font: { size: 10, family: 'Plus Jakarta Sans' },
              boxWidth: 12
            }
          }
        }
      }
    });
  } else {
    radarChartInstance.data.datasets[0].data = studentData;
    radarChartInstance.update();
  }
}

// Marginal Gains List
function updateMarginalGains(impacts) {
  const container = document.getElementById('marginalGainsList');
  if (!container || !impacts) return;

  const labels = {
    'internal_marks': '+10 Internal Marks',
    'assignment_score': '+10 Assignment Score',
    'study_hours': '+1.5 Daily Study Hours',
    'attendance': '+10% Attendance',
    'previous_marks': '+10 Previous Marks'
  };

  container.innerHTML = '';
  // Sort descending by gain
  const sorted = Object.entries(impacts).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([feat, gain]) => {
    const div = document.createElement('div');
    div.className = 'flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800/80';
    div.innerHTML = `
      <div class="flex items-center gap-2">
        <span class="w-1.5 h-1.5 rounded-full ${gain >= 3 ? 'bg-emerald-400' : gain >= 1 ? 'bg-indigo-400' : 'bg-slate-500'}"></span>
        <span class="text-xs text-slate-300 font-medium">${labels[feat] || feat}</span>
      </div>
      <span class="text-xs font-mono font-bold ${gain > 0 ? 'text-emerald-400' : 'text-slate-500'}">
        ${gain > 0 ? `+${gain.toFixed(2)} pts` : '0.00 pts'}
      </span>
    `;
    container.appendChild(div);
  });
}

// AI Recommendations List
function updateRecommendations(recs) {
  const container = document.getElementById('recommendationsList');
  if (!container || !recs) return;

  container.innerHTML = '';

  if (recs.length === 0) {
    container.innerHTML = `<div class="p-4 rounded-xl bg-slate-900/40 text-xs text-slate-400">All student parameters are optimal. Maintain current routines.</div>`;
    return;
  }

  recs.forEach(rec => {
    const isUrgent = rec.priority === 'Urgent' || rec.priority === 'Critical';
    const isPositive = rec.priority === 'Positive';

    const card = document.createElement('div');
    card.className = `p-3.5 rounded-xl border text-xs leading-relaxed transition-all ${
      isUrgent ? 'bg-rose-950/20 border-rose-800/40 text-rose-200' :
      isPositive ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-200' :
      'bg-slate-900/60 border-slate-800 text-slate-300'
    }`;

    card.innerHTML = `
      <div class="flex items-center justify-between mb-1.5">
        <div class="flex items-center gap-2">
          <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
            isUrgent ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' :
            isPositive ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
            'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
          }">
            ${rec.priority}
          </span>
          <span class="font-semibold text-white">${rec.title}</span>
        </div>
        <span class="text-[10px] text-slate-500">${rec.category}</span>
      </div>
      <p class="text-slate-400 text-[11px] mb-2">${rec.description}</p>
      <div class="p-2 rounded-lg bg-slate-950/40 border border-slate-800/50 text-[11px] font-medium text-slate-200 flex items-start gap-1.5">
        <span class="text-indigo-400 font-bold">Action:</span>
        <span>${rec.action}</span>
      </div>
    `;

    container.appendChild(card);
  });
}

// What-If Simulator Execution
async function runWhatIf(featureName) {
  const inputs = getCurrentInputs();
  
  // Highlight active whatif button
  document.querySelectorAll('.whatif-tab').forEach(b => {
    if (b.getAttribute('onclick') && b.getAttribute('onclick').includes(featureName)) {
      b.className = 'whatif-tab px-3.5 py-1.5 rounded-lg bg-indigo-600 text-white text-xs font-semibold transition-all shadow-lg shadow-indigo-600/20';
    } else {
      b.className = 'whatif-tab px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-all';
    }
  });

  try {
    const res = await fetch('/api/what-if', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_inputs: inputs,
        feature: featureName,
        steps: 20
      })
    });

    const data = await res.json();
    if (data.success) {
      renderWhatIfChart(data);
    }
  } catch (err) {
    console.error('What-If error:', err);
  }
}

function renderWhatIfChart(data) {
  const ctx = document.getElementById('whatIfChart');
  if (!ctx) return;

  const labels = data.curve.map(d => d.value);
  const values = data.curve.map(d => d.predicted_score);

  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const delta = maxVal - minVal;

  const minEl = document.getElementById('whatIfMinVal');
  const maxEl = document.getElementById('whatIfMaxVal');
  const deltaEl = document.getElementById('whatIfDeltaVal');
  if (minEl) minEl.textContent = `${minVal.toFixed(1)} / 100`;
  if (maxEl) maxEl.textContent = `${maxVal.toFixed(1)} / 100`;
  if (deltaEl) deltaEl.textContent = `+${delta.toFixed(1)} pts`;

  if (!whatIfChartInstance) {
    whatIfChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: `Predicted Final Score vs ${data.feature_label}`,
          data: values,
          borderColor: '#818cf8',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#6366f1'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 } }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `Score: ${ctx.parsed.y.toFixed(2)}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 10 } }
          },
          y: {
            min: Math.floor(minVal - 5),
            max: Math.ceil(maxVal + 5),
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 10 } }
          }
        }
      }
    });
  } else {
    whatIfChartInstance.data.labels = labels;
    whatIfChartInstance.data.datasets[0].data = values;
    whatIfChartInstance.data.datasets[0].label = `Predicted Final Score vs ${data.feature_label}`;
    whatIfChartInstance.options.scales.y.min = Math.floor(minVal - 5);
    whatIfChartInstance.options.scales.y.max = Math.ceil(maxVal + 5);
    whatIfChartInstance.update();
  }
}

// Feature Importance Chart (scikit-learn Gini / Impurity values)
function renderFeatureImportanceChart(importances) {
  const ctx = document.getElementById('featureImportanceChart');
  if (!ctx) return;

  const labelsMap = {
    'internal_marks': 'Internal Marks',
    'assignment_score': 'Assignment Score',
    'study_hours': 'Daily Study Hours',
    'previous_marks': 'Previous Marks',
    'attendance': 'Attendance Rate'
  };

  // Sort descending
  const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);
  const labels = sorted.map(([k]) => labelsMap[k] || k);
  const data = sorted.map(([, v]) => (v * 100).toFixed(1));

  importanceChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature Weight Percentage (%)',
        data: data,
        backgroundColor: [
          '#6366f1',
          '#8b5cf6',
          '#10b981',
          '#06b6d4',
          '#f59e0b'
        ],
        borderRadius: 8
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Weight: ${ctx.parsed.x}% of Model Decision`
          }
        }
      },
      scales: {
        x: {
          max: 60,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: {
            color: '#94a3b8',
            callback: (v) => `${v}%`,
            font: { family: 'Plus Jakarta Sans', size: 10 }
          }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f1f5f9', font: { family: 'Plus Jakarta Sans', size: 11, weight: 'bold' } }
        }
      }
    }
  });
}

// Setup Event Listeners
function setupEventListeners() {
  const form = document.getElementById('predictionForm');
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      triggerPrediction();
    });
  }

  const resetBtn = document.getElementById('resetInputsBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      applyPreset({
        attendance: 85.0,
        study_hours: 5.0,
        assignment_score: 80.0,
        internal_marks: 75.0,
        previous_marks: 78.0
      });
    });
  }

  const quickSampleBtn = document.getElementById('quickSampleBtn');
  if (quickSampleBtn) {
    let presetIdx = 0;
    quickSampleBtn.addEventListener('click', () => {
      if (modelPresets.length > 0) {
        presetIdx = (presetIdx + 1) % modelPresets.length;
        applyPreset(modelPresets[presetIdx].values);
      }
    });
  }

  // Batch / CSV Upload Listeners
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('csvFileInput');

  if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleCsvFile(e.target.files[0]);
      }
    });

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('border-emerald-500', 'bg-emerald-950/20');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('border-emerald-500', 'bg-emerald-950/20');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-emerald-500', 'bg-emerald-950/20');
      if (e.dataTransfer.files.length > 0) {
        handleCsvFile(e.dataTransfer.files[0]);
      }
    });
  }

  // Load Sample Batch Button
  const loadSampleBatchBtn = document.getElementById('loadSampleBatchBtn');
  if (loadSampleBatchBtn) {
    loadSampleBatchBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/static/sample_students.csv');
        const text = await res.text();
        const blob = new Blob([text], { type: 'text/csv' });
        const file = new File([blob], 'sample_students.csv', { type: 'text/csv' });
        handleCsvFile(file);
      } catch (err) {
        console.error('Failed to load sample CSV:', err);
      }
    });
  }

  // Export CSV Button
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', exportBatchResultsToCsv);
  }

  // Roster Search Filter
  const rosterSearch = document.getElementById('rosterSearch');
  if (rosterSearch) {
    rosterSearch.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      const rows = document.querySelectorAll('#batchTableBody tr');
      rows.forEach(r => {
        const text = r.textContent.toLowerCase();
        r.style.display = text.includes(q) ? '' : 'none';
      });
    });
  }
}

// Upload and Process CSV
async function handleCsvFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (data.success) {
      currentBatchData = data.students;
      displayBatchResults(data);
    } else {
      alert(`Batch processing error: ${data.error}`);
    }
  } catch (err) {
    console.error('Batch upload error:', err);
    alert('Failed to process CSV file.');
  }
}

// Render Batch Analytics and Table
function displayBatchResults(data) {
  const section = document.getElementById('batchResultsSection');
  if (section) section.classList.remove('hidden');

  const s = data.summary;
  document.getElementById('kpiTotalStudents').textContent = s.total_students;
  document.getElementById('kpiAvgScore').textContent = s.average_score.toFixed(1);
  document.getElementById('kpiHighestScore').textContent = s.highest_score.toFixed(1);
  document.getElementById('kpiLowestScore').textContent = s.lowest_score.toFixed(1);
  document.getElementById('kpiPassRate').textContent = `${s.pass_rate}%`;
  document.getElementById('kpiAtRiskCount').textContent = s.at_risk_count;

  const tbody = document.getElementById('batchTableBody');
  if (!tbody) return;

  tbody.innerHTML = '';
  data.students.forEach(st => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-800/40 transition-colors';

    const gradeColor =
      st.grade.startsWith('A') ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' :
      st.grade === 'B' ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30' :
      st.grade === 'C' ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30' :
      st.grade === 'D' ? 'bg-amber-500/20 text-amber-300 border-amber-500/30' :
      'bg-rose-500/20 text-rose-300 border-rose-500/30';

    tr.innerHTML = `
      <td class="py-3 px-4">
        <span class="font-semibold text-white block">${st.name}</span>
        <span class="text-[10px] text-slate-500 font-mono">${st.student_id}</span>
      </td>
      <td class="py-3 px-3 font-mono">${st.attendance.toFixed(1)}%</td>
      <td class="py-3 px-3 font-mono">${st.study_hours.toFixed(1)} hrs</td>
      <td class="py-3 px-3 font-mono">${st.assignment_score.toFixed(1)}</td>
      <td class="py-3 px-3 font-mono">${st.internal_marks.toFixed(1)}</td>
      <td class="py-3 px-3 font-mono">${st.previous_marks.toFixed(1)}</td>
      <td class="py-3 px-4 text-right font-mono font-bold text-sm ${st.predicted_score >= 80 ? 'text-emerald-400' : st.predicted_score >= 60 ? 'text-indigo-300' : 'text-rose-400'}">
        ${st.predicted_score.toFixed(1)}
      </td>
      <td class="py-3 px-3 text-center">
        <span class="px-2 py-0.5 rounded text-[11px] font-extrabold border ${gradeColor}">
          ${st.grade}
        </span>
      </td>
      <td class="py-3 px-3 text-slate-400 text-[11px]">
        ${st.status}
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// Export Results to CSV
function exportBatchResultsToCsv() {
  if (!currentBatchData || currentBatchData.length === 0) {
    alert('No batch records to export.');
    return;
  }

  const headers = ['student_id', 'name', 'attendance', 'study_hours', 'assignment_score', 'internal_marks', 'previous_marks', 'predicted_score', 'grade', 'status', 'gpa'];
  let csvContent = 'data:text/csv;charset=utf-8,' + headers.join(',') + '\n';

  currentBatchData.forEach(row => {
    const line = headers.map(h => `"${row[h] !== undefined ? row[h] : ''}"`).join(',');
    csvContent += line + '\n';
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', 'predicted_student_performance.csv');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Copy cURL sample
function copyCurlCode() {
  const code = `curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "attendance": 85.0,
    "study_hours": 5.0,
    "assignment_score": 80.0,
    "internal_marks": 75.0,
    "previous_marks": 78.0
  }'`;
  navigator.clipboard.writeText(code).then(() => {
    alert('cURL command copied to clipboard!');
  });
}
