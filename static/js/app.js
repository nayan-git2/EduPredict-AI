// EduPredict AI - Student Performance ML Frontend Application
// Supports Dual Mode: Cloud Python API + In-Browser Random Forest Client Engine (for Netlify/Vercel/Offline)

let radarChartInstance = null;
let whatIfChartInstance = null;
let importanceChartInstance = null;
let currentBatchData = [];
let modelPresets = [
  {
    name: '🌟 High Achiever',
    description: 'Top 5% student with consistent high study hours & internal scores',
    values: { attendance: 95.0, study_hours: 7.2, assignment_score: 94.0, internal_marks: 92.0, previous_marks: 90.0 }
  },
  {
    name: '⚖️ Average Student',
    description: 'Typical undergraduate student performing in middle 50th percentile',
    values: { attendance: 80.0, study_hours: 4.5, assignment_score: 72.0, internal_marks: 68.0, previous_marks: 70.0 }
  },
  {
    name: '⚠️ At-Risk Student',
    description: 'Low attendance and internal marks indicating potential course failure',
    values: { attendance: 62.0, study_hours: 1.8, assignment_score: 45.0, internal_marks: 42.0, previous_marks: 48.0 }
  },
  {
    name: '🚀 Hard Worker Improver',
    description: 'High study hours and attendance working to recover from lower past grades',
    values: { attendance: 90.0, study_hours: 7.5, assignment_score: 78.0, internal_marks: 65.0, previous_marks: 52.0 }
  }
];

let localModelData = null;
let debounceTimer = null;

const FEATURES = ['attendance', 'study_hours', 'assignment_score', 'internal_marks', 'previous_marks'];

// Document Ready Initialization
document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  setupInputSync();
  renderPresets(modelPresets);

  // Attempt to load client-side Random Forest model (for Netlify / static deployment)
  await initClientModel();

  // Also check backend API info
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
    }, 120);
  }
}

// Load Client-side Random Forest Model (Netlify compatibility)
async function initClientModel() {
  try {
    const res = await fetch('./static/model.json');
    if (res.ok) {
      localModelData = await res.json();
      console.log('✅ Client-side Random Forest model loaded:', localModelData.n_estimators, 'trees');
      if (localModelData.feature_importances) {
        renderFeatureImportanceChart(localModelData.feature_importances);
      }
    }
  } catch (err) {
    console.warn('Client-side model.json not loaded yet:', err);
  }
}

// Fetch Backend Model Info if available
async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    if (res.ok) {
      const data = await res.json();
      if (data.presets && data.presets.length > 0) {
        modelPresets = data.presets;
        renderPresets(data.presets);
      }
      if (data.feature_importances) {
        renderFeatureImportanceChart(data.feature_importances);
      }
    }
  } catch (err) {
    // Expected on static hosting like Netlify
    console.log('Running on static host (Netlify), using client-side Random Forest engine.');
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

// In-Browser Random Forest Regressor Evaluator (100 Decision Trees)
function predictLocalRandomForest(inputs) {
  if (!localModelData) return null;

  const featureVector = [
    parseFloat(inputs.attendance),
    parseFloat(inputs.study_hours),
    parseFloat(inputs.assignment_score),
    parseFloat(inputs.internal_marks),
    parseFloat(inputs.previous_marks)
  ];

  const treePredictions = [];
  for (let i = 0; i < localModelData.trees.length; i++) {
    const t = localModelData.trees[i];
    let node = 0;
    while (t.feature[node] !== -2) {
      const featIdx = t.feature[node];
      if (featureVector[featIdx] <= t.threshold[node]) {
        node = t.children_left[node];
      } else {
        node = t.children_right[node];
      }
    }
    treePredictions.push(t.value[node]);
  }

  // Mean prediction
  const sum = treePredictions.reduce((a, b) => a + b, 0);
  const rawScore = sum / treePredictions.length;
  const score = Math.max(0, Math.min(100, rawScore));

  // Uncertainty std dev across 100 trees
  const variance = treePredictions.reduce((acc, v) => acc + Math.pow(v - rawScore, 2), 0) / treePredictions.length;
  const stdDev = Math.sqrt(variance);
  const ciLower = Math.max(0, score - 1.96 * stdDev);
  const ciUpper = Math.min(100, score + 1.96 * stdDev);

  // Marginal impacts (+10 points / +1.5 hrs)
  const marginal_impacts = {};
  FEATURES.forEach(feat => {
    const modInputs = { ...inputs };
    if (feat === 'study_hours') {
      modInputs[feat] = Math.min(16, parseFloat(modInputs[feat]) + 1.5);
    } else {
      modInputs[feat] = Math.min(100, parseFloat(modInputs[feat]) + 10);
    }
    const modVec = [
      modInputs.attendance,
      modInputs.study_hours,
      modInputs.assignment_score,
      modInputs.internal_marks,
      modInputs.previous_marks
    ];

    let mSum = 0;
    for (let i = 0; i < localModelData.trees.length; i++) {
      const t = localModelData.trees[i];
      let node = 0;
      while (t.feature[node] !== -2) {
        const featIdx = t.feature[node];
        if (modVec[featIdx] <= t.threshold[node]) {
          node = t.children_left[node];
        } else {
          node = t.children_right[node];
        }
      }
      mSum += t.value[node];
    }
    const mPred = mSum / localModelData.trees.length;
    marginal_impacts[feat] = Math.max(0, parseFloat((mPred - score).toFixed(2)));
  });

  // Grade & Performance Category
  let grade = 'F';
  let category = 'Critical Risk (Academic Warning)';
  let gpa = 0.0;
  let badge_color = 'danger';
  let feedback = 'High risk of course failure. Immediate academic mentoring and intervention required.';

  if (score >= 90) {
    grade = 'A+';
    category = 'High Performance (Distinction)';
    gpa = 4.0;
    badge_color = 'success';
    feedback = 'Outstanding academic mastery with exceptional consistency across all metrics.';
  } else if (score >= 80) {
    grade = 'A';
    category = 'High Performance (Distinction)';
    gpa = 3.7;
    badge_color = 'success';
    feedback = 'Strong comprehension and performance; on track for honors.';
  } else if (score >= 70) {
    grade = 'B';
    category = 'Good Performance (Above Average)';
    gpa = 3.0;
    badge_color = 'info';
    feedback = 'Good overall standing with steady performance across core assessments.';
  } else if (score >= 60) {
    grade = 'C';
    category = 'Average Performance (Moderate)';
    gpa = 2.2;
    badge_color = 'warning';
    feedback = 'Passing performance; targeted improvement in weaker areas will secure better grades.';
  } else if (score >= 50) {
    grade = 'D';
    category = 'Low Performance (Needs Improvement)';
    gpa = 1.5;
    badge_color = 'warning';
    feedback = 'Borderline standing with noticeable vulnerabilities in key evaluated subjects.';
  }

  const recommendations = generateLocalRecommendations(inputs, score);

  return {
    success: true,
    prediction: {
      score: parseFloat(score.toFixed(2)),
      predicted_marks: parseFloat(score.toFixed(2)),
      category: category,
      performance_category: category,
      grade: grade,
      status: category,
      gpa: gpa,
      badge_color: badge_color,
      feedback: feedback,
      uncertainty_std: parseFloat(stdDev.toFixed(2)),
      confidence_interval_95: [parseFloat(ciLower.toFixed(2)), parseFloat(ciUpper.toFixed(2))],
      tree_min: parseFloat(Math.min(...treePredictions).toFixed(2)),
      tree_max: parseFloat(Math.max(...treePredictions).toFixed(2))
    },
    inputs: inputs,
    marginal_impacts: marginal_impacts,
    feature_importances: localModelData.feature_importances,
    recommendations: recommendations
  };
}

// Generate Actionable Academic Recommendations
function generateLocalRecommendations(inputs, score) {
  const recs = [];
  const att = parseFloat(inputs.attendance || 0);
  const study = parseFloat(inputs.study_hours || 0);
  const assign = parseFloat(inputs.assignment_score || 0);
  const internal = parseFloat(inputs.internal_marks || 0);
  const prev = parseFloat(inputs.previous_marks || 0);

  if (internal < 70) {
    recs.append?.({} ) || recs.push({
      priority: internal < 55 ? 'Critical' : 'High',
      category: 'Internal Assessments',
      title: 'Boost Internal Marks (44.9% Model Weight)',
      description: `Internal marks carry the highest weight in the model. Improving from ${internal.toFixed(1)} to 80+ can raise your final score by 6–12 marks.`,
      action: 'Prioritize mid-term preparation, lab evaluations, and continuous unit quizzes.'
    });
  } else if (internal >= 85) {
    recs.push({
      priority: 'Positive',
      category: 'Internal Assessments',
      title: 'Outstanding Internal Standing',
      description: `Your internal marks (${internal.toFixed(1)}) strongly anchor your high performance prediction.`,
      action: 'Maintain your current study rhythm leading into final exams.'
    });
  }

  if (assign < 70) {
    recs.push({
      priority: 'High',
      category: 'Assignments & Coursework',
      title: 'Maximize Assignment Submissions',
      description: `Assignments constitute 28.7% of the model importance. Current score is ${assign.toFixed(1)}.`,
      action: 'Submit all problem sets on time, seek TA feedback before deadlines, and review rubrics carefully.'
    });
  }

  if (study < 3.5) {
    recs.push({
      priority: 'Medium',
      category: 'Study Habits',
      title: 'Increase Daily Study Dedication',
      description: `Current study duration is ${study.toFixed(1)} hrs/day. Average high performers average 5.5 to 7.0 hrs/day.`,
      action: 'Add 1.5 to 2 structured hours using Pomodoro intervals (50m study / 10m break).'
    });
  } else if (study >= 7) {
    recs.push({
      priority: 'Positive',
      category: 'Study Habits',
      title: 'Commendable Study Discipline',
      description: `${study.toFixed(1)} hours/day provides deep retention. Ensure quality rest and spaced repetition.`,
      action: 'Balance intense study sessions with adequate sleep and revision.'
    });
  }

  if (att < 75) {
    recs.push({
      priority: 'Urgent',
      category: 'Attendance Warning',
      title: 'Attendance Below Mandatory 75%',
      description: `Current attendance is ${att.toFixed(1)}%. Falling below 75% incurs institutional penalties and risk of exam debarment.`,
      action: 'Ensure 100% presence in remaining sessions to bring aggregate attendance above 75% immediately.'
    });
  } else {
    recs.push({
      priority: 'Low',
      category: 'Attendance',
      title: 'Healthy Attendance Record',
      description: `Attendance at ${att.toFixed(1)}% satisfies all institutional compliance requirements.`,
      action: 'Continue regular lecture engagement and active classroom participation.'
    });
  }

  return recs;
}

// Execute ML Prediction (Dual Mode)
async function triggerPrediction() {
  const inputs = getCurrentInputs();
  const btn = document.getElementById('predictBtn');
  if (btn) btn.classList.add('opacity-80');

  // Try backend API first (when running Flask locally or on Render)
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs)
    });

    if (res.ok) {
      const data = await res.json();
      if (data.success) {
        updatePredictionDashboard(data);
        if (btn) btn.classList.remove('opacity-80');
        return;
      }
    }
  } catch (err) {
    // API failed or running on static host like Netlify
  }

  // Fallback to client-side Random Forest evaluator
  const localRes = predictLocalRandomForest(inputs);
  if (localRes) {
    updatePredictionDashboard(localRes);
  }

  if (btn) btn.classList.remove('opacity-80');
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

    let strokeColor = '#6366f1';
    if (score >= 80) strokeColor = '#10b981';
    else if (score >= 65) strokeColor = '#6366f1';
    else if (score >= 50) strokeColor = '#f59e0b';
    else strokeColor = '#ef4444';

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

// Radar Chart
function updateRadarChart(inputs) {
  const ctx = document.getElementById('studentRadarChart');
  if (!ctx) return;

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
            pointLabels: { color: '#94a3b8', font: { size: 10, family: 'Plus Jakarta Sans' } }
          }
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#cbd5e1', font: { size: 10, family: 'Plus Jakarta Sans' }, boxWidth: 12 }
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

// Recommendations List
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

// What-If Simulator Execution (Dual Mode)
async function runWhatIf(featureName) {
  const inputs = getCurrentInputs();

  document.querySelectorAll('.whatif-tab').forEach(b => {
    if (b.getAttribute('onclick') && b.getAttribute('onclick').includes(featureName)) {
      b.className = 'whatif-tab px-3.5 py-1.5 rounded-lg bg-indigo-600 text-white text-xs font-semibold transition-all shadow-lg shadow-indigo-600/20';
    } else {
      b.className = 'whatif-tab px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-all';
    }
  });

  // Try backend API first
  try {
    const res = await fetch('/api/what-if', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_inputs: inputs, feature: featureName, steps: 20 })
    });
    if (res.ok) {
      const data = await res.json();
      if (data.success) {
        renderWhatIfChart(data);
        return;
      }
    }
  } catch (err) {}

  // Fallback to client-side sensitivity calculation
  if (!localModelData) return;
  const ranges = {
    'study_hours': { min: 0.5, max: 14.0, label: 'Daily Study Hours' },
    'internal_marks': { min: 30.0, max: 100.0, label: 'Internal Marks' },
    'assignment_score': { min: 30.0, max: 100.0, label: 'Assignment Score' },
    'attendance': { min: 50.0, max: 100.0, label: 'Attendance Rate' },
    'previous_marks': { min: 30.0, max: 100.0, label: 'Previous Exam Marks' }
  };

  const r = ranges[featureName] || { min: 0, max: 100, label: featureName };
  const curve = [];
  const steps = 18;
  for (let i = 0; i <= steps; i++) {
    const v = r.min + (i / steps) * (r.max - r.min);
    const modInputs = { ...inputs, [featureName]: v };
    const p = predictLocalRandomForest(modInputs);
    if (p) {
      curve.push({ value: parseFloat(v.toFixed(1)), predicted_score: p.prediction.score });
    }
  }

  renderWhatIfChart({
    feature: featureName,
    feature_label: r.label,
    curve: curve
  });
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
          legend: { labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 } } },
          tooltip: {
            callbacks: { label: (ctx) => `Score: ${ctx.parsed.y.toFixed(2)}` }
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

// Feature Importance Chart
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

  const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);
  const labels = sorted.map(([k]) => labelsMap[k] || k);
  const data = sorted.map(([, v]) => (v * 100).toFixed(1));

  if (importanceChartInstance) {
    importanceChartInstance.destroy();
  }

  importanceChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature Weight Percentage (%)',
        data: data,
        backgroundColor: ['#6366f1', '#8b5cf6', '#10b981', '#06b6d4', '#f59e0b'],
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
          callbacks: { label: (ctx) => `Weight: ${ctx.parsed.x}% of Model Decision` }
        }
      },
      scales: {
        x: {
          max: 60,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', callback: (v) => `${v}%`, font: { family: 'Plus Jakarta Sans', size: 10 } }
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

  // Batch CSV Upload Listeners
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
        const res = await fetch('./static/sample_students.csv');
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

// Upload and Process CSV (Dual Mode)
async function handleCsvFile(file) {
  // First try backend API
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/batch-predict', { method: 'POST', body: formData });
    if (res.ok) {
      const data = await res.json();
      if (data.success) {
        currentBatchData = data.students;
        displayBatchResults(data);
        return;
      }
    }
  } catch (err) {}

  // Fallback: Parse CSV in browser using client-side Random Forest
  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (lines.length <= 1) {
      alert('CSV file is empty or missing headers.');
      return;
    }

    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    const students = [];
    const scores = [];
    let atRisk = 0;

    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',').map(p => p.trim());
      if (parts.length < 5) continue;

      const row = {};
      headers.forEach((h, idx) => {
        row[h] = parts[idx];
      });

      const inputs = {
        attendance: parseFloat(row.attendance || 75),
        study_hours: parseFloat(row.study_hours || 4),
        assignment_score: parseFloat(row.assignment_score || 70),
        internal_marks: parseFloat(row.internal_marks || 65),
        previous_marks: parseFloat(row.previous_marks || 65)
      };

      const localResult = predictLocalRandomForest(inputs);
      if (localResult) {
        const score = localResult.prediction.score;
        scores.push(score);
        if (score < 60) atRisk++;

        students.push({
          student_id: row.student_id || `STU-${1000 + i}`,
          name: row.name || `Student ${i}`,
          attendance: inputs.attendance,
          study_hours: inputs.study_hours,
          assignment_score: inputs.assignment_score,
          internal_marks: inputs.internal_marks,
          previous_marks: inputs.previous_marks,
          predicted_score: score,
          grade: localResult.prediction.grade,
          status: localResult.prediction.status,
          gpa: localResult.prediction.gpa,
          badge_color: localResult.prediction.badge_color
        });
      }
    }

    if (students.length === 0) {
      alert('Could not parse any valid student records from the CSV.');
      return;
    }

    const sum = scores.reduce((a, b) => a + b, 0);
    const avg = sum / scores.length;
    const max = Math.max(...scores);
    const min = Math.min(...scores);
    const passCount = scores.filter(s => s >= 50).length;

    const batchData = {
      success: true,
      summary: {
        total_students: students.length,
        average_score: avg,
        highest_score: max,
        lowest_score: min,
        pass_rate: parseFloat(((passCount / students.length) * 100).toFixed(1)),
        at_risk_count: atRisk
      },
      students: students
    };

    currentBatchData = students;
    displayBatchResults(batchData);
  };
  reader.readAsText(file);
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
        ${st.predicted_score.toFixed(2)}
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
