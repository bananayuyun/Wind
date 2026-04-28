/**
 * 算法对比页 JS
 * 调用 /api/compare，渲染 4 个指标柱状图 + 1 个时序折线图 + 汇总表格
 */

const ALGO_COLORS = ['#8e8e93', '#ff9f0a', '#0071e3'];  // 灰 / 橙 / 蓝

let charts = {};
let allModels = [];

// ── 初始化：加载模型列表 ───────────────────────────────────────────────────

async function initPage() {
  try {
    const data = await fetchJSON('/api/models');
    allModels = data.models || [];
    populateModelSelect();
  } catch(e) {}

  document.getElementById('gridSize').addEventListener('change', () => {
    populateModelSelect();
    updateModelHint();
  });
  document.getElementById('modelSelect').addEventListener('change', updateModelHint);
}

function populateModelSelect() {
  const sel = document.getElementById('modelSelect');
  const gridSize = parseInt(document.getElementById('gridSize').value);
  const currentVal = sel.value;

  sel.innerHTML = '<option value="">— 不使用（退化为XY）—</option>';

  allModels.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = `${m.model_type.toUpperCase()} ${m.grid_size}×${m.grid_size} — acc ${(m.accuracy*100).toFixed(1)}%`;
    if (m.grid_size !== gridSize) {
      opt.disabled = true;
      opt.textContent += ' ⚠ 不兼容';
    }
    sel.appendChild(opt);
  });

  sel.value = currentVal;
  if (!sel.value) sel.selectedIndex = 0;

  updateModelHint();
}

function updateModelHint() {
  const sel = document.getElementById('modelSelect');
  const gridSize = parseInt(document.getElementById('gridSize').value);
  const noModelHint = document.getElementById('noModelHint');
  const modelMismatchHint = document.getElementById('modelMismatchHint');

  const selectedOpt = sel.options[sel.selectedIndex];
  const hasModel = !!sel.value;
  const isMismatch = hasModel && selectedOpt && selectedOpt.disabled;

  noModelHint.classList.toggle('hidden', hasModel);
  modelMismatchHint.classList.toggle('hidden', !isMismatch);
}

// ── 运行对比 ───────────────────────────────────────────────────────────────

async function runComparison() {
  const modelName = document.getElementById('modelSelect').value;
  const params = {
    grid_size:       parseInt(document.getElementById('gridSize').value),
    traffic_pattern: document.getElementById('trafficPattern').value,
    num_cycles:      parseInt(document.getElementById('numCycles').value),
    injection_rate:  parseFloat(document.getElementById('injRate').value),
    model_name:      modelName,
  };

  updateModelHint();

  // UI 状态
  document.getElementById('cmpBtn').disabled   = true;
  document.getElementById('loadingState').classList.remove('hidden');
  document.getElementById('resultsSection').classList.add('hidden');
  document.getElementById('emptyState').classList.add('hidden');

  try {
    const data = await fetchJSON('/api/compare', {
      method: 'POST',
      body: JSON.stringify(params),
    });

    if (!data.success) throw new Error('对比失败');

    if (data.model_warning) {
      showToast(data.model_warning, 'error', 6000);
    }

    renderResults(data);

    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');
    showToast('对比完成！', 'success');
  } catch(e) {
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('emptyState').classList.remove('hidden');
    showToast('对比出错: ' + e.message, 'error');
  } finally {
    document.getElementById('cmpBtn').disabled = false;
  }
}

// ── 渲染结果 ───────────────────────────────────────────────────────────────

function renderResults(data) {
  renderSummaryTable(data);
  renderBarCharts(data);
  renderTimelineChart(data);
}

// 汇总表格
function renderSummaryTable(data) {
  const algos    = data.algorithms;         // ['XY路由','奇偶转弯','ML自适应']
  const metrics  = data.metrics;            // {avg_latency:[...], throughput:[...], ...}
  const details  = data.details;           // {xy:{name,metrics}, ...}

  // 每列计算最好/最差
  const metaDefs = [
    { key: 'avg_latency',      label: '平均延迟',  lowerBetter: true },
    { key: 'throughput',       label: '吞吐量',    lowerBetter: false },
    { key: 'avg_utilization',  label: '缓冲利用率',lowerBetter: true },
    { key: 'hotspot_count',    label: '热点节点',  lowerBetter: true },
    { key: 'power_mw',         label: '功耗(mW)',  lowerBetter: true },
    { key: 'delivered',        label: '已交付包',  lowerBetter: false },
    { key: 'dropped',          label: '丢包数',    lowerBetter: true },
  ];

  const algoKeys = ['xy', 'odd_even', 'ml_adaptive'];
  const rows = algoKeys.map((key, i) => {
    const m = details[key].metrics;
    const cells = metaDefs.map(def => {
      const vals = algoKeys.map(k => details[k].metrics[def.key] || 0);
      const best  = def.lowerBetter ? Math.min(...vals) : Math.max(...vals);
      const worst = def.lowerBetter ? Math.max(...vals) : Math.min(...vals);
      const v = m[def.key] || 0;
      const cls = v === best ? 'best' : v === worst ? 'worst' : '';
      const display = def.key === 'avg_utilization'
        ? (v * 100).toFixed(1) + '%'
        : def.key === 'throughput'
        ? v.toFixed(4)
        : def.key === 'power_mw'
        ? v.toFixed(2) + ' mW'
        : v;
      return `<td class="${cls}">${display}</td>`;
    }).join('');

    const color = ALGO_COLORS[i];
    return `<tr>
      <td>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};"></span>
          <strong>${algos[i]}</strong>
        </div>
      </td>
      ${cells}
    </tr>`;
  }).join('');

  document.getElementById('summaryBody').innerHTML = rows;
}

// 柱状图
function renderBarCharts(data) {
  const cfgs = [
    { id: 'chartLatency',    key: 'avg_latency',     fmt: v => v.toFixed(2) + ' cycles' },
    { id: 'chartThroughput', key: 'throughput',      fmt: v => v.toFixed(4) + ' pkts/cyc' },
    { id: 'chartUtil',       key: 'avg_utilization', fmt: v => (v*100).toFixed(1) + '%' },
    { id: 'chartHotspot',    key: 'hotspot_count',   fmt: v => v + ' nodes' },
    { id: 'chartPower',      key: 'power_mw',        fmt: v => v.toFixed(2) + ' mW' },
  ];

  cfgs.forEach(cfg => {
    const canvas = document.getElementById(cfg.id);
    if (!canvas) return;

    // 销毁旧图表
    if (charts[cfg.id]) { charts[cfg.id].destroy(); }

    const vals = data.metrics[cfg.key];

    charts[cfg.id] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: data.algorithms,
        datasets: [{
          data: vals,
          backgroundColor: ALGO_COLORS,
          borderRadius: 8,
          borderSkipped: false,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ' ' + cfg.fmt(ctx.raw),
            },
          },
        },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,.04)' } },
        },
      },
    });
  });
}

// 时序折线图
function renderTimelineChart(data) {
  if (charts['timeline']) { charts['timeline'].destroy(); }

  const tl = data.timeline;
  const labels = tl.cycles || [];

  const datasets = data.algorithms.map((name, i) => ({
    label: name,
    data: tl[name] || [],
    borderColor: ALGO_COLORS[i],
    backgroundColor: ALGO_COLORS[i] + '18',
    tension: 0.4,
    fill: false,
    pointRadius: 2,
  }));

  charts['timeline'] = new Chart(document.getElementById('chartTimeline'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        x: {
          title: { display: true, text: '仿真周期', font: { size: 12 } },
          grid: { color: 'rgba(0,0,0,.04)' },
        },
        y: {
          title: { display: true, text: '吞吐量 (pkts/cycle)', font: { size: 12 } },
          beginAtZero: true,
          grid: { color: 'rgba(0,0,0,.04)' },
        },
      },
    },
  });
}

// ── 初始化 ─────────────────────────────────────────────────────────────────

initPage();
