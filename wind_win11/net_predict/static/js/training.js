/**
 * 模型训练页 JS
 * - 发起训练请求 → 订阅 SSE 流 → 实时更新 Chart.js 曲线
 */

let lossChart, accChart;
let evtSource = null;

// ── 初始化图表 ─────────────────────────────────────────────────────────────

function initCharts() {
  const chartDefaults = {
    type: 'line',
    options: {
      animation: { duration: 0 },
      responsive: true,
      plugins: { legend: { position: 'top' } },
      scales: {
        x: { grid: { color: 'rgba(0,0,0,.04)' } },
        y: { grid: { color: 'rgba(0,0,0,.04)' }, beginAtZero: false },
      },
    },
  };

  lossChart = new Chart(document.getElementById('lossChart'), {
    ...chartDefaults,
    data: {
      labels: [],
      datasets: [
        {
          label: 'Train Loss',
          data: [],
          borderColor: '#0071e3',
          backgroundColor: 'rgba(0,113,227,0.07)',
          tension: 0.4,
          fill: true,
          pointRadius: 2,
        },
        {
          label: 'Val Loss',
          data: [],
          borderColor: '#ff9f0a',
          backgroundColor: 'rgba(255,159,10,0.07)',
          tension: 0.4,
          fill: true,
          pointRadius: 2,
        },
      ],
    },
  });

  accChart = new Chart(document.getElementById('accChart'), {
    ...chartDefaults,
    data: {
      labels: [],
      datasets: [
        {
          label: '验证准确率',
          data: [],
          borderColor: '#34c759',
          backgroundColor: 'rgba(52,199,89,0.07)',
          tension: 0.4,
          fill: true,
          pointRadius: 2,
        },
      ],
    },
    options: {
      ...chartDefaults.options,
      scales: {
        ...chartDefaults.options.scales,
        y: { min: 0, max: 1, grid: { color: 'rgba(0,0,0,.04)' } },
      },
    },
  });
}

// ── 添加一个日志行 ─────────────────────────────────────────────────────────

function appendLog(msg, cls = '') {
  const box = document.getElementById('logBox');
  const line = document.createElement('div');
  line.className = cls ? `log-line-${cls}` : '';
  line.textContent = msg;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

// ── 开始训练 ───────────────────────────────────────────────────────────────

async function startTraining() {
  // 收集选中的流量模式
  const patterns = [];
  ['uniform', 'hotspot', 'transpose', 'bit_complement'].forEach(p => {
    const el = document.getElementById(`p_${p === 'bit_complement' ? 'bit' : p}`);
    if (el && el.checked) patterns.push(p);
  });
  if (!patterns.length) {
    showToast('请至少选择一种流量模式', 'error');
    return;
  }

  const config = {
    grid_size:   parseInt(document.getElementById('gridSize').value),
    model_type:  document.getElementById('modelType').value,
    epochs:      parseInt(document.getElementById('epochs').value),
    lr:          parseFloat(document.getElementById('lr').value),
    batch_size:  parseInt(document.getElementById('batchSize').value),
    traffic_patterns: patterns,
    n_cycles_per_pattern: parseInt(document.getElementById('nCycles').value),
  };

  // 重置 UI
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('stopBtn').classList.remove('hidden');
  document.getElementById('progressSection').classList.remove('hidden');
  document.getElementById('completeCard').classList.add('hidden');
  document.getElementById('logBox').innerHTML = '';

  lossChart.data.labels = [];
  lossChart.data.datasets.forEach(d => d.data = []);
  lossChart.update('none');
  accChart.data.labels = [];
  accChart.data.datasets.forEach(d => d.data = []);
  accChart.update('none');

  appendLog(`[开始] grid=${config.grid_size}×${config.grid_size} model=${config.model_type} epochs=${config.epochs}`, 'info');

  // 启动训练
  let sessionId;
  try {
    const res = await fetchJSON('/api/train/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
    if (!res.success) throw new Error('启动训练失败');
    sessionId = res.session_id;
    appendLog(`[信息] 会话 ID: ${sessionId}`, 'info');
  } catch(e) {
    showToast('启动训练失败: ' + e.message, 'error');
    _resetTrainBtn();
    return;
  }

  // 订阅 SSE
  if (evtSource) evtSource.close();
  evtSource = new EventSource(`/api/train/stream/${sessionId}`);

  evtSource.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    if (msg.type === 'heartbeat') return;

    if (msg.type === 'status') {
      appendLog(`[状态] ${msg.message}`, 'info');
      return;
    }

    if (msg.type === 'progress') {
      const { epoch, total_epochs, loss, val_loss, accuracy } = msg;
      const label = `Epoch ${epoch}`;
      const pct   = Math.round(epoch / total_epochs * 100);

      // 图表
      lossChart.data.labels.push(label);
      lossChart.data.datasets[0].data.push(loss);
      lossChart.data.datasets[1].data.push(val_loss);
      lossChart.update('none');

      accChart.data.labels.push(label);
      accChart.data.datasets[0].data.push(accuracy);
      accChart.update('none');

      // 进度条
      document.getElementById('progressBar').style.width  = pct + '%';
      document.getElementById('progressPct').textContent  = pct + '%';
      document.getElementById('progressLabel').textContent = `Epoch ${epoch} / ${total_epochs}`;
      document.getElementById('curLoss').textContent    = loss.toFixed(4);
      document.getElementById('curValLoss').textContent = val_loss.toFixed(4);
      document.getElementById('curAcc').textContent     = (accuracy * 100).toFixed(1) + '%';

      if (epoch % 5 === 0) {
        appendLog(`Epoch ${epoch}/${total_epochs} — loss: ${loss.toFixed(4)} val: ${val_loss.toFixed(4)} acc: ${(accuracy*100).toFixed(1)}%`);
      }
      return;
    }

    if (msg.type === 'complete') {
      evtSource.close();
      appendLog(`[完成] 模型已保存：${msg.model_name}  准确率: ${(msg.final_accuracy*100).toFixed(1)}%`, 'ok');
      document.getElementById('progressBar').style.width = '100%';
      document.getElementById('progressPct').textContent = '100%';
      document.getElementById('completeMsg').textContent =
        `模型：${msg.model_name}  |  准确率：${(msg.final_accuracy*100).toFixed(1)}%  |  最佳 Val Loss：${msg.best_val_loss}`;
      document.getElementById('completeCard').classList.remove('hidden');
      showToast('模型训练完成！', 'success');
      loadModelList();
      _resetTrainBtn();
      return;
    }

    if (msg.type === 'error') {
      evtSource.close();
      appendLog(`[错误] ${msg.message}`, 'err');
      showToast('训练出错: ' + msg.message, 'error');
      _resetTrainBtn();
    }
  };

  evtSource.onerror = () => {
    appendLog('[错误] SSE 连接中断', 'err');
    evtSource.close();
    _resetTrainBtn();
  };
}

// ── 停止训练 ───────────────────────────────────────────────────────────────

function stopTraining() {
  if (evtSource) { evtSource.close(); evtSource = null; }
  appendLog('[停止] 用户手动停止训练', 'err');
  _resetTrainBtn();
}

function _resetTrainBtn() {
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('stopBtn').classList.add('hidden');
}

// ── 模型列表 ───────────────────────────────────────────────────────────────

async function loadModelList() {
  try {
    const data = await fetchJSON('/api/models');
    const el = document.getElementById('modelList');
    if (!data.models.length) {
      el.innerHTML = '<span class="text-sec">暂无已训练模型</span>';
      return;
    }
    const rows = data.models.map(m => `
      <tr>
        <td>${m.name}</td>
        <td><span class="tag ${m.model_type==='lstm'?'tag-blue':'tag-gray'}">${m.model_type.toUpperCase()}</span></td>
        <td>${m.grid_size}×${m.grid_size}</td>
        <td class="${m.accuracy>=0.85?'best':''}">${(m.accuracy*100).toFixed(1)}%</td>
        <td>${m.val_loss.toFixed(4)}</td>
      </tr>`).join('');
    el.innerHTML = `
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>文件名</th><th>类型</th><th>网格</th><th>准确率</th><th>验证Loss</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  } catch(e) {
    document.getElementById('modelList').innerHTML = '<span class="text-sec">加载失败</span>';
  }
}

// ── 初始化 ─────────────────────────────────────────────────────────────────

initCharts();
loadModelList();
