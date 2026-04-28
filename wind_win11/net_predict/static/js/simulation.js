/**
 * NoC Mesh 网络可视化（Canvas）
 * 节点按缓冲利用率着色：绿 → 黄 → 红
 * ML 预测热点以蓝色光晕叠加显示
 */
class NetworkViz {
  constructor(canvasId, gridSize = 4) {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');
    this.gridSize = gridSize;

    this.stateHistory      = [];
    this.hotspotPredictions = [];
    this.currentFrame      = 0;

    this.isPlaying = false;
    this._timer    = null;
    this._fpsInterval = 1000 / 20;   // 默认 20 fps

    this.onFrameChange = null;
    this.onPlayEnd     = null;

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this._drawEmpty();
  }

  // ── 尺寸计算 ──────────────────────────────────────────────────────────────

  resize() {
    const wrap = this.canvas.parentElement;
    const maxSize = Math.min(wrap.clientWidth - 32, 520);
    this.canvas.width  = maxSize;
    this.canvas.height = maxSize;
    this._calcLayout();
    this.draw();
  }

  _calcLayout() {
    const N = this.gridSize;
    this.padding    = N <= 4 ? 32 : 24;
    this.cellSize   = (this.canvas.width - this.padding * 2) / N;
    this.nodeRadius = Math.max(8, this.cellSize * 0.28);
    this.fontSize   = Math.max(9, this.cellSize * 0.18);
  }

  setSpeed(fps) {
    this._fpsInterval = 1000 / Math.max(1, fps);
  }

  // ── 数据加载 ──────────────────────────────────────────────────────────────

  loadData(stateHistory, hotspotPredictions) {
    this.stateHistory       = stateHistory  || [];
    this.hotspotPredictions = hotspotPredictions || [];
    this.currentFrame = 0;
    this.pause();
    this._calcLayout();
    this.draw();
  }

  // ── 颜色工具 ──────────────────────────────────────────────────────────────

  /** 利用率 0–1 → RGB 颜色（绿→黄→红） */
  _utilColor(u) {
    u = Math.max(0, Math.min(1, u));
    if (u < 0.5) {
      const t = u / 0.5;
      return `rgb(${Math.round(52 + t * 203)},${Math.round(199 - t * 40)},${Math.round(89 - t * 89)})`;
    }
    const t = (u - 0.5) / 0.5;
    return `rgb(255,${Math.round(159 - t * 130)},10)`;
  }

  /** 节点中心坐标（x=列, y=行，y=0在上） */
  _nodeXY(x, y) {
    return {
      cx: this.padding + x * this.cellSize + this.cellSize / 2,
      cy: this.padding + y * this.cellSize + this.cellSize / 2,
    };
  }

  // ── 绘制 ──────────────────────────────────────────────────────────────────

  draw() {
    if (!this.stateHistory.length) { this._drawEmpty(); return; }

    const { canvas, ctx, gridSize } = this;
    const state = this.stateHistory[this.currentFrame]       || new Array(gridSize * gridSize).fill(0);
    const preds = this.hotspotPredictions[this.currentFrame] || new Array(gridSize * gridSize).fill(0);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 背景
    ctx.fillStyle = '#f5f5f7';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 链路
    this._drawLinks(ctx, gridSize, state);

    // 节点
    this._drawNodes(ctx, gridSize, state, preds);
  }

  _drawLinks(ctx, N, state) {
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) {
        const { cx, cy } = this._nodeXY(x, y);
        const u = state[x * N + y] || 0;

        // East 链路
        if (x < N - 1) {
          const { cx: cx2, cy: cy2 } = this._nodeXY(x + 1, y);
          const u2 = state[(x + 1) * N + y] || 0;
          this._drawLink(ctx, cx, cy, cx2, cy2, (u + u2) / 2);
        }
        // South 链路
        if (y < N - 1) {
          const { cx: cx2, cy: cy2 } = this._nodeXY(x, y + 1);
          const u2 = state[x * N + y + 1] || 0;
          this._drawLink(ctx, cx, cy, cx2, cy2, (u + u2) / 2);
        }
      }
    }
  }

  _drawLink(ctx, x1, y1, x2, y2, util) {
    const r = Math.round(util * 200);
    const g = Math.round(120 - util * 120);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = `rgba(${r},${g},0,${0.35 + util * 0.45})`;
    ctx.lineWidth   = 1.5 + util * 3;
    ctx.stroke();
  }

  _drawNodes(ctx, N, state, preds) {
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) {
        const idx  = x * N + y;
        const { cx, cy } = this._nodeXY(x, y);
        const util = state[idx] || 0;
        const pred = preds[idx] || 0;
        const r    = this.nodeRadius;

        // 预测热点光晕（蓝色）
        if (pred > 0.3) {
          ctx.beginPath();
          ctx.arc(cx, cy, r * (1.35 + pred * 0.5), 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0,113,227,${pred * 0.22})`;
          ctx.fill();
        }

        // 热点红色光晕
        if (util > 0.75) {
          ctx.beginPath();
          ctx.arc(cx, cy, r * 1.25, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,59,48,${(util - 0.75) * 0.35})`;
          ctx.fill();
        }

        // 节点主体
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = this._utilColor(util);
        ctx.fill();

        // 白色边框
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.lineWidth   = 2;
        ctx.stroke();

        // 节点标签
        ctx.fillStyle    = util > 0.55 ? '#fff' : '#1d1d1f';
        ctx.font         = `600 ${this.fontSize}px -apple-system,sans-serif`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${x},${y}`, cx, cy);

        // 利用率百分比（小字）
        if (this.gridSize <= 4) {
          ctx.fillStyle    = util > 0.55 ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.35)';
          ctx.font         = `${this.fontSize * 0.75}px -apple-system,sans-serif`;
          ctx.fillText(`${Math.round(util * 100)}%`, cx, cy + this.fontSize * 0.85);
        }
      }
    }
  }

  _drawEmpty() {
    const { canvas, ctx } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#f5f5f7';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle   = '#aeaeb2';
    ctx.font        = '14px -apple-system,sans-serif';
    ctx.textAlign   = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('点击「运行仿真」后显示网络拓扑', canvas.width / 2, canvas.height / 2 - 16);
    ctx.font = '12px -apple-system,sans-serif';
    ctx.fillText('节点颜色代表缓冲区利用率', canvas.width / 2, canvas.height / 2 + 12);
  }

  // ── 播放控制 ──────────────────────────────────────────────────────────────

  play() {
    if (this.isPlaying || !this.stateHistory.length) return;
    this.isPlaying = true;

    const tick = () => {
      if (!this.isPlaying) return;
      if (this.currentFrame < this.stateHistory.length - 1) {
        this.currentFrame++;
        this.draw();
        this.onFrameChange?.(this.currentFrame, this.stateHistory.length);
        this._timer = setTimeout(tick, this._fpsInterval);
      } else {
        this.isPlaying = false;
        this.onPlayEnd?.();
      }
    };
    this._timer = setTimeout(tick, this._fpsInterval);
  }

  pause() {
    this.isPlaying = false;
    clearTimeout(this._timer);
  }

  seek(frame) {
    this.pause();
    this.currentFrame = Math.max(0, Math.min(+frame, this.stateHistory.length - 1));
    this.draw();
    this.onFrameChange?.(this.currentFrame, this.stateHistory.length);
  }

  reset() {
    this.seek(0);
  }
}
