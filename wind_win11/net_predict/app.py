"""
Flask 主应用
提供仿真、训练、对比三大功能的 Web 界面与 REST API
"""
from __future__ import annotations

import glob as _glob
import json
import os
import queue
import threading
from typing import Dict, Optional

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import MLConfig, SimConfig
from simulator.mesh_network import MeshNetwork
from simulator.traffic import TrafficGenerator

app = Flask(__name__)

# ── 全局状态 ──────────────────────────────────────────────────────────────────
training_sessions: Dict[str, Dict] = {}
_model_cache: Dict[str, tuple] = {}   # path → (model, checkpoint)


def _load_model(model_path: str):
    """加载并缓存 PyTorch 模型"""
    if model_path in _model_cache:
        return _model_cache[model_path]
    try:
        import torch
        from ml.model import build_model
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        model = build_model(
            ckpt['model_type'],
            ckpt['n_nodes'],
            ckpt.get('seq_len', MLConfig.SEQ_LEN),
            ckpt.get('hidden_size', MLConfig.HIDDEN_SIZE),
            ckpt.get('num_layers', MLConfig.NUM_LAYERS),
        )
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        _model_cache[model_path] = (model, ckpt)
        return model, ckpt
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
# 页面路由
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulation')
def simulation_page():
    return render_template('simulation.html')


@app.route('/training')
def training_page():
    return render_template('training.html')


@app.route('/comparison')
def comparison_page():
    return render_template('comparison.html')


# ══════════════════════════════════════════════════════════════════════════════
# 仿真 API
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    data: Dict = request.json or {}

    grid_size       = int(data.get('grid_size', 4))
    traffic_pattern = data.get('traffic_pattern', 'hotspot')
    num_cycles      = min(int(data.get('num_cycles', 300)), SimConfig.MAX_CYCLES)
    routing_algo    = data.get('routing_algorithm', 'xy')
    injection_rate  = float(data.get('injection_rate', 0.3))
    model_name      = data.get('model_name', '')

    net  = MeshNetwork(size=grid_size, buffer_capacity=SimConfig.BUFFER_CAPACITY)
    tgen = TrafficGenerator(size=grid_size, seed=42)

    # 加载 ML 模型
    ml_model = None
    seq_len = MLConfig.SEQ_LEN
    if routing_algo == 'ml_adaptive' and model_name:
        model_path = os.path.join(MLConfig.MODEL_DIR, model_name)
        if os.path.exists(model_path):
            try:
                ml_model, ckpt = _load_model(model_path)
                seq_len = ckpt.get('seq_len', MLConfig.SEQ_LEN)
                if ckpt.get('grid_size') != grid_size:
                    ml_model = None
            except Exception:
                pass

    state_history: list       = []
    hotspot_predictions: list = []
    metrics_timeline: list    = []
    seq_buffer: list          = []

    for cycle in range(num_cycles):
        pkts = tgen.get_packets(traffic_pattern, injection_rate)
        for src, dst in pkts:
            net.inject_packet(src, dst)

        # ML 预测（需要足够的历史序列）
        hotspot_probs = None
        if ml_model is not None and len(seq_buffer) >= seq_len:
            import torch
            x_in = torch.FloatTensor([seq_buffer[-seq_len:]])
            with torch.no_grad():
                # 模型输出 logits，需经 sigmoid 转为概率
                hotspot_probs = torch.sigmoid(ml_model(x_in)).numpy()[0]
            hotspot_predictions.append(hotspot_probs.tolist())
        else:
            hotspot_predictions.append([0.0] * (grid_size * grid_size))

        state = net.step(routing_algo, hotspot_probs)
        state_history.append(state.tolist())
        seq_buffer.append(state.tolist())

        if (cycle + 1) % 10 == 0:
            metrics_timeline.append({
                'cycle': cycle + 1,
                **net.get_metrics(),
            })

    # 采样包路径（最多 30 条已交付包）
    traces = []
    for pkt in net.delivered[:30]:
        if len(pkt.path) > 1:
            traces.append({
                'src': list(pkt.src),
                'dst': list(pkt.dst),
                'path': [list(p) for p in pkt.path],
                'latency': pkt.latency,
                'hops': pkt.hops,
            })

    return jsonify({
        'success': True,
        'metrics': net.get_metrics(),
        'metrics_timeline': metrics_timeline,
        'state_history': state_history,
        'hotspot_predictions': hotspot_predictions,
        'packet_traces': traces,
        'hotspot_map': net.get_hotspot_map().tolist(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# 训练 API
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    config: Dict = request.json or {}
    session_id = f"train_{int(__import__('time').time() * 1000) % 1_000_000}"

    q: queue.Queue = queue.Queue()

    def _run():
        try:
            from ml.trainer import train_model
            train_model(config, q)
        except Exception as exc:
            q.put({'type': 'error', 'message': str(exc)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    training_sessions[session_id] = {'queue': q, 'thread': t}

    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/train/stream/<session_id>')
def api_train_stream(session_id: str):
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    q: queue.Queue = training_sessions[session_id]['queue']

    def _generate():
        while True:
            try:
                msg = q.get(timeout=120)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get('type') in ('complete', 'error'):
                    break
            except queue.Empty:
                yield 'data: {"type":"heartbeat"}\n\n'

    return Response(
        _generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 对比 API
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        data: Dict = request.json or {}

        grid_size       = int(data.get('grid_size', 4))
        traffic_pattern = data.get('traffic_pattern', 'hotspot')
        num_cycles      = min(int(data.get('num_cycles', 500)), 1000)
        injection_rate  = float(data.get('injection_rate', 0.3))
        model_name      = data.get('model_name', '')

        if grid_size == 8:
            num_cycles = min(num_cycles, 500)

        ml_model = None
        seq_len = MLConfig.SEQ_LEN
        if model_name:
            model_path = os.path.join(MLConfig.MODEL_DIR, model_name)
            if os.path.exists(model_path):
                try:
                    ml_model, ckpt = _load_model(model_path)
                    seq_len = ckpt.get('seq_len', MLConfig.SEQ_LEN)
                    if ckpt.get('grid_size') != grid_size:
                        ml_model = None
                except Exception:
                    pass

        algos = [
            {'key': 'xy',           'name': 'XY路由'},
            {'key': 'odd_even',     'name': '奇偶转弯'},
            {'key': 'ml_adaptive',  'name': 'ML自适应'},
        ]

        results: Dict = {}

        for algo in algos:
            key = algo['key']
            net  = MeshNetwork(size=grid_size, buffer_capacity=SimConfig.BUFFER_CAPACITY)
            tgen = TrafficGenerator(size=grid_size, seed=42)

            seq_buf: list   = []
            timeline: list  = []

            for cycle in range(num_cycles):
                pkts = tgen.get_packets(traffic_pattern, injection_rate)
                for src, dst in pkts:
                    net.inject_packet(src, dst)

                hotspot_probs = None
                if key == 'ml_adaptive' and ml_model and len(seq_buf) >= seq_len:
                    import torch
                    x_in = torch.FloatTensor([seq_buf[-seq_len:]])
                    with torch.no_grad():
                        hotspot_probs = torch.sigmoid(ml_model(x_in)).numpy()[0]

                state = net.step(key, hotspot_probs)
                seq_buf.append(state.tolist())

                if (cycle + 1) % 20 == 0:
                    m = net.get_metrics()
                    timeline.append({
                        'cycle': cycle + 1,
                        'throughput': m['throughput'],
                        'avg_latency': m['avg_latency'],
                        'avg_utilization': m['avg_utilization'],
                        'hotspot_count': m['hotspot_count'],
                    })

            results[key] = {
                'name': algo['name'],
                'metrics': net.get_metrics(),
                'timeline': timeline,
            }

        metric_keys = ['avg_latency', 'throughput', 'avg_utilization', 'hotspot_count', 'power_mw']
        comparison = {
            'algorithms': [a['name'] for a in algos],
            'metrics': {
                mk: [results[a['key']]['metrics'].get(mk, 0) for a in algos]
                for mk in metric_keys
            },
            'timeline': {
                'cycles': [t['cycle'] for t in results['xy']['timeline']],
                **{
                    results[a['key']]['name']: [t['throughput'] for t in results[a['key']]['timeline']]
                    for a in algos
                },
            },
            'details': {
                k: {
                    'name': v['name'],
                    'metrics': v['metrics'],
                }
                for k, v in results.items()
            },
            'ml_available': ml_model is not None,
        }

        return jsonify({'success': True, **comparison})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# 模型管理 API
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/models')
def api_models():
    os.makedirs(MLConfig.MODEL_DIR, exist_ok=True)
    files = sorted(
        _glob.glob(os.path.join(MLConfig.MODEL_DIR, '*.pth')),
        key=os.path.getmtime,
        reverse=True,
    )
    models = []
    for path in files:
        name = os.path.basename(path)
        try:
            import torch
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            models.append({
                'name': name,
                'grid_size': ckpt.get('grid_size', 4),
                'model_type': ckpt.get('model_type', 'lstm'),
                'accuracy': round(float(ckpt.get('final_accuracy', 0)), 4),
                'val_loss': round(float(ckpt.get('best_val_loss', 0)), 4),
            })
        except Exception:
            pass
    return jsonify({'models': models})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data: Dict = request.json or {}
    model_name = data.get('model_name', '')
    state_seq  = data.get('state_sequence')

    if not model_name or state_seq is None:
        return jsonify({'error': '缺少 model_name 或 state_sequence'}), 400

    model_path = os.path.join(MLConfig.MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': '模型文件不存在'}), 404

    try:
        import torch
        model, _ = _load_model(model_path)
        x_in = torch.FloatTensor([state_seq])
        with torch.no_grad():
            probs = torch.sigmoid(model(x_in)).numpy()[0]
        return jsonify({'hotspot_probs': probs.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(MLConfig.MODEL_DIR, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
