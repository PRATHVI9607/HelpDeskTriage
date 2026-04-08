---
title: SupportOps Arena
emoji: 🎯
colorFrom: blue
colorTo: sky
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - it-support
  - environment
  - benchmark
pinned: false
---

# 🎯 SupportOps Arena

**Enterprise IT Incident Triage Reinforcement Learning Environment**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.ai)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

SupportOps Arena is a production-grade, OpenEnv-compliant RL benchmark that simulates enterprise IT support operations. AI agents act as Level-1/2 IT support operators, triaging incomplete and noisy incident tickets.

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run Server

```bash
python app/server.py
# Access dashboard at http://localhost:7860
```

### Run Baseline Agent

```bash
python inference.py
```

### Docker

```bash
docker build -t supportops-arena .
docker run -p 7860:7860 supportops-arena
```

---

## 📖 Environment

### Tasks
- **Easy**: Wi-Fi connectivity (10 steps, 6 scenarios)
- **Medium**: VPN access (16 steps, 6 scenarios, misleading logs)
- **Hard**: Cross-system failures (24 steps, 6 scenarios, contradictory evidence)

### Actions (15 total)
- **Diagnostic**: inspect_network, inspect_logs, check_authentication, etc.
- **Remediation**: flush_dns, reconfigure_client, restart_service, reset_credentials
- **Terminal**: escalate_ticket, resolve_ticket, close_without_fix

### Rewards
- Positive: +0.10 to +1.00 for good actions
- Negative: -0.05 to -0.40 for bad actions

---

## 🔌 API

- `POST /reset` - Start new episode
- `POST /step` - Take action
- `GET /state` - Get full state
- `POST /baseline/run` - Run baseline agent

---

## 🧪 Testing

```bash
pytest
pytest --cov
```

---

## 📊 Performance

**Baseline Agent Scores**:
- Easy: 0.65-0.75
- Medium: 0.40-0.55
- Hard: 0.20-0.35

---

## 📝 License

Apache License 2.0

---

Built for OpenEnv Hackathon 2026
