# ToM-NAS Web Application

A beautiful, intuitive interface for evolving AI that understands minds.

## Quick Start

### Using Docker (Recommended)

```bash
# From the project root directory
docker-compose up --build
```

Then open http://localhost:8000 in your browser.

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn webapp.backend.app:app --reload --host 0.0.0.0 --port 8000
```

## Features

- **One-Click Evolution**: Start experiments with sensible defaults
- **Real-Time Visualization**: Watch fitness and diversity evolve in real-time
- **Interactive Explanations**: Learn about Theory of Mind as you experiment
- **Benchmark Dashboard**: See how evolved models perform on ToM tests
- **Interpretability Insights**: Understand what the models learned

## Architecture

```
webapp/
├── backend/
│   └── app.py          # FastAPI server
├── frontend/
│   ├── templates/
│   │   └── index.html  # Main page
│   └── static/
│       ├── style.css   # Styles
│       └── app.js      # Client-side logic
└── README.md
```

## API Endpoints

- `GET /` - Main application page
- `GET /api/status` - System status
- `GET /api/concepts` - ToM concept explanations
- `POST /api/experiments/quick-start` - Start experiment with preset
- `POST /api/experiments/create` - Create custom experiment
- `GET /api/experiments/{id}` - Get experiment details
- `GET /api/experiments/{id}/stream` - Get live updates
- `POST /api/benchmarks/run` - Run benchmark suite
- `GET /api/interpretability/{id}` - Get model insights
