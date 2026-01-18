# Real-Time Anomaly Detection App

## Projektübersicht
Echtzeit-Anomalieerkennung für industrielle Qualitätskontrolle (PET-Tubes, Spritzgussteile).
Vergleich von PatchCore vs. AD-DINOv3 für MAS-Thesis an der ZHAW.

## Tech Stack
- **Python 3.x** mit Type Hints
- **PyTorch 2.5.1** (CUDA 12.1)
- **Transformers 4.57** für DINOv3-Modelle
- **PySide6** für GUI
- **OpenCV** für Bildverarbeitung
- **FAISS** für Feature-Indexing/Matching
- **IDS Peak SDK** für Industriekameras

## Architektur
```
real_time_anomaly_detection_app/
├── gui/                    # PySide6 GUI
│   ├── handlers/           # Event-Handler
│   └── main_window.py      # Hauptfenster
├── models/                 # Trainierte/vortrainierte Modelle
│   └── facebook_dinov3-*   # DINOv3 Feature Extractors
├── datasets/               # Trainingsdaten, Referenzbilder
├── assets/                 # Icons, Ressourcen
├── projects/               # Projektdateien/Konfigurationen
└── doc/                    # Dokumentation
```

## Coding Standards
- Type Hints für alle Funktionen
- Docstrings im Google-Style
- Tests mit pytest
- Logging statt print()
- Konfiguration via YAML/JSON

## Wichtige Konventionen

### Bildverarbeitung
- Bilder als numpy arrays (H, W, C) oder torch tensors (C, H, W)
- Farbformat: BGR für OpenCV, RGB für PyTorch/PIL
- Normalisierung: ImageNet-Standards für DINOv3

### Anomaly Detection Pipeline
1. **Feature Extraction**: DINOv3 backbone → patch-level features
2. **Memory Bank**: FAISS-Index mit Referenz-Features (gut-Teile)
3. **Scoring**: k-NN Distanz zu nächsten Nachbarn
4. **Visualization**: Heatmap-Overlay auf Originalbild

### Kamera-Integration (IDS)
- `ids_peak` für Kamera-Steuerung
- `ids_peak_ipl` für Bildkonvertierung
- Trigger-Modi: Software, Hardware, Freerun

## Dependencies
```
torch>=2.5.0
torchvision>=0.20.0
transformers>=4.50.0
PySide6>=6.10.0
opencv-python>=4.10.0
faiss-cpu>=1.13.0
ids-peak>=1.13.0
numpy>=2.0.0
```

## Bekannte Einschränkungen
- DINOv3 benötigt min. 8GB VRAM für ViT-L
- IDS-Kameras nur unter Windows mit installierten Treibern
- FAISS-GPU optional (CPU-Version installiert)

## Performance-Ziele
- Inferenz: < 100ms pro Bild (GPU)
- GUI-Responsiveness: 30 FPS Live-Preview
- Memory Bank: bis 100k Features

## Nützliche Befehle
```bash
# Virtuelle Umgebung aktivieren
.venv\Scripts\activate

# Tests ausführen
pytest tests/ -v

# Type-Checking
mypy src/ --ignore-missing-imports

# Linting
ruff check .
```

## TODO / Offene Punkte
- [ ] PatchCore-Implementierung fertigstellen
- [ ] AD-DINOv3 Integration
- [ ] Benchmark-Suite für Vergleich
- [ ] ONNX-Export für Deployment
- [ ] Dokumentation der API
