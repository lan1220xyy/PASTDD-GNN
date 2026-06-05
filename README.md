## Supported Datasets

| Dataset | Description | Nodes | Format |
|---------|-------------|-------|--------|
| NYCTaxi | New York City taxi data | 266 | CSV |
| BJTaxi | Beijing taxi data | 1024 | CSV |
| CCGTaxi | Chicago taxi data | 77 | CSV |
| CCGRide | Chicago ride-haling data | 77 | CSV |

### Data Format

CSV format with:
- **First column**: Timestamp
- **Subsequent columns**: Spatial nodes/regions
- **Temporal resolution**: 30 or 15 minutes between samples
- **Data dimensions**: Time × Nodes × Features (T×N×1)

##  Installation

### Requirements

```bash
torch==2.4.0
numpy=1.24.1
pandas=2.0.3
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Training

```bash
cd model
python Run.py --mode trian
```

### Testing

```bash
cd model
python Run.py --mode test
```

