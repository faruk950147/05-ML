                 RAW DATA
                    │
                    ↓
          ┌──────────────────┐
          │ Data Processing  │
          │                  │
          │ Missing Value    │
          │ Duplicate        │
          │ Wrong Data       │
          │ Encoding         │
          └────────┬─────────┘
                   ↓
              Clean Data
                   │
                   ↓
       ┌────────────────────────┐
       │ Feature Engineering    │
       │                        │
       │ Create New Features    │
       │ Transform Features     │
       │ Select Useful Features │
       └───────────┬────────────┘
                   ↓
             Useful Features
                   │
                   ↓
             Train / Test Split
                   │
                   ↓
              Model Training
                   │
                   ↓
                Prediction
                   │
                   ↓
               Evaluation