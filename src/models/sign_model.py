import torch
import torch.nn as nn

class ASLClassifier(nn.Module):
    """
    Lightweight MLP for static ASL alphabet classification (A-Z except J, Z).
    """
    def __init__(self, input_size=63, num_classes=26, hidden_size=128):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


class ASLDynamicClassifier(nn.Module):
    """
    LSTM-based classifier for dynamic/motion ASL signs (J, Z, and phrases).
    Processes a sequence of two-hand landmark frames (126D per frame).
    """
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use the last hidden state from the top layer
        features = hn[-1]  # shape: (batch_size, hidden_size)
        out = self.classifier(features)
        return out
