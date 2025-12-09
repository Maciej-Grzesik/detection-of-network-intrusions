import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        
        H = 128
        H2 = 64
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(H, H2),
            nn.BatchNorm1d(H2),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(H2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, H2),
            nn.BatchNorm1d(H2),
            nn.ReLU(),

            nn.Linear(H2, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),

            nn.Linear(H, input_dim),
        )

    def forward(self, x):
        x = x.float() 
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def predict(self, x, threshold: float):
        self.eval() 
        with torch.no_grad():
            x = x.float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            recon = self.forward(x)
            
            err = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            
            predictions = (err >= threshold).astype(int)
            
            return predictions, err
        
    @staticmethod
    def load_model_with_weights(input_dim: int, latent_dim: int, weights_path: str, device: str = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        
        state_dict = torch.load(weights_path, map_location=device)
        
        model.load_state_dict(state_dict)
        
        model.eval()
        
        return model