from reconocimiento import Reconocimiento
import cv2
import numpy as np
try:
    from facenet_pytorch import InceptionResnetV1
    import torch
except ImportError:
    print("Warning: FaceNet not installed. Install with: pip install facenet-pytorch torchvision")

class FaceNetReconocimiento(Reconocimiento):
    """Implementación usando FaceNet"""
    
    def __init__(self):
        try:
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.available = True
        except Exception as e:
            self.available = False
            print(f"Error al inicializar FaceNet: {e}")
    
    def generateFaceEmbedding(self, cara):
        if not self.available:
            return None
            
        try:
            # FaceNet espera imágenes de 160x160
            cara_resized = cv2.resize(cara, (160, 160))
            
            # Normalizar y convertir a tensor
            cara_tensor = (cara_resized - 127.5) / 128.0
            cara_tensor = torch.FloatTensor(cara_tensor).permute(2, 0, 1).unsqueeze(0)
            
            # Generar embedding
            with torch.no_grad():
                embedding = self.model(cara_tensor)
                embedding = embedding.numpy()[0]
            
            return embedding
            
        except Exception as e:
            print(f"Error en generación de embedding FaceNet: {e}")
            return None