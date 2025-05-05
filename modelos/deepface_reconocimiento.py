from modelos.reconocimiento import Reconocimiento
import cv2
import time
import numpy as np
try:
    from deepface import DeepFace
except ImportError:
    print("Warning: DeepFace not installed. Install with: pip install deepface")

class DeepFaceReconocimiento(Reconocimiento):
    """Implementación usando DeepFace"""
    
    def __init__(self, model_name="VGG-Face"):
        # Modelos disponibles: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib
        self.model_name = model_name
        try:
            # Verificar que DeepFace está instalado
            from deepface import DeepFace
            self.available = True
        except ImportError:
            self.available = False
            print("DeepFace no está instalado. Por favor instala con: pip install deepface")
        
    def generateFaceEmbedding(self, cara):
        if not self.available:
            return None, 0.0
            
        try:
            # Convertir BGR a RGB para DeepFace
            cara_rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
            
            # Medir tiempo
            start_time = time.time()
            
            # Obtener embedding usando DeepFace
            embedding_result = DeepFace.represent(
                img_path=cara_rgb, 
                model_name=self.model_name,
                enforce_detection=False  # No detectar caras nuevamente
            )
            
            detection_time = time.time() - start_time
            
            if embedding_result and len(embedding_result) > 0:
                embedding = embedding_result[0]['embedding']
                # Convertir a numpy array si no lo es
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                return embedding, detection_time
            else:
                return None, detection_time
            
        except Exception as e:
            print(f"Error en DeepFace {self.model_name}: {e}")
            return None, 0.0