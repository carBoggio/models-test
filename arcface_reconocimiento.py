from reconocimiento import Reconocimiento
import cv2
import insightface
import numpy as np
import time

class ArcFaceReconocimiento(Reconocimiento):
    """Implementación usando ArcFace con insightface"""
    
    def __init__(self):
        try:
            self.app = insightface.app.FaceAnalysis()
            # No redimensionar, usar tamaño original
            self.app.prepare(ctx_id=1, det_size=(640, 640))  # o det_size=None
            self.available = True
        except Exception as e:
            self.available = False
            print(f"Error al inicializar ArcFace: {e}")
    
    def generateFaceEmbedding(self, cara):
        if not self.available:
            return None, 0.0
            
        try:
            h, w = cara.shape[:2]
            
            # Agregar padding alrededor de la cara
            padding = 50  # Puedes ajustar este valor
            cara_con_padding = np.zeros((h + 2*padding, w + 2*padding, 3), dtype=np.uint8)
            cara_con_padding[padding:padding+h, padding:padding+w] = cara
            
            # Medir tiempo de detección
            start_time = time.time()
            faces = self.app.get(cara_con_padding)
            detection_time = time.time() - start_time
            
            if faces and len(faces) > 0:
                embedding = faces[0].embedding
                return embedding, detection_time
            else:
                print(f"No se detectó cara en imagen {h}x{w} con padding {padding}px")
                return None, detection_time
                
        except Exception as e:
            print(f"Error en generación de embedding ArcFace: {e}")
            return None, 0.0