from modelos.reconocimiento import Reconocimiento
import cv2
import insightface
import numpy as np
import time

class ArcFaceReconocimiento(Reconocimiento):
    """Implementación usando ArcFace con insightface"""
    
    def __init__(self):
        try:
            self.app = insightface.app.FaceAnalysis()
            # No usar det_size para máxima precisión
            self.app.prepare(ctx_id=0)  # Sin det_size
            self.available = True
        except Exception as e:
            self.available = False
            print(f"Error al inicializar ArcFace: {e}")
    
    def generateFaceEmbedding(self, cara):
        if not self.available:
            return None, 0.0
            
        try:
            h, w = cara.shape[:2]
            
            # Asegurar dimensiones pares para evitar errores
            if h % 2 != 0:
                h += 1
            if w % 2 != 0:
                w += 1
            
            cara_even = cv2.resize(cara, (w, h))
            
            # Padding adaptativo
            if h > 500 or w > 500:
                padding = 120
            elif h > 300 or w > 300:
                padding = 80
            else:
                padding = 50
                
            # Crear imagen con padding (dimensiones pares)
            final_h = h + 2*padding
            final_w = w + 2*padding
            
            if final_h % 2 != 0:
                final_h += 1
            if final_w % 2 != 0:
                final_w += 1
                
            cara_con_padding = np.zeros((final_h, final_w, 3), dtype=np.uint8)
            cara_con_padding[padding:padding+h, padding:padding+w] = cara_even
            
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