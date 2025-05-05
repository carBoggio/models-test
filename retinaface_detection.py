from detection import Detection
import cv2
try:
    from retinaface import RetinaFace
except ImportError:
    print("Warning: RetinaFace not installed. Install with: pip install retina-face")

class RetinaFaceDetection(Detection):
    """Implementación usando RetinaFace"""
    
    def __init__(self):
        try:
            from retinaface import RetinaFace
            self.available = True
        except ImportError:
            self.available = False
            print("RetinaFace no está instalado. Por favor instala con: pip install retina-face")
    
    def makeRecognition(self, imagen):
        if not self.available:
            return None
            
        imagen = self._load_imagen(imagen)
        if imagen is None:
            return None
            
        # RetinaFace espera RGB
        if len(imagen.shape) == 3:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        else:
            imagen_rgb = imagen
            
        try:
            # Realizar detección
            detecciones = RetinaFace.detect_faces(imagen_rgb)
            
            if detecciones is None:
                return None
                
            caras = []
            # Extraer cada cara usando las cajas delimitadoras
            for key, deteccion in detecciones.items():
                if 'facial_area' in deteccion:
                    x, y, x2, y2 = deteccion['facial_area']
                    # Asegurar límites válidos
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(imagen.shape[1], x2)
                    y2 = min(imagen.shape[0], y2)
                    
                    cara = imagen[y:y2, x:x2]
                    if cara.size > 0:
                        caras.append(cara)
            
            return caras if caras else None
            
        except Exception as e:
            print(f"Error en detección RetinaFace: {e}")
            return None