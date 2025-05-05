from reconocimiento import Reconocimiento


import insightface

class ArcFaceReconocimiento(Reconocimiento):
    """Implementación usando ArcFace con insightface"""
    
    def __init__(self):
        try:
            # Inicializar el modelo ArcFace
            self.app = insightface.app.FaceAnalysis()
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.available = True
        except Exception as e:
            self.available = False
            print(f"Error al inicializar ArcFace: {e}")
    
    def generateFaceEmbedding(self, cara):
        if not self.available:
            return None
            
        try:
            # ArcFace espera BGR (ya lo está por OpenCV)
            # Obtener el embedding directamente
            faces = self.app.get(cara)
            
            if len(faces) > 0:
                # Retorna el embedding de la primera cara detectada
                embedding = faces[0].embedding
                return embedding
            else:
                print("No se detectó ninguna cara en la imagen")
                return None
                
        except Exception as e:
            print(f"Error en generación de embedding ArcFace: {e}")
            return None