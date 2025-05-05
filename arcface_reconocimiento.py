from reconocimiento import Reconocimiento
import cv2
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
            # Mostrar la cara antes de procesarla
            print(f"Dimensiones de la cara: {cara.shape}")
            print(f"Tipo de la cara: {type(cara)}")
            
            # Mostrar la cara con OpenCV
            cv2.imshow('Cara recibida', cara)
            cv2.waitKey(1000)  # Espera 1 segundo
            cv2.destroyAllWindows()
            
            # Guardar la cara en un archivo para inspección
            cv2.imwrite('debug_cara.jpg', cara)
            print("Cara guardada en debug_cara.jpg")
            
            # Comprobar que la imagen sea válida
            if cara.size == 0:
                print("La imagen de la cara está vacía!")
                return None
            
            # Redimensionar si es necesario (ArcFace espera 112x112)
            if cara.shape[0] != 112 or cara.shape[1] != 112:
                cara_resized = cv2.resize(cara, (112, 112))
                print(f"Cara redimensionada a: {cara_resized.shape}")
            else:
                cara_resized = cara
            
            # Obtener el embedding directamente
            faces = self.app.get(cara_resized)
            
            if len(faces) > 0:
                # Retorna el embedding de la primera cara detectada
                embedding = faces[0].embedding
                print(f"Embedding generado con éxito, dimensiones: {embedding.shape}")
                return embedding
            else:
                print("No se detectó ninguna cara en la imagen")
                print("Valor mínimo en imagen:", cara.min())
                print("Valor máximo en imagen:", cara.max())
                print("Tamaño de la imagen:", cara.size)
                return None
                
        except Exception as e:
            print(f"Error en generación de embedding ArcFace: {e}")
            import traceback
            traceback.print_exc()
            return None