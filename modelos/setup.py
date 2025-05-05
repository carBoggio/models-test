import os
import cv2
from collections import defaultdict
from modelos.retinaface_detection import RetinaFaceDetection
from modelos.arcface_reconocimiento import ArcFaceReconocimiento
from modelos.facenet_reconocimiento import FaceNetReconocimiento
from modelos.deepface_reconocimiento import DeepFaceReconocimiento

class Setup:
    def __init__(self, main_folder="caras_buena_definicion", model_name="arcface"):
        """
        Inicializa el Setup con la carpeta principal y el modelo a usar
        
        Args:
            main_folder: Nombre de la carpeta principal
            model_name: Nombre del modelo a usar ("arcface", "facenet", "deepface")
        """
        # Obtener la ruta absoluta de la carpeta principal
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.main_folder_path = os.path.join(parent_dir, main_folder)
        
        self.model_name = model_name
        
        # Instanciar todos los modelos
        self.detector = RetinaFaceDetection()
        self.models = {
            "arcface": ArcFaceReconocimiento(),
            "facenet": FaceNetReconocimiento(),
            "deepface": DeepFaceReconocimiento()
        }
        
        # Verificar si la carpeta existe
        if not os.path.exists(self.main_folder_path):
            raise ValueError(f"La carpeta {self.main_folder_path} no existe")
        
        # Verificar si el modelo existe
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible. Opciones: {list(self.models.keys())}")
            
        self.selected_model = self.models[model_name]
    
    def generateMapWithEmbeddings(self):
        """
        Genera un mapa con los embeddings de todas las subcarpetas
        
        Returns:
            dict: {nombre_carpeta: lista de listas de embeddings}
        """
        embeddings_map = {}
        
        # Listar todas las subcarpetas
        subcarpetas = [f for f in os.listdir(self.main_folder_path) 
                      if os.path.isdir(os.path.join(self.main_folder_path, f))]
        
        print(f"Encontradas {len(subcarpetas)} subcarpetas: {subcarpetas}")
        
        for subcarpeta in subcarpetas:
            subcarpeta_path = os.path.join(self.main_folder_path, subcarpeta)
            print(f"\nProcesando carpeta: {subcarpeta}")
            
            # Lista para almacenar embeddings de esta persona
            person_embeddings = []
            
            # Obtener todos los archivos de imagen en la subcarpeta
            image_files = []
            for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                image_files.extend([f for f in os.listdir(subcarpeta_path) 
                                  if f.lower().endswith(ext)])
            
            print(f"  Encontradas {len(image_files)} imágenes")
            
            # Procesar cada imagen
            for img_file in image_files:
                img_path = os.path.join(subcarpeta_path, img_file)
                print(f"    Procesando: {img_file}")
                
                # Detectar caras en la imagen
                faces = self.detector.makeRecognition(img_path)
                
                # Lista de embeddings para esta imagen
                image_embeddings = []
                
                if faces:
                    print(f"      Caras detectadas: {len(faces)}")
                    
                    for i, face in enumerate(faces):
                        # Generar embedding según el modelo
                        if self.model_name in ["arcface", "deepface"]:
                            embedding, inference_time = self.selected_model.generateFaceEmbedding(face)
                        else:  # facenet
                            embedding = self.selected_model.generateFaceEmbedding(face)
                        
                        if embedding is not None:
                            # Convertir a lista si es numpy array
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                            else:
                                embedding = list(embedding)
                            
                            image_embeddings.append(embedding)
                            print(f"        Cara {i+1}: Embedding generado")
                        else:
                            print(f"        Cara {i+1}: Error al generar embedding")
                else:
                    print(f"      No se detectaron caras")
                
                if image_embeddings:
                    person_embeddings.append(image_embeddings)
            
            # Guardar embeddings de esta persona
            if person_embeddings:
                embeddings_map[subcarpeta] = person_embeddings
                print(f"  Total embeddings generados: {sum(len(img_emb) for img_emb in person_embeddings)}")
            else:
                print(f"  No se generaron embeddings para {subcarpeta}")
        
        return embeddings_map

# Ejemplo de uso
if __name__ == "__main__":
    # Crear setup para generar embeddings con ArcFace
    setup = Setup(main_folder="caras_buena_definicion", model_name="arcface")
    
    # Generar el mapa de embeddings
    embeddings_map = setup.generateMapWithEmbeddings()
    
    # Mostrar resumen
    print("\n=== RESUMEN ===")
    for person, embeddings_list in embeddings_map.items():
        total_faces = sum(len(img_embeddings) for img_embeddings in embeddings_list)
        print(f"{person}: {len(embeddings_list)} imágenes, {total_faces} caras")
    
    # Opcional: Guardar el mapa en JSON
    import json
    with open('embeddings_map_generated.json', 'w') as f:
        json.dump(embeddings_map, f, indent=4)
    print("\nEmbeddings guardados en 'embeddings_map_generated.json'")