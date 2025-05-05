import json
import os
import cv2
from random_forest import RandomForest
from modelos.retinaface_detection import RetinaFaceDetection
from modelos.arcface_reconocimiento import ArcFaceReconocimiento
from modelos.setup import Setup

class TrainAndTest:
    def __init__(self, embeddings_file="embeddings_arcface.json", model_name="arcface"):
        """
        Inicializa el sistema de entrenamiento y prueba
        
        Args:
            embeddings_file: Archivo JSON con embeddings de entrenamiento
            model_name: Modelo usado para generar embeddings
        """
        self.embeddings_file = embeddings_file
        self.model_name = model_name
        
        # Inicializar detector y modelo de reconocimiento
        self.detector = RetinaFaceDetection()
        
        if model_name == "arcface":
            self.recognition_model = ArcFaceReconocimiento()
        else:
            raise ValueError(f"Modelo {model_name} no implementado en este ejemplo")
        
        # Cargar embeddings de entrenamiento
        self.load_training_data()
        
        # Entrenar Random Forest
        self.train_random_forest()
    
    def load_training_data(self):
        """Carga los embeddings desde el JSON"""
        try:
            with open(self.embeddings_file, 'r') as f:
                self.training_embeddings = json.load(f)
            print(f"Embeddings de entrenamiento cargados desde '{self.embeddings_file}'")
            print(f"Personas encontradas: {list(self.training_embeddings.keys())}")
        except FileNotFoundError:
            print(f"Error: Archivo '{self.embeddings_file}' no encontrado")
            # Generar embeddings si no existen
            self.generate_training_embeddings()
    
    def generate_training_embeddings(self):
        """Genera embeddings de entrenamiento si no existen"""
        print("Generando embeddings de entrenamiento...")
        setup = Setup(
            main_folder="caras_buena_definicion",
            model_name=self.model_name,
            output_file=self.embeddings_file
        )
        self.training_embeddings = setup.generateMapWithEmbeddings()
    
    def train_random_forest(self):
        """Entrena el modelo Random Forest con los embeddings"""
        print("\n=== Entrenando Random Forest ===")
        
        # Preparar datos para Random Forest
        embeddings_for_rf = {}
        
        for person, images_embeddings in self.training_embeddings.items():
            # Aplanar las listas de listas
            person_embeddings = []
            for image_embeddings in images_embeddings:
                person_embeddings.extend(image_embeddings)
            embeddings_for_rf[person] = person_embeddings
        
        # Crear y entrenar Random Forest
        self.rf_model = RandomForest(embeddings_for_rf)
        print("Random Forest entrenado exitosamente")
    
    def process_test_images(self, test_folder="caras_test"):
        """
        Procesa imágenes de test y calcula confianza
        
        Args:
            test_folder: Carpeta con imágenes de test
        """
        # Obtener ruta absoluta de la carpeta de test
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Buscar la carpeta de test en diferentes ubicaciones
        possible_paths = [
            test_folder,  # Ruta absoluta o en el directorio actual
            os.path.join(current_dir, test_folder),  # En el directorio actual
            os.path.join(os.path.dirname(current_dir), test_folder),  # En el directorio padre
        ]
        
        test_folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                test_folder_path = path
                break
        
        if test_folder_path is None:
            print(f"Error: No se encontró la carpeta de test en ninguna de estas ubicaciones:")
            for path in possible_paths:
                print(f"  - {path}")
            return
        
        print(f"\n=== Procesando imágenes de test en '{test_folder_path}' ===")
        
        # Obtener todas las imágenes de test
        test_images = []
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            test_images.extend([f for f in os.listdir(test_folder_path) 
                              if f.lower().endswith(ext)])
        
        print(f"Encontradas {len(test_images)} imágenes de test")
        
        results = []
        
        # Procesar cada imagen
        for img_file in test_images:
            img_path = os.path.join(test_folder_path, img_file)
            print(f"\n--- Procesando: {img_file} ---")
            
            # Detectar caras
            faces = self.detector.makeRecognition(img_path)
            
            if not faces:
                print("No se detectaron caras")
                results.append({
                    'image': img_file,
                    'faces': [],
                    'status': 'no_faces_detected'
                })
                continue
            
            print(f"Caras detectadas: {len(faces)}")
            
            image_results = {
                'image': img_file,
                'faces': []
            }
            
            # Procesar cada cara
            for i, face in enumerate(faces):
                print(f"  Cara {i+1}:")
                
                # Generar embedding
                embedding, inference_time = self.recognition_model.generateFaceEmbedding(face)
                
                if embedding is None:
                    print("    Error al generar embedding")
                    image_results['faces'].append({
                        'face_id': i,
                        'status': 'embedding_error'
                    })
                    continue
                
                # Convertir a lista si es numpy array
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                else:
                    embedding = list(embedding)
                
                # Predecir con Random Forest
                predictions = self.rf_model.detect(embedding)
                
                # Mostrar resultados
                print("    Predicciones:")
                for person, confidence in predictions.items():
                    print(f"      {person}: {confidence:.3f}")
                
                # Guardar resultados
                image_results['faces'].append({
                    'face_id': i,
                    'predictions': predictions,
                    'inference_time': float(inference_time)
                })
            
            results.append(image_results)
        
        # Guardar resultados en JSON
        self.save_test_results(results)
        
        # Mostrar resumen
        self.print_summary(results)
        
        return results
    
    def save_test_results(self, results):
        """Guarda los resultados de test en JSON"""
        output_file = "test_results.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResultados guardados en '{output_file}'")
        except Exception as e:
            print(f"Error al guardar resultados: {e}")
    
    def print_summary(self, results):
        """Imprime un resumen de los resultados"""
        print("\n=== RESUMEN DE RESULTADOS ===")
        
        total_images = len(results)
        total_faces = sum(len(r['faces']) for r in results if 'faces' in r)
        
        print(f"Imágenes procesadas: {total_images}")
        print(f"Total de caras detectadas: {total_faces}")
        
        # Contar predicciones por persona
        person_counts = {}
        high_confidence_counts = {}
        
        for result in results:
            for face in result.get('faces', []):
                if 'predictions' in face:
                    # Contar la persona con mayor confianza
                    best_person = max(face['predictions'].items(), key=lambda x: x[1])
                    person, confidence = best_person
                    
                    person_counts[person] = person_counts.get(person, 0) + 1
                    
                    if confidence > 0.5:
                        high_confidence_counts[person] = high_confidence_counts.get(person, 0) + 1
        
        print("\nPredicciones por persona:")
        for person, count in person_counts.items():
            high_conf = high_confidence_counts.get(person, 0)
            print(f"  {person}: {count} predicciones ({high_conf} con confianza > 0.5)")

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia con parámetros fijos
    trainer = TrainAndTest(
        embeddings_file="modelos/embeddings_map_generated.json",
        model_name="arcface"
    )
    
    # Procesar imágenes de test
    results = trainer.process_test_images(test_folder="caras_test")