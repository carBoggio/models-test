import json
import os
import cv2
from random_forest import RandomForest
from modelos.retinaface_detection import RetinaFaceDetection
from modelos.arcface_reconocimiento import ArcFaceReconocimiento
from modelos.facenet_reconocimiento import FaceNetReconocimiento
from modelos.deepface_reconocimiento import DeepFaceReconocimiento
from modelos.setup import Setup

class MultiModelTrainAndTest:
    def __init__(self):
        """Inicializa el sistema con múltiples modelos"""
        # Modelos disponibles
        self.models = {
            "arcface": ArcFaceReconocimiento(),
            "facenet": FaceNetReconocimiento(),
            "deepface": DeepFaceReconocimiento()
        }
        
        # Detector común
        self.detector = RetinaFaceDetection()
        
        # Diccionario para almacenar modelos Random Forest
        self.rf_models = {}
        
    def train_model(self, model_name):
        """Entrena un modelo específico"""
        embeddings_file = f"modelos/embeddings_map_generated_{model_name}.json"
        
        # Cargar o generar embeddings
        if os.path.exists(embeddings_file):
            print(f"Cargando embeddings de {model_name}...")
            with open(embeddings_file, 'r') as f:
                training_embeddings = json.load(f)
        else:
            print(f"Generando embeddings para {model_name}...")
           
            setup = Setup(
                main_folder="caras_buena_definicion",
                model_name=model_name
            )
            training_embeddings = setup.generateMapWithEmbeddings()
            
            # Renombrar el archivo generado
            default_file = "embeddings_map.json"
            if os.path.exists(default_file):
                os.rename(default_file, embeddings_file)        # Preparar datos para Random Forest
        embeddings_for_rf = {}
        for person, images_embeddings in training_embeddings.items():
            person_embeddings = []
            for image_embeddings in images_embeddings:
                person_embeddings.extend(image_embeddings)
            embeddings_for_rf[person] = person_embeddings
        
        # Entrenar Random Forest
        self.rf_models[model_name] = RandomForest(embeddings_for_rf)
        print(f"Random Forest entrenado para {model_name}")
    
    def train_all_models(self):
        """Entrena todos los modelos"""
        for model_name in self.models.keys():
            print(f"\n=== Entrenando {model_name.upper()} ===")
            self.train_model(model_name)
    
    def process_test_images(self, test_folder="caras_test"):
        """Procesa imágenes de test con todos los modelos"""
        # Buscar carpeta de test
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            test_folder,
            os.path.join(current_dir, test_folder),
            os.path.join(os.path.dirname(current_dir), test_folder),
        ]
        
        test_folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                test_folder_path = path
                break
        
        if test_folder_path is None:
            print(f"Error: No se encontró la carpeta de test")
            return
        
        # Obtener todas las imágenes de test
        test_images = []
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            test_images.extend([f for f in os.listdir(test_folder_path) 
                              if f.lower().endswith(ext)])
        
        # Procesar con cada modelo
        for model_name in self.models.keys():
            print(f"\n=== Procesando con {model_name.upper()} ===")
            results = self.process_with_model(model_name, test_images, test_folder_path)
            
            # Guardar resultados en JSON separado
            output_file = f"test_results_{model_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Resultados guardados en '{output_file}'")
            
            # Mostrar resumen
            self.print_summary(results, model_name)
    
    def process_with_model(self, model_name, test_images, test_folder_path):
        """Procesa imágenes con un modelo específico"""
        results = []
        
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
                    'status': 'no_faces_detected',
                    'model': model_name
                })
                continue
            
            print(f"Caras detectadas: {len(faces)}")
            
            image_results = {
                'image': img_file,
                'faces': [],
                'model': model_name
            }
            
            # Procesar cada cara
            for i, face in enumerate(faces):
                print(f"  Cara {i+1}:")
                
                # Generar embedding según el modelo
                model = self.models[model_name]
                
                if model_name in ["arcface", "deepface"]:
                    embedding, inference_time = model.generateFaceEmbedding(face)
                else:  # facenet
                    embedding = model.generateFaceEmbedding(face)
                    inference_time = 0.0
                
                if embedding is None:
                    print("    Error al generar embedding")
                    image_results['faces'].append({
                        'face_id': i,
                        'status': 'embedding_error'
                    })
                    continue
                
                # Convertir a lista
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                else:
                    embedding = list(embedding)
                
                # Predecir con Random Forest
                rf_model = self.rf_models[model_name]
                predictions = rf_model.detect(embedding)
                
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
        
        return results
    
    def print_summary(self, results, model_name):
        """Imprime un resumen de resultados para un modelo"""
        print(f"\n=== RESUMEN DE RESULTADOS - {model_name.upper()} ===")
        
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
                    best_person = max(face['predictions'].items(), key=lambda x: x[1])
                    person, confidence = best_person
                    
                    person_counts[person] = person_counts.get(person, 0) + 1
                    
                    if confidence > 0.5:
                        high_confidence_counts[person] = high_confidence_counts.get(person, 0) + 1
        
        print("\nPredicciones por persona:")
        for person, count in person_counts.items():
            high_conf = high_confidence_counts.get(person, 0)
            print(f"  {person}: {count} predicciones ({high_conf} con confianza > 0.5)")
    
    def compare_models_performance(self):
        """Compara el rendimiento de todos los modelos"""
        comparison = {}
        
        # Leer resultados de cada modelo
        for model_name in self.models.keys():
            results_file = f"test_results_{model_name}.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                # Calcular métricas
                total_faces = sum(len(r['faces']) for r in results if 'faces' in r)
                correct_predictions = 0
                confidence_sum = 0
                
                for result in results:
                    for face in result.get('faces', []):
                        if 'predictions' in face:
                            # Asumiendo que el nombre de la imagen indica la persona correcta
                            image_name = result['image'].split('_')[0]
                            best_person = max(face['predictions'].items(), key=lambda x: x[1])
                            person, confidence = best_person
                            
                            if person.lower() == image_name.lower():
                                correct_predictions += 1
                            confidence_sum += confidence
                
                comparison[model_name] = {
                    'total_faces': total_faces,
                    'correct_predictions': correct_predictions,
                    'accuracy': correct_predictions / total_faces if total_faces > 0 else 0,
                    'average_confidence': confidence_sum / total_faces if total_faces > 0 else 0
                }
        
        # Guardar comparación
        with open('models_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=4)
        
        print("\n=== COMPARACIÓN DE MODELOS ===")
        for model, metrics in comparison.items():
            print(f"\n{model.upper()}:")
            print(f"  Total de caras: {metrics['total_faces']}")
            print(f"  Predicciones correctas: {metrics['correct_predictions']}")
            print(f"  Precisión: {metrics['accuracy']:.2%}")
            print(f"  Confianza promedio: {metrics['average_confidence']:.3f}")

# Ejemplo de uso
if __name__ == "__main__":
    # Crear sistema multi-modelo
    multi_trainer = MultiModelTrainAndTest()
    
    # Entrenar todos los modelos
    multi_trainer.train_all_models()
    
    # Procesar imágenes de test
    multi_trainer.process_test_images(test_folder="caras_test")
    
    # Comparar rendimiento
    multi_trainer.compare_models_performance()