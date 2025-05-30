import os
import cv2
import numpy as np
import time
from pathlib import Path
import json
from collections import defaultdict

# Importar los modelos
from retinaface_detection import RetinaFaceDetection
from facenet_reconocimiento import FaceNetReconocimiento
from deepface_reconocimiento import DeepFaceReconocimiento
from arcface_reconocimiento import ArcFaceReconocimiento

class TestTimeDifferentModels:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.detector = RetinaFaceDetection()
        self.arcface = ArcFaceReconocimiento()
        self.facenet = FaceNetReconocimiento()
        self.deepface = DeepFaceReconocimiento()
        
        # Diccionario para almacenar resultados
        self.embeddings_map = defaultdict(list)
        self.timing_results = {
            'detection': [],
            'arcface': [],
            'facenet': [],
            'deepface': []
        }
    
    def load_images(self):
        """Cargar todas las imágenes del folder"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = []
        
        for file in os.listdir(self.image_folder):
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(self.image_folder, file))
        
        print(f"Encontradas {len(image_files)} imágenes")
        return image_files
    
    def process_image(self, image_path):
        """Procesar una imagen y extraer caras"""
        print(f"\n=== Procesando: {os.path.basename(image_path)} ===")
        
        # Detección de caras
        start_time = time.time()
        faces = self.detector.makeRecognition(image_path)
        detection_time = time.time() - start_time
        
        self.timing_results['detection'].append({
            'image': os.path.basename(image_path),
            'time': detection_time,
            'faces_found': len(faces) if faces else 0
        })
        
        print(f"Tiempo de detección: {detection_time:.4f} segundos")
        print(f"Caras encontradas: {len(faces) if faces else 0}")
        
        if faces:
            for i, face in enumerate(faces):
                print(f"\n  --- Cara {i+1} ---")
                print(f"  Dimensiones: {face.shape}")
                
                # Generar embedding con ArcFace (ahora devuelve tupla)
                arcface_embedding, arcface_inference_time = self.arcface.generateFaceEmbedding(face)
                
                self.timing_results['arcface'].append({
                    'image': os.path.basename(image_path),
                    'face_id': i,
                    'inference_time': arcface_inference_time
                })
                
                if arcface_embedding is not None:
                    print(f"  ArcFace - Tiempo de inferencia: {arcface_inference_time:.4f} segundos")
                else:
                    print(f"  ArcFace - ERROR: No se generó embedding")
                
                # Generar embedding con FaceNet
                start_time = time.time()
                facenet_embedding = self.facenet.generateFaceEmbedding(face)
                facenet_time = time.time() - start_time
                
                self.timing_results['facenet'].append({
                    'image': os.path.basename(image_path),
                    'face_id': i,
                    'time': facenet_time
                })
                
                if facenet_embedding is not None:
                    print(f"  FaceNet - Tiempo: {facenet_time:.4f} segundos")
                else:
                    print(f"  FaceNet - ERROR: No se generó embedding")
                
                # Generar embedding con DeepFace (ahora devuelve tupla)
                deepface_embedding, deepface_inference_time = self.deepface.generateFaceEmbedding(face)
                
                self.timing_results['deepface'].append({
                    'image': os.path.basename(image_path),
                    'face_id': i,
                    'inference_time': deepface_inference_time
                })
                
                if deepface_embedding is not None:
                    print(f"  DeepFace - Tiempo de inferencia: {deepface_inference_time:.4f} segundos")
                else:
                    print(f"  DeepFace - ERROR: No se generó embedding")
                
                # Guardar embeddings
                image_key = f"{os.path.basename(image_path)}_face_{i}"
                
                if arcface_embedding is not None:
                    self.embeddings_map['arcface'].append({
                        'key': image_key,
                        'embedding': arcface_embedding.tolist() if hasattr(arcface_embedding, 'tolist') else list(arcface_embedding)
                    })
                
                if facenet_embedding is not None:
                    self.embeddings_map['facenet'].append({
                        'key': image_key,
                        'embedding': facenet_embedding.tolist() if hasattr(facenet_embedding, 'tolist') else list(facenet_embedding)
                    })
                
                if deepface_embedding is not None:
                    self.embeddings_map['deepface'].append({
                        'key': image_key,
                        'embedding': deepface_embedding.tolist() if hasattr(deepface_embedding, 'tolist') else list(deepface_embedding)
                    })
        
        else:
            print("No se encontraron caras en esta imagen")
    
    def run_tests(self):
        """Ejecutar todos los tests"""
        print("=== Iniciando Test de Modelos ===")
        print(f"Carpeta de imágenes: {self.image_folder}\n")
        
        image_files = self.load_images()
        
        if not image_files:
            print("No se encontraron imágenes para procesar")
            return
        
        # Procesar cada imagen
        for image_file in image_files:
            self.process_image(image_file)
        
        # Mostrar resumen
        self.print_summary()
        
        # Guardar resultados
        self.save_results()
    
    def print_summary(self):
        """Imprimir resumen de resultados"""
        print("\n" + "="*50)
        print("RESUMEN DE RESULTADOS")
        print("="*50)
        
        # Resumen de detección
        if self.timing_results['detection']:
            avg_detection = sum(r['time'] for r in self.timing_results['detection']) / len(self.timing_results['detection'])
            total_faces = sum(r['faces_found'] for r in self.timing_results['detection'])
            print(f"\nDetección (RetinaFace):")
            print(f"  Tiempo promedio: {avg_detection:.4f} segundos")
            print(f"  Total de caras detectadas: {total_faces}")
        
        # Resumen de ArcFace
        if self.timing_results['arcface']:
            # Filtrar solo los exitosos
            arcface_success = [r for r in self.timing_results['arcface'] if r['inference_time'] > 0 and r['inference_time'] < 1.0]
            if arcface_success:
                avg_arcface = sum(r['inference_time'] for r in arcface_success) / len(arcface_success)
                print(f"\nArcFace:")
                print(f"  Tiempo promedio de inferencia: {avg_arcface:.4f} segundos")
                print(f"  Éxitos: {len(arcface_success)} de {len(self.timing_results['arcface'])}")
                print(f"  Total de embeddings generados: {len(self.embeddings_map['arcface'])}")
            else:
                print(f"\nArcFace: No se generaron embeddings exitosos")
        
        # Resumen de FaceNet
        if self.timing_results['facenet']:
            facenet_success = [r for r in self.timing_results['facenet'] if r['time'] > 0]
            if facenet_success:
                avg_facenet = sum(r['time'] for r in facenet_success) / len(facenet_success)
                print(f"\nFaceNet:")
                print(f"  Tiempo promedio por cara: {avg_facenet:.4f} segundos")
                print(f"  Éxitos: {len(facenet_success)} de {len(self.timing_results['facenet'])}")
                print(f"  Total de embeddings generados: {len(self.embeddings_map['facenet'])}")
            else:
                print(f"\nFaceNet: No se generaron embeddings exitosos")
        
        # Resumen de DeepFace
        if self.timing_results['deepface']:
            deepface_success = [r for r in self.timing_results['deepface'] if r['inference_time'] > 0 and r['inference_time'] < 10.0]
            if deepface_success:
                avg_deepface = sum(r['inference_time'] for r in deepface_success) / len(deepface_success)
                print(f"\nDeepFace:")
                print(f"  Tiempo promedio de inferencia: {avg_deepface:.4f} segundos")
                print(f"  Éxitos: {len(deepface_success)} de {len(self.timing_results['deepface'])}")
                print(f"  Total de embeddings generados: {len(self.embeddings_map['deepface'])}")
            else:
                print(f"\nDeepFace: No se generaron embeddings exitosos")
    
    def save_results(self):
        """Guardar resultados en archivos"""
        # Crear carpeta de resultados
        results_folder = 'test_results'
        os.makedirs(results_folder, exist_ok=True)
        
        # Guardar timings
        with open(os.path.join(results_folder, 'timing_results.json'), 'w') as f:
            json.dump(self.timing_results, f, indent=4)
        
        # Guardar embeddings
        with open(os.path.join(results_folder, 'embeddings_map.json'), 'w') as f:
            json.dump(dict(self.embeddings_map), f, indent=4)
        
        print(f"\nResultados guardados en '{results_folder}/'")
        
        # Guardar resumen en texto
        with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
            f.write("RESUMEN DE TEST DE MODELOS\n")
            f.write("="*50 + "\n\n")
            
            # Escribir tiempos promedio
            for model_name in ['detection', 'arcface', 'facenet', 'deepface']:
                results = self.timing_results[model_name]
                if results:
                    if model_name == 'detection':
                        avg_time = sum(r['time'] for r in results) / len(results)
                        f.write(f"Detection (RetinaFace):\n")
                        f.write(f"  Tiempo promedio: {avg_time:.4f} segundos\n")
                        f.write(f"  Total de operaciones: {len(results)}\n\n")
                    else:
                        time_key = 'inference_time' if model_name in ['arcface', 'deepface'] else 'time'
                        success = [r for r in results if r[time_key] > 0 and r[time_key] < 10.0]
                        if success:
                            avg_time = sum(r[time_key] for r in success) / len(success)
                            f.write(f"{model_name.capitalize()}:\n")
                            f.write(f"  Tiempo promedio de inferencia: {avg_time:.4f} segundos\n")
                            f.write(f"  Éxitos: {len(success)} de {len(results)}\n\n")
                        else:
                            f.write(f"{model_name.capitalize()}: No se generaron embeddings exitosos\n\n")
            
            # Escribir cantidad de embeddings
            f.write("\nEmbeddings generados por modelo:\n")
            for model, embeddings in self.embeddings_map.items():
                f.write(f"  {model}: {len(embeddings)} embeddings\n")

# Ejecución del script
if __name__ == "__main__":
    import sys
    
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(parent_dir)  # Subir un nivel más
    image_folder = os.path.join(parent_dir, "caras_buena_definicion")
    
    # Verificar si la carpeta existe
    if not os.path.exists(image_folder):
        print(f"Error: La carpeta '{image_folder}' no existe.")
        print("Carpetas disponibles en el directorio actual:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        sys.exit(1)
    
    # Crear y ejecutar el tester
    tester = TestTimeDifferentModels(image_folder)
    tester.run_tests()