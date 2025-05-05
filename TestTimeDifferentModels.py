import os
import cv2
import numpy as np
import time
from pathlib import Path
import json
from collections import defaultdict

# Importar los modelos
from retinaface_detection import RetinaFaceDetection
from arcface_reconocimiento import ArcFaceReconocimiento
from facenet_reconocimiento import FaceNetReconocimiento

class TestTimeDifferentModels:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.detector = RetinaFaceDetection()
        self.arcface = ArcFaceReconocimiento()
        self.facenet = FaceNetReconocimiento()
        
        # Diccionario para almacenar resultados
        self.embeddings_map = defaultdict(list)
        self.timing_results = {
            'detection': [],
            'arcface': [],
            'facenet': []
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
                
                # Generar embedding con ArcFace
                start_time = time.time()
                arcface_embedding = self.arcface.generateFaceEmbedding(face)
                arcface_time = time.time() - start_time
                
                self.timing_results['arcface'].append({
                    'image': os.path.basename(image_path),
                    'face_id': i,
                    'time': arcface_time
                })
                
                # Generar embedding con FaceNet
                start_time = time.time()
                facenet_embedding = self.facenet.generateFaceEmbedding(face)
                facenet_time = time.time() - start_time
                
                self.timing_results['facenet'].append({
                    'image': os.path.basename(image_path),
                    'face_id': i,
                    'time': facenet_time
                })
                
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
            avg_arcface = sum(r['time'] for r in self.timing_results['arcface']) / len(self.timing_results['arcface'])
            print(f"\nArcFace:")
            print(f"  Tiempo promedio por cara: {avg_arcface:.4f} segundos")
            print(f"  Total de embeddings generados: {len(self.embeddings_map['arcface'])}")
        
        # Resumen de FaceNet
        if self.timing_results['facenet']:
            avg_facenet = sum(r['time'] for r in self.timing_results['facenet']) / len(self.timing_results['facenet'])
            print(f"\nFaceNet:")
            print(f"  Tiempo promedio por cara: {avg_facenet:.4f} segundos")
            print(f"  Total de embeddings generados: {len(self.embeddings_map['facenet'])}")
    
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
            for model, results in self.timing_results.items():
                if results:
                    avg_time = sum(r['time'] for r in results) / len(results)
                    f.write(f"{model.capitalize()}:\n")
                    f.write(f"  Tiempo promedio: {avg_time:.4f} segundos\n")
                    f.write(f"  Total de operaciones: {len(results)}\n\n")
            
            # Escribir cantidad de embeddings
            f.write("\nEmbeddings generados por modelo:\n")
            for model, embeddings in self.embeddings_map.items():
                f.write(f"  {model}: {len(embeddings)} embeddings\n")

# Función main para ejecutar el test
def main():
    # Especifica aquí la ruta a tu carpeta de imágenes
    image_folder = "./caras_buena_definicion_taylor"  # Cambia esto por tu ruta
    
    # Crear el tester
    tester = TestTimeDifferentModels(image_folder)
    
    # Ejecutar los tests
    tester.run_tests()

if __name__ == "__main__":
    # También puedes especificar la carpeta como argumento de línea de comandos
    import sys
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:
        image_folder = "images"  # Carpeta por defecto
    
    tester = TestTimeDifferentModels(image_folder)
    tester.run_tests()