import os
import json
import numpy as np
from collections import defaultdict

class ReconocedorFacial:
    def __init__(self, embeddings_por_modelo):
        self.embeddings_por_modelo = embeddings_por_modelo
        self.modelo_actual = None
    
    def usar_modelo(self, nombre_modelo):
        if nombre_modelo not in self.embeddings_por_modelo:
            raise ValueError(f"Modelo '{nombre_modelo}' no encontrado.")
        self.modelo_actual = nombre_modelo
    
    def distancia_euclidiana(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)
    
    def calcular_similitud(self, embedding, ref_embedding):
        """Calcula la similitud entre dos embeddings (1 - distancia normalizada)"""
        dist = self.distancia_euclidiana(embedding, ref_embedding)
        # Convertir distancia a similitud (0-1 donde 1 es idéntico)
        similitud = np.exp(-dist / 10)  # El divisor controla la pendiente de la transformación
        return similitud


def cargar_embeddings(ruta):
    """Carga los embeddings desde un archivo JSON"""
    print(f"Cargando embeddings desde {ruta}...")
    try:
        with open(ruta, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error al cargar embeddings: {e}")
        return None


def obtener_nombre_imagen(indice, persona):
    """Genera un nombre para la imagen basado en su índice"""
    return f"foto{indice+1}"


def calcular_similitudes_por_persona(embeddings_ref, embeddings_test):
    """
    Calcula la similitud entre cada foto de prueba y cada foto de referencia
    
    Args:
        embeddings_ref: Embeddings de referencia
        embeddings_test: Embeddings de prueba
        
    Returns:
        dict: Similitudes organizadas por persona, modelo y persona_ref
    """
    # Inicializar resultados
    resultados = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Modelos disponibles
    modelos_disponibles = sorted(set(embeddings_ref.keys()).intersection(embeddings_test.keys()))
    print(f"Modelos disponibles: {modelos_disponibles}")
    
    # Inicializar reconocedor
    reconocedor = ReconocedorFacial(embeddings_ref)
    
    # Para cada modelo
    for modelo in modelos_disponibles:
        print(f"\nCalculando similitudes para modelo: {modelo}")
        reconocedor.usar_modelo(modelo)
        
        # Para cada persona en test
        for persona_test, embeddings_test_persona in embeddings_test[modelo].items():
            print(f"  Procesando persona de prueba: {persona_test}")
            
            # Para cada persona en referencia
            for persona_ref, embeddings_ref_persona in embeddings_ref[modelo].items():
                print(f"    Comparando con fotos de referencia de: {persona_ref}")
                
                # Clave para las fotos de referencia
                clave_ref = f"fotos_{persona_ref}"
                
                # Para cada índice de embedding de prueba (por imagen)
                for i_test, embedding_test in enumerate(embeddings_test_persona):
                    nombre_foto_test = obtener_nombre_imagen(i_test, persona_test)
                    
                    # Para cada índice de embedding de referencia (por imagen)
                    for i_ref, embedding_ref_list in enumerate(embeddings_ref_persona):
                        # Puede haber múltiples embeddings por imagen de referencia
                        # Tomamos la similitud máxima encontrada para esta imagen
                        max_similitud = 0
                        
                        for embedding_ref in embedding_ref_list:
                            similitud = reconocedor.calcular_similitud(embedding_test, embedding_ref)
                            max_similitud = max(max_similitud, similitud)
                        
                        # Guardar el resultado para esta foto de referencia
                        nombre_foto_ref = obtener_nombre_imagen(i_ref, persona_ref)
                        resultados[persona_test][modelo][clave_ref][nombre_foto_ref] = max_similitud
    
    return resultados


def main():
    # Rutas a los embeddings
    ruta_ref = 'embeddings_results/all_embeddings_map.json'
    ruta_test = 'test_results/all_embeddings_map.json'
    
    # Cargar embeddings
    embeddings_ref = cargar_embeddings(ruta_ref)
    embeddings_test = cargar_embeddings(ruta_test)
    
    if not embeddings_ref or not embeddings_test:
        print("Error al cargar embeddings. Verificar rutas.")
        return
    
    # Calcular similitudes
    resultados = calcular_similitudes_por_persona(embeddings_ref, embeddings_test)
    
    # Convertir defaultdict a dict regular para JSON
    resultados_dict = {}
    for persona, modelos in resultados.items():
        resultados_dict[persona] = {}
        for modelo, personas_ref in modelos.items():
            resultados_dict[persona][modelo] = {}
            for persona_ref, similitudes in personas_ref.items():
                resultados_dict[persona][modelo][persona_ref] = dict(similitudes)
    
    # Guardar resultados
    ruta_salida = 'similitudes_detalladas.json'
    with open(ruta_salida, 'w') as f:
        json.dump(resultados_dict, f, indent=2)
    
    print(f"\nResultados guardados en {ruta_salida}")
    
    # Mostrar un resumen de los resultados
    print("\n=== RESUMEN DE SIMILITUDES ===")
    for persona_test, modelos in resultados_dict.items():
        print(f"\nPersona de prueba: {persona_test}")
        for modelo, personas_ref in modelos.items():
            print(f"  Modelo: {modelo}")
            for persona_ref, similitudes in personas_ref.items():
                print(f"    Fotos de referencia de {persona_ref}:")
                # Ordenar similitudes de mayor a menor
                similitudes_ordenadas = sorted(similitudes.items(), key=lambda x: x[1], reverse=True)
                for foto_ref, similitud in similitudes_ordenadas[:3]:  # Mostrar las 3 mejores
                    print(f"      {foto_ref}: {similitud:.4f}")


if __name__ == "__main__":
    main()