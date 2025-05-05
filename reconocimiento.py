from abc import ABC, abstractmethod
import time

class Reconocimiento(ABC):
    """Interfaz abstracta para reconocimiento facial"""
    
    @abstractmethod
    def generateFaceEmbedding(self, cara):
        """
        Método abstracto para generar embedding de una cara
        
        Args:
            cara: Imagen de una cara en formato numpy array
            
        Returns:
            Embedding de la cara o None si hay error
        """
        pass
    
    def _measure_time(self, cara):
        """Helper method para medir el tiempo de generación de embedding"""
        inicio = time.time()
        embedding = self.generateFaceEmbedding(cara)
        fin = time.time()
        tiempo = fin - inicio
        print(f"Tiempo de generación de embedding: {tiempo:.4f} segundos")
        return embedding