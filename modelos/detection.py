from abc import ABC, abstractmethod
import cv2
import numpy as np

class Detection(ABC):
    """Interfaz abstracta para detección facial"""
    
    @abstractmethod
    def makeRecognition(self, imagen):
        """
        Método abstracto para extraer caras de una imagen
        
        Args:
            imagen: Imagen en formato numpy array o string path
            
        Returns:
            Lista de imágenes de caras o None si no encuentra ninguna
        """
        pass
    
    def _load_imagen(self, imagen):
        """Helper method para cargar imagen si se pasa un path"""
        if isinstance(imagen, str):
            return cv2.imread(imagen)
        return imagen