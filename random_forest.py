from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class RandomForest:
    def __init__(self, embeddings_map):
        """
        Inicializa el clasificador Random Forest
        
        Args:
            embeddings_map: Dict {nombre: list de listas de embeddings}
        """
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.train(embeddings_map)
    
    def train(self, embeddings_map):
        """
        Entrena el modelo con los embeddings proporcionados
        
        Args:
            embeddings_map: Dict {nombre: list de listas de embeddings}
        """
        X = []  # Lista de todos los embeddings
        y = []  # Lista de labels correspondientes
        
        # Concatenar todos los embeddings y sus labels
        for nombre, listas_embeddings in embeddings_map.items():
            for embedding_list in listas_embeddings:
                X.append(embedding_list)
                y.append(nombre)
        
        # Convertir a numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Entrenar el modelo
        self.rf.fit(X, y_encoded)
        
        print(f"Modelo entrenado con {len(X)} embeddings de {len(set(y))} personas")
    
    def detect(self, embedding):
        """
        Detecta y retorna las 3 personas más probables con sus confianzas
        
        Args:
            embedding: Lista o array con un embedding
            
        Returns:
            dict: Mapa con las 3 personas y sus confianzas
        """
        # Convertir embedding a formato correcto
        embedding = np.array(embedding).reshape(1, -1)
        
        # Obtener probabilidades
        probabilities = self.rf.predict_proba(embedding)[0]
        
        # Obtener índices de las 3 probabilidades más altas
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # Crear mapa de resultados
        result = {}
        
        for i, idx in enumerate(top_3_indices):
            person_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            result[person_name] = confidence
        
        return result

