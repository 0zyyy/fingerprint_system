
import numpy as np
import pickle
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate: float = 0.0005
    epochs: int = 100
    batch_size: int = 16
    momentum: float = 0.9
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    validation_split: float = 0.2

class ActivationFunctions:
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

class Layer:
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        if activation == 'relu':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        
        self.bias = np.zeros((1, output_size))
        
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)
        
        self.input_cache = None
        self.output_cache = None
        self.z_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        self.z_cache = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.output_cache = ActivationFunctions.relu(self.z_cache)
        elif self.activation == 'sigmoid':
            self.output_cache = ActivationFunctions.sigmoid(self.z_cache)
        elif self.activation == 'softmax':
            self.output_cache = ActivationFunctions.softmax(self.z_cache)
        else:
            self.output_cache = self.z_cache
        
        return self.output_cache
    
    def backward(self, grad_output: np.ndarray, learning_rate: float, 
                 momentum: float = 0.9, weight_decay: float = 0.0) -> np.ndarray:
        batch_size = grad_output.shape[0]
        
        if self.activation == 'relu':
            grad_z = grad_output * ActivationFunctions.relu_derivative(self.z_cache)
        elif self.activation == 'sigmoid':
            grad_z = grad_output * ActivationFunctions.sigmoid_derivative(self.z_cache)
        else:
            grad_z = grad_output
        
        grad_weights = np.dot(self.input_cache.T, grad_z) / batch_size
        grad_bias = np.sum(grad_z, axis=0, keepdims=True) / batch_size
        grad_input = np.dot(grad_z, self.weights.T)
        
        if weight_decay > 0:
            grad_weights += weight_decay * self.weights
        
        self.velocity_w = momentum * self.velocity_w - learning_rate * grad_weights
        self.velocity_b = momentum * self.velocity_b - learning_rate * grad_bias
        
        self.weights += self.velocity_w
        self.bias += self.velocity_b
        
        return grad_input

class FingerprintClassifier:
    
    def __init__(self, input_dim: int = 1764, hidden_layers: List[int] = [512, 256], 
                 output_dim: int = 10):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.layers = []
        
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                activation = 'relu'
            else:
                activation = 'softmax'
            
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger = logging.getLogger('FingerprintANN')
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                 learning_rate: float, momentum: float, weight_decay: float) -> float:
        batch_size = X.shape[0]
        
        epsilon = 1e-7
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / batch_size
        
        if weight_decay > 0:
            l2_loss = 0
            for layer in self.layers:
                l2_loss += np.sum(layer.weights ** 2)
            loss += 0.5 * weight_decay * l2_loss
        
        grad = (y_pred - y_true) / batch_size
        
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate, momentum, weight_decay)
        
        return loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              config: Optional[TrainingConfig] = None) -> dict:
        if config is None:
            config = TrainingConfig()
        
        y_train_onehot = self._to_onehot(y_train, self.output_dim)
        if X_val is not None and y_val is not None:
            y_val_onehot = self._to_onehot(y_val, self.output_dim)
        
        n_samples = X_train.shape[0]
        n_batches = (n_samples + config.batch_size - 1) // config.batch_size
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting training: {config.epochs} epochs, "
                        f"batch_size={config.batch_size}, lr={config.learning_rate}")
        
        for epoch in range(config.epochs):
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            epoch_correct = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                y_pred = self.forward(X_batch)
                
                
                batch_loss = self.backward(
                    X_batch, y_batch, y_pred,
                    config.learning_rate, config.momentum, config.weight_decay
                )
                
                epoch_loss += batch_loss * (end_idx - start_idx)
                epoch_correct += np.sum(np.argmax(y_pred, axis=1) == 
                                       np.argmax(y_batch, axis=1))
            
            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('models/best_model.pkl')
                else:
                    patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                self.logger.info(f"Epoch {epoch + 1}/{config.epochs} - "
                               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                self.logger.info(f"Epoch {epoch + 1}/{config.epochs} - "
                               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            if (epoch + 1) % 20 == 0:
                config.learning_rate *= 0.9
                self.logger.info(f"Learning rate decreased to {config.learning_rate:.6f}")
        
        return self.history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        y_pred = self.forward(X)
        y_onehot = self._to_onehot(y, self.output_dim)
        
        loss = -np.sum(y_onehot * np.log(y_pred + 1e-7)) / X.shape[0]
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        
        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[float, int]:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        y_pred = self.forward(X)
        class_id = np.argmax(y_pred, axis=1)[0]
        confidence = np.max(y_pred)
        
        return confidence, class_id
    
    def _to_onehot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        n_samples = y.shape[0]
        onehot = np.zeros((n_samples, num_classes))
        onehot[np.arange(n_samples), y] = 1
        return onehot
    
    def save_model(self, filepath: str):
        model_data = {
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'output_dim': self.output_dim
            },
            'weights': [(layer.weights, layer.bias) for layer in self.layers],
            'history': self.history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        for i, (weights, bias) in enumerate(model_data['weights']):
            self.layers[i].weights = weights
            self.layers[i].bias = bias
        
        self.history = model_data.get('history', {})
        self.logger.info(f"Model loaded from {filepath}")