"""
Artificial Neural Network with Backpropagation for Fingerprint Recognition
Implements forward pass and backpropagation from scratch
"""

import numpy as np
import pickle
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    learning_rate: float = 0.0005
    epochs: int = 100
    batch_size: int = 16
    momentum: float = 0.9
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    validation_split: float = 0.2

class ActivationFunctions:
    """Activation functions and their derivatives"""
    
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
    """Single layer in the neural network"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """Initialize layer with Xavier/He initialization"""
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # He initialization for ReLU, Xavier for others
        if activation == 'relu':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        
        self.bias = np.zeros((1, output_size))
        
        # For momentum
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)
        
        # Cache for backpropagation
        self.input_cache = None
        self.output_cache = None
        self.z_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        self.input_cache = x
        self.z_cache = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.output_cache = ActivationFunctions.relu(self.z_cache)
        elif self.activation == 'sigmoid':
            self.output_cache = ActivationFunctions.sigmoid(self.z_cache)
        elif self.activation == 'softmax':
            self.output_cache = ActivationFunctions.softmax(self.z_cache)
        else:  # linear
            self.output_cache = self.z_cache
        
        return self.output_cache
    
    def backward(self, grad_output: np.ndarray, learning_rate: float, 
                 momentum: float = 0.9, weight_decay: float = 0.0) -> np.ndarray:
        """Backward pass - compute gradients and update weights"""
        batch_size = grad_output.shape[0]
        
        # Apply activation derivative
        if self.activation == 'relu':
            grad_z = grad_output * ActivationFunctions.relu_derivative(self.z_cache)
        elif self.activation == 'sigmoid':
            grad_z = grad_output * ActivationFunctions.sigmoid_derivative(self.z_cache)
        else:  # linear or softmax (softmax derivative handled in loss)
            grad_z = grad_output
        
        # Compute gradients
        grad_weights = np.dot(self.input_cache.T, grad_z) / batch_size
        grad_bias = np.sum(grad_z, axis=0, keepdims=True) / batch_size
        grad_input = np.dot(grad_z, self.weights.T)
        
        # Add L2 regularization
        if weight_decay > 0:
            grad_weights += weight_decay * self.weights
        
        # Update weights with momentum
        self.velocity_w = momentum * self.velocity_w - learning_rate * grad_weights
        self.velocity_b = momentum * self.velocity_b - learning_rate * grad_bias
        
        self.weights += self.velocity_w
        self.bias += self.velocity_b
        
        return grad_input

class FingerprintClassifier:
    """Complete ANN implementation with backpropagation for fingerprint recognition"""
    
    def __init__(self, input_dim: int = 1764, hidden_layers: List[int] = [512, 256], 
                 output_dim: int = 10):
        """
        Initialize the neural network
        
        Args:
            input_dim: HOG feature vector size (1764)
            hidden_layers: List of hidden layer sizes [512, 256]
            output_dim: Number of subjects/classes (10)
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.layers = []
        
        # Build network architecture
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:
                # Hidden layers use ReLU
                activation = 'relu'
            else:
                # Output layer uses softmax
                activation = 'softmax'
            
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger = logging.getLogger('FingerprintANN')
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the entire network"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                 learning_rate: float, momentum: float, weight_decay: float) -> float:
        """
        Backward pass - backpropagation through the network
        
        Args:
            X: Input features
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            learning_rate: Learning rate
            momentum: Momentum coefficient
            weight_decay: L2 regularization coefficient
        
        Returns:
            loss: Cross-entropy loss
        """
        batch_size = X.shape[0]
        
        # Compute cross-entropy loss
        epsilon = 1e-7  # Small value to prevent log(0)
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / batch_size
        
        # Add L2 regularization to loss
        if weight_decay > 0:
            l2_loss = 0
            for layer in self.layers:
                l2_loss += np.sum(layer.weights ** 2)
            loss += 0.5 * weight_decay * l2_loss
        
        # Gradient of loss w.r.t softmax output
        grad = (y_pred - y_true) / batch_size
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate, momentum, weight_decay)
        
        return loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              config: Optional[TrainingConfig] = None) -> dict:
        """
        Train the neural network using backpropagation
        
        Args:
            X_train: Training features (n_samples, 1764)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            config: Training configuration
        
        Returns:
            history: Training history
        """
        if config is None:
            config = TrainingConfig()
        
        # Convert labels to one-hot encoding
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
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            epoch_correct = 0
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass (weight updates happen here)
                batch_loss = self.backward(
                    X_batch, y_batch, y_pred,
                    config.learning_rate, config.momentum, config.weight_decay
                )
                
                epoch_loss += batch_loss * (end_idx - start_idx)
                epoch_correct += np.sum(np.argmax(y_pred, axis=1) == 
                                       np.argmax(y_batch, axis=1))
            
            # Calculate epoch metrics
            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
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
            
            # Learning rate decay
            if (epoch + 1) % 20 == 0:
                config.learning_rate *= 0.9
                self.logger.info(f"Learning rate decreased to {config.learning_rate:.6f}")
        
        return self.history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on given data"""
        y_pred = self.forward(X)
        y_onehot = self._to_onehot(y, self.output_dim)
        
        loss = -np.sum(y_onehot * np.log(y_pred + 1e-7)) / X.shape[0]
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        
        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[float, int]:
        """
        Predict class and confidence for input features
        
        Returns:
            confidence: Maximum probability
            class_id: Predicted class
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        y_pred = self.forward(X)
        class_id = np.argmax(y_pred, axis=1)[0]
        confidence = np.max(y_pred)
        
        return confidence, class_id
    
    def _to_onehot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert labels to one-hot encoding"""
        n_samples = y.shape[0]
        onehot = np.zeros((n_samples, num_classes))
        onehot[np.arange(n_samples), y] = 1
        return onehot
    
    def save_model(self, filepath: str):
        """Save model weights and architecture"""
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
        """Load model weights"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        for i, (weights, bias) in enumerate(model_data['weights']):
            self.layers[i].weights = weights
            self.layers[i].bias = bias
        
        self.history = model_data.get('history', {})
        self.logger.info(f"Model loaded from {filepath}")