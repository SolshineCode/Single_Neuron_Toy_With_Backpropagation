## **Single Neuron Toy Model with Backpropagation**

This script implements a simple neuron model using NumPy. It demonstrates how a neuron learns through forward propagation, error computation, and backpropagation. It’s an accessible way to explore neural networks and mechanistic interpretability without complex frameworks. (Code is handwritten, some code comments along with most of the documentation are LLM generated.)

### **Features**
- **Forward Propagation**: Calculates neuron output using weights, bias, and a sigmoid activation function.
- **Backpropagation**: Updates weights and bias based on Mean Squared Error (MSE).
- **Custom Parameters**: Allows adjustment of learning rate, epochs, and initial parameters.
- **Test Cases**: Includes examples to validate the model's behavior.
- **Training Visualization**: Includes a version of the script where matplotlib (must be installed) is used to show visualizations of the training process, further illustrating the concept for mech. intrep. purposes.

---

### **Function Overview**

#### `train_neuron()`
Trains a single neuron using labeled input data.

**Parameters**:
- `features` (`np.ndarray`): Input data, shape (n_samples, n_features).
- `labels` (`np.ndarray`): True labels, shape (n_samples,).
- `initial_weights` (`list` or `np.ndarray`): Initial weights, length = n_features.
- `initial_bias` (`float`): Initial bias value.
- `learning_rate` (`float`): Step size for gradient descent.
- `epochs` (`int`): Number of iterations for training.

**Returns**:
- `weights` (`list`): Final weights after training.
- `bias` (`float`): Final bias after training.
- `mse_values` (`list`): MSE for each epoch.

---

### **Usage**

```python
import numpy as np
from neuronWithBackprop import train_neuron

# Example data
features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
labels = np.array([1, 0, 0])

# Train the neuron
final_weights, final_bias, mse_values = train_neuron(
    features=features,
    labels=labels,
    initial_weights=[0.1, -0.2],
    initial_bias=0.0,
    learning_rate=0.1,
    epochs=10
)

print("Final Weights:", final_weights)
print("Final Bias:", final_bias)
print("MSE History:", mse_values)
```

---

### **Test Cases**
The script includes the following test cases:
1. **Standard Inputs**: Validates functionality with typical inputs.
2. **Non-Array Inputs**: Demonstrates automatic array conversion.
3. **Extended Epochs**: Illustrates learning over multiple epochs.

Example from the script:
```python
updated_weights, updated_bias, mse_values = train_neuron(
    features=[[2.0, 2.0], [2.0, 2.0], [-1.0, -2.0]],
    labels=[1, 1, 0],
    initial_weights=[0.3, -0.2],
    initial_bias=0.0,
    learning_rate=0.1,
    epochs=6
)
print(f"Updated Weights: {updated_weights}")
print(f"Updated Bias: {updated_bias}")
print(f"MSE Values: {mse_values}")
```
