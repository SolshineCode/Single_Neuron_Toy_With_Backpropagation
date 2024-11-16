import math
import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	
	# Make sure features and labels are arrays
	features = np.array(features)
	labels = np.array(labels)
 
	#Neuron inference:
	def inference(weights, features, bias):
		#weights dot features
		weightfeatures = np.dot(features,weights)
		#plus bias
		weightfeaturesbiased = np.add(weightfeatures,bias)
		#sigmoid it
		generation = 1 / (1 + np.exp(-weightfeaturesbiased))
		return generation
	
	#set neuron parameters ahead of time to intials, pretraining
	weights = np.array(initial_weights)
	bias = np.array(initial_bias)
	
	#BackProp:
 	# Here we could divide the dataset into batches
 
	mse_values = []  # mse empty list initialization

	# Loop through each epoch
	while epochs > 0:
		#Run inference
		generation = inference(weights, features, bias)  # Store the output of inference
		#Calculate MSE
		Errorrate = np.subtract(generation, labels)
		MSE = np.square(Errorrate).mean()
		mse_values.append(MSE)  # Append MSE to the list
  
		# Backprop modify weights and bias
		learning_rate = 0.01  # Define a learning rate
		# Calculate gradients
		gradients = np.dot(features.T, (generation - labels)) / len(labels)  # Gradient for weights
		weights -= learning_rate * gradients  # Update weights
		bias -= learning_rate * np.mean(generation - labels)  # Update bias
		epochs = epochs - 1
  
	return weights, bias, mse_values
     
		
	updated_weights = weights
	updated_bias = bias
	
	updated_weights = np.around(updated_weights, decimals = 4).tolist()
	updated_bias = np.around(updated_bias, decimals = 4).tolist()
	#Take MSE array out into list
	mse_values = np.around(mse_values, decimals = 4).tolist()

	return updated_weights, updated_bias, mse_values


# Test case
updated_weights, updated_bias, mse_values = train_neuron(
    features=np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), 
    labels=np.array([1, 0, 0]),
    initial_weights=[0.1, -0.2],
    initial_bias=0.0,
    learning_rate=0.1,
    epochs=2
)
print("Test case 1 passed!")
print(f"Updated weights: {updated_weights}")
print(f"Updated biases: {updated_bias}")
print(f"MSE values: {mse_values}")


# Test case without arraying the inputs first 
updated_weights, updated_bias, mse_values = train_neuron(
    features=[[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], 
    labels=[1, 0, 0],
    initial_weights=[0.1, -0.2],
    initial_bias=0.0,
    learning_rate=0.1,
    epochs=2
)
print("Test case 2 passed!")
print(f"Updated weights: {updated_weights}")
print(f"Updated biases: {updated_bias}")
print(f"MSE values: {mse_values}")


# Test case without arraying the inputs first 
updated_weights, updated_bias, mse_values = train_neuron(
    features=[[2.0, 2.0], [2.0, 2.0], [-1.0, -2.0]], 
    labels=[1, 1, 0],
    initial_weights=[0.3, -0.2],
    initial_bias=0.0,
    learning_rate=0.1,
    epochs=6
)
print("Test case 3 passed!")
print(f"Updated weights: {updated_weights}")
print(f"Updated biases: {updated_bias}")
print(f"MSE values: {mse_values}")