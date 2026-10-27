# Neural Networks and Deep Learning: Course Assignments

Programming assignments from the Coursera / deeplearning.ai course "Neural Networks and Deep Learning" (course 1 of the Deep Learning Specialization), exported from Jupyter notebooks to Python scripts. Neural networks implemented from scratch in NumPy.

Year: 2020 · Stack: Python, NumPy, h5py, matplotlib, scikit-learn · Context: online course (Coursera, deeplearning.ai) taken during university studies; the completion certificate is dated 2020-04-27

## What's inside

All files are in `course/`, in the order of the course weeks:

- `LogisticRegressionWithaNeuralNetworkMindset.py` - logistic regression as a single-neuron network, a cat / non-cat image classifier: `sigmoid`, `initialize_with_zeros`, `propagate` (forward and backward pass), `optimize` (gradient descent), `predict`, and a `model` function that ties them together.
- `PlanarDataClassificationWithOneHiddenLayer.py` - a 2-class "flower" planar dataset. First scikit-learn logistic regression as a baseline, then a hand-written network with one hidden layer (4 units): `layer_sizes`, `initialize_parameters`, `forward_propagation`, `compute_cost`, `backward_propagation`, `update_parameters`, `nn_model`, `predict`.
- `BuildingYourDeepNeuralNetworkStepbyStep.py` - the building blocks of an L-layer network: `initialize_parameters_deep`, `linear_forward`, `linear_activation_forward` (ReLU / sigmoid), `L_model_forward`, `compute_cost`, `linear_backward`, `linear_activation_backward`, `L_model_backward`, `update_parameters`.
- `DeepNeuralNetworkApplication.py` - uses those blocks for cat vs non-cat classification: a `two_layer_model` and an `L_layer_model` with layers `[12288, 20, 7, 5, 1]` (64x64x3 images flattened to 12288 inputs).
- `sertificateNN.pdf` - the course completion certificate.

## Interesting bits

- `BuildingYourDeepNeuralNetworkStepbyStep.py`: forward propagation keeps a `cache` per layer and `L_model_backward` walks the layers in reverse, so backpropagation is written out by hand without any framework.
- `PlanarDataClassificationWithOneHiddenLayer.py`: the logistic regression baseline is compared to the hidden-layer network on the same non-linearly-separable data, using `plot_decision_boundary`.
- `DeepNeuralNetworkApplication.py`: the same dataset is trained with a 2-layer and a 4-layer model, with the learning rate `0.0075` and 2500-3000 iterations; a comment in the file records an earlier try with learning rate `0.009`.

## Run

Not re-tested here. The scripts are notebook exports: they call `get_ipython()` magics and import course-provided helper modules (`lr_utils`, `planar_utils`, `testCases_v2`, `testCases_v4a`, `dnn_utils_v2`, `dnn_app_utils_v3`) and `.h5` datasets that are not in this repository. To run them, copy the course notebook files next to them, then in Jupyter / IPython:

```
pip install numpy h5py matplotlib scikit-learn scipy pillow
ipython course/LogisticRegressionWithaNeuralNetworkMindset.py
```

## Note

Written in 2020, before AI coding assistants. Kept as-is for history.
