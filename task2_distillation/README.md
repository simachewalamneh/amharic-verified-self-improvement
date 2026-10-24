# Task 2: Knowledge distillation
Teacher (MLP, all gold train labels) -> student (linear, smaller vocabulary). Loss: alpha*CE + (1-alpha)*T^2*KL(teacher||student).
Sweep T in {1,2,4}, alpha in {0.3,0.5,0.7}; 3 seeds; (T, alpha) chosen on dev and saved to `results/task2/best.json`.
Reports macro-F1, ECE, parameters, size (KB), single-example CPU latency (forward pass, precomputed features).
Run: `python -m task2_distillation.run`.
