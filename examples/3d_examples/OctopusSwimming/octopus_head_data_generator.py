import numpy as np

"""for head only from Cathy"""
"""[[x,y,z],r]"""
body_data = [
    [[-0.10000, 0.000000, -0.10000], 0.05000],
    [[0.0000, 0.010000, 0.00000], 0.1000],
    [[0.05, 0.02, 0.05000], 0.13000],
    [[0.1, 0.05, 0.1500], 0.15000],
    [[0.15, 0.06, 0.2000], 0.10000],
    [[0.2, 0.08, 0.250], 0.050000],
]

element_positions = np.zeros((3, 6))
radius = np.zeros((6))

for i, data in enumerate(body_data):
    element_positions[:, i] = data[0]
    radius[i] = data[1]

tangents = element_positions[:, 1:] - element_positions[:, :-1]
lengths = np.linalg.norm(tangents, axis=0, keepdims=True)[0]
tangents /= lengths
base_length = lengths.sum()

non_dimensional_lengths = lengths.cumsum()

interior_node_position = 0.5 * (element_positions[:, 1:] + element_positions[:, :-1])
positions = np.zeros((3, 7))
# Fill exterior nodes
positions[:, 0] = element_positions[:, 0] - tangents[:, 0] * lengths[0] / 2
positions[:, -1] = element_positions[:, -1] + tangents[:, -1] * lengths[-1] / 2
# Fill interior nodes
positions[:, 1:-1] = interior_node_position[:, :]

element_tangents = tangents.copy()
tangents = positions[:, 1:] - positions[:, :-1]
lengths = np.linalg.norm(tangents, axis=0, keepdims=True)[0]
tangents /= lengths
base_length = lengths.sum()
non_dimensional_lengths = np.hstack((0, lengths.cumsum())) / base_length

# Data we need to create head
slender_ratio = base_length / radius[0] / 2
non_dimensional_radius = np.hstack((radius / radius[0], radius[-1] / radius[0] / 2))
n_elem = tangents.shape[-1]
# tangents as well.

np.savez(
    "octopus_head_data.npz",
    slenderness_ratio=slender_ratio,
    non_dimensional_radius=non_dimensional_radius,
    non_dimensional_lengths=non_dimensional_lengths,
)
