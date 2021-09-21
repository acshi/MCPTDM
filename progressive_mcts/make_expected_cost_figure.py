#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

dy = -12
initial_dx = 40  # to the left and the right
example_dx = initial_dx * 2
branching_factor = 2
max_depth = 4
circle_r = 6.5

plt.gcf().set_figwidth(6.4 * 3)
plt.gcf().set_figheight(6.4 / 2 * 3)
plt.gca().set_aspect(1)


def mean_or_zero(vals):
    if len(vals) == 0:
        return 0
    return np.sum(vals) / len(vals)


class Node:
    def __init__(self, true_marginal_cost, observed_marginal_costs, children=[]):
        self.depth = 0
        self.true_marginal_cost = true_marginal_cost
        self.true_intermediate_cost = 0
        self.marginal_costs = observed_marginal_costs
        self.marginal_cost = mean_or_zero(observed_marginal_costs)
        self.intermediate_costs = self.marginal_costs
        self.intermediate_cost = self.marginal_cost
        self.expected_cost = None
        self.chosen_child_i = None
        self.children = children

        self.calculate_all()

    def calculate_all(self, parent=None, parent_intermediate_costs=[]):
        if parent:
            self.depth = parent.depth + 1

            self.intermediate_costs = [parent_intermediate_costs[i] +
                                       m for (i, m) in enumerate(self.marginal_costs)]
            self.intermediate_cost = mean_or_zero(self.intermediate_costs)
            # self.marginal_cost = self.intermediate_cost - parent.intermediate_cost
            self.true_intermediate_cost = parent.true_intermediate_cost + self.true_marginal_cost
        else:
            self.depth = 0
            self.intermediate_costs = self.marginal_costs
            self.intermediate_cost = self.marginal_cost
            # self.marginal_cost = self.intermediate_cost
            self.true_intermediate_cost = self.marginal_cost

        parent_costs_i = 0
        for child in self.children:
            child_costs_n = len(child.marginal_costs)
            start_i = parent_costs_i
            end_i = start_i + child_costs_n
            child.calculate_all(self, self.intermediate_costs[start_i:end_i])
            parent_costs_i = end_i

        if len(self.children) == 0:
            self.costs = [self.intermediate_cost]
        else:
            self.costs = list(chain.from_iterable(c.costs for c in self.children))

    def mean_cost(self):
        return np.sum(self.costs) / len(self.costs)

    def min_child_expected_cost(self):
        if len(self.children) == 0:
            return None
        else:
            best_child_i = np.argmin([c.expected_cost for c in self.children])
            expected_cost = self.children[best_child_i].expected_cost
            return (best_child_i, expected_cost)


tree = Node(0.0, [0.0] * 8, children=[
    Node(20.0, [0.0, 40.0, 0.0, 0.0], children=[
        Node(2.0, [2.0, 2.0]),
        Node(4.0, [0.0, 4.0]),
    ]),
    Node(4.0, [3.0, 5.0, 2.0, 2.0], children=[
        Node(21.0, [19.0, 23.0]),
        Node(3.0, [3.0, 3.0]),
    ])
])


def draw_level(node, depth, start_x, start_y, in_best_path):
    plt.gca().add_artist(plt.Circle((start_x, start_y), circle_r, fill=True,
                                    zorder=100, edgecolor="black", facecolor="white", clip_on=False))
    display_str = str(node.display).replace(".0", "")
    fontsize = 16.0 if len(display_str) > 4 else 20.0
    plt.text(start_x, start_y, display_str, fontsize=fontsize,
             zorder=101, horizontalalignment='center', verticalalignment='center')

    dx = initial_dx / branching_factor**depth
    for (child_i, child) in enumerate(node.children):
        end_x = start_x + dx / (branching_factor - 1) * child_i - 0.5 * dx
        end_y = start_y + dy

        is_best_child = child_i == node.chosen_child_i
        child_best_path = in_best_path and is_best_child
        linewidth = 8 if child_best_path else 1

        plt.plot([start_x, end_x], [start_y, end_y], '-', color="black", linewidth=linewidth)

        draw_level(child, depth + 1, end_x, end_y, child_best_path)


def calculate_expected_costs_and_display(name, node):
    for child in node.children:
        calculate_expected_costs_and_display(name, child)
    if name == "Classic expected-cost":
        node.expected_cost = node.mean_cost()
        if len(node.children) > 0:
            node.chosen_child_i = node.min_child_expected_cost()[0]
        node.display = node.expected_cost
    elif name == "Expectimax expected-cost":
        (node.chosen_child_i, node.expected_cost) = node.min_child_expected_cost() or (None, node.mean_cost())
        node.display = node.expected_cost
    elif name == "Lower-bound expected-cost":
        (chosen_child_i, expected_cost) = node.min_child_expected_cost() or (None, 0)
        if node.intermediate_cost > expected_cost:
            (node.chosen_child_i, node.expected_cost) = (None, node.intermediate_cost)
        else:
            (node.chosen_child_i, node.expected_cost) = (chosen_child_i, expected_cost)
        node.display = node.expected_cost
    elif name == "Marginal expected-cost":
        (node.chosen_child_i, node.expected_cost) = node.min_child_expected_cost() or (None, 0)
        node.expected_cost += node.marginal_cost
        node.display = node.expected_cost
    elif name == "True marginal and intermediate costs":
        node.display = f"M = {node.true_marginal_cost}\nI = {node.true_intermediate_cost}"
    elif name == "True marginal costs":
        node.display = node.true_marginal_cost
    elif name == "True intermediate costs":
        node.display = node.true_intermediate_cost
    elif name == "Sampled intermediate costs":
        if node.depth == 0:
            node.display = ""
        else:
            # node.display = ", ".join(str(c) for c in node.intermediate_costs)
            node.display = ""
            for i in range(0, len(node.intermediate_costs), 2):
                if i > 0:
                    node.display += ",\n"
                node.display += str(node.intermediate_costs[i]) + ", "
                node.display += str(node.intermediate_costs[i + 1])
            node.display += "\nx̄ = " + str(node.intermediate_cost)
    else:
        node.display = 42


def draw_example(name, start_x, start_y, letter):
    plt.text(start_x - initial_dx * 0.8, start_y, letter, fontsize=30.0,
             zorder=101, horizontalalignment='center', verticalalignment='center')
    plt.text(start_x, start_y + 10, name, fontsize=20.0,
             zorder=101, horizontalalignment='center', verticalalignment='center')
    calculate_expected_costs_and_display(name, tree)
    draw_level(tree, 0, start_x, start_y, True)


set_dy = dy * 2 - 30

start_x = 0
start_y = 0
letter = "A"
draw_example("True marginal and intermediate costs", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)
# draw_example("True marginal costs", start_x, start_y, letter)
# start_x += example_dx
# letter = chr(ord(letter) + 1)
# draw_example("True intermediate costs", start_x, start_y, letter)
# start_x += example_dx
# letter = chr(ord(letter) + 1)
draw_example("Sampled intermediate costs", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)
draw_example("Classic expected-cost", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)


start_x = 0
start_y += set_dy
draw_example("Expectimax expected-cost", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)

# start_x = 0
# start_y += set_dy
draw_example("Lower-bound expected-cost", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)
draw_example("Marginal expected-cost", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)

plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(f"figures/pdf/expected_cost_comparison.pdf",
            bbox_inches="tight", pad_inches=0)
plt.savefig(f"figures/expected_cost_comparison.png")
