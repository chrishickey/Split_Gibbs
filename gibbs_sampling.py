import os, random, string
import operator
import numpy as np
from functools import reduce
BASE_DIR = os.path.join(os.path.dirname(__file__), "data_files")
COG160 = os.path.join(BASE_DIR, "COG160.fasta")
K = 4

def prod(factors):
    return reduce(operator.mul, factors, 1)

def cog_get_data(cog_file):

    full_file = ""
    with open(cog_file, "r") as fr:
        full_file = fr.read()

    data_output = []
    for data in full_file.split(">"):
        if not data:
            continue
        _, value = data.split("\n", 1)
        value = value.replace("X", "")
        data_output.append(value.replace("\n", ""))
    return data_output


cog160 = cog_get_data(COG160)


def get_background_probabilities(dataset=cog160):
    string_set = "".join(dataset)
    return {c: (string_set.count(c) + 1) / (len(string_set) + 20) for c in set(list(string_set))}


def get_random_indices():
    random_indices = []
    for cog in cog160:
        random_indices.append(random.choice(range(len(cog) - K + 1)))
    return random_indices


def get_all_chars():
    return sorted(list(get_background_probabilities(cog160).keys()), key=lambda c: string.ascii_uppercase.index(c))


def get_model(cog, indices, k):
    total_set = get_all_chars()
    model =[([0] * k) for _ in range(len(total_set))]
    subsequences = [cog_seq[index:index + k] for cog_seq, index in zip(cog, indices)]
    for i in range(k):
        all_at_index = [s[i] for s in subsequences]
        for j in range(len(total_set)):
            model[j][i] = (all_at_index.count(total_set[j]) + 1) / (len(subsequences) + len(total_set))

    return model


def get_distribution(sequence, model, k=K):

    num_possible = len(sequence) - k + 1
    background_set = get_all_chars()
    background_probabilities = get_background_probabilities()
    all_sub = []
    for i in range(num_possible):
        all_sub.append(sequence[i:i + k])
    probabilities = []
    for sub in all_sub:
        model_generated = []
        random_generated = []
        for index in range(len(sub)):
            c_index = background_set.index(sub[index])
            random_generated.append(background_probabilities[sub[index]])
            model_generated.append(model[c_index][index])
        probabilities.append(prod(model_generated) / prod(random_generated))
    return [prob / sum(probabilities) for prob in probabilities]


def compare_models(current_model, last_model=None, error=.01):

    if last_model:
        for i in range(len(current_model)):
            for j in range(len(current_model[0])):
                if abs(current_model[i][j] - last_model[i][j]) > error:
                    return False

    return bool(last_model)


def run_sample(cog160, random_indices, k=K, max_iterations=200):

    current_model = get_model(cog160, random_indices, k)
    last_model = None
    iteration = 0
    while (not compare_models(current_model, last_model)) and iteration < max_iterations:
        for i in range(len(random_indices)):

            model = get_model(cog160[:i] + cog160[i+1:], random_indices[:i] + random_indices[i+1:], k)
            distributions = get_distribution(cog160[i], model)
            index_choice = np.random.choice(list(range(len(distributions))), 1, p=distributions)
            random_indices[i] = index_choice[0]
        last_model = current_model
        current_model = get_model(cog160, random_indices, k)
        iteration += 1
        print(iteration)

    return current_model, random_indices


if __name__ == "__main__":

    cm, rm = run_sample(cog160, get_random_indices())
    for seq, index in zip(cog160, rm):
        print(seq[index: index + K])




