import os, random, string
import operator
import numpy as np
from math import log2
from frozendict import OrderedDict
from functools import reduce
import threading
BASE_DIR = os.path.join(os.path.dirname(__file__), "data_files")
COG160 = os.path.join(BASE_DIR, "COG160.fasta")
COG1 = os.path.join(BASE_DIR, "COG1.fasta")
COG161 = os.path.join(BASE_DIR, "COG161.fasta")
SUB_SEQ_FILE = os.path.join(BASE_DIR, "best_seq{}.txt")
MODEL_FILE = os.path.join(BASE_DIR, "model_file{}.txt")
OBJECTIVE_FUNCTION_FILE = os.path.join(BASE_DIR, "objective_function{}.txt")
K = 3
SPLITS = 6
TOTAL_SPLIT_ITERATIONS = 25
TOTAL_ITERATIONS = 50


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

SEQUENCES = cog_get_data(COG160) + cog_get_data(COG161) + cog_get_data(COG1)

def prod(factors):
    return reduce(operator.mul, factors, 1)


def _split(seqs, n):
    k, m = divmod(len(seqs), n)
    return (seqs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

########################################################################################
# Add in some function that reads in the whichever data set
# we are going to use here and returns that dataset as a list of sequences
########################################################################################

class GibbsCalculculator(object):

    def __init__(self, sequences=SEQUENCES, background_model=None, current_model=None,
                 k=K, total_split_iterations=TOTAL_SPLIT_ITERATIONS, total_iterations=TOTAL_ITERATIONS,
                 identifier=""):
        self.data_seq = sequences
        self.background_probs = background_model if background_model \
            else self._get_background_probabilities()
        self.k = k
        indices = self._get_random_indices() if not current_model \
            else self.get_random_indices_from_model(current_model)
        self.dict_of_seq_indices_pairs = OrderedDict(zip(self.data_seq, indices))
        self.current_model = current_model or self.get_model(self.sequences, self.indices)
        self.total_iterations = total_iterations
        self.total_split_iterations = total_split_iterations
        self.identifier = identifier
        if not current_model and self.identifier:
            open(OBJECTIVE_FUNCTION_FILE.format("{}_{}".format(self.k, self.identifier)), "w").close()

    def run_split_iteration(self, splits=SPLITS):

        total_set = self._get_all_chars()
        iteration = 0
        last_model = None
        while (not self.compare_models(self.current_model, last_model)) and iteration < self.total_split_iterations:
            combined_model = [([0] * self.k) for _ in range(len(total_set))]
            sequences_to_split = self.sequences
            random.shuffle(sequences_to_split)
            seq_sets = _split(sequences_to_split, splits)
            calculators = []
            threads = []

            for seqs in seq_sets:
                new_sub_calculator = GibbsCalculculator(seqs, self.background_probs,
                                                        self.current_model if last_model else None,
                                                        k=self.k, total_iterations=self.total_iterations)
                calculators.append(new_sub_calculator)
                t = threading.Thread(target=lambda gc: gc.run_sample(write_result=False), args=(new_sub_calculator, ))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()
            models = [c.current_model for c in calculators]

            for model in models:
                for i in range(len(combined_model)):
                    for j in range(len(combined_model[0])):
                        combined_model[i][j] += model[i][j]

            for i in range(len(combined_model)):
                for j in range(len(combined_model[0])):
                    combined_model[i][j] = combined_model[i][j] / splits
            last_model = self.current_model
            self.current_model = combined_model
            self.write_objective_func()
            self.write_max_prob_sequences()
            self.write_model()
            iteration += 1

##################################################################################
    def calculate_objective_function(self):
        """
        TODO: Use self.current_model, self.sequences and self.indices to maximise the objective function
        # self.current_model is a 4 * 20 matrix with each position being the likelihood of the kmer appearing
        # at that corresponding position in
        """
        probs_from_random = []
        for seq in self.max_prob_sequences:
            probs_from_random.append(
                prod([self.background_probs[character] for character in seq]))
        return sum([log2(maxx / rand) for maxx, rand in zip(self.max_probs, probs_from_random)])
####################################################################################
    @property
    def indices(self):
        return list(self.dict_of_seq_indices_pairs.values())

    @property
    def sequences(self):
        return list(self.dict_of_seq_indices_pairs.keys())

    @property
    def max_probs(self):
        probs = []
        for seq in self.sequences:
            dist = self.get_distribution(seq, self.current_model)
            probs.append(max(dist))

        return probs

    @property
    def max_prob_sequences(self):
        seqs = []
        for seq, prob in zip(self.sequences, self.max_probs):
            i = self.get_distribution(seq, self.current_model).index(prob)
            seqs.append(seq[i: i + self.k])

        return seqs

    def get_random_indices_from_model(self, model):
        indices = []
        for seq in self.data_seq:
            distributions = self.get_distribution(seq, model)
            index_choice = np.random.choice(list(range(len(distributions))), 1, p=distributions)
            indices.append(index_choice[0])
        return indices

    def _get_background_probabilities(self):
        string_set = "".join(self.data_seq)
        return {c: (string_set.count(c) + 1) / (len(string_set) + 20) for c in set(list(string_set))}

    def _get_random_indices(self):
        random_indices = []
        for seq in self.data_seq:
            random_indices.append(random.choice(range(len(seq) - self.k + 1)))
        return random_indices

    def _get_all_chars(self):
        return sorted(list(self._get_background_probabilities().keys()),
                      key=lambda c: string.ascii_uppercase.index(c))

    def get_model(self, seqs, indices):
        total_set = self._get_all_chars()
        model = [([0] * self.k) for _ in range(len(total_set))]
        subsequences = [cog_seq[index:index + self.k] for cog_seq, index in zip(seqs, indices)]
        for i in range(self.k):
            all_at_index = [s[i] for s in subsequences]
            for j in range(len(total_set)):
                model[j][i] = (all_at_index.count(total_set[j]) + 1) / (len(subsequences) + len(total_set))

        return model

    def run_sample(self, write_result=True):

        last_model = None
        iteration = 0
        while (not self.compare_models(self.current_model, last_model)) and iteration < self.total_iterations:
            for i in range(len(self.indices)):
                model = self.get_model(self.sequences[:i] + self.sequences[i + 1:], self.indices[:i] + self.indices[i + 1:])
                distributions = self.get_distribution(self.sequences[i], model)
                index_choice = np.random.choice(list(range(len(distributions))), 1, p=distributions)
                self.dict_of_seq_indices_pairs[self.sequences[i]] = index_choice[0]
            last_model = self.current_model
            self.current_model = self.get_model(self.sequences, self.indices)
            if write_result:
                self.write_objective_func()
                self.write_max_prob_sequences()
                self.write_model()
            iteration += 1

    def get_distribution(self, sequence, model):

        num_possible = len(sequence) - self.k + 1
        background_set = self._get_all_chars()

        all_sub = []
        for i in range(num_possible):
            all_sub.append(sequence[i:i + self.k])
        probabilities = []
        for sub in all_sub:
            model_generated = []
            random_generated = []
            for index in range(len(sub)):
                c_index = background_set.index(sub[index])
                random_generated.append(self.background_probs[sub[index]])
                model_generated.append(model[c_index][index])
            probabilities.append(prod(model_generated) / prod(random_generated))
        return [prob / sum(probabilities) for prob in probabilities]

    @staticmethod
    def compare_models(current_model, last_model=None, error=.01):

        if last_model:
            for i in range(len(current_model)):
                for j in range(len(current_model[0])):
                    if abs(current_model[i][j] - last_model[i][j]) > error:
                        return False

        return bool(last_model)

    def print_max_prob_sequences(self):
        print("\n")
        for seq in self.max_prob_sequences:
            print(seq)
        print("\n")

    def write_max_prob_sequences(self, seq_file=SUB_SEQ_FILE):
        seq_file = seq_file.format("{}_{}".format(self.k, self.identifier))
        with open(seq_file, "w+") as fh:
            for seq in self.max_prob_sequences:
                fh.write("{}\n".format(seq))

    def write_model(self, model_file=MODEL_FILE):
        model_file = model_file.format("{}_{}".format(self.k, self.identifier))
        with open(model_file, "w+") as fh:
            for probs in self.current_model:
                fh.write("{}\n".format(probs))

    def write_objective_func(self, obj_file=OBJECTIVE_FUNCTION_FILE):
        obj_file = obj_file.format("{}_{}".format(self.k, self.identifier))
        with open(obj_file, "a+") as fh:
            fh.write("{}\n".format(self.calculate_objective_function()))


if __name__ == "__main__":

    for k in range(3, 7):
        for i in range(5):
            gc = GibbsCalculculator(identifier="{}SPLIT".format(i), k=k)
            gc.run_split_iteration()

        for i in range(5):
            gc = GibbsCalculculator(identifier="{}STANDARD".format(i), k=k, total_iterations=150)
            gc.run_sample()

