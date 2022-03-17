import itertools
import random
import torch
import pandas as pd
from tqdm import tqdm


class ItemSet:
    """
        A class containing all possible samples of the Hierarchical Reference Game.
        The properties are given by the object dictionary and are stored as a number.

        Terminology:
        sender_objects: the sender's input object (without relevance vector)
        sample: it contains the sender_object, plus target and distractors for the receiver
        items:  the final output which will be passed to the game:
                contains sender_input, referential label and receiver_input

        Items can be accessed like this:
        symbolic_set = SymbolicSet(...) # Or ImageSet
        symbolic_set[i]     ... returns the ith sample (important for pytorch train loader)
        symoblic_set[i,j]   ... returns the item with the ith sender object and the jth relevance vector
    """
    def __init__(self, properties_dim, predefined_relevance_vectors,
                 intentional_distractor_size=10,
                 random_distractor_size=0,
                 sample_scaling=10,
                 upsample=True,
                 balanced_distractors=False):
        """
        Note: game size is intentional_distractors_size + random_distractor_size + 1 (for the target)
        :param intentional_distractor_size :The number of distractors based on alternative relevance_vectors
        :param random_distractor_size: The number of distractors sampled randomly
        :param sample_scaling: Indicates how many different items are there for the very same sender input.
        """

        self.properties_dim = properties_dim
        self.sample_scaling = sample_scaling
        if random_distractor_size: raise NotImplementedError
        self.intentional_distractor_size = intentional_distractor_size
        self.random_distractor_size = random_distractor_size

        self.objects = self._get_all_different_objects(properties_dim)
        if upsample and predefined_relevance_vectors:
            raise ValueError("cannot upsample with predefined relevance_vectors")
        if not upsample and balanced_distractors:
            raise ValueError("balanced distractors works only with upsample")

        self.upsample = upsample
        self.balanced_distractors = balanced_distractors

        if predefined_relevance_vectors:
            self.relevance_vectors = predefined_relevance_vectors
        else:
            # Method is called in subclass
            self.relevance_vectors = self._get_all_possible_relevance_vectors()

        self.summed_up_dim = sum(self.properties_dim)

    @staticmethod
    def _get_all_different_objects(properties_dim):
        """
        Returns all different combinations of the properties as a DataFrame
        """
        list_of_dim = [range(0, dim) for dim in properties_dim]
        # Each object is a row
        all_objects = list(itertools.product(*list_of_dim))

        return pd.DataFrame(all_objects)

    def _many_hot_encoding(self, pd_series):
        """
        Outputs a binary one dim vector
        """
        output = torch.zeros([self.summed_up_dim])
        start = 0

        for elem, dim in zip(pd_series, self.properties_dim):
            output[start + elem] = 1
            start += dim

        return output

    def get_datasets(self, split_ratio):
        """
        Note: Generates training and test data sets. The test set has different sender_objects than the training set.
        :param split_ratio Tuple of ratios (train, val, test) of the samples in train validation and test set
        """

        if sum(split_ratio) != 1:
            raise ValueError

        train_ratio, val_ratio, test_ratio = split_ratio

        # Shuffle sender indices
        object_indices = torch.randperm(len(self.objects)).tolist()
        ratio = int(len(self.objects)*(train_ratio + val_ratio))

        train_and_val = []
        print("Creating train_ds and val_ds...")
        for object_idx in tqdm(object_indices[:ratio]):
            for _ in range(self.sample_scaling):
                for relevance in self.relevance_vectors:
                    if self.upsample:
                        relevance = random.choice(relevance)
                    train_and_val.append(self.get_item(object_idx, relevance, self._many_hot_encoding))

        test = []
        print("\nCreating test_ds...")
        for object_idx in tqdm(object_indices[ratio:]):
            for _ in range(self.sample_scaling):
                for relevance in self.relevance_vectors:
                    if self.upsample:
                        relevance = random.choice(relevance)
                    test.append(self.get_item(object_idx, relevance, self._many_hot_encoding))

        # Calculating how many train
        train_samples = int(len(train_and_val)*(train_ratio/(train_ratio+val_ratio)))
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])

        # Write important information about the dataset
        train.dimensions = self.properties_dim
        train.n_intentional_distractors = self.intentional_distractor_size
        train.n_random_distractors = self.random_distractor_size
        train.sample_scaling = self.sample_scaling

        return train, val, test

    def get_zero_shot_datasets(self, split_ratio):
        """
        Note: Generates train, val and test data. The test data set has different sender_objects than the training set.
        :param split_ratio Tuple of ratios (train, val) of the samples should be in the training and validation sets.
        """

        if sum(split_ratio) != 1:
            raise ValueError

        # For each category, one attribute will be chosen for zero shot
        # The attributes will be taken from a random object
        zero_shot_object = pd.Series([0 for _ in self.properties_dim])  # self.objects.sample().iloc[0]

        train_ratio, val_ratio = split_ratio

        train_and_val = []
        test = []

        print("Creating train_ds and val_ds...")
        for object_idx in tqdm(range(len(self.objects))):
            # Checks if current object contain some of the zero shot attributes
            contains_zero_shot_attributes = zero_shot_object.eq(self.objects.iloc[object_idx])

            for _ in range(self.sample_scaling):
                for relevance in self.relevance_vectors:
                    if self.upsample:
                        relevance = random.choice(relevance)

                    # Cast relevance to boolean pd.Series
                    casted_relevance = pd.Series(relevance).astype(bool)

                    # If any is applied to zero shot attribute, add it to test ds
                    if (casted_relevance & contains_zero_shot_attributes).any():
                        test.append(self.get_item(object_idx, relevance, self._many_hot_encoding))
                    else:
                        train_and_val.append(self.get_item(object_idx, relevance, self._many_hot_encoding))

        # Train val split
        train_samples = int(len(train_and_val)*train_ratio)
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])

        # Write important information about the dataset
        train.dimensions = self.properties_dim
        train.n_intentional_distractors = self.intentional_distractor_size
        train.n_random_distractors = self.random_distractor_size
        train.sample_scaling = self.sample_scaling

        print("Length of test ds:", len(test))

        return train, val, test

    def get_item(self, object_idx, relevance, encoding_func):

        # Get samples
        if self.balanced_distractors:
            sender_input, target, distractors = self.get_sample_with_balanced_distractors(object_idx, relevance)
        else:
            sender_input, target, distractors = self.get_sample(object_idx, relevance)

        # Generate label and receiver output by shuffling target and distractors
        receiver_input = distractors + [target]
        random.shuffle(receiver_input)
        # hack around list.index() because it doesnt work here
        label = [i for i, x in enumerate(receiver_input) if x.equals(target)][0]

        # Encode
        sender_input = encoding_func(sender_input)
        try:
            receiver_input = torch.stack([encoding_func(elem) for elem in receiver_input], dim=0)
        except TypeError:
            receiver_input = [encoding_func(elem) for elem in receiver_input]

        # Needs to have the structure sender_input, label, receiver_input
        try:
            return torch.cat([sender_input, torch.tensor(relevance, dtype=torch.float)]), label, receiver_input
        except TypeError:
            return (sender_input, relevance), label, receiver_input


class AnyItemSet(ItemSet):

    def __init__(self,
                 properties_dim,
                 predefined_relevance_vectors=None,
                 distractors=10,
                 random_distractors=0,
                 sample_scaling=10,
                 upsample=True,
                 balanced_distractors=False):
        ItemSet.__init__(self,
                         properties_dim,
                         predefined_relevance_vectors,
                         distractors,
                         random_distractors,
                         sample_scaling,
                         upsample,
                         balanced_distractors)

    def _get_all_possible_relevance_vectors(self):
        # Create all possible relevance_vectors (many-hot-encoded)
        list_of_dim = [range(0, 2) for dim in self.properties_dim]
        relevance_vectors = list(itertools.product(*list_of_dim))

        # Remove last relevance vector, because everything is irrelevant (1,1,1,..)
        relevance_vectors.pop()

        if not self.upsample:
            return relevance_vectors
        # Order the relevance_vectors by the amount of irrelevant attributes
        ordered_relevance_vectors = []

        for number_of_any in range(len(self.properties_dim)):
            grouped_relevance_vectors = []

            for intention in relevance_vectors:
                if sum(intention) == number_of_any:
                    grouped_relevance_vectors.append(intention)

            ordered_relevance_vectors.append(grouped_relevance_vectors)

        return ordered_relevance_vectors

    def get_sample_with_balanced_distractors(self, sender_object_idx, target_relevance):
        sender_object = self.objects.iloc[sender_object_idx]

        def change_to_any(object, relevance, different_value=False):
            for i, value in object.items():
                if relevance[i]:
                    new_value = random.randint(0, self.properties_dim[i]-1)

                    # Sample value until it is different if different value is True
                    if different_value:
                        while new_value == value:
                            new_value = random.randint(0, self.properties_dim[i]-1)
                    object.iloc[i] = new_value

        target = sender_object.copy()
        change_to_any(target, target_relevance)

        # Get indeces of relevant attributes (zeros)
        same_value_idx_list = []
        for i, v in enumerate(target_relevance):
            if not v:
                same_value_idx_list.append(i)

        distractors = []
        for _i in range(self.intentional_distractor_size):
            distractor = sender_object.copy()

            # sample relevance vectors by number of relevant attributes
            distractor_relevance = random.choice(self.relevance_vectors)
            distractor_relevance = random.choice(distractor_relevance)

            # replace any with different value
            change_to_any(distractor, distractor_relevance, different_value=True)

            # To ensure that the distractor is not possible target, one further attribute is changed
            idx = random.choice(same_value_idx_list)
            actual_value = sender_object.iloc[idx]
            while True:
                random_value = random.randint(0, self.properties_dim[idx]-1)
                if random_value != actual_value:
                    # Replace with another value
                    distractor.iloc[idx] = random_value
                    break

            distractors.append(distractor)

        return sender_object, target, distractors

    def get_sample(self, sender_object_idx, relevance):
        sender_object = self.objects.iloc[sender_object_idx]

        def change_to_any(object):
            for i, item in object.items():
                if relevance[i]:
                    object.iloc[i] = random.randint(0, self.properties_dim[i]-1)

        target = sender_object.copy()
        change_to_any(target)

        # Get zero indices of relevance vector
        same_value_idx_list = []
        for i, v in enumerate(relevance):
            if not v:
                same_value_idx_list.append(i)

        distractors = []
        for _i in range(self.intentional_distractor_size):
            distractor = sender_object.copy()
            change_to_any(distractor)

            # The attribute that should be changed
            idx = random.choice(same_value_idx_list)
            actual_value = distractor.iloc[idx]
            while True:
                random_value = random.randint(0, self.properties_dim[idx]-1)
                if random_value != actual_value:
                    # Replace with another value
                    distractor.iloc[idx] = random_value
                    break

            distractors.append(distractor)

        return sender_object, target, distractors


def turn_to_series(dataframe):
    """
    Takes all rows of a dataframe convert them to series and returns list of that serieses.
    """
    output = []

    for _, row in dataframe.iterrows():
        output.append(row)

    return output
